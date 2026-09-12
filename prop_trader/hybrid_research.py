"""Frozen-split TimesFM+TA calibration, reliability metrics and portfolio replay."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from peft import PeftModel
from transformers import TimesFm2_5ModelForPrediction
from .__main__ import load
from .forecast_time import SECONDS
from .timesfm_run import windows
from .evaluator import fewshot_forecast, horizon_metrics, momentum_forecast, write_predictions
from .technical import FEATURES, feature_frame, HybridCalibrator
from .downstream import RidgeHead, head_features, uncertainty_targets, volatility_targets
from .reliable_backtest import portfolio


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def evaluate(interval, source=Path('data/current'), root=Path('runs/forecast5'), out_root=Path('runs/hybrid')):
    # out_root defaults to the exact path predict_symbol() reads live calibrators from
    # (runs/hybrid/{interval}/calibrator.json) -- pass a different out_root for any ad-hoc/
    # research run (e.g. a different source or root) so it never overwrites the calibrator the
    # live trader is actively using. (Incident 2026-09-11: an ad-hoc Upbit-data comparison run
    # with default out_root silently clobbered the production 1h calibrator, blocking every live
    # 1h forecast with "Technical calibrator belongs to another model training snapshot" until
    # the correct calibrator was regenerated.)
    if root!=Path('runs/forecast5') and out_root==Path('runs/hybrid'):
        raise ValueError('root is non-default but out_root was left at the live path -- pass an explicit out_root')
    run=root/interval;out=out_root/interval;out.mkdir(parents=True,exist_ok=True)
    path=source/f'{interval}.csv'
    protocol=json.loads((run/'protocol.json').read_text())
    if digest(path)!=protocol['data_sha256']:
        raise ValueError('Data differ from original training split; use the frozen data/current snapshot')
    data=load(path);all_bars=defaultdict(list)
    for b in data:all_bars[b.symbol].append(b)
    splits=windows(path,protocol['context'],5,SECONDS[interval],True)
    validation,test=splits['validation'],splits['test']
    if max(r[2]['target_end'] for r in validation)>=min(r[2]['target_start'] for r in test):
        raise ValueError('Calibration/test label overlap')
    signature=dict(data_sha256=digest(path),adapter_sha256=digest(run/'adapter/adapter_model.safetensors'),
                   context=protocol['context'],horizon=5,shots=8,seed=42,features=FEATURES,
                   alpha=10.,split_ranges=protocol['split_ranges'],interval=interval)
    (out/'protocol.json').write_text(json.dumps(signature,indent=2))
    cache=out/'raw_predictions.npz';cache_meta=out/'cache_signature.json'
    if cache.exists() and cache_meta.exists() and json.loads(cache_meta.read_text())==signature:
        with np.load(cache) as cached:val_raw,test_raw=cached['validation'],cached['test']
    else:
        torch.set_num_threads(4);torch.manual_seed(42)
        device=protocol['device']
        base=TimesFm2_5ModelForPrediction.from_pretrained(protocol['model'],dtype=torch.float32,local_files_only=True).to(device)
        model=PeftModel.from_pretrained(base,run/'adapter',is_trainable=True).to(device).eval()
        initial={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        print(interval,'calibration forecasts',len(validation),flush=True)
        val_raw=fewshot_forecast(model,initial,validation,all_bars,protocol['context'],5,8,device)
        print(interval,'test forecasts',len(test),flush=True)
        test_raw=fewshot_forecast(model,initial,test,all_bars,protocol['context'],5,8,device)
        np.savez(cache,validation=val_raw,test=test_raw)
        cache_meta.write_text(json.dumps(signature,indent=2))
        del model,base
    frames={s:feature_frame(b) for s,b in all_bars.items()}
    def features(items):
        x=np.stack([frames[r[2]['symbol']].loc[r[2]['context_end']].to_numpy() for r in items])
        if not np.isfinite(x).all():raise ValueError('Undefined calibration/test indicators')
        return x
    vx,tx=features(validation),features(test)
    vy,ty=np.stack([r[1] for r in validation]),np.stack([r[1] for r in test])
    hybrid=HybridCalibrator().fit(val_raw,vx,vy)
    technical=HybridCalibrator().fit(np.zeros_like(val_raw),vx,vy)
    artifact=dict(model=hybrid.to_dict(),interval=interval,horizon=5,
        calibration_end=max(r[2]['target_end'] for r in validation),
        reference_ta_source='ta/ta (checked-in 0.11.0)',
        training_data_sha256=signature['data_sha256'],deployment_status='research_unverified')
    (out/'calibrator.json').write_text(json.dumps(artifact,indent=2))
    # Two more downstream heads, same validation-only fit and hash lock as the calibrator above,
    # but predicting conviction/risk sizing rather than correcting the price path -- see
    # downstream.py's module docstring.
    ema_gap_idx=FEATURES.index('ema_gap')
    trend_agree=np.sign(vx[:,ema_gap_idx])*np.sign(val_raw[:,-1])
    head_x=np.concatenate([val_raw,vx,trend_agree[:,None]],axis=1)
    delta=timedelta(seconds=SECONDS[interval])
    val_anchors=np.array([r[2]['anchor'] for r in validation])
    val_window_bars=[]
    for _,_,meta in validation:
        start=datetime.fromisoformat(meta['target_start'])
        by_ts={b.timestamp:b for b in all_bars[meta['symbol']]}
        val_window_bars.append([by_ts[(start+n*delta).isoformat()] for n in range(5)])
    uncertainty_head=RidgeHead().fit(head_x,uncertainty_targets(val_raw,vy))
    volatility_head=RidgeHead().fit(head_x,volatility_targets(val_anchors,val_window_bars))
    head_meta=dict(interval=interval,horizon=5,calibration_end=artifact['calibration_end'],
        training_data_sha256=signature['data_sha256'],feature_dim=head_x.shape[1],
        deployment_status='research_unverified')
    (out/'uncertainty_head.json').write_text(json.dumps(dict(model=uncertainty_head.to_dict(),**head_meta),indent=2))
    (out/'volatility_head.json').write_text(json.dumps(dict(model=volatility_head.to_dict(),**head_meta),indent=2))
    predictions=dict(naive=np.zeros_like(ty),momentum=np.stack([momentum_forecast(r[0],5) for r in test]),
        timesfm_fewshot=test_raw,technical_only=technical.predict(np.zeros_like(test_raw),tx),
        hybrid=hybrid.predict(test_raw,tx))
    write_predictions(out/'predictions.csv',test,predictions,5)
    grouped=defaultdict(list)
    with (out/'predictions.csv').open() as f:
        for row in csv.DictReader(f):grouped[row['model']].append(row)
    results={}
    for name,pred in predictions.items():
        metrics=horizon_metrics(pred,ty,np.array([r[2]['anchor'] for r in test]),has_direction=name!='naive')
        summary,trades,curve=portfolio(grouped[name],data,interval)
        write_csv(out/f'{name}_trades.csv',trades);write_csv(out/f'{name}_equity.csv',curve)
        results[name]=dict(forecast=metrics,portfolio=summary)
    stress,_,_=portfolio(grouped['hybrid'],data,interval,slippage=.003)
    results['hybrid_cost_stress']=dict(portfolio=stress)
    # Aggregate all symbols at each origin before paired block bootstrap.
    diffs=defaultdict(list)
    for i,(_,_,meta) in enumerate(test):
        diffs[meta['target_start']].append(float(np.abs(predictions['hybrid'][i]-ty[i]).mean()-np.abs(test_raw[i]-ty[i]).mean()))
    blocks=np.array([np.mean(v) for _,v in sorted(diffs.items())]);rng=np.random.default_rng(42)
    draws=[]
    for _ in range(1000):
        starts=rng.integers(len(blocks),size=(len(blocks)+1)//2)
        sample=np.concatenate([blocks[(np.arange(2)+s)%len(blocks)] for s in starts])[:len(blocks)]
        draws.append(sample.mean())
    reliability=dict(origin_groups=len(blocks),paired_mae_difference=float(blocks.mean()),
        paired_mae_difference_ci95=np.quantile(draws,[.025,.975]).tolist(),
        interpretation='Negative difference favors hybrid. Evaluation period was already inspected in prior research; this is a development regression test, not a new untouched holdout.',
        execution='Next candle open entry; 5th candle close exit; long-only; 20% capital/order; max 3 positions; fees+slippage each leg.',
        limitations=['small correlated sample','retrospective six-coin selection','no orderbook/market-impact model','no intrabar stop simulation'])
    (out/'results.json').write_text(json.dumps(dict(models=results,reliability=reliability),indent=2))
    print(interval,json.dumps({k:v['portfolio']['return_fraction'] for k,v in results.items()}),flush=True)
    return results,reliability


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--intervals',nargs='+',default=['1m','1h','1d'])
    args=parser.parse_args();lines=['# TimesFM + 기술적 지표 신뢰성 검증','',
        '기존 검증 구간에서만 보정 모델을 학습하고 기존 평가 구간에서 비교했습니다. 이 평가 구간은 이전 연구에서 확인한 데이터이므로 새로운 미관측 검증이라고 주장하지 않습니다.', '',
        '| 주기 | 모델 | 로그 MAE | DA@5 | 포트폴리오 순수익률 | MDD | 거래 수 |','|---|---|---:|---:|---:|---:|---:|']
    notes=[]
    for interval in args.intervals:
        results,reliability=evaluate(interval)
        for name,r in results.items():
            p=r['portfolio'];f=r.get('forecast')
            mae=f'{f["mae"]:.6f}' if f else 'N/A'
            da=f'{f["per_horizon"][-1]["direction_accuracy"]:.1%}' if f and name!='naive' else 'N/A'
            lines.append(f'| {interval} | {name} | {mae} | {da} | {p["return_fraction"]:.2%} | {p["max_drawdown"]:.2%} | {p["trades"]} |')
        notes+=['',f'{interval}: 시점 그룹 {reliability["origin_groups"]}개. Hybrid−TimesFM MAE 차이 95% 블록 부트스트랩 구간: {reliability["paired_mae_difference_ci95"]}. 음수일수록 Hybrid 우위.','']
    lines+=notes
    lines+=['','기본 비용은 편도 수수료 0.1%+슬리피지 0.1%, 스트레스는 슬리피지 0.3%입니다. 이는 실험 가정이며 업비트 계정의 실제 수수료가 아닙니다. 매수 예상 순수익률이 0.2%를 초과한 후보만 순위대로 진입하고 현금·동시 보유 한도를 적용했습니다. 숏은 없습니다. 최대 낙폭은 봉 시가 평가 기준이며 봉 내부 손실은 더 클 수 있습니다.',
        '', '기술적 지표는 RSI·MACD·ATR·Bollinger·EMA·ADX·CMF·거래량 비율·과거 수익률·봉 범위를 사용합니다. 미래 값으로 결측치를 채우지 않습니다. Hybrid는 TimesFM 내부의 native feature가 아니라 학습된 잔차 보정 계층입니다.',
        '', '비용 스트레스는 증가한 비용으로 진입 조건과 투자 수량도 다시 계산하므로 거래 구성과 거래 수가 달라집니다. 따라서 일부 구간에서는 오히려 수익률이 높아질 수 있으며, 고정 거래의 비용만 비교한 결과가 아닙니다.',
        '', '분류된 종목과 작은 구간의 결과만으로 신뢰성이 확보됐다고 볼 수 없습니다. 신뢰구간과 비용 스트레스, raw TimesFM·technical-only·naive·momentum 비교를 함께 확인하세요.']
    Path('runs/hybrid/REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()

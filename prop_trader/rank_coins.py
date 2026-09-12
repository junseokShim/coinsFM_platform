"""Rank same-horizon Upbit KRW forecasts by estimated net return, never place orders."""
import argparse
import csv
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path

import numpy as np
from .technical import HybridCalibrator, FEATURES, latest_features
from .forecast_time import SECONDS
from .__main__ import load


def buy_vwap(book,notional):
    if not math.isfinite(notional) or notional<=0:raise ValueError('Invalid notional')
    remain=notional;qty=0.
    units=sorted(book['orderbook_units'],key=lambda u:u['ask_price'])
    for unit in units:
        price,size=float(unit['ask_price']),float(unit['ask_size'])
        if not math.isfinite(price+size) or price<=0 or size<0:raise ValueError('Invalid orderbook')
        spend=min(remain,price*size);qty+=spend/price;remain-=spend
        if remain<=1e-7:return notional/qty
    raise ValueError('Insufficient visible ask liquidity')


def rank_records(records,snapshot,notional=10000.,fee=.001,exit_slippage=.001,now=None):
    if not 0<=fee<.1 or not 0<=exit_slippage<.1:raise ValueError('Invalid cost assumption')
    now=now or datetime.now(timezone.utc)
    intervals={r['interval'] for r in records}
    if len(intervals)!=1:raise ValueError('Rank only comparable horizons within one interval')
    if len({r['as_of'] for r in records})!=1:raise ValueError('Rank only forecasts from the same origin')
    rows=[]
    for record in records:
        symbol=record['symbol'];book=snapshot['orderbooks'][symbol]
        # notional is the complete cash budget, including the entry fee.
        entry=buy_vwap(book,notional/(1+fee))
        projected=float(record['hybrid_prices'][-1])
        if not math.isfinite(projected) or projected<=0:raise ValueError('Invalid forecast price')
        gross=projected/entry-1
        net=projected*(1-exit_slippage)*(1-fee)/(entry*(1+fee))-1
        expires=datetime.fromisoformat(record['as_of'])+timedelta(seconds=SECONDS[record['interval']])
        age=now.timestamp()-float(book['timestamp'])/1000
        stale=now>=expires or age>60 or age< -5
        quote_expiry=datetime.fromtimestamp(float(book['timestamp'])/1000,timezone.utc)+timedelta(seconds=60)
        uncertainty=record.get('predicted_uncertainty')
        rows.append(dict(symbol=symbol,exchange='upbit',quote='KRW',interval=record['interval'],horizon=5,
            as_of=record['as_of'],expected_gross_return=gross,expected_net_return=net,
            expected_pnl_krw=notional*net,notional_krw=notional,estimated_entry_vwap=entry,
            predicted_close_t5=projected,forecast_prices=record['hybrid_prices'],
            timesfm_prices=record['forecast_prices'],technical_features=record['technical_features'],
            fee_assumption=fee,exit_slippage_assumption=exit_slippage,stale=stale,
            orderbook_timestamp=book['timestamp'],valid_until=min(expires,quote_expiry).isoformat(),
            positive_after_costs=net>0,model_status='research_unverified_cross_exchange',
            predicted_uncertainty=uncertainty,predicted_stop_fraction=record.get('predicted_stop_fraction'),
            reason='Binance-trained adapter and calibration transferred to Upbit KRW; not validated for live execution'))
    # Conviction-adjusted: same downstream.UncertaintyHead penalty multiframe.decide() applies to
    # the live ranking (UNCERTAINTY_PENALTY there), so this dashboard table orders candidates the
    # same way live trading would rank them, not by raw point estimate alone. None (head not
    # trained yet) is treated as zero penalty -- unchanged from the original sort.
    from .multiframe import UNCERTAINTY_PENALTY
    rows.sort(key=lambda r:(-(r['expected_net_return']-UNCERTAINTY_PENALTY*(r['predicted_uncertainty'] or 0.)),r['symbol']))
    for i,r in enumerate(rows,1):r['rank']=i
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--interval',choices=['1m','1h','1d'],default='1h')
    p.add_argument('--notional',type=float,default=10000);p.add_argument('--fee',type=float,default=.001)
    p.add_argument('--exit-slippage',type=float,default=.001)
    p.add_argument('--refresh-orderbooks',action='store_true',help='Refresh public Upbit orderbooks after inference')
    args=p.parse_args()
    import torch
    from transformers import TimesFm2_5ModelForPrediction
    from .timesfm_predict import predict_symbol
    interval=args.interval;run=Path('runs/forecast5')/interval
    snapshot=json.loads(Path(f'data/upbit/{interval}_snapshot.json').read_text())
    protocol=json.loads((run/'protocol.json').read_text())
    artifact=json.loads(Path(f'runs/hybrid/{interval}/calibrator.json').read_text())
    if artifact['interval']!=interval or artifact['horizon']!=5:raise ValueError('Incompatible hybrid calibrator')
    if artifact['training_data_sha256']!=protocol['data_sha256']:raise ValueError('Stale calibrator from another training snapshot')
    calibrator=HybridCalibrator.from_dict(artifact['model'])
    candles=load(Path(f'data/upbit/{interval}.csv'))
    records=[];excluded=[]
    torch.manual_seed(42);torch.set_num_threads(4)
    base=TimesFm2_5ModelForPrediction.from_pretrained(protocol['model'],dtype=torch.float32,local_files_only=True).to(protocol['device'])
    for symbol in sorted({b.symbol for b in candles}):
        bars=[b for b in candles if b.symbol==symbol]
        if artifact['calibration_end']>=bars[-1].timestamp:raise ValueError('Calibration has future data relative to ranking origin')
        try:features=latest_features(bars)
        except ValueError as e:
            excluded.append(dict(symbol=symbol,reason=str(e)));continue
        rec,base=predict_symbol(base,run,protocol,bars,bars[-1].timestamp,8,protocol['device'],use_technical=False)
        raw=np.log(np.array(rec['forecast_prices'])/rec['last_close'])
        fused=calibrator.predict(raw[None,:],np.array([[features[k] for k in FEATURES]]))[0]
        rec['hybrid_prices']=(rec['last_close']*np.exp(fused)).tolist()
        rec['technical_features']=features;records.append(rec)
        print(symbol,'TimesFM + technical forecast complete',flush=True)
    if args.refresh_orderbooks:
        from .upbit_broker import UpbitClient
        books=UpbitClient().request('GET','/v1/orderbook',dict(markets=','.join(r['symbol'] for r in records)),private=False)
        snapshot['orderbooks']={b['market']:b for b in books}
    ranked=rank_records(records,snapshot,args.notional,args.fee,args.exit_slippage)
    out=Path('runs/rankings');out.mkdir(parents=True,exist_ok=True)
    (out/f'{interval}_forecast_records.json').write_text(json.dumps(records,indent=2))
    (out/f'{interval}_execution_snapshot.json').write_text(json.dumps(snapshot,indent=2))
    result=dict(generated_at=datetime.now(timezone.utc).isoformat(),interval=interval,
        sorting='expected_net_return descending; same 5-candle horizon only',rankings=ranked,excluded=excluded,
        note='Model estimates, not calibrated expected income. Fees/slippage are assumptions; stale rows cannot be treated as current candidates.')
    (out/f'{interval}.json').write_text(json.dumps(result,indent=2))
    columns=['rank','symbol','interval','as_of','expected_net_return','expected_pnl_krw','predicted_close_t5','stale','model_status']
    with (out/f'{interval}.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(ranked)
    lines=['# 업비트 예상 순수익률 순위','',
        f'주기 {interval}, 향후 5개 봉, 가정 투자금 {args.notional:,.0f} KRW. 수수료 편도 {args.fee:.2%}, 예상 매도 슬리피지 {args.exit_slippage:.2%}.', '',
        '| 순위 | 종목 | 예상 순수익률 | 예상 손익 KRW | 데이터 만료 |','|---:|---|---:|---:|---|']
    for r in ranked:lines.append(f'| {r["rank"]} | {r["symbol"]} | {r["expected_net_return"]:.2%} | {r["expected_pnl_krw"]:,.0f} | {r["stale"]} |')
    lines+=['','순위는 TimesFM+기술적 지표 모델 추정치입니다. 실현 수익이나 교정된 확률이 아닙니다. 업비트 호가로 진입 가능 가격을 계산했지만, Binance 학습 모델의 업비트 이전 성능은 미검증입니다. 전체 업비트가 아니라 현재 거래대금 상위에서 수집에 성공한 종목만 비교합니다. 주문을 자동 실행하지 않습니다.']
    (out/f'{interval}.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()

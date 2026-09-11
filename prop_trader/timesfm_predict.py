"""Load historical LoRA, adapt on K completed recent examples, forecast next horizon."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from peft import PeftModel
from transformers import TimesFm2_5ModelForPrediction
from .__main__ import load
from .forecast_time import forecast_points,path_stats,SECONDS


def support_windows(bars,context,horizon,shots):
    """Only fully observed targets; no artificial concatenation of examples."""
    if len(bars)<context+horizon*shots:raise ValueError('Insufficient recent history')
    windows=[]
    for end in range(len(bars)-horizon*(shots-1),len(bars)+1,horizon):
        start=end-horizon;anchor=bars[start-1].close
        x=[math.log(b.close/anchor) for b in bars[start-context:start]]
        y=[math.log(b.close/anchor) for b in bars[start:end]]
        windows.append((x,y))
    return windows


def predict_symbol(base,run_dir,protocol,bars,as_of,shots,device,use_technical=True,save_adapter=True):
    """Historical LoRA -> recent K-shot adaptation -> next-horizon close forecast for one symbol.

    `bars` must be sorted, gap-free, completed candles ending exactly at `as_of`
    (the OPEN timestamp of the last completed candle). The adapter loaded from
    `run_dir/adapter` is re-initialized per call so per-symbol few-shot updates
    never leak into another symbol's weights.
    """
    from datetime import datetime,timedelta,timezone
    context,horizon=protocol['context'],protocol['horizon']
    interval=protocol.get('interval','1h')
    delta=timedelta(seconds=SECONDS[interval])
    if not bars or not 1<=shots<=64:raise ValueError('Nonempty bars and shots 1..64 required')
    if bars[-1].timestamp!=as_of:raise ValueError('Missing last completed bar')
    if any(datetime.fromisoformat(b.timestamp)-datetime.fromisoformat(a.timestamp)!=delta for a,b in zip(bars,bars[1:])):
        raise ValueError('Gap or wrong timeframe')
    technical_features=None
    artifact=None
    calibrator_path=Path('runs/hybrid')/interval/'calibrator.json'
    if use_technical and calibrator_path.exists():
        from .technical import latest_features,HybridCalibrator,FEATURES
        artifact=json.loads(calibrator_path.read_text())
        if artifact['interval']!=interval or artifact['horizon']!=horizon:
            raise ValueError('Technical calibrator does not match forecast horizon')
        if artifact['calibration_end']>=as_of:
            raise ValueError('Technical calibration contains data after prediction cutoff')
        if artifact['training_data_sha256']!=protocol['data_sha256']:
            raise ValueError('Technical calibrator belongs to another model training snapshot')
        technical_features=latest_features(bars)
    model=PeftModel.from_pretrained(base,run_dir/'adapter',is_trainable=True).to(device)
    anchor=bars[-1].close
    recent=torch.tensor([[math.log(b.close/anchor) for b in bars[-context:]]],dtype=torch.float32,device=device)
    model.eval()
    with torch.no_grad():
        before=model(past_values=recent,forecast_context_len=context,truncate_negative=False).mean_predictions[0,:horizon].cpu().tolist()
    support=support_windows(bars,context,horizon,shots)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-5)
    model.train();losses=[]
    for offset in range(0,len(support),4):
        batch=support[offset:offset+4]
        x=torch.tensor([r[0] for r in batch],dtype=torch.float32,device=device)
        y=torch.tensor([r[1] for r in batch],dtype=torch.float32,device=device)
        optimizer.zero_grad()
        loss=model(past_values=x,future_values=y,forecast_context_len=context,truncate_negative=False).loss
        if not torch.isfinite(loss):raise ValueError('Non-finite few-shot loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        after_out=model(past_values=recent,forecast_context_len=context,truncate_negative=False)
        after=after_out.mean_predictions[0,:horizon].cpu().tolist()
        # TimesFM 2.5 natively predicts 9 quantiles (indices 1..9 of the last dim, config.quantiles
        # = 0.1..0.9; index 0 is a separate point-forecast channel, unused here) alongside the
        # point forecast -- no separate variance model needed for forecast uncertainty.
        quantile_log_returns=None
        if after_out.full_predictions is not None and after_out.full_predictions.shape[-1]>=10:
            quantile_log_returns=after_out.full_predictions[0,:horizon,1:10].cpu().tolist()
    if not all(math.isfinite(v) for v in before+after):raise ValueError('Non-finite forecast')
    as_of_close=datetime.fromisoformat(as_of)+delta
    prices=[anchor*math.exp(v) for v in after]
    raw_prices=list(prices)
    if artifact is not None:
        values=HybridCalibrator.from_dict(artifact['model']).predict(np.array([after]),
            np.array([[technical_features[k] for k in FEATURES]]))[0]
        prices=(anchor*np.exp(values)).tolist()
        if not all(math.isfinite(v) and v>0 for v in prices):raise ValueError('Invalid hybrid forecast')
    quantile_prices=None
    forecast_uncertainty=None
    if quantile_log_returns is not None and all(math.isfinite(v) for step in quantile_log_returns for v in step):
        # Shift each quantile step by the same calibration delta applied to the point forecast
        # (delta = calibrated - raw, in price space), so the reported band stays centered on
        # `prices` without pretending the calibrator itself was fit on quantile targets.
        delta=[c-r for c,r in zip(prices,raw_prices)]
        quantile_prices=[[anchor*math.exp(q)+d for q in step] for step,d in zip(quantile_log_returns,delta)]
        if all(math.isfinite(p) and p>0 for step in quantile_prices for p in step):
            q10,q90=quantile_prices[-1][0],quantile_prices[-1][-1]
            forecast_uncertainty=(q90-q10)/anchor
        else:
            quantile_prices=None
    points=forecast_points(as_of,interval,prices)
    stats=path_stats(anchor,prices)
    generated_at=datetime.now(timezone.utc).isoformat()
    record=dict(symbol=bars[-1].symbol,as_of=as_of_close.isoformat(),last_candle_open=as_of,
                generated_at=generated_at,prediction_time=generated_at,interval=interval,target='close',
                context=context,horizon=horizon,shots=shots,last_close=anchor,
                lora_log_return=before[-1],few_shot_log_return=after[-1],
                forecast_prices=prices,support_losses=losses,
                timesfm_prices=raw_prices,technical_features=technical_features,
                predictor='timesfm_lora_fewshot_plus_ta' if technical_features is not None else 'timesfm_lora_fewshot',
                points=points,
                forecast=[dict(horizon=p['step'],timestamp=p['candle_close'],predicted_close=p['predicted_close']) for p in points],
                max_predicted_return=stats['max_predicted_return'],min_predicted_return=stats['min_predicted_return'],
                forecast_slope=stats['forecast_slope'],forecast_consistency=stats['forecast_consistency'],
                quantile_prices=quantile_prices,forecast_uncertainty=forecast_uncertainty,
                note='Forecast after last historical bar; future targets not evaluated')
    for i,r in enumerate(stats['return_by_step'],1):record[f'return_t{i}']=r
    if save_adapter:
        model.save_pretrained(run_dir/'recent_adapters'/bars[-1].symbol)
    return record,model.unload()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,default=Path('runs/timesfm'))
    p.add_argument('--csv',type=Path,default=Path('data/binance/candles_recent.csv'))
    p.add_argument('--shots',type=int,default=8)
    p.add_argument('--as-of',help='Last completed candle OPEN timestamp; default latest data candle')
    args=p.parse_args()
    if not 1<=args.shots<=64:raise ValueError('shots must be 1..64')
    protocol=json.loads((args.run/'protocol.json').read_text())
    torch.manual_seed(42);torch.set_num_threads(4)
    device=protocol['device']
    series=defaultdict(list)
    bars_loaded=load(args.csv)
    if not bars_loaded:raise ValueError('Empty candle file')
    if args.as_of is None:args.as_of=max(b.timestamp for b in bars_loaded)
    for bar in bars_loaded:
        if bar.timestamp<=args.as_of:series[bar.symbol].append(bar)
    if not series:raise ValueError('No data at cutoff')
    results=[]
    base=TimesFm2_5ModelForPrediction.from_pretrained(protocol['model'],dtype=torch.float32,local_files_only=True).to(device)
    for symbol,bars in sorted(series.items()):
        record,base=predict_symbol(base,args.run,protocol,bars,args.as_of,args.shots,device)
        results.append(record)
        print(symbol,'recent examples adapted:',args.shots,flush=True)
    (args.run/'recent_predictions.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()

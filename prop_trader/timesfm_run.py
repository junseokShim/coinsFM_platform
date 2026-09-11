"""Actual LoRA training and rolling recent-context forecasts; no live orders."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import TimesFm2_5ModelForPrediction
from peft import LoraConfig, get_peft_model, PeftModel
from .__main__ import load
from .forecast_time import SECONDS

MODEL = 'google/timesfm-2.5-200m-transformers'


def windows(path, context, horizon, interval_seconds=3600, rolling_split=False):
    series=defaultdict(list)
    for b in load(path):
        series[b.symbol].append(b)
    splits={name:[] for name in ('train','validation','test')}
    from datetime import datetime
    if rolling_split:
        times=sorted({b.timestamp for bars in series.values() for b in bars})
        train_end,val_end=times[int(len(times)*.8)],times[int(len(times)*.9)]
    for symbol,bars in sorted(series.items()):
        for a,b in zip(bars,bars[1:]):
            if (datetime.fromisoformat(b.timestamp)-datetime.fromisoformat(a.timestamp)).total_seconds()!=interval_seconds:
                raise ValueError(f'Gap or duplicate in {symbol}')
        for i in range(context,len(bars)-horizon+1):
            start,end=bars[i].timestamp,bars[i+horizon-1].timestamp
            split=('train' if start>='2026-05-01' and end<'2026-08-01' else
                   'validation' if start>='2026-08-01' and end<'2026-08-15' else
                   'test' if start>='2026-08-15' and end<'2026-09-01' else None)
            if rolling_split:
                split=('train' if end<train_end else 'validation' if start>=train_end and end<val_end
                       else 'test' if start>=val_end else None)
            if split is None or (split!='train' and (i%horizon!=0 if rolling_split else start[11:13]!='00')):
                continue
            # Anchor using last observed close only; targets never affect scaling.
            anchor=bars[i-1].close
            x=np.array([math.log(b.close/anchor) for b in bars[i-context:i]],dtype=np.float32)
            y=np.array([math.log(b.close/anchor) for b in bars[i:i+horizon]],dtype=np.float32)
            splits[split].append((x,y,dict(symbol=symbol,context_end=bars[i-1].timestamp,
                                          target_start=start,target_end=end,anchor=anchor)))
    if not all(splits.values()):
        raise ValueError('Each chronological split must have complete windows')
    return splits


def loader(items,batch,shuffle=False):
    return DataLoader(TensorDataset(torch.tensor(np.stack([r[0] for r in items])),
                                   torch.tensor(np.stack([r[1] for r in items]))),
                      batch_size=batch,shuffle=shuffle)


def validation(model,items,batch,device,context):
    model.eval();total=0;count=0
    with torch.no_grad():
        for x,y in loader(items,batch):
            loss=model(past_values=x.to(device),future_values=y.to(device),forecast_context_len=context,truncate_negative=False).loss
            if not torch.isfinite(loss):
                raise ValueError('Non-finite validation loss')
            total+=loss.item()*len(x);count+=len(x)
    return total/count


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',type=Path,default=Path('data/binance/candles_recent.csv'))
    p.add_argument('--out',type=Path,default=Path('runs/timesfm'))
    p.add_argument('--context',type=int,default=128)
    p.add_argument('--horizon',type=int,default=6)
    p.add_argument('--samples',type=int,default=96)
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--model',default=MODEL)
    p.add_argument('--interval',choices=list(SECONDS),default='1h')
    p.add_argument('--rolling-split',action='store_true')
    p.add_argument('--device',choices=['cpu','cuda','mps'],default=None,
                    help='Override auto-selected device; used to keep a background retrain off the GPU while live inference uses it')
    args=p.parse_args()
    if args.context<32 or args.context%32 or not 1<=args.horizon<=128 or min(args.samples,args.epochs,args.batch)<1:
        raise ValueError('Invalid training dimensions')
    args.out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42);random.seed(42);np.random.seed(42)
    torch.set_num_threads(4)
    device=args.device or ('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    seconds=SECONDS[args.interval]
    splits=windows(args.csv,args.context,args.horizon,seconds,args.rolling_split)
    train=random.sample(splits['train'],min(args.samples,len(splits['train'])))
    protocol=dict(model=args.model,context=args.context,horizon=args.horizon,seed=42,interval=args.interval,
        training_samples=len(train),epochs=args.epochs,batch=args.batch,device=device,
        splits={k:len(v) for k,v in splits.items()},data_sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        split_ranges={k:dict(first_target=min(r[2]['target_start'] for r in v),last_target=max(r[2]['target_end'] for r in v)) for k,v in splits.items()},
        recent_context='Contiguous observations updated at each origin, not labeled example prompting',
        selection='Minimum validation loss; final test never used for selection',
        limitation='Small training pilot; pretraining overlap unknown; forecasting metrics only')
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print('Loading model',device,flush=True)
    base=TimesFm2_5ModelForPrediction.from_pretrained(args.model,dtype=torch.float32,local_files_only=True)
    base.to(device)
    model=get_peft_model(base,LoraConfig(r=4,lora_alpha=8,target_modules='all-linear',lora_dropout=.05,bias='none'))
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Trainable parameters:',trainable,flush=True)
    before={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-4)
    log=[];best=float('inf');steps=0
    initial=validation(model,splits['validation'],args.batch,device,args.context)
    print('Initial validation:',initial,flush=True)
    for epoch in range(args.epochs):
        model.train();losses=[]
        for x,y in loader(train,args.batch,True):
            optimizer.zero_grad()
            loss=model(past_values=x.to(device),future_values=y.to(device),forecast_context_len=args.context,truncate_negative=False).loss
            if not torch.isfinite(loss): raise ValueError('Non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            losses.append(loss.item());steps+=1
            if steps%4==0:print('Step',steps,'loss',round(loss.item(),6),flush=True)
        val=validation(model,splits['validation'],args.batch,device,args.context)
        log.append(dict(epoch=epoch+1,train_loss=sum(losses)/len(losses),validation_loss=val))
        print(log[-1],flush=True)
        if val<best:
            best=val;model.save_pretrained(args.out/'adapter')
    changed=any(not torch.equal(before[n],p.detach().cpu()) for n,p in model.named_parameters() if n in before)
    if not changed: raise AssertionError('LoRA weights did not update')
    (args.out/'training.json').write_text(json.dumps(dict(log=log,initial_validation_loss=initial,
        best_validation_loss=best,steps=steps,trainable_parameters=trainable,weights_changed=changed),indent=2))
    # Load the chosen adapter; disable_adapter provides a genuinely independent base comparison.
    base=model.unload()
    model=PeftModel.from_pretrained(base,args.out/'adapter').to(device).eval()
    forecasts=[]
    from contextlib import nullcontext
    for name,length,disabled in [('base_recent',args.context,True),('lora_recent',args.context,False),('lora_short_context',32,False)]:
        with (model.disable_adapter() if disabled else nullcontext()),torch.no_grad():
            for offset in range(0,len(splits['test']),args.batch):
                items=splits['test'][offset:offset+args.batch]
                x=torch.tensor(np.stack([r[0][-length:] for r in items])).to(device)
                pred=model(past_values=x,forecast_context_len=length,truncate_negative=False).mean_predictions[:,:args.horizon].float().cpu().numpy()
                if not np.isfinite(pred).all():raise ValueError('Non-finite predictions')
                for row,forecast in zip(items,pred):
                    x,y,meta=row
                    forecasts.append(dict(model=name,**meta,predicted_log_return=float(forecast[-1]),
                        actual_log_return=float(y[-1]),path_mae=float(np.abs(forecast-y).mean())))
        print('Evaluated',name,flush=True)
    for x,y,meta in splits['test']:
        forecasts.append(dict(model='last_price',**meta,predicted_log_return=0.,actual_log_return=float(y[-1]),path_mae=float(np.abs(y).mean())))
    with (args.out/'forecasts.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(forecasts[0]));w.writeheader();w.writerows(forecasts)
    metrics={}
    for name in sorted({r['model'] for r in forecasts}):
        rows=[r for r in forecasts if r['model']==name]
        metrics[name]=dict(windows=len(rows),path_mae=float(np.mean([r['path_mae'] for r in rows])),
            endpoint_mae=float(np.mean([abs(r['predicted_log_return']-r['actual_log_return']) for r in rows])),
            direction_accuracy=float(np.mean([np.sign(r['predicted_log_return'])==np.sign(r['actual_log_return']) for r in rows])))
    (args.out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':main()

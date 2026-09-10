"""Causal historical MTF predictions and 10s decision replay on minute-open proxies.

Not tick-accurate: no historical depth/latency/order events exist for this pilot.
Calls the production decision and sizing rules without changing live artifacts.
"""
import argparse
import bisect
import csv
import hashlib
import json
import math
import shutil
import time
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path

from .__main__ import load
from .forecast_time import SECONDS
from .multiframe import decide, buy_budget, exit_fraction
from .execution_rules import timestamp, minimum_entry
from .live_trader import evaluate_position

ROOT=Path('runs/bear_research/mtf')
SOURCE=Path('data/bear_research/pilot')
START=timestamp('2026-09-09T18:00:00+00:00')
END=timestamp('2026-09-09T22:00:00+00:00')


def series():
    result={}
    for iv in ('1m','1h','1d'):
        by=defaultdict(list)
        for b in load(SOURCE/(iv+'.csv')):by[b.symbol].append(b)
        result[iv]=by
    return result


def generate():
    import numpy as np
    import torch
    from transformers import TimesFm2_5ModelForPrediction
    from .technical import HybridCalibrator, latest_features, FEATURES
    from .timesfm_predict import predict_symbol
    ROOT.mkdir(parents=True,exist_ok=True)
    snapshot=ROOT/'models'
    protocols={};calibrators={};artifacts={};shots={}
    for iv in ('1m','1h','1d'):
        dest=snapshot/iv
        if not dest.exists():
            dest.mkdir(parents=True)
            shutil.copytree(Path('runs/forecast5')/iv/'adapter',dest/'adapter')
            shutil.copyfile(Path('runs/forecast5')/iv/'protocol.json',dest/'protocol.json')
            shutil.copyfile(Path('runs/hybrid')/iv/'calibrator.json',dest/'calibrator.json')
            best_shots=Path('runs/forecast5')/iv/'best_shots.json'
            if best_shots.exists():
                shutil.copyfile(best_shots,dest/'best_shots.json')
        protocols[iv]=json.loads((dest/'protocol.json').read_text())
        artifacts[iv]=json.loads((dest/'calibrator.json').read_text())
        if artifacts[iv]['training_data_sha256']!=protocols[iv]['data_sha256']:
            raise ValueError('Calibration/model snapshot mismatch')
        # Frozen with the rest of the snapshot: a backtest must use the shots count that was
        # actually best-known at the time, not whatever a later sweep happens to prefer.
        chosen=json.loads((dest/'best_shots.json').read_text()).get('best_shots',8) if (dest/'best_shots.json').exists() else 8
        shots[iv]=chosen if isinstance(chosen,int) and 1<=chosen<=64 else 8
        calibrators[iv]=HybridCalibrator.from_dict(artifacts[iv]['model'])
    data=series();indices={iv:{m:[timestamp(b.timestamp) for b in bars] for m,bars in by.items()}
                           for iv,by in data.items()}
    output=ROOT/'forecasts.jsonl'
    done=set()
    if output.exists():
        for line in output.open():
            r=json.loads(line);done.add((r['symbol'],r['interval'],r['as_of']))
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(42)
    base=TimesFm2_5ModelForPrediction.from_pretrained(protocols['1h']['model'],dtype=torch.float32,local_files_only=True).to('cpu')
    count=0;failures=Counter();durations=[]
    with output.open('a') as out:
        for origin in range(int(START),int(END),60):
            cursor=float(origin)
            for market in sorted(data['1m']):
                for iv in ('1d','1h','1m'):
                    cutoff=origin//SECONDS[iv]*SECONDS[iv]
                    iso=datetime.fromtimestamp(cutoff,timezone.utc).isoformat()
                    key=(market,iv,iso)
                    if key in done:continue
                    done.add(key)
                    end=bisect.bisect_left(indices[iv].get(market,[]),cutoff)
                    bars=data[iv].get(market,[])[max(0,end-200):end]
                    if len(bars)<168 or timestamp(bars[-1].timestamp)+SECONDS[iv]!=cutoff:
                        failures['history_missing']+=1;continue
                    if any(timestamp(b.timestamp)-timestamp(a.timestamp)!=SECONDS[iv] for a,b in zip(bars,bars[1:])):
                        failures['history_gap']+=1;continue
                    if timestamp(artifacts[iv]['calibration_end'])>=timestamp(bars[-1].timestamp):
                        raise ValueError('Calibration leakage')
                    if timestamp(protocols[iv]['split_ranges']['train']['last_target'])>=timestamp(bars[-1].timestamp):
                        raise ValueError('Training leakage')
                    start=time.perf_counter()
                    rec,base=predict_symbol(base,snapshot/iv,protocols[iv],bars,bars[-1].timestamp,shots[iv],'cpu',use_technical=False,save_adapter=False)
                    feat=latest_features(bars)
                    raw=np.log(np.array(rec['forecast_prices'])/rec['last_close'])
                    fused=calibrators[iv].predict(raw[None,:],np.array([[feat[k] for k in FEATURES]]))[0]
                    rec['forecast_prices']=(rec['last_close']*np.exp(fused)).tolist()
                    rec['technical_features']=feat
                    elapsed=time.perf_counter()-start;durations.append(elapsed)
                    cursor+=elapsed
                    # No predictions are available before their observed sequential compute time.
                    rec['available_at']=cursor;rec['compute_seconds']=elapsed
                    out.write(json.dumps(rec)+'\n');out.flush();count+=1
            if (origin-int(START))%600==0:
                print('historical',datetime.fromtimestamp(origin,timezone.utc).isoformat(),'new forecasts',count,'skipped',dict(failures),flush=True)
    manifest=dict(start=START,end=END,markets=sorted(data['1m']),new_forecasts=count,failures=dict(failures),
        latency_seconds=dict(median=float(np.median(durations)) if durations else None,
                             p95=float(np.quantile(durations,.95)) if durations else None),
        inputs={iv:hashlib.sha256((SOURCE/(iv+'.csv')).read_bytes()).hexdigest() for iv in data},
        limitation='Minute-open price held constant for six 10s decisions; synthetic spread/depth. Fixed 3-major universe. Not a live tick replay.')
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))


def replay(fee=.001, slip=.001, capital=76000., cap=8000.):
    data=series()['1m'];book={m:{timestamp(b.timestamp):b for b in bars} for m,bars in data.items()}
    forecasts=sorted([json.loads(l) for l in (ROOT/'forecasts.jsonl').open()],key=lambda r:r['available_at'])
    cache=defaultdict(dict);index=0;cash=capital;positions={};cooldowns={};retry={};events=[];curve=[];reasons=Counter()
    peak=capital;dd=0.;ticks_with_all=0
    def sell(m,price,ts,reason,fraction=1):
        nonlocal cash
        p=positions[m];fraction=exit_fraction(fraction,p['qty'],price)
        qty=p['qty']*fraction
        if qty*price*(1-slip)<5000:return
        amount=qty*price*(1-slip)*(1-fee);cost=p['cost']*fraction
        cash+=amount;p['qty']-=qty;p['cost']-=cost
        events.append(dict(symbol=m,side='SELL',ts=ts,amount=amount,pnl=amount-cost,reason=reason))
        cooldowns[m]=ts+3600;retry[m]=ts+60
        if p['qty']<1e-8:del positions[m]
    for ts in range(int(START),int(END),10):
        while index<len(forecasts) and forecasts[index]['available_at']<=ts:
            r=forecasts[index];cache[r['symbol']][r['interval']]=r;index+=1
        prices={m:bars[ts//60*60].open for m,bars in book.items() if ts//60*60 in bars}
        candidates=[];exited=set()
        for m,price in prices.items():
            ob=dict(timestamp=ts*1000,orderbook_units=[dict(ask_price=price*(1+slip),ask_size=1e12,bid_price=price*(1-slip),bid_size=1e12)])
            d=decide(cache[m],ob,ts,max(minimum_entry(),min(cap,cash)),held=m in positions)
            reasons[d['reason']]+=1;ticks_with_all+=int(d['complete'])
            if m in positions:
                action,reason=evaluate_position(positions[m],price,ts)
                fraction=1.
                if action=='STAY' and d['action']=='SELL' and ts>=retry.get(m,0):
                    action,reason,fraction='SELL',d['reason'],d['sell_fraction']
                if action=='SELL':sell(m,price,ts,reason,fraction);exited.add(m)
            elif d['action']=='BUY':candidates.append((m,d,price,ob))
        for m,d,price,ob in sorted(candidates,key=lambda r:(-r[1]['signal_strength'],-r[1]['score'])):
            if m in exited or ts<cooldowns.get(m,0) or len(positions)>=6:continue
            budget=buy_budget(d,cash,cap)
            if not budget:continue
            d=decide(cache[m],ob,ts,budget)
            if d['action']!='BUY':continue
            fill=price*(1+slip);qty=budget/(fill*(1+fee));cash-=budget
            positions[m]=dict(qty=qty,cost=budget,entry_price=fill,stop=fill*.975,target=fill*1.05,
                              horizon_expires=timestamp(cache[m]['1h']['as_of'])+18000)
            events.append(dict(symbol=m,side='BUY',ts=ts,amount=budget,pnl=0.,reason=d['reason']))
        value=cash+sum(prices.get(m,p['entry_price'])*p['qty'] for m,p in positions.items())
        peak=max(peak,value);dd=max(dd,1-value/peak)
        curve.append(dict(ts=ts,equity=value,cash=cash))
    # End-of-study liquidation is reported separately from strategy exits.
    for m in list(positions):
        last=max(t for t in book[m] if t<END)
        sell(m,book[m][last].close,END,'study_end')
    final=cash+sum(book[m][max(t for t in book[m] if t<END)].close*p['qty'] for m,p in positions.items())
    summary=dict(return_pct=(final/capital-1)*100,final_equity=final,max_drawdown_pct=dd*100,
        buys=sum(e['side']=='BUY' for e in events),sells=sum(e['side']=='SELL' for e in events),
        realized_pnl=sum(e['pnl'] for e in events),remaining_dust=len(positions),
        complete_symbol_ticks=ticks_with_all,reasons=dict(reasons),fee=fee,slippage=slip,
        buy_hold_pct={m:(book[m][max(t for t in book[m] if t<END)].close/book[m][START].open-1)*100 for m in book})
    suffix='base' if slip==.001 else 'stress'
    (ROOT/f'{suffix}_results.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    for name,rows in [('trades',events),('equity',curve)]:
        path=ROOT/f'{suffix}_{name}.csv'
        with path.open('w',newline='') as f:
            if rows:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('MTF_REPLAY',suffix,json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--generate',action='store_true');a=p.parse_args()
    if a.generate:generate()
    replay();replay(slip=.003)

"""Public Upbit KRW universe and completed candles for ranking (no credentials)."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from .upbit_broker import UpbitClient
from .forecast_time import SECONDS
from .engine import Bar


def collect(interval='1h',top=12,out=Path('data/upbit')):
    if interval not in ('1m','1h','1d') or not 1<=top<=30:
        raise ValueError('Supported intervals 1m/1h/1d; top 1..30')
    out.mkdir(parents=True,exist_ok=True)
    client=UpbitClient();metadata=client.request('GET','/v1/market/all',dict(is_details='true'),private=False)
    markets=[r['market'] for r in metadata if r['market'].startswith('KRW-')
             and r.get('market_warning','NONE')!='CAUTION'
             and not r.get('market_event',{}).get('warning',False)]
    tickers=[]
    for i in range(0,len(markets),50):
        time.sleep(.15)
        tickers.extend(client.request('GET','/v1/ticker',dict(markets=','.join(markets[i:i+50])),private=False))
    selected=sorted(tickers,key=lambda r:(-r['acc_trade_price_24h'],r['market']))[:top]
    cutoff=datetime.now(timezone.utc)
    seconds=SECONDS[interval]
    boundary=datetime.fromtimestamp(int(cutoff.timestamp())//seconds*seconds,timezone.utc)
    rows=[];accepted=[];excluded=[];books={}
    for ticker in selected:
        symbol=ticker['market']
        endpoint='/v1/candles/days' if interval=='1d' else f'/v1/candles/minutes/{seconds//60}'
        time.sleep(.15)
        raw=client.request('GET',endpoint,dict(market=symbol,count=200,to=boundary.isoformat()),private=False)
        bars=sorted([Bar(datetime.fromisoformat(r['candle_date_time_utc']).replace(tzinfo=timezone.utc).isoformat(),
            symbol,float(r['opening_price']),float(r['high_price']),float(r['low_price']),
            float(r['trade_price']),float(r['candle_acc_trade_volume'])) for r in raw],key=lambda b:b.timestamp)
        if len(bars)<168 or (datetime.fromisoformat(bars[-1].timestamp).timestamp()+seconds!=boundary.timestamp()) or any(
            datetime.fromisoformat(b.timestamp).timestamp()-datetime.fromisoformat(a.timestamp).timestamp()!=seconds for a,b in zip(bars,bars[1:])):
            excluded.append(dict(symbol=symbol,reason='Missing recent complete candles or no-trade gaps'));continue
        time.sleep(.15)
        book=client.request('GET','/v1/orderbook',dict(markets=symbol),private=False)[0]
        books[symbol]=book;accepted.append(ticker)
        rows.extend([[b.timestamp,b.symbol,b.open,b.high,b.low,b.close,b.volume] for b in bars])
    if not rows:raise ValueError('No eligible complete series')
    path=out/f'{interval}.csv'
    with path.open('w') as f:
        w=csv.writer(f);w.writerow(['timestamp','symbol','open','high','low','close','volume']);w.writerows(sorted(rows))
    snapshot=dict(exchange='upbit',quote='KRW',interval=interval,as_of=boundary.isoformat(),
        collected_at=datetime.now(timezone.utc).isoformat(),selection='Top 24h KRW traded notional, excluding warnings; retrospective current universe',
        tickers=accepted,orderbooks=books,excluded=excluded,data_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (out/f'{interval}_snapshot.json').write_text(json.dumps(snapshot,indent=2))
    print(f'{interval}: {len(accepted)} markets, {len(excluded)} excluded',flush=True)
    return path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--interval',default='1h');p.add_argument('--top',type=int,default=12)
    args=p.parse_args();collect(args.interval,args.top)

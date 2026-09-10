"""Public, paginated Upbit research snapshots; never overwrite live candle files."""
import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from .upbit_broker import UpbitClient, BrokerError
from .forecast_time import SECONDS

MARKETS = ('KRW-BTC','KRW-ETH','KRW-XRP')


def collect(root, interval, start, end, markets=MARKETS, warmup=200):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    path=root/(interval+'.csv')
    if path.exists() and path.with_suffix('.manifest.json').exists():
        meta=json.loads(path.with_suffix('.manifest.json').read_text())
        if (meta['start'],meta['end'],meta['markets'])==(start,end,list(markets)):
            if hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']:return path
    client=UpbitClient(access='',secret='')
    step=SECONDS[interval]
    floor=datetime.fromisoformat(start).timestamp()-warmup*step
    rows=[]
    for market in markets:
        cursor=datetime.fromisoformat(end).timestamp();seen={};pages=0
        while cursor>floor:
            endpoint='/v1/candles/days' if interval=='1d' else f'/v1/candles/minutes/{step//60}'
            time.sleep(.35)
            for attempt in range(4):
                try:
                    raw=client.request('GET',endpoint,dict(market=market,count=200,
                        to=datetime.fromtimestamp(cursor,timezone.utc).isoformat()),private=False)
                    break
                except BrokerError:
                    if attempt==3:raise
                    time.sleep(20)
            if not raw:break
            earliest=cursor
            for r in raw:
                ts=datetime.fromisoformat(r['candle_date_time_utc']).replace(tzinfo=timezone.utc)
                earliest=min(earliest,ts.timestamp())
                if floor<=ts.timestamp()<datetime.fromisoformat(end).timestamp():
                    seen[ts.isoformat()]=[ts.isoformat(),market,r['opening_price'],r['high_price'],r['low_price'],r['trade_price'],r['candle_acc_trade_volume']]
            if earliest>=cursor:raise ValueError('Pagination did not advance')
            cursor=earliest;pages+=1
            if pages%20==0:print(root.name,market,interval,'pages',pages,flush=True)
        rows.extend(seen.values())
        print(root.name,market,interval,'candles',len(seen),flush=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['timestamp','symbol','open','high','low','close','volume']);w.writerows(sorted(rows))
    tmp.replace(path)
    path.with_suffix('.manifest.json').write_text(json.dumps(dict(start=start,end=end,markets=list(markets),
        interval=interval,warmup=warmup,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source='Upbit public historical candles; fixed currently-listed majors, survivorship bias'),indent=2))
    return path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['pilot','2022','2025'],required=True)
    a=p.parse_args();root=Path('data/bear_research')/a.dataset
    if a.dataset=='pilot':
        for iv in ('1m','1h','1d'):
            collect(root,iv,'2026-09-09T18:00:00+00:00','2026-09-09T22:00:00+00:00')
    else:
        collect(root,'1h',f'{a.dataset}-01-01T00:00:00+00:00',f'{int(a.dataset)+1}-01-01T00:00:00+00:00')

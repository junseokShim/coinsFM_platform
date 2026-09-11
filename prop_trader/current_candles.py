"""Read-only, completed candles with a fixed exchange-server cutoff."""
import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from .download import SYMBOLS
from .forecast_time import SECONDS

# Currently fetched/trained timeframes; add a key here (and train an adapter)
# to support another entry from forecast_time.SECONDS without other changes.
SUPPORTED=('1m','1h','1d')
INTERVALS={k:SECONDS[k] for k in SUPPORTED}
BASE='https://data-api.binance.vision/api/v3/'


def request(endpoint,params=None):
    url=BASE+endpoint+('?' + urlencode(params) if params else '')
    with urlopen(url,timeout=30) as response:
        return json.load(response),url


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('data/current'))
    p.add_argument('--count',type=int,default=800)
    p.add_argument('--symbols',nargs='+',default=SYMBOLS,help='Override the default 6-symbol research universe, e.g. for a matched-symbol comparison against another data source')
    args=p.parse_args()
    if not 256<=args.count<=999:raise ValueError('count must be 256..999')
    server,_=request('time');cutoff=server['serverTime']
    args.out.mkdir(parents=True,exist_ok=True)
    def fetch(job):
        interval,symbol=job
        rows,url=request('klines',dict(symbol=symbol,interval=interval,limit=args.count+1,endTime=cutoff))
        raw=[r for r in rows if r[6]<cutoff][-args.count:]
        if len(raw)!=args.count:raise ValueError(f'Insufficient history for {symbol}/{interval}')
        expected=INTERVALS[interval]*1000
        if any(b[0]-a[0]!=expected for a,b in zip(raw,raw[1:])):raise ValueError('Noncontinuous candles')
        if cutoff-(raw[-1][0]+expected)>=expected:raise ValueError('Stale exchange candle response')
        digest=hashlib.sha256(json.dumps(raw).encode()).hexdigest()
        return interval,[[datetime.fromtimestamp(r[0]/1000,timezone.utc).isoformat(),symbol,*r[1:6]] for r in raw],dict(url=url,sha256=digest,symbol=symbol,interval=interval)
    data={k:[] for k in INTERVALS};sources=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for interval,rows,meta in pool.map(fetch,[(i,s) for i in INTERVALS for s in args.symbols]):
            data[interval].extend(rows);sources.append(meta)
    for interval,rows in data.items():
        rows.sort(key=lambda r:(r[0],r[1]))
        path=args.out/f'{interval}.csv'
        with path.open('w') as f:
            w=csv.writer(f);w.writerow(['timestamp','symbol','open','high','low','close','volume']);w.writerows(rows)
    (args.out/'manifest.json').write_text(json.dumps(dict(cutoff_ms=cutoff,
        cutoff_utc=datetime.fromtimestamp(cutoff/1000,timezone.utc).isoformat(),sources=sources,
        files={i:hashlib.sha256((args.out/f'{i}.csv').read_bytes()).hexdigest() for i in data}),indent=2))
    print(f'Saved {len(INTERVALS)} intervals, {len(args.symbols)} symbols,',args.count,'completed candles each')


if __name__=='__main__':main()

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


def fetch_series(symbol,interval,count,pages,cutoff,expected,page_cap=999):
    """Up to `pages` sequential <=page_cap-candle klines requests (Binance's own ceiling is 999;
    page_cap is only lowered by tests, to exercise pagination without a 999-row fixture),
    walking backward from `cutoff`, stitched into one ascending, gap-checked series of exactly
    `count` completed candles.

    Only the first (most recent) page can contain a candle that hasn't closed yet -- server time
    `cutoff` can fall mid-candle. Every older page's endTime is itself a prior candle's open
    time, so everything at or before it is already a completed historical candle; re-applying
    the close-time-vs-cutoff filter there would incorrectly drop the boundary candle and split
    the stitched series (this broke once during development -- see the accompanying test)."""
    raw=[];page_end=cutoff;urls=[]
    for page_num in range(pages):
        page_size=min(page_cap,count-len(raw))
        if page_size<=0:break
        rows,url=request('klines',dict(symbol=symbol,interval=interval,limit=page_size+1,endTime=page_end))
        urls.append(url)
        rows=[r for r in rows if r[6]<cutoff] if page_num==0 else rows
        page=rows[-page_size:]
        if not page:break
        raw=page+raw
        page_end=page[0][0]-1
        if len(page)<page_size:break  # exchange ran out of history earlier than this page's target
    raw=raw[-count:]
    if len(raw)!=count:raise ValueError(f'Insufficient history for {symbol}/{interval}: got {len(raw)}, wanted {count}')
    if any(b[0]-a[0]!=expected for a,b in zip(raw,raw[1:])):raise ValueError('Noncontinuous candles')
    if cutoff-(raw[-1][0]+expected)>=expected:raise ValueError('Stale exchange candle response')
    digest=hashlib.sha256(json.dumps(raw).encode()).hexdigest()
    return raw,dict(urls=urls,sha256=digest,symbol=symbol,interval=interval)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('data/current'))
    p.add_argument('--count',type=int,default=800)
    p.add_argument('--pages',type=int,default=1,help='Fetch this many sequential <=999-candle requests per '
        'symbol/interval, walking backward from the cutoff, to cover more history than one klines request allows '
        '(a single request caps at 999). Default 1 preserves the original single-request 256..999 behavior '
        'exactly; pages>1 raises the effective --count ceiling to 999*pages.')
    p.add_argument('--intervals',nargs='+',default=SUPPORTED,choices=SUPPORTED,
        help='Subset of intervals to fetch, e.g. --intervals 1h to extend one timeframe without refetching the others')
    p.add_argument('--symbols',nargs='+',default=SYMBOLS,help='Override the default 6-symbol research universe, e.g. for a matched-symbol comparison against another data source')
    args=p.parse_args()
    if args.pages<1:raise ValueError('pages must be >=1')
    if not 256<=args.count<=999*args.pages:raise ValueError(f'count must be 256..{999*args.pages}')
    intervals={k:INTERVALS[k] for k in args.intervals}
    server,_=request('time');cutoff=server['serverTime']
    args.out.mkdir(parents=True,exist_ok=True)
    def fetch(job):
        interval,symbol=job
        raw,meta=fetch_series(symbol,interval,args.count,args.pages,cutoff,intervals[interval]*1000)
        return interval,[[datetime.fromtimestamp(r[0]/1000,timezone.utc).isoformat(),symbol,*r[1:6]] for r in raw],meta
    data={k:[] for k in intervals};sources=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for interval,rows,meta in pool.map(fetch,[(i,s) for i in intervals for s in args.symbols]):
            data[interval].extend(rows);sources.append(meta)
    for interval,rows in data.items():
        rows.sort(key=lambda r:(r[0],r[1]))
        path=args.out/f'{interval}.csv'
        with path.open('w') as f:
            w=csv.writer(f);w.writerow(['timestamp','symbol','open','high','low','close','volume']);w.writerows(rows)
    (args.out/'manifest.json').write_text(json.dumps(dict(cutoff_ms=cutoff,
        cutoff_utc=datetime.fromtimestamp(cutoff/1000,timezone.utc).isoformat(),sources=sources,
        files={i:hashlib.sha256((args.out/f'{i}.csv').read_bytes()).hexdigest() for i in data}),indent=2))
    print(f'Saved {len(intervals)} intervals, {len(args.symbols)} symbols,',args.count,'completed candles each')


if __name__=='__main__':main()

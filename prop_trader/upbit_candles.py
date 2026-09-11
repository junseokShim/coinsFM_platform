"""Read-only, completed Upbit KRW-market candles for LoRA backbone training.

Same CSV contract as current_candles.py (timestamp,symbol,open,high,low,close,volume per
interval file, plus a manifest.json with source hashes) so timesfm_run.py and
hybrid_research.py need no changes -- only the data source moves from Binance USDT pairs
to native Upbit KRW pairs, closing the cross-exchange domain-shift gap flagged in
TRADING_INTEGRATION.md (research_unverified_cross_exchange).
"""
import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .forecast_time import SECONDS
from .upbit_broker import BrokerError, UpbitClient

SUPPORTED = ('1m', '1h', '1d')
INTERVALS = {k: SECONDS[k] for k in SUPPORTED}
# download.py's Binance meme-coin research universe (DOGE/SHIB/PEPE/FLOKI/BONK/WIF) doesn't
# transfer cleanly to Upbit-native training: FLOKI/BONK have no Upbit market at all, and among
# the rest only DOGE/SHIB have >=800 days of Upbit KRW listing history -- PEPE (~667d), TRUMP
# (~576d), RAY (~450d) and WIF (~129d, only listed 2026-05) all fall short of the 800-candle
# window `count` needs for the 1d interval (verified live 2026-09-11). Picked instead: DOGE/SHIB
# (the meme coins that DO have enough history) plus XRP/BTC/ETH/IOST, all confirmed >=800 days --
# also a deliberately more diverse volatility mix (majors + meme) than a pure meme-coin universe.
SYMBOLS = ['KRW-DOGE', 'KRW-SHIB', 'KRW-XRP', 'KRW-BTC', 'KRW-ETH', 'KRW-IOST']
ENDPOINT = {'1m': '/v1/candles/minutes/1', '1h': '/v1/candles/minutes/60', '1d': '/v1/candles/days'}


def fill_gaps(raw, step):
    """Upbit only returns a candle when a trade actually happened, unlike Binance which
    synthesizes a zero-volume candle (open=high=low=close=prior close) for silent periods.
    Replicate that convention here so training data has the same gap-free contract the rest of
    the pipeline (timesfm_run.windows, resident_forecasts.completed_bars) already assumes."""
    if not raw:
        return raw
    filled = [raw[0]]
    for stamp, r in raw[1:]:
        prev_stamp, prev_r = filled[-1]
        gap = round((stamp-prev_stamp)/step)
        for i in range(1, gap):
            synth_time = prev_stamp + i*step
            close = float(prev_r['trade_price'])
            filled.append((synth_time, dict(candle_date_time_utc=datetime.fromtimestamp(
                synth_time, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S'), opening_price=close,
                high_price=close, low_price=close, trade_price=close, candle_acc_trade_volume=0.)))
        filled.append((stamp, r))
    return filled


def paced_request(client, endpoint, params):
    # This is a one-time offline bulk-download job sharing Upbit's per-IP rate budget with the
    # live resident inference worker, which must never be starved by it -- so back off patiently
    # on 429 (UpbitClient._blocked_until is a per-process class attribute; separate process from
    # the live worker, so this never blocks live trading) rather than retrying aggressively.
    for attempt in range(8):
        try:
            return client.request('GET', endpoint, params, private=False)
        except BrokerError:
            if attempt == 7:
                raise
            time.sleep(5. * (attempt+1))
    raise AssertionError('unreachable')


def fetch_symbol(client, symbol, interval, count, cutoff):
    step = INTERVALS[interval]
    endpoint = ENDPOINT[interval]
    rows = []
    to = cutoff
    exhausted = False
    # Fetch real candles a page at a time and re-check the gap-filled length after each page --
    # rather than guessing a fixed extra-padding percentage up front (which either wastes
    # history unnecessarily on gap-free intervals like 1d, or under-fetches on gappy ones like
    # 1m), so this works right up to a symbol's actual listing-history limit either way.
    while len(fill_gaps(sorted(rows, key=lambda x: x[0]), step)) < count:
        if exhausted:
            raise ValueError(f'Insufficient history for {symbol}/{interval}')
        page = paced_request(client, endpoint, dict(market=symbol, count=200,
                              to=datetime.fromtimestamp(to, timezone.utc).isoformat()))
        time.sleep(.3)  # extra headroom beyond UpbitClient's own .15s pacing
        if not page:
            raise ValueError(f'Insufficient history for {symbol}/{interval}')
        for r in page:
            stamp = datetime.fromisoformat(r['candle_date_time_utc']).replace(tzinfo=timezone.utc).timestamp()
            if stamp + step > cutoff:
                continue  # forming/not-yet-completed candle
            rows.append((stamp, r))
        to = min(datetime.fromisoformat(r['candle_date_time_utc']).replace(tzinfo=timezone.utc).timestamp()
                  for r in page) - step
        exhausted = len(page) < 200
    rows.sort(key=lambda x: x[0])
    raw = fill_gaps(rows, step)[-count:]
    if len(raw) != count:
        raise ValueError(f'Insufficient history for {symbol}/{interval}')
    if any(b[0]-a[0] != step for a, b in zip(raw, raw[1:])):
        raise ValueError(f'Noncontinuous candles for {symbol}/{interval}')
    out = [[datetime.fromtimestamp(t, timezone.utc).isoformat(), symbol,
            float(r['opening_price']), float(r['high_price']), float(r['low_price']),
            float(r['trade_price']), float(r['candle_acc_trade_volume'])] for t, r in raw]
    digest = hashlib.sha256(json.dumps(out).encode()).hexdigest()
    return out, dict(source='upbit', symbol=symbol, interval=interval, sha256=digest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=Path('data/current_upbit'))
    p.add_argument('--count', type=int, default=800)
    p.add_argument('--symbols', nargs='+', default=SYMBOLS)
    args = p.parse_args()
    if not 256 <= args.count <= 999:
        raise ValueError('count must be 256..999')
    client = UpbitClient(access='', secret='')
    cutoff = int(time.time())
    args.out.mkdir(parents=True, exist_ok=True)
    data = {k: [] for k in INTERVALS}
    sources = []
    for interval in INTERVALS:
        aligned_cutoff = cutoff // INTERVALS[interval] * INTERVALS[interval]
        for symbol in args.symbols:
            rows, meta = fetch_symbol(client, symbol, interval, args.count, aligned_cutoff)
            data[interval].extend(rows)
            sources.append(meta)
    for interval, rows in data.items():
        rows.sort(key=lambda r: (r[0], r[1]))
        path = args.out / f'{interval}.csv'
        with path.open('w') as f:
            w = csv.writer(f)
            w.writerow(['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
            w.writerows(rows)
    (args.out / 'manifest.json').write_text(json.dumps(dict(
        cutoff_utc=datetime.fromtimestamp(cutoff, timezone.utc).isoformat(), source='upbit',
        sources=sources, symbols=args.symbols,
        files={i: hashlib.sha256((args.out/f'{i}.csv').read_bytes()).hexdigest() for i in data}), indent=2))
    print(f'Saved {len(INTERVALS)} intervals, {len(args.symbols)} symbols, {args.count} completed Upbit candles each')


if __name__ == '__main__':
    main()

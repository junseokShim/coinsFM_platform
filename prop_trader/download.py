"""Download and verify a fixed research universe, without exchange credentials."""
import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

SYMBOLS = ['DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'FLOKIUSDT', 'BONKUSDT', 'WIFUSDT']
ROOT = Path('data/binance')


def fetch(job):
    symbol, month = job
    name = f'{symbol}-1h-{month}.zip'
    url = f'https://data.binance.vision/data/spot/monthly/klines/{symbol}/1h/{name}'
    path = ROOT / name
    if not path.exists():
        payload = urllib.request.urlopen(url, timeout=30).read()
        checksum = urllib.request.urlopen(url + '.CHECKSUM', timeout=30).read().decode().split()[0]
        if hashlib.sha256(payload).hexdigest() != checksum:
            raise ValueError(f'Checksum mismatch: {name}')
        path.write_bytes(payload)
        path.with_suffix('.sha256').write_text(checksum)
    payload = path.read_bytes()
    checksum = hashlib.sha256(payload).hexdigest()
    if checksum != path.with_suffix('.sha256').read_text().strip():
        raise ValueError(f'Cached checksum mismatch: {name}')
    rows = []
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        for r in csv.reader(io.StringIO(z.read(z.namelist()[0]).decode())):
            epoch = int(r[0])
            ts = datetime.fromtimestamp(epoch / (1e6 if epoch > 10**14 else 1e3), timezone.utc)
            rows.append([ts.isoformat(), symbol, *r[1:6]])
    return rows, dict(url=url, sha256=checksum, rows=len(rows))


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    # WIF spot history starts in March 2024: use full months from April onward.
    months = [f'2024-{m:02d}' for m in range(4, 13)] + [f'2025-{m:02d}' for m in range(1, 13)]
    jobs = [(s, m) for s in SYMBOLS for m in months]
    rows, sources = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (batch, meta) in enumerate(pool.map(fetch, jobs), 1):
            rows.extend(batch)
            sources.append(meta)
            if i % 12 == 0:
                print(f'Downloaded {i}/{len(jobs)} archives', flush=True)
    rows.sort(key=lambda r: (r[0], r[1]))
    with (ROOT / 'candles.csv').open('w') as f:
        w = csv.writer(f)
        w.writerow(['timestamp','symbol','open','high','low','close','volume'])
        w.writerows(rows)
    (ROOT / 'manifest.json').write_text(json.dumps(dict(
        retrieved_at=datetime.now(timezone.utc).isoformat(), symbols=SYMBOLS,
        interval='1h', sources=sources, rows=len(rows),
        selection='Fixed retrospective meme-token convenience sample; survivorship and selection bias'), indent=2))
    print(f'Saved {len(rows)} rows', flush=True)


if __name__ == '__main__':
    main()

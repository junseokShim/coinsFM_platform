import argparse
import csv
import json
import random
from datetime import datetime, timedelta, timezone
from itertools import groupby
from pathlib import Path
from .engine import Bar, Config, Desk


def load(path):
    bars = []
    with path.open() as f:
        for row in csv.DictReader(f):
            ts = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
            if ts.tzinfo is None:
                raise ValueError('CSV timestamps must include timezone')
            bars.append(Bar(ts.astimezone(timezone.utc).isoformat(), row['symbol'],
                            **{k: float(row[k]) for k in ('open','high','low','close','volume')}))
    return sorted(bars, key=lambda b: (b.timestamp, b.symbol))


def demo():
    rng = random.Random(42)
    prices = {'ALPHA': 10., 'BETA': 20., 'GAMMA': 5.}
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    bars = []
    for i in range(360):
        for j, (s, p) in enumerate(prices.items()):
            shock = .045 * (1 if (i // 45 + j) % 2 else -1) if i % 45 == 30 else 0
            close = p * (1 + rng.uniform(-.006, .006) + shock)
            bars.append(Bar((start + timedelta(minutes=i)).isoformat(), s, p,
                max(p, close) * 1.003, min(p, close) * .997, close,
                rng.uniform(800, 1200) * (5 if shock else 1)))
            prices[s] = close
    return bars


def main():
    parser = argparse.ArgumentParser(description='Offline prop trading research desk')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--demo', action='store_true')
    source.add_argument('--csv', type=Path)
    parser.add_argument('--out', type=Path, default=Path('runs/latest'))
    parser.add_argument('--allow-short', action='store_true', help='Synthetic short simulation only')
    args = parser.parse_args()
    config = Config(allow_short=args.allow_short)
    desk = Desk(config)
    for _, rows in groupby(demo() if args.demo else load(args.csv), key=lambda b: b.timestamp):
        desk.step(list(rows))
    report = desk.finish()
    report['data_source'] = 'synthetic_demo' if args.demo else str(args.csv)
    report['config'] = vars(config)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2))
    (args.out / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in desk.events))
    with (args.out / 'equity.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp','equity','positions','halted'])
        writer.writeheader()
        writer.writerows(desk.curve)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

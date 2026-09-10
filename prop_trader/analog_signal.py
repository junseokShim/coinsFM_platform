"""Historical pattern-matching confirmation gate: before trusting a forecast, check how
similar historical setups (same technical regime + the model's own view) actually played out.

The analog database is built purely from resident_forecasts.py's forecast_history.jsonl
archive. Consecutive archived records for the same (symbol, interval) trace the REALIZED price
path -- each record's `last_close` is a completed candle's actual close, not a forecast -- so
the record exactly `horizon` completed bars later tells us what really happened next. No
separate historical candle database is needed, and coverage grows automatically the longer the
resident worker keeps running.

This is a confirmation gate, not a signal generator: it only ever downgrades a BUY that the
multiframe rules already approved to STAY, never the reverse.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

from .execution_rules import timestamp
from .forecast_time import SECONDS
from .technical import FEATURES

# Technical regime + the model's own view of this forecast, all scale-free so pooling across
# symbols (BTC's RSI=70 and DOGE's RSI=70 count as the same "regime") is meaningful.
FORECAST_FEATURE_NAMES = ('forecast_slope_pct', 'forecast_consistency', 'few_shot_log_return')
FEATURE_NAMES = tuple(FEATURES) + FORECAST_FEATURE_NAMES


def feature_vector(record):
    """None on any missing/non-finite input -- callers must treat that as 'cannot compare'."""
    tf = record.get('technical_features') or {}
    try:
        last_close = float(record['last_close'])
        if last_close <= 0:
            return None
        values = [float(tf[k]) for k in FEATURES]
        values.append(float(record['forecast_slope']) / last_close)
        values.append(float(record['forecast_consistency']))
        values.append(float(record['few_shot_log_return']))
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    return values


def load_history(path):
    rows = []
    if not Path(path).exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def build_analog_db(rows, interval, horizon=5):
    """[{vector, realized_return, symbol, as_of}, ...] for every archived forecast whose
    horizon has already completed in the SAME record stream. A record only counts if the
    record exactly `horizon` bars later exists -- a gap (retry, disconnection, symbol dropped
    from the universe for a while) correctly excludes that point rather than mislabeling it
    against a longer, noisier gap."""
    if interval not in SECONDS:
        raise ValueError(f'Unsupported interval: {interval}')
    step = SECONDS[interval] * horizon
    by_symbol = defaultdict(dict)
    for r in rows:
        if r.get('interval') != interval:
            continue
        try:
            by_symbol[r['symbol']][timestamp(r['as_of'])] = r
        except (KeyError, ValueError, TypeError):
            continue
    db = []
    for symbol, by_time in by_symbol.items():
        for t, r in by_time.items():
            future = by_time.get(t + step)
            if future is None:
                continue
            vector = feature_vector(r)
            if vector is None:
                continue
            try:
                realized_return = float(future['last_close']) / float(r['last_close']) - 1
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                continue
            if not math.isfinite(realized_return):
                continue
            db.append(dict(vector=vector, realized_return=realized_return, symbol=symbol, as_of=r['as_of']))
    return db


def _standardize_params(db):
    n = len(db[0]['vector'])
    means = [sum(row['vector'][i] for row in db) / len(db) for i in range(n)]
    variances = [sum((row['vector'][i] - means[i]) ** 2 for row in db) / len(db) for i in range(n)]
    stds = [math.sqrt(v) or 1. for v in variances]
    return means, stds


def nearest_analogs(current_vector, db, k=30):
    """k nearest db rows to current_vector by z-scored Euclidean distance (each feature scaled
    by the db's own mean/std, so no dimension dominates just for having larger raw units)."""
    if not db or current_vector is None or len(current_vector) != len(db[0]['vector']):
        return []
    means, stds = _standardize_params(db)
    def z(vector):
        return [(v - m) / s for v, m, s in zip(vector, means, stds)]
    target = z(current_vector)
    scored = [(math.dist(target, z(row['vector'])), row) for row in db]
    scored.sort(key=lambda pair: pair[0])
    return [row for _, row in scored[:k]]


def analog_signal(current_vector, db, k=30):
    """Empirical outcome distribution of the k nearest historical analogs."""
    neighbors = nearest_analogs(current_vector, db, k)
    n = len(neighbors)
    if n == 0:
        return dict(n=0, win_rate=None, mean_return=None, median_return=None)
    returns = sorted(row['realized_return'] for row in neighbors)
    mid = n // 2
    median = returns[mid] if n % 2 else (returns[mid - 1] + returns[mid]) / 2
    return dict(n=n, win_rate=sum(1 for r in returns if r > 0) / n,
                mean_return=sum(returns) / n, median_return=median)


def confirms_buy(signal, min_win_rate=0.55, min_mean_return=0., min_n=15):
    """Pure gate. Refuses to vouch on a thin sample rather than pretending confidence, and
    only confirms when the historical analogs actually agree this setup has worked out."""
    if signal['n'] < min_n:
        return False, f"유사 사례 부족({signal['n']}개, 최소 {min_n}개 필요)"
    if signal['win_rate'] < min_win_rate:
        return False, f"과거 유사 패턴 승률 {signal['win_rate']:.0%} < 기준 {min_win_rate:.0%}"
    if signal['mean_return'] < min_mean_return:
        return False, f"과거 유사 패턴 평균 수익률 {signal['mean_return']:.2%} < 기준 {min_mean_return:.2%}"
    return True, f"유사 사례 {signal['n']}개 · 승률 {signal['win_rate']:.0%} · 평균 {signal['mean_return']:.2%}"

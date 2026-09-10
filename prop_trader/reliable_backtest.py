"""Cash-constrained spot portfolio replay of nonoverlapping five-candle forecasts."""
from collections import defaultdict
from datetime import datetime, timedelta
import math

import numpy as np
from .forecast_time import SECONDS


def portfolio(rows, bars, interval, fee=.001, slippage=.001, capital=10_000., max_positions=3,
              allocation=.2, min_edge=.002):
    if not (capital > 0 and 0 <= fee < .1 and 0 <= slippage < .1 and 0 < allocation <= 1 and max_positions >= 1):
        raise ValueError('Invalid portfolio configuration')
    delta = timedelta(seconds=SECONDS[interval])
    book = {(b.symbol, b.timestamp): b for b in bars}
    entries, closes = defaultdict(list), defaultdict(list)
    all_origins = sorted({r['target_start'] for r in rows})
    used = set()
    for r in rows:
        key = (r['symbol'], r['target_start'])
        if key in used:
            raise ValueError('Duplicate forecast origin')
        used.add(key)
        start, end = (datetime.fromisoformat(r[k]) for k in ('target_start', 'target_end'))
        if end - start != 4 * delta:
            raise ValueError('Expected exactly five target candles')
        # Features end at prior close. Execution uses actual NEXT candle open.
        if datetime.fromisoformat(r['context_end']) + delta != start:
            raise ValueError('Noncausal forecast boundary')
        for n in range(5):
            if (r['symbol'], (start + n * delta).isoformat()) not in book:
                raise ValueError('Missing holding-period candle')
        entries[start].append(r)
    if not entries:
        raise ValueError('No forecast windows')
    first = min(entries)
    last = max(datetime.fromisoformat(r['target_end']) + delta for r in rows)
    timeline = [first + n * delta for n in range(int((last-first)/delta)+1)]
    cash, positions, events, curve = capital, {}, [], []
    peak, dd = capital, 0.
    for ts in timeline:
        # Exits at the just-completed fifth candle close precede new open entries.
        for symbol in sorted(list(positions)):
            p = positions[symbol]
            if p['exit_time'] == ts:
                price = book[(symbol, (ts-delta).isoformat())].close * (1-slippage)
                proceeds = p['qty'] * price * (1-fee)
                cash += proceeds
                events.append(dict(symbol=symbol, entry_time=p['entry_time'], exit_time=ts.isoformat(),
                    forecast_origin=p['entry_time'], pnl=proceeds-p['spent'], spent=p['spent'],
                    net_return=proceeds/p['spent']-1))
                del positions[symbol]
        def equity(at_open=False):
            value = cash
            for symbol, p in positions.items():
                key = (symbol, ts.isoformat() if at_open else (ts-delta).isoformat())
                if key not in book:
                    raise ValueError('Missing mark-to-market candle')
                mark = book[key].open if at_open else book[key].close
                value += p['qty'] * mark
            return value
        candidates = sorted(entries.get(ts, []), key=lambda r: (-float(r['pred_5']), r['symbol']))
        for r in candidates:
            symbol = r['symbol']
            pred = float(r['pred_5'])
            if not math.isfinite(pred):
                raise ValueError('Nonfinite forecast')
            # Exact proportional costs, not a quoted exchange price translation.
            estimated = math.exp(pred)*(1-slippage)*(1-fee)/((1+slippage)*(1+fee))-1
            if estimated <= min_edge or symbol in positions or len(positions) >= max_positions:
                continue
            budget = min(cash, max(0, equity(at_open=True))*allocation)
            if budget <= 0:
                continue
            price = book[(symbol, ts.isoformat())].open*(1+slippage)
            qty = budget/(price*(1+fee))
            cash -= budget
            positions[symbol] = dict(qty=qty, spent=budget, entry_time=ts.isoformat(),
                exit_time=datetime.fromisoformat(r['target_end'])+delta)
        value = equity(at_open=True) if positions and ts < last else cash
        peak = max(peak, value); dd = max(dd, 1-value/peak)
        curve.append(dict(timestamp=ts.isoformat(), equity=value, cash=cash, positions=len(positions)))
    if positions or abs(sum(t['pnl'] for t in events) - (cash-capital)) > 1e-6:
        raise AssertionError('Portfolio accounting reconciliation failed')
    gains = sum(max(0,t['pnl']) for t in events)
    losses = -sum(min(0,t['pnl']) for t in events)
    # Confidence on origin-level PnL blocks preserves shared market shocks among coins.
    pnl_by_origin = defaultdict(float)
    for e in events:
        pnl_by_origin[e['forecast_origin']] += e['pnl']
    values = np.array([pnl_by_origin[o] for o in all_origins])
    rng = np.random.default_rng(42)
    draws = []
    for _ in range(1000):
        starts = rng.integers(0,len(values),size=math.ceil(len(values)/2))
        sample = np.concatenate([values[(np.arange(2)+s)%len(values)] for s in starts])[:len(values)]
        draws.append(sample.sum()/capital)
    ci = np.quantile(draws,[.025,.975]).tolist()
    return dict(final_equity=cash, return_fraction=cash/capital-1, max_drawdown=dd,
        trades=len(events), wins=sum(t['pnl']>0 for t in events),
        profit_factor=gains/losses if losses else None, origin_blocks=len(values),
        bootstrap_return_ci95=ci, fee=fee, slippage=slippage,
        note='Long-only, 5-candle holding, mark-to-market at candle opens; no intrabar stops. Block bootstrap is descriptive, not a profit guarantee.'), events, curve

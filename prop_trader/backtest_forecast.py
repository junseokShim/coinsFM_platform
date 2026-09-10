"""Turn TimesFM forecast signals into a simulated trading backtest. No orders are sent.

Reuses `eval_predictions.csv` written by evaluator.py (raw per-window predicted and
actual log returns from the same walk-forward test split) -- no model inference here,
so this is fast and can be re-run cheaply with different threshold/notional settings.

Simulation and its stated simplifications:
- One trade per non-HOLD test window: enter at the window's anchor close, exit at the
  ACTUAL close of candle t+5 (not the predicted one). Direction from decide_signal().
- No intrabar stop-loss/take-profit: eval_predictions.csv only has close-path returns,
  not OHLC, so we cannot know the path within the holding period. This can understate
  risk relative to a strategy that would have been stopped out intrabar.
- Fixed notional per trade, no compounding, no concurrent-position or margin limits.
  This measures signal quality (would the direction calls have made money net of
  cost), not a portfolio-level capital simulation.
- Round-trip cost = 2 x (fee + slippage), taken from prop_trader.engine.Config defaults
  (0.1% fee, 0.1% slippage per leg), applied to every non-HOLD trade.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .engine import Config
from .forecast_time import path_stats, decide_signal

_cost = Config()
ROUND_TRIP_COST = 2 * (_cost.fee + _cost.slippage)


def load_predictions(path):
    by_model = defaultdict(list)
    with path.open() as f:
        for row in csv.DictReader(f):
            by_model[row['model']].append(row)
    return by_model


def simulate(rows, threshold, min_consistency, notional, starting_capital):
    trades = []
    for r in rows:
        anchor = float(r['anchor'])
        pred_prices = [anchor * np.exp(float(r[f'pred_{h}'])) for h in range(1, 6)]
        stats = path_stats(anchor, pred_prices)
        signal = decide_signal(stats, threshold=threshold, min_consistency=min_consistency)
        if signal == 'HOLD':
            continue
        actual_final_return = np.exp(float(r['actual_5'])) - 1
        raw_return = actual_final_return if signal == 'LONG' else -actual_final_return
        net_return = raw_return - ROUND_TRIP_COST
        trades.append(dict(symbol=r['symbol'], target_start=r['target_start'], target_end=r['target_end'],
                            signal=signal, predicted_t5_return=stats['return_by_step'][-1],
                            raw_return=raw_return, net_return=net_return, pnl=notional * net_return))
    trades.sort(key=lambda t: t['target_end'])

    equity = starting_capital
    peak = starting_capital
    max_dd = 0.0
    curve = []
    for t in trades:
        equity += t['pnl']
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1)
        curve.append(dict(target_end=t['target_end'], equity=equity, drawdown=equity / peak - 1))

    n = len(trades)
    returns = np.array([t['net_return'] for t in trades]) if trades else np.array([])
    wins = int((returns > 0).sum()) if n else None
    summary = dict(
        trades=n, hold_windows=len(rows) - n,
        win_rate=(wins / n if n else None),
        total_pnl=equity - starting_capital,
        total_return_pct=(equity / starting_capital - 1) * 100,
        avg_net_return_pct=float(returns.mean()) * 100 if n else None,
        return_std_pct=float(returns.std()) * 100 if n else None,
        sharpe_like=float(returns.mean() / returns.std()) if n > 1 and returns.std() > 0 else None,
        max_drawdown_pct=max_dd * 100, final_equity=equity, round_trip_cost_pct=ROUND_TRIP_COST * 100)
    return summary, trades, curve


def buy_and_hold_benchmark(rows):
    """Reference only: average actual t+5 return with no signal, no cost, every window."""
    returns = [np.exp(float(r['actual_5'])) - 1 for r in rows]
    return dict(avg_return_pct=float(np.mean(returns)) * 100 if returns else None, windows=len(returns))


def write_csv(path, rows, fieldnames):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def run_interval(run_dir, threshold, min_consistency, notional, starting_capital):
    pred_path = run_dir / 'eval_predictions.csv'
    if not pred_path.exists():
        raise FileNotFoundError(f'{pred_path} missing; run `python -m prop_trader.evaluator` first')
    by_model = load_predictions(pred_path)
    summaries = {}
    for model, rows in by_model.items():
        summary, trades, curve = simulate(rows, threshold, min_consistency, notional, starting_capital)
        summary['buy_and_hold'] = buy_and_hold_benchmark(rows)
        summaries[model] = summary
        out_dir = run_dir / 'backtest' / model
        out_dir.mkdir(parents=True, exist_ok=True)
        if trades:
            write_csv(out_dir / 'trades.csv', trades, list(trades[0]))
            write_csv(out_dir / 'equity.csv', curve, list(curve[0]))
        (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summaries


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--roots', nargs='+', type=Path,
                   default=[Path('runs/forecast5/1m'), Path('runs/forecast5/1h'), Path('runs/forecast5/1d')])
    p.add_argument('--threshold', type=float, default=0.005, help='Default signal threshold for all intervals')
    p.add_argument('--interval-threshold', action='append', default=[], metavar='INTERVAL=VALUE',
                    help='Per-interval override, e.g. --interval-threshold 1m=0.001 (predicted 1m moves are much '
                         'smaller than 1h/1d, so a single global threshold under- or over-fires by timeframe)')
    p.add_argument('--min-consistency', type=float, default=0.5)
    p.add_argument('--notional', type=float, default=1000.0)
    p.add_argument('--starting-capital', type=float, default=10_000.0)
    args = p.parse_args()
    overrides = dict(kv.split('=') for kv in args.interval_threshold)
    thresholds = {k: float(v) for k, v in overrides.items()}

    all_summaries = {}
    for run_dir in args.roots:
        interval = run_dir.name
        threshold = thresholds.get(interval, args.threshold)
        summaries = run_interval(run_dir, threshold, args.min_consistency, args.notional, args.starting_capital)
        all_summaries[interval] = summaries
        print(interval, f'(threshold={threshold:.3%})', json.dumps(summaries, indent=2), flush=True)

    order = ['naive', 'momentum', 'zero_shot', 'lora', 'lora_fewshot']
    threshold_note = ', '.join(f'{interval} {thresholds.get(interval, args.threshold):.2%}' for interval in all_summaries)
    lines = ['# Forecast-signal trading backtest (walk-forward test split)', '',
             f'Signal: t+5 predicted return beyond the per-interval threshold ({threshold_note}) with path '
             f'consistency >= {args.min_consistency:.0%} and no adverse excursion past the threshold '
             '(see `forecast_time.decide_signal`). Thresholds differ by timeframe because predicted move size '
             'scales with the interval -- 1m rarely predicts a 0.5% move, so a single global threshold would '
             'just never fire on 1m (0 trades for every model).',
             f'Cost: {ROUND_TRIP_COST:.2%} round-trip (fee+slippage, both legs, from `engine.Config` defaults).',
             f'Sizing: fixed {args.notional:.0f} USDT notional per trade, no compounding, no concurrent-position '
             'limit -- this measures signal quality, not a portfolio simulation. Exit is the actual close at t+5, '
             'not simulated intrabar stop-loss/take-profit (only close-path forecasts are available).', '',
             '| Interval | Model | Trades | Win rate | Avg net return/trade | Total return | Max DD | Buy&hold avg |',
             '|---|---|---:|---:|---:|---:|---:|---:|']

    def fmt_pct(v):
        return f'{v:.2f}%' if v is not None else 'N/A'

    for interval in ('1m', '1h', '1d'):
        for model in order:
            s = all_summaries.get(interval, {}).get(model)
            if s is None:
                continue
            win_rate = f"{s['win_rate']:.1%}" if s['win_rate'] is not None else 'N/A'
            lines.append(f"| {interval} | {model} | {s['trades']} | {win_rate} | "
                         f"{fmt_pct(s['avg_net_return_pct'])} | {fmt_pct(s['total_return_pct'])} | "
                         f"{fmt_pct(s['max_drawdown_pct'])} | {fmt_pct(s['buy_and_hold']['avg_return_pct'])} |")

    lines += ['', 'Trades = non-HOLD windows out of 96 test windows per interval (6 symbols combined).',
              'Buy&hold avg = mean actual t+5 return with no signal and no cost, for reference only -- it is',
              'not a strategy (always long, ignores cost). A model beating naive/momentum on direction accuracy',
              '(see EVAL_REPORT.md) does not automatically mean it is profitable here: few trades, small sample,',
              'no slippage-on-size modeling, and no intrabar stop-loss all bias this in ways not corrected for.']
    Path('runs/forecast5/BACKTEST_REPORT.md').write_text('\n'.join(lines) + '\n')
    print('Wrote runs/forecast5/BACKTEST_REPORT.md', flush=True)


if __name__ == '__main__':
    main()

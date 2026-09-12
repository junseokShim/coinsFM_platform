"""Walk-forward search for a stop-loss/take-profit rule, using real intrabar OHLC.

Why this exists: STOP_FRACTION/REWARD_MULTIPLE in live_trader.py (2.5%/2.5%) were set by
looking once at the realized entry->exit range of 29 live horizon-timeout trades (see the
7b93b95 commit message) -- not backtested, not compared against any alternative, not
out-of-sample validated. backtest_forecast.py can't fill that gap either: it only has
eval_predictions.csv's close-path returns, no OHLC, so it deliberately never simulates
intrabar stops (see its own module docstring). This module joins the same eval_predictions.csv
against the frozen data/current/{interval}.csv candles to get real highs/lows, and grid-searches
an actual exit rule with a train/validation/test split so the winner isn't just curve-fit.

Two exit families, evaluated on the same candidate trades and the same intrabar fill rule
(stop wins on a same-bar hit; gaps fill at open -- matches engine.Desk.step exactly):

  fixed    - stop_fraction and target_fraction=stop_fraction*reward_multiple, constant for
             every trade. This is what live_trader.py does today; the grid includes the
             current live pair (2.5%/2.5%) as one candidate among many, not a privileged default.
  adaptive - stop_fraction fixed, but the take-profit is set per trade from CoinsFM's own
             5-step predicted path: target = anchor + shrink*(predicted_peak - anchor).
             shrink is grid-searched on the train split only, exactly like the fixed grid, so a
             systematically overconfident forecast doesn't just hand back an unreachable target.
             If the predicted path never rises above the actual entry fill, there is no
             synthetic target for that trade -- it falls back to stop + horizon timeout only.

Reuses eval_predictions.csv (already-computed walk-forward forecasts, no model inference) for
model == 'lora_fewshot', the model actually deployed live. Only windows that pass the live
entry gate (net expected return > ENTRY_THRESHOLD after cost) are included, so the trade
population matches what live_trader.py would actually have entered. Fixed notional per trade,
no compounding, no concurrent-position limit or cash constraint -- this isolates exit-rule
quality the same way backtest_forecast.py isolates signal quality (see reliable_backtest.py's
portfolio() for the cash-constrained layer; re-check any winning rule there before deploying).

Sample size: eval_predictions.csv's walk-forward test split is small (order of 100 windows per
interval across 6 symbols), and the entry gate filters it down further -- 1m has essentially
zero qualifying windows (matches BACKTEST_REPORT.md: 1m has no edge at all), 1h and 1d have a
few dozen at most. A single chronological train/validation/test split would leave a handful of
test trades, so parameter selection instead uses expanding-window walk-forward cross-validation
(grid-search on candidates[:i], evaluate strictly on candidates[i:i+step], slide forward): every
window after an initial seed serves as an out-of-sample test point in some fold, and no fold's
choice ever sees its own test data. This is still a small-sample result, not proof of future
profitability -- it is a better-supported candidate than a constant picked by eyeballing 29
trades once, nothing more. Same-bar stop/target priority is an assumption (real tick-level path
is unknown, matching engine.Desk's existing convention), and OHLC candles do not capture
intra-candle execution risk from thin order books.
"""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .__main__ import load
from .forecast_time import SECONDS

FEE = 0.001
SLIPPAGE = 0.001
MIN_EDGE = 0.002  # matches ENTRY_THRESHOLD in live_trader.py / min_edge in reliable_backtest.portfolio
STARTING_CAPITAL = 10_000.
NOTIONAL = 1_000.

STOP_GRID = (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05)
REWARD_GRID = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
SHRINK_GRID = (0.2, 0.35, 0.5, 0.7, 0.85, 1.0)
LIVE_BASELINE = dict(stop_fraction=0.025, reward_multiple=1.0)  # current live_trader.py constants


def load_candidates(run_dir, csv_path, interval, min_edge=MIN_EDGE):
    """lora_fewshot rows from eval_predictions.csv that clear the entry gate, joined with
    the actual five holding-period candles' OHLC (chronologically sorted, oldest first).

    min_edge defaults to ENTRY_THRESHOLD (0.2%), but the live server currently runs with
    --entry-threshold 0 (any post-cost-positive edge) -- pass min_edge=0 to match that
    population exactly instead of a threshold the live config no longer uses."""
    pred_path = run_dir / 'eval_predictions.csv'
    if not pred_path.exists():
        raise FileNotFoundError(f'{pred_path} missing; run `python -m prop_trader.evaluator` first')
    book = {(b.symbol, b.timestamp): b for b in load(csv_path)}
    delta = timedelta(seconds=SECONDS[interval])
    candidates = []
    with pred_path.open() as f:
        for row in csv.DictReader(f):
            if row['model'] != 'lora_fewshot':
                continue
            anchor = float(row['anchor'])
            pred_prices = [anchor * math.exp(float(row[f'pred_{h}'])) for h in range(1, 6)]
            est_net = pred_prices[-1] * (1 - SLIPPAGE) * (1 - FEE) / (anchor * (1 + SLIPPAGE) * (1 + FEE)) - 1
            if est_net <= min_edge:
                continue
            start = datetime.fromisoformat(row['target_start'])
            path, complete = [], True
            for n in range(5):
                key = (row['symbol'], (start + n * delta).isoformat())
                if key not in book:
                    complete = False
                    break
                path.append(book[key])
            if not complete:
                continue
            candidates.append(dict(symbol=row['symbol'], target_start=row['target_start'],
                anchor=anchor, pred_prices=pred_prices, predicted_net_return=est_net, path=path))
    candidates.sort(key=lambda c: (c['target_start'], c['symbol']))
    return candidates


def expanding_folds(candidates, min_train=4, n_folds=4):
    """candidates[:i] trains fold i's chosen config; candidates[i:i+step] is that fold's test --
    strictly later in time, never seen by the grid search that picked the config for it. Every
    window after the initial min_train seed ends up as an out-of-sample test point in exactly
    one fold, which matters a lot when the whole candidate set is a few dozen trades."""
    n = len(candidates)
    if n < min_train + 2:
        return []
    step = max(1, (n - min_train) // n_folds)
    cuts = list(range(min_train, n, step))
    if cuts[-1] != n:
        cuts.append(n)
    return [(candidates[:cuts[i]], candidates[cuts[i]:cuts[i + 1]])
            for i in range(len(cuts) - 1) if candidates[cuts[i]:cuts[i + 1]]]


def _walk(path, fill_price, stop_level, target_level):
    for i, bar in enumerate(path):
        stopped = bar.low <= stop_level
        hit_target = target_level is not None and bar.high >= target_level
        if stopped:
            # Stop wins when both levels occur in one OHLC bar; gaps fill at open (engine.Desk.step).
            raw_exit = min(bar.open, stop_level)
            return raw_exit * (1 - SLIPPAGE), 'stop_loss', i + 1
        if hit_target:
            return target_level * (1 - SLIPPAGE), 'take_profit', i + 1
    return path[-1].close * (1 - SLIPPAGE), 'horizon_timeout', len(path)


def simulate_fixed(candidate, stop_fraction, target_fraction):
    fill_price = candidate['path'][0].open * (1 + SLIPPAGE)
    stop_level = fill_price * (1 - stop_fraction)
    target_level = fill_price * (1 + target_fraction) if target_fraction is not None else None
    exit_price, reason, bars_held = _walk(candidate['path'], fill_price, stop_level, target_level)
    net_return = exit_price * (1 - FEE) / (fill_price * (1 + FEE)) - 1
    return dict(symbol=candidate['symbol'], target_start=candidate['target_start'],
                reason=reason, bars_held=bars_held, net_return=net_return)


def simulate_adaptive(candidate, stop_fraction, shrink):
    """Take-profit is CoinsFM's own predicted peak price, shrunk toward the anchor by a factor
    fit out-of-sample -- not a fixed percentage applied identically to every trade."""
    fill_price = candidate['path'][0].open * (1 + SLIPPAGE)
    anchor, peak_price = candidate['anchor'], max(candidate['pred_prices'])
    raw_target = anchor + shrink * (peak_price - anchor)
    # No synthetic target if the shrunk forecast doesn't even clear the actual entry fill.
    target_level = raw_target if raw_target > fill_price * 1.0005 else None
    stop_level = fill_price * (1 - stop_fraction)
    exit_price, reason, bars_held = _walk(candidate['path'], fill_price, stop_level, target_level)
    net_return = exit_price * (1 - FEE) / (fill_price * (1 + FEE)) - 1
    return dict(symbol=candidate['symbol'], target_start=candidate['target_start'],
                reason=reason, bars_held=bars_held, net_return=net_return)


def summarize(trades):
    n = len(trades)
    if n == 0:
        return dict(trades=0, win_rate=None, avg_net_return_pct=None, total_pnl=None,
                    total_return_pct=None, max_drawdown_pct=None, profit_factor=None, reasons={})
    returns = np.array([t['net_return'] for t in trades])
    pnls = returns * NOTIONAL
    ordered = sorted(trades, key=lambda t: t['target_start'])
    equity = STARTING_CAPITAL
    peak = STARTING_CAPITAL
    max_dd = 0.0
    for t in ordered:
        equity += t['net_return'] * NOTIONAL
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1)
    gains = float(pnls[pnls > 0].sum())
    losses = float(-pnls[pnls < 0].sum())
    return dict(trades=n, win_rate=float((returns > 0).mean()), avg_net_return_pct=float(returns.mean()) * 100,
                total_pnl=float(pnls.sum()), total_return_pct=float(pnls.sum()) / STARTING_CAPITAL * 100,
                max_drawdown_pct=max_dd * 100, profit_factor=(gains / losses if losses > 0 else None),
                reasons=dict(Counter(t['reason'] for t in trades)))


def _rank_key(summary):
    return summary['total_pnl'] if summary['total_pnl'] is not None else float('-inf')


def grid_search_fixed(train):
    results = []
    for stop in STOP_GRID:
        for mult in REWARD_GRID:
            target = stop * mult
            trades = [simulate_fixed(c, stop, target) for c in train]
            results.append(dict(stop_fraction=stop, reward_multiple=mult, target_fraction=target,
                                 **summarize(trades)))
    results.sort(key=_rank_key, reverse=True)
    return results


def grid_search_adaptive(train):
    results = []
    for stop in STOP_GRID:
        for shrink in SHRINK_GRID:
            trades = [simulate_adaptive(c, stop, shrink) for c in train]
            results.append(dict(stop_fraction=stop, shrink=shrink, **summarize(trades)))
    results.sort(key=_rank_key, reverse=True)
    return results


def cv_evaluate(folds, grid_search_fn, simulate_fn, param_keys):
    """Grid-search each fold's train slice, apply that fold's single winner to that fold's own
    (strictly later, unseen) test slice, and pool every fold's test trades into one out-of-
    sample sample. Returns the pooled trades plus the winning params per fold (their spread is
    itself informative: if the grid search picks a wildly different config every fold, that is
    a sign the "winner" is noise, not signal)."""
    pooled_trades, chosen_per_fold = [], []
    for train, test in folds:
        ranked = grid_search_fn(train)
        params = {k: ranked[0][k] for k in param_keys}
        pooled_trades.extend(simulate_fn(c, **params) for c in test)
        chosen_per_fold.append(params)
    return dict(chosen_per_fold=chosen_per_fold, pooled_trades=pooled_trades,
                pooled_summary=summarize(pooled_trades))


def paired_bootstrap_ci(candidate_trades_a, candidate_trades_b, n_draws=1000, seed=42):
    """Block bootstrap (block size 2, grouped by origin timestamp) on the per-trade net_return
    difference a-b, both simulated over the identical candidate set -- so this is a paired
    comparison, not two independent samples. Mirrors reliable_backtest.portfolio's bootstrap."""
    by_origin = defaultdict(list)
    for ta, tb in zip(candidate_trades_a, candidate_trades_b):
        by_origin[ta['target_start']].append(ta['net_return'] - tb['net_return'])
    origins = sorted(by_origin)
    blocks = np.array([np.mean(by_origin[o]) for o in origins])
    if len(blocks) == 0:
        return dict(origin_groups=0, mean_diff=None, ci95=None)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_draws):
        starts = rng.integers(len(blocks), size=(len(blocks) + 1) // 2)
        sample = np.concatenate([blocks[(np.arange(2) + s) % len(blocks)] for s in starts])[:len(blocks)]
        draws.append(float(sample.mean()))
    return dict(origin_groups=len(blocks), mean_diff=float(blocks.mean()),
                ci95=np.quantile(draws, [.025, .975]).tolist())


def run_interval(run_dir, csv_path, interval, min_edge=MIN_EDGE):
    candidates = load_candidates(run_dir, csv_path, interval, min_edge=min_edge)
    folds = expanding_folds(candidates)
    out = dict(interval=interval, total_candidates=len(candidates), folds=len(folds), min_edge=min_edge)
    if not folds:
        out['error'] = 'Too few entry-gate-qualifying windows for cross-validation (need >=6)'
        return out

    fixed_cv = cv_evaluate(folds, grid_search_fixed, simulate_fixed, ('stop_fraction', 'target_fraction'))
    adaptive_cv = cv_evaluate(folds, grid_search_adaptive, simulate_adaptive, ('stop_fraction', 'shrink'))

    # Same pooled out-of-sample population (union of every fold's test slice, same order) so the
    # fixed live-baseline/horizon-only rules -- which need no fitting -- are directly comparable
    # trade-for-trade against the CV-selected rules.
    pooled_test = [c for _, test in folds for c in test]
    live_target = LIVE_BASELINE['stop_fraction'] * LIVE_BASELINE['reward_multiple']
    test_trades = dict(
        live_baseline=[simulate_fixed(c, LIVE_BASELINE['stop_fraction'], live_target) for c in pooled_test],
        horizon_only=[simulate_fixed(c, LIVE_BASELINE['stop_fraction'], None) for c in pooled_test],
        fixed_chosen=fixed_cv['pooled_trades'],
        adaptive_chosen=adaptive_cv['pooled_trades'],
    )
    test_summary = {name: summarize(trades) for name, trades in test_trades.items()}
    ci = dict(
        fixed_vs_live_baseline=paired_bootstrap_ci(test_trades['fixed_chosen'], test_trades['live_baseline']),
        adaptive_vs_live_baseline=paired_bootstrap_ci(test_trades['adaptive_chosen'], test_trades['live_baseline']),
        adaptive_vs_fixed=paired_bootstrap_ci(test_trades['adaptive_chosen'], test_trades['fixed_chosen']),
    )
    out.update(
        fixed_chosen_per_fold=fixed_cv['chosen_per_fold'], adaptive_chosen_per_fold=adaptive_cv['chosen_per_fold'],
        live_baseline=LIVE_BASELINE, test=test_summary, test_bootstrap_ci95=ci)
    return out


def write_report(all_results):
    lines = ['# 청산 규칙 탐색: 고정 손절/익절 그리드 vs CoinsFM 예측경로 기반 적응형 익절', '',
             '`STOP_FRACTION=2.5%, REWARD_MULTIPLE=1`(현재 `live_trader.py` 값)은 라이브 거래 29건의 실현폭을',
             '한 번 육안으로 보고 정한 값이며 백테스트나 표본외 검증을 거치지 않았다. 이 리포트는',
             '`eval_predictions.csv`(실제 배포 모델 `lora_fewshot`의 걸어온-미래 예측)를 실제 캔들 고가/저가와',
             '결합해 그 값을 실제로 검증하고, 대안(다른 고정 쌍, 예측경로 기반 적응형 목표가)과 비교한다.',
             '', '방법: 진입 게이트(비용 차감 후 기대수익 > 0.2%)를 통과하는 구간만 사용. 표본이 작아(주기당',
             '수 건~수십 건) 단일 train/validation/test 3분할 대신 확장형 walk-forward 교차검증을 쓴다:',
             '매 fold는 그 이전 구간(candidates[:i])에서만 그리드서치로 설정을 고르고, 그 설정을 strictly',
             '이후 구간(candidates[i:i+step])에서만 평가한다. 모든 fold의 표본외 결과를 모아 하나의 pooled',
             '테스트 성과로 보고한다 — 모든 거래가 정확히 한 번, 자신을 고르는 데 쓰이지 않은 규칙으로 평가된다.',
             '동일 봉에서 손절·익절이 모두 닿으면 손절 우선, 갭은 시가 체결(`engine.Desk.step`과 동일 규칙).',
             '표본이 작다는 점을 유의하라 — 아래 표본 크기를 항상 함께 보라.', '']
    for r in all_results:
        interval = r['interval']
        lines.append(f"## {interval}")
        lines.append('')
        lines.append(f"진입 게이트(min_edge={r.get('min_edge', MIN_EDGE):.2%}) 통과 표본 "
                     f"{r['total_candidates']}건, {r.get('folds', 0)}개 fold")
        lines.append('')
        if 'error' in r:
            lines.append(f"**{r['error']}** -- 이 주기는 표본이 너무 작아 결과를 신뢰할 수 없어 생략한다.")
            lines.append('')
            continue
        lines.append('### fold별로 선택된 설정 (매 fold, 그 이전 구간만으로 그리드서치)')
        lines.append('')
        lines.append('| fold | 고정: stop/target | 적응형: stop/shrink |')
        lines.append('|---:|---|---|')
        for i, (fp, ap) in enumerate(zip(r['fixed_chosen_per_fold'], r['adaptive_chosen_per_fold']), 1):
            lines.append(f"| {i} | {fp['stop_fraction']:.1%} / {fp['target_fraction']:.1%} | "
                          f"{ap['stop_fraction']:.1%} / {ap['shrink']:.2f} |")
        lines.append('')
        lines.append('### pooled 표본외 성과 (모든 fold의 test 거래를 합산, 자신을 고르는 데 쓰이지 않은 설정으로만 평가)')
        lines.append('')
        lines.append('| 설정 | 거래수 | 승률 | 총손익 | 총수익률 | MDD | Profit factor | stop_loss/tp/timeout |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|---|')
        labels = dict(live_baseline='현재 라이브 (2.5%/2.5%)', horizon_only='익절 없음(손절+만료만)',
                       fixed_chosen='고정 그리드 (CV 선택)', adaptive_chosen='적응형(예측경로, CV 선택)')
        for key, label in labels.items():
            s = r['test'][key]
            wr = f"{s['win_rate']:.1%}" if s['win_rate'] is not None else 'N/A'
            pf = f"{s['profit_factor']:.2f}" if s['profit_factor'] is not None else 'N/A'
            reasons = s['reasons']
            reason_str = f"{reasons.get('stop_loss', 0)}/{reasons.get('take_profit', 0)}/{reasons.get('horizon_timeout', 0)}"
            lines.append(f"| {label} | {s['trades']} | {wr} | {s['total_pnl']:.1f} | {s['total_return_pct']:.2f}% | "
                          f"{s['max_drawdown_pct']:.2f}% | {pf} | {reason_str} |")
        lines.append('')
        ci = r['test_bootstrap_ci95']
        for key, label in (('fixed_vs_live_baseline', '고정 CV선택값 - 현재 라이브'),
                            ('adaptive_vs_live_baseline', '적응형 CV선택값 - 현재 라이브'),
                            ('adaptive_vs_fixed', '적응형 CV선택값 - 고정 CV선택값')):
            c = ci[key]
            if c['ci95'] is None:
                continue
            lines.append(f"- {label}: 거래당 순수익률 차이 평균 {c['mean_diff']:.4%}, 95% 블록부트스트랩 구간 "
                          f"[{c['ci95'][0]:.4%}, {c['ci95'][1]:.4%}] ({c['origin_groups']}개 시점 그룹)")
        lines.append('')
    lines += ['---', '',
              '**한계**: pooled 표본외 거래 수가 주기당 한 자릿수~10여 건으로 작다 — 하나의 나쁜 혹은 좋은',
              '거래가 순위를 뒤집을 수 있다. 신뢰구간이 0을 포함하면 우위를 주장할 근거가 없다는 뜻이다.',
              '고정 명목가·복리 없음·동시 포지션 한도 없음이므로 포트폴리오 자본 시뮬레이션이 아니라 청산',
              '규칙 자체의 품질만 격리해서 본다(`reliable_backtest.portfolio`가 자본 제약 레이어). 동일 봉 내',
              '손절·익절 우선순위는 가정이며 실제 틱 경로는 알 수 없다. 여기서 선택된 설정이라도',
              '`live_server.py --mode paper`로 먼저 모의 검증한 뒤에만 실거래 상수를 바꿔야 한다.']
    Path('runs/forecast5/EXIT_SWEEP_REPORT.md').write_text('\n'.join(lines) + '\n')
    print('Wrote runs/forecast5/EXIT_SWEEP_REPORT.md', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--roots', nargs='+', type=Path,
                   default=[Path('runs/forecast5/1m'), Path('runs/forecast5/1h'), Path('runs/forecast5/1d')])
    p.add_argument('--csv-dir', type=Path, default=Path('data/current'))
    p.add_argument('--min-edge', type=float, default=MIN_EDGE,
                    help='Entry gate: minimum cost-adjusted expected net return to count as a candidate trade. '
                         'Defaults to ENTRY_THRESHOLD (0.2%%); pass 0 to match a live server run with '
                         '--entry-threshold 0 instead.')
    args = p.parse_args()
    all_results = []
    for run_dir in args.roots:
        interval = run_dir.name
        csv_path = args.csv_dir / f'{interval}.csv'
        print('Sweeping exit rules for', interval, flush=True)
        result = run_interval(run_dir, csv_path, interval, min_edge=args.min_edge)
        (run_dir / 'exit_sweep.json').write_text(json.dumps(result, indent=2))
        all_results.append(result)
    write_report(all_results)


if __name__ == '__main__':
    main()

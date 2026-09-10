"""Pick the few-shot adaptation count (K completed recent examples) with the best walk-forward
accuracy, per interval, and write it where resident_forecasts.py / mtf_backtest.py read it.

Reuses evaluator.evaluate() unchanged (never retrains the backbone LoRA) -- just calls it once
per candidate `shots` value and compares the lora_fewshot model's walk-forward accuracy. Ranked
primarily by direction accuracy at the full 5-candle horizon (what an entry actually keys off
of), tie-broken by lower MAE.
"""
import argparse
import json
from pathlib import Path

CANDIDATES = (4, 8, 12, 16, 24, 32)


def rank(results):
    def key(r):
        da5 = r['da5']
        return (da5 if da5 is not None else -1., -r['mae'])
    return sorted(results, key=key, reverse=True)


def sweep(run_dir, csv_path, rolling_split=True, candidates=CANDIDATES):
    # Imported lazily: pure helpers here (rank) stay importable/testable without torch/peft,
    # which only the heavy .venv-timesfm interpreter has installed.
    from .evaluator import evaluate
    if not candidates:
        raise ValueError('candidates must be non-empty')
    results = []
    interval = None
    for shots in candidates:
        evaluation = evaluate(run_dir, csv_path, rolling_split, shots, save_predictions=False)
        interval = evaluation['interval']
        m = evaluation['models']['lora_fewshot']
        results.append(dict(shots=shots, mae=m['mae'], rmse=m['rmse'],
                             da5=m['per_horizon'][-1]['direction_accuracy'],
                             top_k_precision=m['top_k_directional_precision'], windows=m['windows']))
    ranked = rank(results)
    best = ranked[0]
    return dict(interval=interval, candidates=results, ranked_shots=[r['shots'] for r in ranked],
                best_shots=best['shots'], best=best)


def main():
    p = argparse.ArgumentParser(description='Sweep few-shot adaptation counts and pick the most accurate one')
    p.add_argument('--roots', nargs='+', type=Path,
                    default=[Path('runs/forecast5/1m'), Path('runs/forecast5/1h'), Path('runs/forecast5/1d')])
    p.add_argument('--csv-dir', type=Path, default=Path('data/live'))
    p.add_argument('--candidates', nargs='+', type=int, default=list(CANDIDATES))
    p.add_argument('--rolling-split', action='store_true', default=True)
    args = p.parse_args()
    for run_dir in args.roots:
        interval = run_dir.name
        csv_path = args.csv_dir / f'{interval}.csv'
        print('Sweeping', interval, 'shots in', args.candidates, flush=True)
        result = sweep(run_dir, csv_path, args.rolling_split, args.candidates)
        (run_dir / 'best_shots.json').write_text(json.dumps(result, indent=2))
        print(interval, '-> best_shots =', result['best_shots'], json.dumps(result['best'], indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Full-horizon, multi-baseline evaluation: MAE/RMSE/MAPE, DA@1..DA@5, Top-K precision.

Reuses the already-trained adapter in `--run` (never retrains). Reconstructs
the same chronological test split as timesfm_run.windows() and, for every
test window, produces 5-step forecasts from five candidate models:

  naive     - flat: predicted close stays at the last observed close
  momentum  - linear extrapolation of the recent average per-step log return
  zero_shot - pretrained TimesFM + LoRA disabled (no crypto adaptation at all)
  lora      - pretrained TimesFM + historical LoRA, latest context only
  lora_fewshot - lora + online adaptation on the 8 most recent completed
                 examples that end at-or-before this window's context_end
                 (walk-forward: the same protocol as timesfm_fewshot_eval.py)

All five operate on the same log-return-from-anchor target used in training,
so predicted/actual close prices are anchor*exp(log_return) and every target
is a candle that had already closed before the model saw it (see windows()
and support_windows() for the boundary checks).
"""
import argparse
import csv
import json
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
from peft import PeftModel
from transformers import TimesFm2_5ModelForPrediction

from .timesfm_run import windows
from .timesfm_predict import support_windows
from .__main__ import load
from .forecast_time import SECONDS


def momentum_forecast(x, horizon, lookback=10):
    """x is the anchored context (x[-1]==0); extrapolate the recent mean step return."""
    tail = x[-lookback:]
    step = float(np.diff(tail).mean()) if len(tail) > 1 else 0.0
    return np.array([step * h for h in range(1, horizon + 1)], dtype=np.float32)


def batched_forecast(model, items, context, horizon, device, disable_adapter):
    preds = []
    with (model.disable_adapter() if disable_adapter else nullcontext()), torch.no_grad():
        for offset in range(0, len(items), 8):
            batch = items[offset:offset + 8]
            x = torch.tensor(np.stack([r[0][-context:] for r in batch])).to(device)
            out = model(past_values=x, forecast_context_len=context, truncate_negative=False)
            pred = out.mean_predictions[:, :horizon].float().cpu().numpy()
            if not np.isfinite(pred).all():
                raise ValueError('Non-finite predictions')
            preds.append(pred)
    return np.concatenate(preds, axis=0)


def fewshot_forecast(model, initial_state, test, all_bars, context, horizon, shots, device):
    """Walk-forward: reset to the frozen LoRA before every origin, adapt on K
    completed examples ending at-or-before that origin's context_end, forecast, discard."""
    preds = []
    for x, y, meta in test:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in initial_state:
                    p.copy_(initial_state[n])
        past = [b for b in all_bars[meta['symbol']] if b.timestamp <= meta['context_end']]
        support = support_windows(past, context, horizon, shots)
        model.train()
        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-5)
        # Matches predict_symbol's training loop exactly (all support windows, batches of 4) so
        # sweeping `shots` here measures the same thing production few-shot adaptation does.
        # (For shots=8 this is offsets [0, 4], identical to the old hardcoded (0, 4) pair --
        # only other `shots` values behave differently now.)
        for offset in range(0, len(support), 4):
            sample = support[offset:offset + 4]
            sx = torch.tensor([r[0] for r in sample], dtype=torch.float32, device=device)
            sy = torch.tensor([r[1] for r in sample], dtype=torch.float32, device=device)
            opt.zero_grad()
            loss = model(past_values=sx, future_values=sy, forecast_context_len=context, truncate_negative=False).loss
            if not torch.isfinite(loss):
                raise ValueError('Non-finite online loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(past_values=torch.tensor(x[None, :], device=device), forecast_context_len=context,
                          truncate_negative=False).mean_predictions[0, :horizon].cpu().numpy()
        if not np.isfinite(pred).all():
            raise ValueError('Non-finite forecast')
        preds.append(pred)
    return np.stack(preds)


def horizon_metrics(pred, actual, anchors, top_k_fraction=0.2, has_direction=True):
    """pred/actual: (windows, horizon) cumulative log returns from anchor. anchors: (windows,) close price.

    `has_direction=False` for a model whose prediction is always flat (e.g.
    naive persistence): sign(0) vs. sign(actual) is not a real directional
    call, it just measures how often the true return happens to be exactly
    zero, so direction accuracy and Top-K precision are undefined (None)
    rather than a misleadingly low number.
    """
    n, horizon = pred.shape
    per_horizon = []
    for h in range(horizon):
        err = pred[:, h] - actual[:, h]
        pred_close = anchors * np.exp(pred[:, h])
        actual_close = anchors * np.exp(actual[:, h])
        mape = float(np.mean(np.abs(pred_close - actual_close) / actual_close)) * 100
        da = float(np.mean(np.sign(pred[:, h]) == np.sign(actual[:, h]))) if has_direction else None
        per_horizon.append(dict(horizon=h + 1, mae=float(np.mean(np.abs(err))),
                                 rmse=float(np.sqrt(np.mean(err ** 2))), mape_pct=mape, direction_accuracy=da))
    overall_err = pred - actual
    k = max(1, int(round(n * top_k_fraction)))
    if has_direction:
        order = np.argsort(-np.abs(pred[:, -1]))[:k]
        top_k_precision = float(np.mean(np.sign(pred[order, -1]) == np.sign(actual[order, -1])))
    else:
        top_k_precision = None
    return dict(windows=n, mae=float(np.mean(np.abs(overall_err))), rmse=float(np.sqrt(np.mean(overall_err ** 2))),
                per_horizon=per_horizon, top_k_fraction=top_k_fraction, top_k_windows=k,
                top_k_directional_precision=top_k_precision)


def evaluate(run_dir, csv_path, rolling_split, shots=8, save_predictions=True):
    protocol = json.loads((run_dir / 'protocol.json').read_text())
    context, horizon, interval = protocol['context'], protocol['horizon'], protocol.get('interval', '1h')
    device = protocol['device']
    seconds = SECONDS[interval]
    test = windows(csv_path, context, horizon, seconds, rolling_split)['test']
    all_bars = {}
    for bar in load(csv_path):
        all_bars.setdefault(bar.symbol, []).append(bar)
    x_all = np.stack([r[0] for r in test])
    y_all = np.stack([r[1] for r in test])
    anchors = np.array([r[2]['anchor'] for r in test])

    predictions = {}
    results = {}
    predictions['naive'] = np.zeros_like(y_all)
    results['naive'] = horizon_metrics(predictions['naive'], y_all, anchors, has_direction=False)
    predictions['momentum'] = np.stack([momentum_forecast(x, horizon) for x in x_all])
    results['momentum'] = horizon_metrics(predictions['momentum'], y_all, anchors)

    torch.manual_seed(42)
    base = TimesFm2_5ModelForPrediction.from_pretrained(protocol['model'], dtype=torch.float32, local_files_only=True).to(device)
    model = PeftModel.from_pretrained(base, run_dir / 'adapter', is_trainable=True).to(device).eval()

    predictions['zero_shot'] = batched_forecast(model, test, context, horizon, device, disable_adapter=True)
    results['zero_shot'] = horizon_metrics(predictions['zero_shot'], y_all, anchors)
    predictions['lora'] = batched_forecast(model, test, context, horizon, device, disable_adapter=False)
    results['lora'] = horizon_metrics(predictions['lora'], y_all, anchors)

    initial_state = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    predictions['lora_fewshot'] = fewshot_forecast(model, initial_state, test, all_bars, context, horizon, shots, device)
    results['lora_fewshot'] = horizon_metrics(predictions['lora_fewshot'], y_all, anchors)

    if save_predictions:
        write_predictions(run_dir / 'eval_predictions.csv', test, predictions, horizon)

    return dict(interval=interval, context=context, horizon=horizon, shots=shots, models=results)


def write_predictions(path, test, predictions, horizon):
    """One row per (model, test window): raw predicted/actual log returns and the anchor
    price, so a downstream backtest can turn a signal into an actual trade P&L without
    re-running any model inference."""
    fieldnames = ['model', 'symbol', 'context_end', 'target_start', 'target_end', 'anchor'] + \
                 [f'pred_{h}' for h in range(1, horizon + 1)] + [f'actual_{h}' for h in range(1, horizon + 1)]
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for name, pred in predictions.items():
            for (x, y, meta), p in zip(test, pred):
                row = dict(model=name, symbol=meta['symbol'], context_end=meta['context_end'],
                           target_start=meta['target_start'], target_end=meta['target_end'], anchor=meta['anchor'])
                row.update({f'pred_{h}': float(p[h - 1]) for h in range(1, horizon + 1)})
                row.update({f'actual_{h}': float(y[h - 1]) for h in range(1, horizon + 1)})
                w.writerow(row)


def to_rows(evaluation):
    interval = evaluation['interval']
    for name, m in evaluation['models'].items():
        da = {f"da{h['horizon']}": h['direction_accuracy'] for h in m['per_horizon']}
        yield dict(interval=interval, model=name, windows=m['windows'], mae=m['mae'], rmse=m['rmse'],
                   top_k_precision=m['top_k_directional_precision'], **da)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--roots', nargs='+', type=Path, default=[Path('runs/forecast5/1m'), Path('runs/forecast5/1h'), Path('runs/forecast5/1d')])
    p.add_argument('--csv-dir', type=Path, default=Path('data/live'))
    p.add_argument('--rolling-split', action='store_true', default=True)
    p.add_argument('--shots', type=int, default=8)
    args = p.parse_args()
    all_rows = []
    for run_dir in args.roots:
        interval = run_dir.name
        csv_path = args.csv_dir / f'{interval}.csv'
        print('Evaluating', interval, flush=True)
        evaluation = evaluate(run_dir, csv_path, args.rolling_split, args.shots)
        (run_dir / 'eval_full.json').write_text(json.dumps(evaluation, indent=2))
        rows = list(to_rows(evaluation))
        all_rows.extend(rows)
        print(json.dumps(rows, indent=2), flush=True)
    lines = ['# Baseline vs. TimesFM comparison (walk-forward test split)', '',
             '5-step close forecasts. MAE/RMSE are mean/root-mean-square absolute error of the',
             'cumulative log return from the last observed close, averaged over all 5 horizons.',
             'DA@h = direction accuracy at horizon h (sign of predicted vs. actual return from entry).',
             'Top-K precision = direction accuracy on the 20% of windows with the largest |predicted t+5 return|.',
             '', '| Interval | Model | Windows | MAE | RMSE | DA@1 | DA@5 | Top-K precision |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    def fmt(v):
        return f'{v:.1%}' if v is not None else 'N/A'

    order = ['naive', 'momentum', 'zero_shot', 'lora', 'lora_fewshot']
    for interval in ('1m', '1h', '1d'):
        for name in order:
            row = next((r for r in all_rows if r['interval'] == interval and r['model'] == name), None)
            if row is None:
                continue
            lines.append(f"| {interval} | {name} | {row['windows']} | {row['mae']:.6f} | {row['rmse']:.6f} | "
                          f"{fmt(row['da1'])} | {fmt(row['da5'])} | {fmt(row['top_k_precision'])} |")
    lines += ['', 'Naive and momentum need no model. zero_shot = pretrained TimesFM 2.5 with the crypto',
              'LoRA disabled. lora = pretrained + historical per-timeframe LoRA on the latest context.',
              'lora_fewshot = lora + walk-forward online adaptation on the 8 most recently completed',
              'examples at each origin. If lora_fewshot does not beat naive/momentum on DA@5 and',
              'Top-K precision here, that is reported as-is and is not treated as an improvement.']
    Path('runs/forecast5/EVAL_REPORT.md').write_text('\n'.join(lines) + '\n')
    print('Wrote runs/forecast5/EVAL_REPORT.md', flush=True)


if __name__ == '__main__':
    main()

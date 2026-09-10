"""Single-symbol CLI: historical LoRA + recent few-shot -> next-horizon close forecast.

    HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 \\
        .venv-timesfm/bin/python -m prop_trader.forecast --symbol DOGEUSDT --interval 1h [--plot]

Research signal only. No exchange order is ever sent from this module.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import TimesFm2_5ModelForPrediction

from .__main__ import load
from .forecast_time import SECONDS, path_stats, decide_signal
from .timesfm_predict import predict_symbol


def run(symbol, interval, run_root=Path('runs/forecast5'), csv=None, shots=8, threshold=0.01):
    if interval not in SECONDS:
        raise ValueError(f'Unsupported timeframe: {interval} (known: {", ".join(SECONDS)})')
    run_dir = run_root / interval
    protocol_path = run_dir / 'protocol.json'
    if not protocol_path.exists():
        raise ValueError(f'No trained adapter for {interval} at {run_dir} (run timesfm_run first)')
    protocol = json.loads(protocol_path.read_text())
    csv = csv or Path('data/live') / f'{interval}.csv'
    if not csv.exists():
        raise ValueError(f'{csv} not found; run `python -m prop_trader.current_candles --out data/live` first')
    bars = [b for b in load(csv) if b.symbol == symbol]
    if not bars:
        raise ValueError(f'No candles for {symbol} in {csv}')
    as_of = bars[-1].timestamp
    torch.manual_seed(42)
    device = protocol['device']
    base = TimesFm2_5ModelForPrediction.from_pretrained(protocol['model'], dtype=torch.float32, local_files_only=True).to(device)
    record, _ = predict_symbol(base, run_dir, protocol, bars, as_of, shots, device)
    stats = path_stats(record['last_close'], record['forecast_prices'])
    signal = decide_signal(stats, threshold=threshold)
    return record, stats, signal


def render_text(symbol, interval, record, stats, signal):
    lines = [f'{symbol} | {interval}', '', 'Current:', f"{record['last_close']:.8g}", '', 'Forecast:']
    for point, ret in zip(record['forecast'], stats['return_by_step']):
        lines.append(f"T+{point['horizon']}  {point['predicted_close']:.8g}  {ret:+.2%}")
    lines += ['', f"Model: {record.get('predictor','timesfm_lora_fewshot')}",
              f"Path: min {stats['min_predicted_return']:+.2%} · max {stats['max_predicted_return']:+.2%} "
                  f"· consistency {stats['forecast_consistency']:.0%}", '', 'Signal:', signal, '',
              '연구용 신호이며 실제 주문은 전송하지 않습니다.']
    return '\n'.join(lines)


def plot(symbol, interval, record, bars, stats, out_path, history=60):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from datetime import datetime

    tail = bars[-history:]
    hist_times = [datetime.fromisoformat(b.timestamp) for b in tail]
    hist_close = [b.close for b in tail]
    fc_times = [datetime.fromisoformat(record['as_of'])] + [datetime.fromisoformat(p['timestamp']) for p in record['forecast']]
    fc_close = [record['last_close']] + record['forecast_prices']

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(hist_times, hist_close, color='#2563eb', label='Observed close')
    ax.plot(fc_times, fc_close, color='#f59e0b', linestyle='--', marker='o', label='Predicted close (t+1..t+5)')
    ax.axvline(hist_times[-1], color='#94a3b8', linestyle=':', label='Forecast boundary')
    ax.axhline(record['last_close'], color='#94a3b8', linestyle=':', linewidth=1)
    t5_return = stats['return_by_step'][-1]
    ax.annotate(f't+5 {t5_return:+.2%}', xy=(fc_times[-1], fc_close[-1]), xytext=(10, 10),
                textcoords='offset points', color='#b45309', fontweight='bold')
    ax.set_title(f'{symbol} | {interval} | as-of {record["as_of"]}')
    ax.set_ylabel('Close (USDT)')
    ax.legend(loc='best')
    ax.grid(alpha=.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', required=True)
    p.add_argument('--interval', required=True, choices=list(SECONDS))
    p.add_argument('--run-root', type=Path, default=Path('runs/forecast5'))
    p.add_argument('--csv', type=Path, default=None)
    p.add_argument('--shots', type=int, default=8)
    p.add_argument('--threshold', type=float, default=0.01)
    p.add_argument('--plot', action='store_true')
    p.add_argument('--plot-out', type=Path, default=None)
    args = p.parse_args()

    record, stats, signal = run(args.symbol, args.interval, args.run_root, args.csv, args.shots, args.threshold)
    print(render_text(args.symbol, args.interval, record, stats, signal))

    if args.plot:
        bars = [b for b in load(args.csv or Path('data/live') / f'{args.interval}.csv') if b.symbol == args.symbol]
        out_path = args.plot_out or Path('runs/forecast5') / args.interval / f'{args.symbol}_forecast.png'
        saved = plot(args.symbol, args.interval, record, bars, stats, out_path)
        print(f'\nSaved chart: {saved}')


if __name__ == '__main__':
    main()

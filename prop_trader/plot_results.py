"""Optional matplotlib chart for the completed experiment."""
import csv
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path('runs/backtest')
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for name in ('default', 'selected'):
        with (root / f'holdout_{name}_equity.csv').open() as f:
            rows = list(csv.DictReader(f))
        times = [datetime.fromisoformat(r['timestamp']) for r in rows]
        values = [float(r['equity']) for r in rows]
        peak, dd = 10000, []
        for value in values:
            peak = max(peak, value)
            dd.append((value/peak-1)*100)
        axes[0].plot(times, values, label=name)
        axes[1].plot(times, dd, label=name)
    axes[0].axhline(10000, color='gray', linestyle='--', label='cash')
    axes[0].set(ylabel='Equity (USDT)', title='2025 H2 holdout | 6 meme tokens | 1h spot long-only')
    axes[1].set(ylabel='Drawdown (%)', xlabel='UTC | Fees and slippage included')
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(root / 'holdout.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()

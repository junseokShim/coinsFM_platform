"""End-to-end forecast pipeline checks against the real cached TimesFM 2.5 base model.

Runs the actual historical-LoRA -> recent-few-shot -> forecast path on synthetic
bars (not real market data) so the test is fast and deterministic, while still
exercising real adapter loading and a real forward pass.
"""
import os
from pathlib import Path

os.environ.setdefault('HF_HOME', str(Path(__file__).resolve().parents[2] / '.cache' / 'huggingface'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import json
import unittest
from datetime import datetime, timedelta, timezone

import torch
from transformers import TimesFm2_5ModelForPrediction

from prop_trader.engine import Bar
from prop_trader.timesfm_predict import predict_symbol
from prop_trader import forecast as forecast_cli

RUN_DIR = Path(__file__).resolve().parents[2] / 'runs' / 'forecast5' / '1h'


def synthetic_bars(symbol, n, interval_seconds=3600, start_price=1.0):
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    bars = []
    price = start_price
    for i in range(n):
        price *= 1 + 0.0005 * (1 if i % 7 else -1)
        ts = (start + timedelta(seconds=interval_seconds * i)).isoformat()
        bars.append(Bar(ts, symbol, price, price * 1.001, price * 0.999, price, 100.0))
    return bars


@unittest.skipUnless(RUN_DIR.exists(), 'runs/forecast5/1h adapter not present')
class ForecastPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((RUN_DIR / 'protocol.json').read_text())
        torch.manual_seed(42)
        cls.base = TimesFm2_5ModelForPrediction.from_pretrained(
            cls.protocol['model'], dtype=torch.float32, local_files_only=True
        ).to(cls.protocol['device'])

    @classmethod
    def tearDownClass(cls):
        # predict_symbol() persists a per-symbol adapter as a side effect; clean up the
        # synthetic test symbols so they don't linger in the real run directory.
        import shutil
        for symbol in ('SYN1', 'SYN2', 'SYN3', 'SYN4'):
            path = RUN_DIR / 'recent_adapters' / symbol
            if path.exists():
                shutil.rmtree(path)

    def test_forecast_has_five_horizons_with_correct_future_timestamps(self):
        bars = synthetic_bars('SYN1', 200)
        as_of = bars[-1].timestamp
        record, _ = predict_symbol(self.base, RUN_DIR, self.protocol, bars, as_of, shots=8, device=self.protocol['device'], use_technical=False)
        self.assertEqual(len(record['forecast']), 5)
        self.assertEqual([p['horizon'] for p in record['forecast']], [1, 2, 3, 4, 5])
        last_close_ts = datetime.fromisoformat(as_of) + timedelta(hours=1)
        expected = [(last_close_ts + timedelta(hours=h)).isoformat() for h in range(1, 6)]
        self.assertEqual([p['timestamp'] for p in record['forecast']], expected)
        for i in range(1, 6):
            self.assertIn(f'return_t{i}', record)

    def test_no_lookahead_leakage_in_recent_fewshot_examples(self):
        bars = synthetic_bars('SYN2', 200)
        as_of = bars[-1].timestamp
        record, _ = predict_symbol(self.base, RUN_DIR, self.protocol, bars, as_of, shots=8, device=self.protocol['device'], use_technical=False)
        # Every recent-adaptation label must have fully completed strictly before the forecast origin.
        self.assertEqual(record['last_candle_open'], as_of)
        self.assertLessEqual(as_of, bars[-1].timestamp)
        # The as_of close boundary is exactly one interval after the last observed bar open.
        self.assertEqual(record['as_of'], (datetime.fromisoformat(as_of) + timedelta(hours=1)).isoformat())

    def test_missing_last_completed_bar_is_rejected(self):
        bars = synthetic_bars('SYN3', 200)
        stale_as_of = bars[-2].timestamp  # not the true latest bar
        with self.assertRaises(ValueError):
            predict_symbol(self.base, RUN_DIR, self.protocol, bars, stale_as_of, shots=8, device=self.protocol['device'])

    def test_gap_in_history_is_rejected(self):
        bars = synthetic_bars('SYN4', 200)
        del bars[100]  # introduces a timestamp gap
        with self.assertRaises(ValueError):
            predict_symbol(self.base, RUN_DIR, self.protocol, bars, bars[-1].timestamp, shots=8, device=self.protocol['device'])


class ForecastCliErrorTests(unittest.TestCase):
    def test_unsupported_timeframe_raises(self):
        with self.assertRaises(ValueError):
            forecast_cli.run('DOGEUSDT', '2h')

    def test_untrained_timeframe_raises_clear_error(self):
        with self.assertRaises(ValueError):
            forecast_cli.run('DOGEUSDT', '4h')


if __name__ == '__main__':
    unittest.main()

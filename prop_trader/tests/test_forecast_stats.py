import unittest
from prop_trader.forecast_time import forecast_points, path_stats, decide_signal, SECONDS


class ForecastPointsTests(unittest.TestCase):
    def test_unsupported_timeframe_rejected(self):
        with self.assertRaises(ValueError):
            forecast_points('2026-09-09T00:00:00+00:00', '2h', [1] * 5)

    def test_new_timeframes_are_wired_generically(self):
        for interval in ('3m', '5m', '15m', '30m', '4h'):
            rows = forecast_points('2026-09-09T00:00:00+00:00', interval, [1, 2, 3, 4, 5])
            self.assertEqual(len(rows), 5)


class PathStatsTests(unittest.TestCase):
    def test_ideal_long_has_high_consistency_and_no_drawdown(self):
        stats = path_stats(100, [101, 103, 105, 107, 110])
        self.assertAlmostEqual(stats['return_by_step'][-1], 0.10)
        self.assertEqual(stats['forecast_consistency'], 1.0)
        self.assertGreater(stats['min_predicted_return'], 0)

    def test_bad_long_dips_below_entry_before_recovering(self):
        # Same t+5 return as the ideal case, but it enters underwater first (path risk).
        stats = path_stats(100, [90, 91, 93, 97, 105])
        self.assertAlmostEqual(stats['return_by_step'][-1], 0.05)
        self.assertLess(stats['min_predicted_return'], 0)

    def test_reversal_path_has_lower_consistency(self):
        stats = path_stats(100, [105, 102, 98, 101, 108])
        self.assertLess(stats['forecast_consistency'], 1.0)

    def test_rejects_nonpositive_price(self):
        with self.assertRaises(ValueError):
            path_stats(0, [1, 2, 3, 4, 5])


class SignalTests(unittest.TestCase):
    def test_ideal_long_signals_long(self):
        stats = path_stats(100, [101, 103, 105, 107, 110])
        self.assertEqual(decide_signal(stats, threshold=0.01), 'LONG')

    def test_choppy_path_with_early_drawdown_does_not_signal_long(self):
        stats = path_stats(100, [90, 91, 93, 97, 105])
        self.assertEqual(decide_signal(stats, threshold=0.01), 'HOLD')

    def test_small_move_is_hold(self):
        stats = path_stats(100, [100.1, 100.2, 100.1, 100.3, 100.2])
        self.assertEqual(decide_signal(stats, threshold=0.01), 'HOLD')

    def test_symmetric_down_path_signals_short(self):
        stats = path_stats(100, [99, 97, 95, 93, 90])
        self.assertEqual(decide_signal(stats, threshold=0.01), 'SHORT')


if __name__ == '__main__':
    unittest.main()

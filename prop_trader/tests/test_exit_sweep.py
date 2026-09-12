import unittest
from prop_trader.engine import Bar
from prop_trader.exit_sweep import simulate_fixed, simulate_adaptive, expanding_folds, summarize


def candle(ts, o, h, l, c, symbol='A'):
    return Bar(ts, symbol, o, h, l, c, 100)


def candidate(path, anchor=100., pred_prices=None):
    return dict(symbol='A', target_start=path[0].timestamp, anchor=anchor,
                pred_prices=pred_prices or [100.] * 5, path=path)


class SimulateFixedTests(unittest.TestCase):
    def test_stop_wins_when_both_levels_hit_same_bar(self):
        # Entry fill ~100.1 (open*(1+slippage)); stop 1% below, target 1% above -- one bar
        # spans both. Stop must win (matches engine.Desk.step's same-bar priority).
        c = candidate([candle('t0', 100, 103, 97, 100)])
        trade = simulate_fixed(c, stop_fraction=.01, target_fraction=.01)
        self.assertEqual(trade['reason'], 'stop_loss')
        self.assertLess(trade['net_return'], 0)

    def test_gap_down_fills_at_open_not_stop_level(self):
        # Bar opens well below the stop level -- fill must be the worse open price, not the
        # nominal stop, and the resulting loss must be larger than a clean stop-level fill.
        c = candidate([candle('t0', 100, 100, 100, 100), candle('t1', 90, 91, 85, 90)])
        trade = simulate_fixed(c, stop_fraction=.02, target_fraction=None)
        self.assertEqual(trade['reason'], 'stop_loss')
        self.assertLess(trade['net_return'], -.02)

    def test_target_hit_without_stop_touch(self):
        c = candidate([candle('t0', 100, 103, 99.5, 102)])
        trade = simulate_fixed(c, stop_fraction=.02, target_fraction=.02)
        self.assertEqual(trade['reason'], 'take_profit')
        self.assertGreater(trade['net_return'], 0)

    def test_horizon_timeout_when_neither_level_touched(self):
        c = candidate([candle(f't{i}', 100, 101, 99, 100) for i in range(5)])
        trade = simulate_fixed(c, stop_fraction=.05, target_fraction=.05)
        self.assertEqual(trade['reason'], 'horizon_timeout')
        self.assertEqual(trade['bars_held'], 5)

    def test_none_target_means_stop_and_horizon_only(self):
        c = candidate([candle('t0', 100, 200, 99, 150)])  # would have hit a real target
        trade = simulate_fixed(c, stop_fraction=.05, target_fraction=None)
        self.assertEqual(trade['reason'], 'horizon_timeout')


class SimulateAdaptiveTests(unittest.TestCase):
    def test_uses_predicted_peak_shrunk_toward_anchor(self):
        # anchor 100, predicted peak 110 -> shrink .5 => synthetic target ~105.
        path = [candle('t0', 100, 106, 99, 101)]
        c = candidate(path, anchor=100., pred_prices=[101, 103, 110, 108, 105])
        trade = simulate_adaptive(c, stop_fraction=.05, shrink=.5)
        self.assertEqual(trade['reason'], 'take_profit')

    def test_falls_back_to_horizon_when_forecast_never_clears_entry(self):
        # Predicted path never rises above the actual (slippage-adjusted) entry fill.
        path = [candle('t0', 100, 100.2, 99, 100)]
        c = candidate(path, anchor=100., pred_prices=[99, 98, 97, 96, 95])
        trade = simulate_adaptive(c, stop_fraction=.05, shrink=1.0)
        self.assertEqual(trade['reason'], 'horizon_timeout')


class ExpandingFoldsTests(unittest.TestCase):
    def test_every_fold_test_slice_is_strictly_after_its_train_slice(self):
        items = list(range(20))
        folds = expanding_folds(items, min_train=4, n_folds=4)
        self.assertTrue(folds)
        for train, test in folds:
            self.assertLess(max(train), min(test))

    def test_too_few_candidates_yields_no_folds(self):
        self.assertEqual(expanding_folds(list(range(3))), [])


class SummarizeTests(unittest.TestCase):
    def test_empty_trades(self):
        s = summarize([])
        self.assertEqual(s['trades'], 0)
        self.assertIsNone(s['win_rate'])


if __name__ == '__main__':
    unittest.main()

import unittest
from prop_trader.backtest_forecast import simulate, buy_and_hold_benchmark, ROUND_TRIP_COST


def row(symbol, target_start, target_end, anchor, preds, actual_5):
    r = dict(symbol=symbol, target_start=target_start, target_end=target_end, anchor=str(anchor))
    for h, p in enumerate(preds, 1):
        r[f'pred_{h}'] = str(p)
    r['actual_5'] = str(actual_5)
    return r


class SimulateTests(unittest.TestCase):
    def test_long_signal_with_favorable_actual_move_is_a_winning_trade(self):
        import math
        rows = [row('A', 't0', 't5', 100.0, [.005, .01, .015, .02, .03], math.log(1.04))]
        summary, trades, curve = simulate(rows, threshold=0.01, min_consistency=0.5, notional=1000, starting_capital=10_000)
        self.assertEqual(summary['trades'], 1)
        self.assertEqual(trades[0]['signal'], 'LONG')
        self.assertAlmostEqual(trades[0]['raw_return'], 0.04, places=4)
        self.assertAlmostEqual(trades[0]['net_return'], 0.04 - ROUND_TRIP_COST, places=4)

    def test_hold_signal_produces_no_trade(self):
        import math
        rows = [row('A', 't0', 't5', 100.0, [0, 0, 0, 0, 0.001], math.log(1.05))]
        summary, trades, curve = simulate(rows, threshold=0.01, min_consistency=0.5, notional=1000, starting_capital=10_000)
        self.assertEqual(summary['trades'], 0)
        self.assertEqual(summary['hold_windows'], 1)
        self.assertIsNone(summary['win_rate'])

    def test_equity_curve_reflects_costs_on_a_flat_actual_move(self):
        import math
        rows = [row('A', 't0', 't5', 100.0, [.005, .01, .015, .02, .03], math.log(1.0))]
        summary, trades, curve = simulate(rows, threshold=0.01, min_consistency=0.5, notional=1000, starting_capital=10_000)
        self.assertEqual(summary['trades'], 1)
        self.assertLess(summary['total_pnl'], 0)  # zero raw move, cost-only loss
        self.assertAlmostEqual(curve[-1]['equity'], 10_000 - 1000 * ROUND_TRIP_COST, places=4)


class BuyAndHoldTests(unittest.TestCase):
    def test_averages_actual_return_across_all_windows_including_holds(self):
        import math
        rows = [row('A', 't0', 't5', 100.0, [0] * 5, math.log(1.1)),
                row('B', 't0', 't5', 100.0, [0] * 5, math.log(0.9))]
        result = buy_and_hold_benchmark(rows)
        self.assertEqual(result['windows'], 2)
        self.assertAlmostEqual(result['avg_return_pct'], 0.0, places=2)


if __name__ == '__main__':
    unittest.main()

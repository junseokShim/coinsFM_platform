import unittest
from prop_trader.engine import Bar, Config
from prop_trader.research import run


class ResearchTests(unittest.TestCase):
    def test_warmup_does_not_trade_and_pnl_reconciles(self):
        rows = []
        for day, close, volume in [(1, 100, 100), (2, 100, 100), (3, 103, 400), (4, 103, 100)]:
            ts = f'2025-01-{day:02d}T00:00:00+00:00'
            rows.append((ts, [Bar(ts, 'A', close, close+1, close-1, close, volume)]))
        c = Config(lookback=2)
        # Signal on Jan 3 belongs to warmup and cannot create Jan 4 entry.
        no_trade = run(rows, c, '2025-01-04', '2025-01-05')
        self.assertEqual(no_trade['trades'], 0)
        active = run(rows, c, '2025-01-03', '2025-01-05')
        self.assertEqual(active['trades'], 1)
        self.assertLess(active['net_pnl'], 0)
        self.assertAlmostEqual(active['final_equity']-c.capital, active['net_pnl'])

    def test_end_boundary_excludes_future_bars(self):
        ts = '2025-01-01T00:00:00+00:00'
        rows = [(ts, [Bar(ts, 'A', 100, 101, 99, 100, 100)])]
        a = run(rows, Config(), '2025-01-01', '2025-01-02')
        future = '2025-01-02T00:00:00+00:00'
        rows.append((future, [Bar(future, 'A', 1000, 1001, 999, 1000, 100000)]))
        self.assertEqual(a, run(rows, Config(), '2025-01-01', '2025-01-02'))

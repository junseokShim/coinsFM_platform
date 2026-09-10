import unittest
from prop_trader.engine import Bar, Config, Desk


def bar(i, o=100, h=101, l=99, c=100, v=100, symbol='A'):
    return Bar(f'2025-01-01T00:{i:02d}:00+00:00', symbol, o, h, l, c, v)


class EngineTests(unittest.TestCase):
    def ready(self, **kwargs):
        d = Desk(Config(lookback=2, **kwargs))
        d.step([bar(0)])
        d.step([bar(1)])
        d.step([bar(2, h=104, c=103, v=400)])
        return d

    def test_signal_is_not_same_bar_fill(self):
        d = self.ready()
        self.assertFalse(d.positions)
        d.step([bar(3, o=104, h=105, l=103, c=104)])
        self.assertAlmostEqual(d.positions['A']['entry'], 104 * 1.001)

    def test_both_levels_hit_stop_wins_and_costs_reconcile(self):
        d = self.ready()
        d.step([bar(3, o=103, h=120, l=90, c=103)])
        exits = [e for e in d.events if e['kind'] == 'exit']
        self.assertEqual(exits[0]['reason'], 'stop')
        self.assertAlmostEqual(d.cash - d.cfg.capital, exits[0]['pnl'])
        self.assertLess(d.cash, d.cfg.capital)

    def test_gap_stop_uses_open(self):
        d = self.ready()
        d.step([bar(3, o=103, h=104, l=102, c=103)])
        d.step([bar(4, o=90, h=91, l=89, c=90)])
        exit_event = next(e for e in d.events if e['kind'] == 'exit')
        self.assertAlmostEqual(exit_event['price'], 90 * .999)

    def test_daily_halt_closes_and_blocks_new_entries(self):
        d = self.ready(daily_loss_limit=.001)
        d.step([bar(3, o=103, h=104, l=90, c=103)])
        self.assertTrue(d.halted)
        self.assertFalse(d.positions)
        self.assertFalse(d.pending)

    def test_short_opt_in(self):
        for enabled in (False, True):
            d = Desk(Config(lookback=2, allow_short=enabled))
            d.step([bar(0)])
            d.step([bar(1)])
            d.step([bar(2, l=95, c=96, v=400)])
            self.assertEqual(bool(d.pending), enabled)
            if enabled:
                d.step([bar(3, o=96, h=97, l=95, c=96)])
                self.assertEqual(d.positions['A']['side'], -1)
                d.finish()
                pnl = sum(e['pnl'] for e in d.events if e['kind'] == 'exit')
                self.assertAlmostEqual(d.cash - d.cfg.capital, pnl)

    def test_missing_position_bar_rejected(self):
        d = self.ready()
        d.step([bar(3, o=103, h=104, l=102, c=103)])
        with self.assertRaises(ValueError):
            d.step([bar(4, symbol='B')])

    def test_duplicate_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            Desk().step([bar(0), bar(0)])
        with self.assertRaises(ValueError):
            bar(0, c=float('nan'))

    def test_future_does_not_change_prior_decisions(self):
        a, b = self.ready(), self.ready()
        prior = list(a.events)
        a.step([bar(3, o=103, h=120, l=90, c=110)])
        b.step([bar(3, o=103, h=104, l=102, c=103)])
        self.assertEqual(a.events[:len(prior)], prior)
        self.assertEqual(b.events[:len(prior)], prior)


if __name__ == '__main__':
    unittest.main()

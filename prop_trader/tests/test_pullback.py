import unittest
from prop_trader.engine import Bar, Config
from prop_trader.pullback import PullbackDesk


class PullbackTests(unittest.TestCase):
    def test_recovery_required_and_overextended_move_rejected(self):
        d=PullbackDesk(Config(lookback=168))
        d.market_ok=True
        past=[Bar(str(i),'A',90,91,89,90,100) for i in range(144)]
        past += [Bar(str(i),'A',101,102,100,101,100) for i in range(23)]
        past += [Bar('last','A',100,100.5,99.5,100,100)]
        recovery=Bar('now','A',100,102,99,101.5,100)
        self.assertEqual(d.signal('A',recovery,past)['side'],1)
        d.market_ok=False
        self.assertIsNone(d.signal('A',recovery,past))
        d.market_ok=True
        chase=Bar('now','A',100,110,99,109,500)
        self.assertIsNone(d.signal('A',chase,past))

from datetime import datetime,timezone
import unittest
from prop_trader.rank_coins import buy_vwap,rank_records


class RankingTests(unittest.TestCase):
    def test_liquidity_depth_vwap(self):
        book={'orderbook_units':[{'ask_price':100,'ask_size':1},{'ask_price':110,'ask_size':1}]}
        self.assertAlmostEqual(buy_vwap(book,210),105)
        with self.assertRaises(ValueError):buy_vwap(book,220)

    def test_net_sort_and_stale_flag(self):
        now=datetime(2026,1,1,1,tzinfo=timezone.utc)
        snapshot={'orderbooks':{s:dict(timestamp=now.timestamp()*1000,orderbook_units=[dict(ask_price=100,ask_size=1000)]) for s in ('KRW-A','KRW-B')}}
        records=[dict(symbol=s,interval='1h',as_of='2026-01-01T01:00:00+00:00',
            hybrid_prices=[p]*5,forecast_prices=[p]*5,technical_features={}) for s,p in [('KRW-A',99),('KRW-B',105)]]
        rows=rank_records(records,snapshot,now=now)
        self.assertEqual(rows[0]['symbol'],'KRW-B')
        self.assertFalse(rows[0]['stale'])
        self.assertLess(rows[1]['expected_net_return'],0)
        later=datetime(2026,1,1,3,tzinfo=timezone.utc)
        self.assertTrue(rank_records(records,snapshot,now=later)[0]['stale'])

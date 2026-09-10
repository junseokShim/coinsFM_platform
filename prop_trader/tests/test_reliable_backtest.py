import unittest
from datetime import datetime,timedelta,timezone
import math
from prop_trader.engine import Bar
from prop_trader.reliable_backtest import portfolio


def fixture(symbols=('A',)):
    start=datetime(2026,1,1,tzinfo=timezone.utc)
    rows=[];bars=[]
    for s in symbols:
        rows.append(dict(symbol=s,context_end=(start-timedelta(hours=1)).isoformat(),
            target_start=start.isoformat(),target_end=(start+timedelta(hours=4)).isoformat(),pred_5=str(math.log(1.1))))
        bars.extend(Bar((start+timedelta(hours=i)).isoformat(),s,120,121,119,120,100) for i in range(5))
    return rows,bars


class ReliableBacktestTests(unittest.TestCase):
    def test_next_open_not_prior_anchor_and_cost_reconciliation(self):
        rows,bars=fixture();summary,trades,curve=portfolio(rows,bars,'1h')
        self.assertEqual(summary['trades'],1)
        self.assertLess(summary['return_fraction'],0)
        self.assertAlmostEqual(sum(t['pnl'] for t in trades),summary['final_equity']-10000)

    def test_position_limit_cash_and_deterministic_ranking(self):
        rows,bars=fixture(('A','B','C','D'))
        summary,trades,curve=portfolio(rows,bars,'1h',max_positions=2)
        self.assertEqual(summary['trades'],2)
        self.assertTrue(all(p['positions']<=2 and p['cash']>=0 for p in curve))
        self.assertEqual({t['symbol'] for t in trades},{'A','B'})

    def test_missing_candle_rejected(self):
        rows,bars=fixture()
        with self.assertRaises(ValueError):portfolio(rows,bars[:-1],'1h')

    def test_future_context_rejected(self):
        rows,bars=fixture();rows[0]['context_end']=rows[0]['target_start']
        with self.assertRaises(ValueError):portfolio(rows,bars,'1h')

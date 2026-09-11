import json
import tempfile
import unittest
from pathlib import Path

from prop_trader.loss_report import bucket, report


class BucketTests(unittest.TestCase):
    def test_stop_loss(self):
        self.assertEqual(bucket('손절(stop-loss)'), 'stop_loss')

    def test_take_profit(self):
        self.assertEqual(bucket('익절(take-profit)'), 'take_profit')

    def test_horizon_timeout(self):
        self.assertEqual(bucket('예측 만료 청산(horizon)'), 'horizon_timeout')

    def test_other_is_signal_exit(self):
        self.assertEqual(bucket('다중 주기 예측의 보유 조건 이탈'), 'signal_exit')


def entry(**kw):
    base = dict(ts='2026-09-10T00:00:00+00:00', mode='live')
    base.update(kw)
    return base


class ReportTests(unittest.TestCase):
    def test_pairs_buy_and_sell_fills_via_submitted_reason(self):
        lines = [
            entry(symbol='KRW-BTC', action='BUY_SUBMITTED', identifier='b1', reason='기대수익 1%, 순위 1'),
            entry(symbol='KRW-BTC', action='BUY', identifier='b1', reason='체결 확인', fee_krw=5., pnl_krw=None),
            entry(symbol='KRW-BTC', action='SELL_SUBMITTED', identifier='s1', reason='손절(stop-loss)'),
            entry(symbol='KRW-BTC', action='SELL', identifier='s1', reason='체결 확인', fee_krw=3., pnl_krw=-100.),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'decisions.jsonl'
            path.write_text('\n'.join(json.dumps(l) for l in lines)+'\n')
            rows, totals = report(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['bucket'], 'stop_loss')
        self.assertEqual(totals['stop_loss']['pnl'], -100.)
        self.assertEqual(totals['stop_loss']['fee'], 8.)
        self.assertEqual(totals['stop_loss']['gross'], -92.)

    def test_fifo_pairs_multiple_buys_per_symbol(self):
        lines = [
            entry(symbol='KRW-ETH', action='BUY_SUBMITTED', identifier='b1', reason='기대수익 1%, 순위 1'),
            entry(symbol='KRW-ETH', action='BUY', identifier='b1', reason='체결 확인', fee_krw=1., pnl_krw=None),
            entry(symbol='KRW-ETH', action='BUY_SUBMITTED', identifier='b2', reason='기대수익 1%, 순위 1'),
            entry(symbol='KRW-ETH', action='BUY', identifier='b2', reason='체결 확인', fee_krw=2., pnl_krw=None),
            entry(symbol='KRW-ETH', action='SELL_SUBMITTED', identifier='s1', reason='익절(take-profit)'),
            entry(symbol='KRW-ETH', action='SELL', identifier='s1', reason='체결 확인', fee_krw=1., pnl_krw=50.),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'decisions.jsonl'
            path.write_text('\n'.join(json.dumps(l) for l in lines)+'\n')
            rows, _ = report(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['buy_fee_krw'], 1.)  # FIFO: the first (oldest) open buy pairs first


if __name__ == '__main__':
    unittest.main()

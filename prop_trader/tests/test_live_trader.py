import unittest
from unittest.mock import Mock

from prop_trader.live_trader import LiveConfig, evaluate_entry, evaluate_position, net_external_flow


def position(entry=100., stop=97.5, target=105., horizon=1000.):
    return dict(symbol='KRW-BTC', qty=1., entry_price=entry, entry_time=0., stop=stop, target=target,
                horizon_expires=horizon, identifier='x')


class LiveConfigTests(unittest.TestCase):
    def test_rejects_bad_mode(self):
        with self.assertRaises(ValueError):
            LiveConfig(mode='yolo')

    def test_rejects_out_of_range_tick(self):
        with self.assertRaises(ValueError):
            LiveConfig(tick_seconds=1)

    def test_rejects_too_fast_refresh(self):
        with self.assertRaises(ValueError):
            LiveConfig(refresh_seconds=10)

    def test_rejects_negative_entry_threshold(self):
        with self.assertRaises(ValueError):
            LiveConfig(entry_threshold=-0.001)

    def test_rejects_universe_top_out_of_range(self):
        with self.assertRaises(ValueError):
            LiveConfig(universe_top=101)
        with self.assertRaises(ValueError):
            LiveConfig(universe_top=0)
        LiveConfig(universe_top=40)  # within the new resident-worker-era ceiling; must not raise

    def test_rejects_too_frequent_retrain(self):
        with self.assertRaises(ValueError):
            LiveConfig(retrain_seconds=60)

    def test_accepts_defaults(self):
        cfg = LiveConfig()
        self.assertEqual(cfg.mode, 'paper')
        self.assertEqual(cfg.max_positions, 3)


class EvaluatePositionTests(unittest.TestCase):
    def test_stop_loss_fires_below_stop(self):
        action, reason = evaluate_position(position(), 97.0, now_ts=500)
        self.assertEqual(action, 'SELL')
        self.assertIn('손절', reason)

    def test_take_profit_fires_above_target(self):
        action, reason = evaluate_position(position(), 106.0, now_ts=500)
        self.assertEqual(action, 'SELL')
        self.assertIn('익절', reason)

    def test_horizon_expiry_forces_exit(self):
        action, reason = evaluate_position(position(), 101.0, now_ts=1000)
        self.assertEqual(action, 'SELL')
        self.assertIn('만료', reason)

    def test_stay_when_within_band(self):
        action, reason = evaluate_position(position(), 101.0, now_ts=500)
        self.assertEqual(action, 'STAY')


class EvaluateEntryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = LiveConfig()

    def row(self, **overrides):
        base = dict(valid_until='2099-01-01T00:00:00+00:00', rank=1, symbol='KRW-ETH', expected_net_return=0.01, stale=False, positive_after_costs=True)
        base.update(overrides)
        return base

    def test_no_candidate(self):
        self.assertEqual(evaluate_entry(None, self.cfg)[0], 'STAY')

    def test_stale_row_skipped(self):
        self.assertEqual(evaluate_entry(self.row(stale=True), self.cfg)[0], 'STAY')

    def test_negative_after_costs_skipped(self):
        self.assertEqual(evaluate_entry(self.row(positive_after_costs=False), self.cfg)[0], 'STAY')

    def test_below_threshold_skipped(self):
        self.assertEqual(evaluate_entry(self.row(expected_net_return=0.0001), self.cfg)[0], 'STAY')

    def test_buys_above_threshold(self):
        action, reason = evaluate_entry(self.row(expected_net_return=0.01), self.cfg)
        self.assertEqual(action, 'BUY')
        self.assertIn('순위 1', reason)


def deposit(amount, done_at, state='ACCEPTED'):
    return dict(currency='KRW', amount=str(amount), done_at=done_at, state=state)


def withdraw(amount, done_at, fee=0, state='DONE'):
    return dict(currency='KRW', amount=str(amount), fee=str(fee), done_at=done_at, state=state)


class NetExternalFlowTests(unittest.TestCase):
    def client(self, deposits=(), withdraws=()):
        fake = Mock()
        def call(method, path, params=None, private=True):
            if path == '/v1/deposits':
                return list(deposits)
            if path == '/v1/withdraws':
                return list(withdraws)
            raise AssertionError(f'unexpected call {path}')
        fake.request.side_effect = call
        return fake

    def test_deposit_after_baseline_counts(self):
        c = self.client(deposits=[deposit(66000, '2026-09-09T22:21:33+09:00')])
        self.assertEqual(net_external_flow(c, '2026-09-09T12:00:00+00:00'), 66000.)

    def test_deposit_before_baseline_excluded(self):
        c = self.client(deposits=[deposit(66000, '2026-09-09T05:00:00+09:00')])
        self.assertEqual(net_external_flow(c, '2026-09-09T12:00:00+00:00'), 0.)

    def test_non_accepted_deposit_excluded(self):
        c = self.client(deposits=[deposit(10000, '2026-09-09T22:21:33+09:00', state='REJECTED')])
        self.assertEqual(net_external_flow(c, '2026-09-09T12:00:00+00:00'), 0.)

    def test_withdrawal_subtracts_amount_plus_fee(self):
        c = self.client(withdraws=[withdraw(5000, '2026-09-09T22:21:33+09:00', fee=100)])
        self.assertEqual(net_external_flow(c, '2026-09-09T12:00:00+00:00'), -5100.)

    def test_mixed_deposits_and_withdrawals_net_out(self):
        # Baseline is 12:00 UTC = 21:00 KST: the 19:33 KST deposit predates it and is excluded,
        # the 22:21 KST deposit and 23:00 KST withdrawal both postdate it and are included.
        c = self.client(
            deposits=[deposit(66000, '2026-09-09T22:21:33+09:00'), deposit(10000, '2026-09-09T19:33:44+09:00')],
            withdraws=[withdraw(1000, '2026-09-09T23:00:00+09:00', fee=0)])
        self.assertEqual(net_external_flow(c, '2026-09-09T12:00:00+00:00'), 65000.)


if __name__ == '__main__':
    unittest.main()

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from prop_trader.execution_rules import reprice, minimum_entry, fills
from prop_trader.live_trader import LiveTrader, LiveConfig, evaluate_entry
from prop_trader.upbit_broker import BrokerError, order_payload, submit
from prop_trader.tests.test_upbit_broker import client as broker_client


def row(symbol='KRW-BTC', predicted=110, origin=None):
    origin = origin if origin is not None else int(time.time())//3600*3600
    return dict(symbol=symbol, interval='1h', horizon=5,
                as_of=datetime.fromtimestamp(origin, timezone.utc).isoformat(),
                forecast_prices=[predicted]*5, rank=1, stale=False,
                positive_after_costs=True, expected_net_return=.1,
                valid_until='2099-01-01T00:00:00+00:00', estimated_entry_vwap=100)


def book(symbol='KRW-BTC', price=100):
    return dict(market=symbol, timestamp=time.time()*1000,
                orderbook_units=[dict(ask_price=price, ask_size=10000)])


class ExecutionRulesTests(unittest.TestCase):
    def test_expired_and_missing_expiry_fail_closed(self):
        r=row(); r['valid_until']='2020-01-01T00:00:00+00:00'
        self.assertEqual(evaluate_entry(r, LiveConfig())[0], 'STAY')
        del r['valid_until']
        self.assertEqual(evaluate_entry(r, LiveConfig())[0], 'STAY')

    def test_chasing_rally_removes_edge(self):
        r=reprice(row(predicted=110), book(price=112), '1h', time.time(), 8000)
        self.assertLess(r['expected_net_return'], 0)
        self.assertEqual(evaluate_entry(r, LiveConfig())[0], 'STAY')

    def test_stale_book_and_forecast_rejected(self):
        b=book(); b['timestamp']-=11000
        with self.assertRaises(ValueError): reprice(row(), b, '1h', time.time(), 8000)
        with self.assertRaises(ValueError): reprice(row(origin=0), book(), '1h', time.time(), 8000)

    def test_minimum_size_has_exit_buffer(self):
        self.assertGreater(minimum_entry(), 5000)
        self.assertGreater(minimum_entry()/(1.001)*.975*.999, 5000)

    def test_owned_exit_can_exceed_buy_cap_but_not_quantity(self):
        c=broker_client()
        submit(c, order_payload('KRW-BTC','sell','.00012','exit'), 'test',
               max_krw=8000, owned_volume='.00012')
        with self.assertRaises(BrokerError):
            submit(c, order_payload('KRW-BTC','sell','.00013','exit2'), 'test',
                   max_krw=8000, owned_volume='.00012')

    def test_fill_requires_matching_details(self):
        with self.assertRaises(ValueError): fills(dict(executed_volume='2', trades=[]))

    def test_quote_expiry_during_validation_never_submits(self):
        c=broker_client()
        with self.assertRaises(BrokerError):
            submit(c, order_payload('KRW-BTC','buy',8000,'expired'), 'test', deadline=0)
        self.assertFalse(any(call.args[0]=='POST' for call in c.request.call_args_list))


class TraderSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.patch=patch('prop_trader.live_trader.LIVE_ROOT', self.root)
        self.patch.start(); self.addCleanup(self.patch.stop)
        self.client=Mock()
        self.client.request.side_effect=self.request
        self.orders={}
        self.posts=[]
        self.ranking=dict(rankings=[row()])

    def request(self, method, path, params=None, private=True):
        if path=='/v1/accounts':
            return [dict(currency='KRW', balance='100000', locked='0')]
        if path in ('/v1/deposits','/v1/withdraws'): return []
        if path=='/v1/market/all': return [dict(market='KRW-BTC')]
        if path=='/v1/ticker': return [dict(market='KRW-BTC', trade_price=100)]
        if path=='/v1/orderbook': return [book()]
        if path=='/v1/order': return self.orders[params['identifier']]
        if path=='/v1/orders/chance':
            return dict(market=dict(state='active',bid_types=['price'],ask_types=['market'],
                                   bid=dict(min_total='5000'),ask=dict(min_total='5000')),
                        bid_fee='0.001',bid_account=dict(balance='100000'),ask_account=dict(balance='1000'))
        if path=='/v1/orders':
            self.posts.append(params)
            self.orders[params['identifier']]=dict(state='wait', executed_volume='0',uuid='fake')
            return self.orders[params['identifier']]
        if path=='/v1/orders/test': return {}
        raise AssertionError(path)

    def trader(self, mode='paper', budget=8000):
        t=LiveTrader(LiveConfig(mode=mode, max_krw_per_trade=budget), self.client)
        t.journal_path=self.root/'orders.sqlite'
        t.latest_ranking=lambda: self.ranking
        return t

    def test_stop_does_not_rebuy_same_tick_or_after_restart(self):
        t=self.trader(); t.tick()
        self.assertIn('KRW-BTC',t.positions)
        t.positions['KRW-BTC']['stop']=101
        t.tick()
        self.assertNotIn('KRW-BTC',t.positions)
        restarted=self.trader(); restarted.tick()
        self.assertNotIn('KRW-BTC',restarted.positions)

    def test_small_budget_skips_entry_without_order(self):
        t=self.trader(budget=5000); t.tick()
        self.assertFalse(t.positions)
        self.assertFalse(self.posts)

    def test_exchange_test_does_not_create_phantom_position(self):
        t=self.trader('test'); t.tick()
        self.assertFalse(t.positions)
        self.assertFalse(t.pending)
        self.assertFalse(self.posts)
        self.assertTrue(any(c.args[1]=='/v1/orders/test' for c in self.client.request.call_args_list))

    def test_wait_then_terminal_partial_buy_restart_exactly_once(self):
        t=self.trader('live'); t.tick()
        self.assertEqual(len(self.posts),1)
        self.assertFalse(t.positions)
        t.tick(); self.assertEqual(len(self.posts),1)
        identifier=next(iter(t.pending))
        self.orders[identifier]=dict(state='cancel',executed_volume='70',paid_fee='7',
                                    trades=[dict(volume='70',funds='7000')])
        restarted=self.trader('live'); restarted.tick()
        self.assertFalse(restarted.pending)
        self.assertEqual(restarted.positions['KRW-BTC']['qty'],70)
        self.assertEqual(restarted.positions['KRW-BTC']['cost_krw'],7007)
        restarted.tick()
        self.assertEqual(restarted.positions['KRW-BTC']['qty'],70)
        self.assertEqual(len(self.posts),1)

    def test_timeout_keeps_durable_pending_and_never_reposts(self):
        original=self.client.request.side_effect
        def timeout(method,path,*args,**kwargs):
            if path=='/v1/orders':
                original(method,path,*args,**kwargs)
                raise BrokerError('timeout')
            return original(method,path,*args,**kwargs)
        self.client.request.side_effect=timeout
        t=self.trader('live'); t.tick()
        self.assertTrue(t.pending)
        restarted=self.trader('live'); restarted.tick()
        self.assertEqual(len(self.posts),1)

    def test_partial_sell_keeps_remainder_and_cost(self):
        t=self.trader('live')
        t.positions['KRW-BTC']=dict(qty=100,cost_krw=10010,entry_price=100)
        t.pending['sell']=dict(market='KRW-BTC',side='sell')
        self.orders['sell']=dict(state='cancel',executed_volume='60',paid_fee='6',
                                 trades=[dict(volume='60',funds='6000')])
        t._reconcile()
        self.assertEqual(t.positions['KRW-BTC']['qty'],40)
        self.assertAlmostEqual(t.positions['KRW-BTC']['cost_krw'],4004)
        t._reconcile()
        self.assertEqual(t.positions['KRW-BTC']['qty'],40)

    def test_sell_submission_keeps_position_until_confirmed(self):
        t=self.trader('live')
        t.started_tick_ts=time.time()
        t.positions['KRW-BTC']=dict(qty=100,cost_krw=10010,entry_price=100,
                                   forecast_as_of=row()['as_of'])
        normal=self.client.request.side_effect
        def held(method,path,*args,**kwargs):
            if path=='/v1/accounts':
                return [dict(currency='BTC',balance='100',locked='0')]
            return normal(method,path,*args,**kwargs)
        self.client.request.side_effect=held
        t._sell('KRW-BTC',100,'익절')
        self.assertEqual(t.positions['KRW-BTC']['qty'],100)
        self.assertEqual(len(t.pending),1)
        identifier=next(iter(t.pending))
        self.orders[identifier]=dict(state='done',executed_volume='100',paid_fee='10',
                                    trades=[dict(volume='100',funds='10000')])
        t._reconcile()
        self.assertNotIn('KRW-BTC',t.positions)

    def test_dust_sell_retained_without_post(self):
        t=self.trader(); t.started_tick_ts=time.time()
        t.positions['KRW-BTC']=dict(qty=40,cost_krw=5000,entry_price=125)
        t._sell('KRW-BTC',100,'손절')
        self.assertIn('KRW-BTC',t.positions)
        self.assertGreater(t.retry_after['KRW-BTC'],time.time())
        self.assertFalse(self.posts)

    def test_live_dust_sell_triggers_rescue_buy(self):
        t=self.trader('live'); t.started_tick_ts=time.time()
        t.positions['KRW-BTC']=dict(qty=0.00005,cost_krw=4900,entry_price=98000000,
                                   horizon_expires=t.started_tick_ts,forecast_as_of=row()['as_of'])
        normal=self.client.request.side_effect
        def tiny_balance(method,path,*args,**kwargs):
            if path=='/v1/accounts':
                return [dict(currency='BTC',balance='0.00005',locked='0')]
            return normal(method,path,*args,**kwargs)
        self.client.request.side_effect=tiny_balance
        t._sell('KRW-BTC',98000000,'손절')
        self.assertIn('KRW-BTC',t.positions)  # original tracked position untouched
        self.assertEqual(len(self.posts),1)
        self.assertEqual(self.posts[0]['market'],'KRW-BTC')
        self.assertEqual(self.posts[0]['side'],'bid')
        self.assertTrue(t.positions['KRW-BTC'].get('rescue_attempted_at'))
        self.assertGreater(t.retry_after['KRW-BTC'],time.time())

    def test_dust_rescue_respects_cooldown(self):
        t=self.trader('live'); t.started_tick_ts=time.time()
        t.positions['KRW-BTC']=dict(qty=0.00005,cost_krw=4900,entry_price=98000000,
                                   horizon_expires=t.started_tick_ts,forecast_as_of=row()['as_of'],
                                   rescue_attempted_at=time.time())
        normal=self.client.request.side_effect
        def tiny_balance(method,path,*args,**kwargs):
            if path=='/v1/accounts':
                return [dict(currency='BTC',balance='0.00005',locked='0')]
            return normal(method,path,*args,**kwargs)
        self.client.request.side_effect=tiny_balance
        t._sell('KRW-BTC',98000000,'손절')
        self.assertFalse(self.posts)  # cooldown still active, no repeat rescue buy

    def test_dust_rescue_skipped_in_paper_mode(self):
        t=self.trader(); t.started_tick_ts=time.time()
        t.positions['KRW-BTC']=dict(qty=0.00005,cost_krw=4900,entry_price=98000000)
        t._sell('KRW-BTC',98000000,'손절')
        self.assertFalse(self.posts)

    def test_rescue_buy_fill_accumulates_into_existing_position(self):
        t=self.trader('live')
        t.positions['KRW-BTC']=dict(symbol='KRW-BTC',qty=0.00005,cost_krw=4900,entry_price=98000000,
                                    entry_time=0,stop=95000000,target=102000000,horizon_expires=0,
                                    identifier='orig',forecast_as_of=None,reconciled=True)
        t.pending['rescue']=dict(identifier='rescue',market='KRW-BTC',side='buy',created_at=1,horizon_expires=1)
        self.orders['rescue']=dict(state='done',executed_volume='0.00006',paid_fee='5.75',
                                  trades=[dict(volume='0.00006',funds='5750')])
        t._reconcile()
        p=t.positions['KRW-BTC']
        self.assertAlmostEqual(p['qty'],0.00011)
        self.assertAlmostEqual(p['cost_krw'],4900+5750+5.75)
        self.assertEqual(p['identifier'],'orig')  # original position identity preserved, not the rescue's

    def test_new_negative_forecast_closes_position(self):
        t=self.trader(); t.tick()
        self.ranking=dict(rankings=[row(predicted=95)])
        t.tick()
        self.assertFalse(t.positions)

    def test_legacy_position_uses_actual_fill_not_estimate(self):
        t=self.trader('live')
        t.positions['KRW-BTC']=dict(identifier='old',qty=98,entry_price=100)
        self.orders['old']=dict(state='done',executed_volume='100',paid_fee='10',
                               trades=[dict(volume='100',funds='10000')])
        t._migrate_positions([dict(currency='BTC',balance='100',locked='0')])
        self.assertEqual(t.positions['KRW-BTC']['qty'],100)
        self.assertEqual(t.positions['KRW-BTC']['cost_krw'],10010)

    def test_corrupt_state_does_not_start_empty(self):
        t=self.trader('live')
        t.state_path.write_text('{bad')
        with self.assertRaises(ValueError): self.trader('live')

    def test_forecast_expiry_is_target_time_not_entry_plus_five_hours(self):
        t=self.trader(); t.tick()
        from prop_trader.execution_rules import timestamp
        self.assertEqual(t.positions['KRW-BTC']['horizon_expires'],
                         timestamp(self.ranking['rankings'][0]['as_of'])+18000)


if __name__=='__main__': unittest.main()

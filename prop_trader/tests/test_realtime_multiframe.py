import time
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock

from prop_trader.multiframe import decide, buy_budget, exit_fraction, candidate_score, partial_qualifies, partial_budget, hourly_only
from prop_trader.market_stream import MarketStream
from prop_trader.resident_forecasts import next_job, boundary, completed_bars
from prop_trader.live_trader import LiveConfig, LiveTrader
from prop_trader.analog_signal import FEATURE_NAMES
from prop_trader.technical import FEATURES
from prop_trader.tests import test_execution_safety as fixtures
row, book = fixtures.row, fixtures.book


def frames(minute=110, hour=110, day=110):
    out={}
    for iv, predicted in zip(('1m','1h','1d'),(minute,hour,day)):
        r=row(predicted=predicted, origin=boundary(iv,time.time()))
        r.update(interval=iv,technical_features={'atr14': .01}, last_close=100, forecast_consistency=1,
                 forecast=[dict(horizon=i+1,predicted_close=predicted) for i in range(5)])
        out[iv]=r
    return out


def live_book():
    b=book(); b['orderbook_units'][0].update(bid_price=100,bid_size=10000)
    return b


class MultiFrameTests(unittest.TestCase):
    def decision(self, **kwargs):
        held=kwargs.pop('held',False)
        return decide(frames(**kwargs),live_book(),time.time(),8000,held=held)

    def test_all_three_required_to_buy(self):
        self.assertEqual(self.decision()['action'],'BUY')
        for iv in ('1m','1h','1d'):
            rows=frames(); del rows[iv]
            self.assertEqual(decide(rows,live_book(),time.time(),8000)['action'],'STAY')

    def test_each_frame_can_veto_entry(self):
        for arg in ('minute','hour','day'):
            self.assertEqual(self.decision(**{arg:99})['action'],'STAY')

    def test_stale_minute_vetoes_fresh_longer_frames(self):
        r=frames(); r['1m']['as_of']='2020-01-01T00:00:00+00:00'
        self.assertFalse(decide(r,live_book(),time.time(),8000)['complete'])

    def test_minute_noise_does_not_force_exit_when_hour_positive(self):
        self.assertEqual(self.decision(minute=99,held=True)['action'],'STAY')

    def test_day_down_exits_and_mild_hour_decline_scales_out(self):
        self.assertEqual(self.decision(day=99,held=True)['sell_fraction'],1)
        self.assertEqual(self.decision(hour=99.7,held=True)['sell_fraction'],.5)

    def test_sizing_scales_without_exceeding_cap_or_cash(self):
        weak=dict(action='BUY',complete=True,signal_strength=.1)
        strong=dict(weak,signal_strength=.9)
        self.assertGreater(buy_budget(strong,100000,20000),buy_budget(weak,100000,20000))
        self.assertLessEqual(buy_budget(strong,10000,20000),9000)
        self.assertEqual(buy_budget(weak,100000,8000),0)
        self.assertEqual(exit_fraction(.5,80,100),1)
        self.assertEqual(exit_fraction(.5,200,100),.5)

    def test_partial_qualifies_when_hourly_alone_is_positive(self):
        d=self.decision(day=99)  # daily vetoes the full gate, hourly edge is still strongly positive
        self.assertEqual(d['action'],'STAY')
        self.assertTrue(partial_qualifies(d,.002))

    def test_partial_does_not_qualify_a_full_buy_or_incomplete_decision(self):
        self.assertFalse(partial_qualifies(self.decision(),.002))  # already a full BUY
        self.assertFalse(partial_qualifies(dict(complete=False,action='STAY'),.002))

    def test_partial_budget_is_positive_but_smaller_than_full_buy_budget(self):
        full=buy_budget(self.decision(),100000,20000)
        partial=partial_budget(self.decision(day=99),100000,20000,.002)
        self.assertGreater(full,0);self.assertGreater(partial,0)
        self.assertLess(partial,full)

    def test_partial_budget_zero_when_hourly_edge_too_thin(self):
        d=self.decision(hour=100,day=99)  # hourly ~breakeven after costs, still vetoed by day
        self.assertEqual(partial_budget(d,100000,20000,.002),0)

    def test_partial_budget_nonzero_at_intermediate_signal_strength(self):
        # Regression: half of buy_budget's own ceiling routinely undercut the exchange minimum
        # well before strength reached 1 (e.g. ~0.35 needed the ceiling itself just to clear the
        # floor at an 8000 cap), which silently zeroed out any partial candidate that wasn't
        # near-maximum strength even with a perfectly good hourly edge.
        d=dict(action='STAY',complete=True,returns={'1h':.0272},signal_strength=.347)
        self.assertGreaterEqual(partial_budget(d,76232.,8000,0),5000)

    def test_hourly_only_reads_the_hourly_frame_without_needing_1m_or_1d(self):
        rows=frames();del rows['1d'];del rows['1m']  # only 1h left, as for a thinly-traded market
        d=hourly_only(rows,live_book(),time.time(),8000)
        self.assertTrue(d['complete'])
        self.assertEqual(d['action'],'STAY')  # never claims full conviction on its own
        self.assertGreater(d['returns']['1h'],0)
        self.assertTrue(partial_qualifies(d,.002))

    def test_hourly_only_incomplete_when_1h_itself_is_missing(self):
        d=hourly_only({},live_book(),time.time(),8000)
        self.assertFalse(d['complete'])

    def test_hourly_only_never_outranks_full_conviction_sizing(self):
        full=self.decision()  # all three frames agree -> full BUY
        rows=frames();del rows['1d'];del rows['1m']
        partial=hourly_only(rows,live_book(),time.time(),8000)
        self.assertGreater(buy_budget(full,100000,8000),partial_budget(partial,100000,8000,.002))

    def test_candidate_score_boosts_upward_movers_over_flat_equal_volume(self):
        flat = dict(acc_trade_price_24h=1_000_000, signed_change_rate=0.)
        pumping = dict(acc_trade_price_24h=1_000_000, signed_change_rate=.15)
        self.assertGreater(candidate_score(pumping), candidate_score(flat))

    def test_candidate_score_never_boosts_downward_movers(self):
        flat = dict(acc_trade_price_24h=1_000_000, signed_change_rate=0.)
        dumping = dict(acc_trade_price_24h=1_000_000, signed_change_rate=-.15)
        self.assertEqual(candidate_score(dumping), candidate_score(flat))

    def test_candidate_score_tolerates_missing_or_bad_fields(self):
        self.assertEqual(candidate_score({}), 0.)
        self.assertEqual(candidate_score(dict(acc_trade_price_24h='not-a-number')), 0.)
        self.assertEqual(candidate_score(dict(acc_trade_price_24h=-5, signed_change_rate=.1)), 0.)


class StreamTests(unittest.TestCase):
    def test_disconnect_and_old_quotes_are_never_tradable(self):
        stream=MarketStream();stream.connected=True
        stream.accept(dict(type='ticker',code='KRW-BTC',timestamp=100000,trade_price=100),received=100)
        self.assertIn('KRW-BTC',stream.snapshot(now=105)['tickers'])
        self.assertFalse(stream.snapshot(now=111)['tickers'])
        stream.connected=False
        self.assertFalse(stream.snapshot(now=105)['tickers'])

    def test_out_of_order_and_invalid_prices_are_ignored(self):
        stream=MarketStream();stream.connected=True
        for stamp,price in [(100000,100),(99000,99),(101000,float('nan'))]:
            stream.accept(dict(type='ticker',code='KRW-BTC',timestamp=stamp,trade_price=price),received=101)
        self.assertEqual(stream.snapshot(now=102)['tickers']['KRW-BTC']['trade_price'],100)


class ResidentScheduleTests(unittest.TestCase):
    def test_held_first_and_once_per_completed_bar(self):
        now=1788992110
        markets=['KRW-HELD','KRW-BTC'];attempted=set()
        for iv in ('1d','1h','1m'):
            key=next_job(markets,{},attempted,now)
            self.assertEqual(key[:2],('KRW-HELD',iv));attempted.add(key)
        self.assertEqual(next_job(markets,{},attempted,now)[:2],('KRW-BTC','1d'))
        self.assertEqual(next_job(['KRW-HELD'],{},attempted,now),None)
        self.assertEqual(next_job(['KRW-HELD'],{},attempted,now+60)[1],'1m')

    def test_forming_bar_not_used(self):
        client=Mock();client.request.return_value=[dict(candle_date_time_utc='2026-09-10T01:00:00',
            opening_price=1,high_price=1,low_price=1,trade_price=1,candle_acc_trade_volume=1)]
        cutoff=datetime(2026,9,10,1,tzinfo=timezone.utc).timestamp()
        with self.assertRaises(ValueError):completed_bars(client,'KRW-BTC','1h',cutoff)


class RealtimeIntegrationTests(unittest.TestCase):
    setUp = fixtures.TraderSafetyTests.setUp
    request = fixtures.TraderSafetyTests.request
    trader = fixtures.TraderSafetyTests.trader
    def test_realtime_path_uses_three_frames_and_adaptive_budget(self):
        t=self.trader()
        runtime=Mock()
        feed=dict(connected=True,error=None,books={'KRW-BTC':live_book()},
                  tickers={'KRW-BTC':dict(market='KRW-BTC',trade_price=100,acc_trade_price_24h=100000)})
        runtime.stream.snapshot.return_value=feed
        runtime.forecasts.return_value=dict(records={'KRW-BTC':frames()})
        t.runtime=runtime
        t.tick()
        self.assertIn('KRW-BTC',t.positions)
        self.assertLessEqual(t.positions['KRW-BTC']['cost_krw'],8000)
        self.assertTrue(t.multiframe_decisions['KRW-BTC']['complete'])
        from prop_trader.live_server import build_dashboard_payload
        dashboard=build_dashboard_payload(t)
        self.assertEqual(set(dashboard['forecasts']), {'1m','1h','1d'})
        self.assertTrue(dashboard['rankings']['1m']['rankings'])
        runtime.forecasts.return_value=dict(records={'KRW-BTC':frames(day=99)})
        t.tick()
        self.assertFalse(t.positions)

    def test_untracked_dust_with_no_ticker_does_not_block_equity_logging(self):
        # Regression: an untradeable holding with no active market (e.g. ETHW/ETHF/STRK -- a
        # delisted/forked asset that can never get a WebSocket ticker price) used to mark the
        # whole equity snapshot incomplete every single tick, forever, freezing 누적 수익률 while
        # real trades kept happening. Only a holding this bot actually tracks should block it.
        t=self.trader('live')
        def with_dust(method,path,params=None,private=True):
            if path=='/v1/accounts':
                return [dict(currency='KRW',balance='50000',locked='0'),
                        dict(currency='ZZZ',balance='1',locked='0')]
            return self.request(method,path,params,private)
        self.client.request.side_effect=with_dust
        runtime=Mock()
        runtime.stream.snapshot.return_value=dict(connected=True,error=None,books={},tickers={})
        runtime.forecasts.return_value=dict(records={})
        t.runtime=runtime
        t.tick()
        rows=t.equity_path.read_text().strip().splitlines()
        self.assertGreater(len(rows),1)  # header + at least one appended row

    def test_realtime_partial_conviction_buy_when_only_hourly_agrees(self):
        t=self.trader()  # fixtures default budget=8000
        runtime=Mock()
        feed=dict(connected=True,error=None,books={'KRW-BTC':live_book()},
                  tickers={'KRW-BTC':dict(market='KRW-BTC',trade_price=100,acc_trade_price_24h=100000)})
        runtime.stream.snapshot.return_value=feed
        runtime.forecasts.return_value=dict(records={'KRW-BTC':frames(day=99)})  # daily vetoes the full gate
        t.runtime=runtime
        t.tick()
        self.assertIn('KRW-BTC',t.positions)
        # At this account size (8000 cap), half-conviction collapses to the exchange minimum --
        # still a real, meaningfully smaller trade than the 8000 a full-conviction buy would use.
        self.assertLess(t.positions['KRW-BTC']['cost_krw'],8000)
        self.assertGreaterEqual(t.positions['KRW-BTC']['cost_krw'],5000)
        self.assertEqual(t.multiframe_decisions['KRW-BTC']['action'],'PARTIAL_CANDIDATE')

    def test_realtime_missing_frame_falls_back_to_partial_not_full_buy(self):
        # A missing frame (many smaller Upbit markets can never produce a gap-free 1m series)
        # no longer blocks a buy outright -- it falls back to the smaller hourly-only partial
        # conviction size, never the full-conviction budget.
        t=self.trader();runtime=Mock()
        runtime.stream.snapshot.return_value=dict(connected=True,error=None,books={'KRW-BTC':live_book()},
            tickers={'KRW-BTC':dict(market='KRW-BTC',trade_price=100,acc_trade_price_24h=100000)})
        r=frames();del r['1d']
        runtime.forecasts.return_value=dict(records={'KRW-BTC':r})
        t.runtime=runtime;t.tick()
        self.assertIn('KRW-BTC',t.positions)
        self.assertLess(t.positions['KRW-BTC']['cost_krw'],8000)  # partial, not the full 8000 cap

    def test_realtime_all_frames_missing_prevents_any_buy(self):
        t=self.trader();runtime=Mock()
        runtime.stream.snapshot.return_value=dict(connected=True,error=None,books={'KRW-BTC':live_book()},
            tickers={'KRW-BTC':dict(market='KRW-BTC',trade_price=100,acc_trade_price_24h=100000)})
        runtime.forecasts.return_value=dict(records={'KRW-BTC':{}})
        t.runtime=runtime;t.tick()
        self.assertFalse(t.positions)


def full_technical(**overrides):
    values=dict(rsi14=55.,macd_hist=.1,atr14=.01,bb_position=.5,ema_gap=.01,adx14=20.,
                cmf20=.05,volume_ratio=1.2,return1=.001,return5=.005,return20=.02,range_fraction=.3)
    values.update(overrides)
    return {k:values[k] for k in FEATURES}


def analog_frames():
    # feature_vector() needs a complete technical_features dict plus forecast_slope/
    # few_shot_log_return -- the shared frames() fixture only sets atr14, so build our own.
    out=frames()
    out['1h'].update(technical_features=full_technical(),forecast_slope=1.,few_shot_log_return=.01)
    return out


class AnalogGateIntegrationTests(unittest.TestCase):
    """The self-activation contract: analog_gate_enabled must be a no-op below analog_min_n,
    and only start filtering BUYs once enough same-symbol horizon-apart pairs exist -- so
    turning it on never needs to be timed against how much history has accumulated."""
    setUp = fixtures.TraderSafetyTests.setUp
    request = fixtures.TraderSafetyTests.request

    def trader(self):
        cfg=LiveConfig(mode='paper',max_krw_per_trade=8000,analog_gate_enabled=True,
                       analog_min_n=2,analog_min_win_rate=.6)
        t=LiveTrader(cfg,self.client)
        t.journal_path=self.root/'orders.sqlite'
        return t

    def tick_with_db(self, t, db):
        runtime=Mock()
        runtime.stream.snapshot.return_value=dict(connected=True,error=None,books={'KRW-BTC':live_book()},
            tickers={'KRW-BTC':dict(market='KRW-BTC',trade_price=100,acc_trade_price_24h=100000)})
        runtime.forecasts.return_value=dict(records={'KRW-BTC':analog_frames()})
        t.runtime=runtime
        t._analog_db=lambda: db
        t.tick()

    def analog_row(self, realized_return):
        return dict(vector=[0.]*len(FEATURE_NAMES),realized_return=realized_return,symbol='X',as_of='x')

    def test_noop_below_min_n(self):
        t=self.trader()
        self.tick_with_db(t,[self.analog_row(-.5)])  # 1 analog, below min_n=2 -> gate has no say
        self.assertIn('KRW-BTC',t.positions)

    def test_blocks_once_enough_bad_analogs(self):
        t=self.trader()
        self.tick_with_db(t,[self.analog_row(-.05) for _ in range(5)])  # 5 >= min_n, all losers
        self.assertNotIn('KRW-BTC',t.positions)
        self.assertEqual(t.multiframe_decisions['KRW-BTC']['action'],'STAY')
        self.assertIn('과거 유사 패턴',t.multiframe_decisions['KRW-BTC']['reason'])

    def test_allows_once_enough_good_analogs(self):
        t=self.trader()
        self.tick_with_db(t,[self.analog_row(.05) for _ in range(5)])  # 5 >= min_n, all winners
        self.assertIn('KRW-BTC',t.positions)


if __name__=='__main__': unittest.main()

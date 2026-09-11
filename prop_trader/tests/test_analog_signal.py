import unittest
from datetime import datetime, timedelta, timezone

from prop_trader.analog_signal import (
    feature_vector, build_analog_db, nearest_analogs, analog_signal, confirms_buy, too_uncertain, FEATURE_NAMES,
)
from prop_trader.technical import FEATURES

BASE = datetime(2026, 9, 10, tzinfo=timezone.utc)


def technical(**overrides):
    values = dict(rsi14=55., macd_hist=.1, atr14=.02, bb_position=.5, ema_gap=.01, adx14=20.,
                  cmf20=.05, volume_ratio=1.2, return1=.001, return5=.005, return20=.02, range_fraction=.3)
    values.update(overrides)
    return {k: values[k] for k in FEATURES}


def record(symbol, hour_offset, last_close, interval='1h', slope=1., consistency=1., few_shot=0.01, tf=None):
    as_of = BASE + timedelta(hours=hour_offset)
    return dict(symbol=symbol, interval=interval, as_of=as_of.isoformat(), last_close=last_close,
                forecast_slope=slope, forecast_consistency=consistency, few_shot_log_return=few_shot,
                technical_features=technical(**(tf or {})))


class FeatureVectorTests(unittest.TestCase):
    def test_valid_record_has_expected_length(self):
        vec = feature_vector(record('KRW-BTC', 0, 100.))
        self.assertEqual(len(vec), len(FEATURE_NAMES))
        self.assertTrue(all(isinstance(v, float) for v in vec))

    def test_missing_technical_features_returns_none(self):
        r = record('KRW-BTC', 0, 100.)
        del r['technical_features']
        self.assertIsNone(feature_vector(r))

    def test_missing_forecast_field_returns_none(self):
        r = record('KRW-BTC', 0, 100.)
        del r['few_shot_log_return']
        self.assertIsNone(feature_vector(r))

    def test_nonfinite_value_returns_none(self):
        r = record('KRW-BTC', 0, 100., tf=dict(rsi14=float('nan')))
        self.assertIsNone(feature_vector(r))

    def test_nonpositive_last_close_returns_none(self):
        self.assertIsNone(feature_vector(record('KRW-BTC', 0, 0.)))


class BuildAnalogDbTests(unittest.TestCase):
    def test_pairs_records_exactly_horizon_apart(self):
        rows = [record('KRW-BTC', 0, 100.), record('KRW-BTC', 5, 110.)]
        db = build_analog_db(rows, '1h', horizon=5)
        self.assertEqual(len(db), 1)
        self.assertAlmostEqual(db[0]['realized_return'], .10)
        self.assertEqual(db[0]['symbol'], 'KRW-BTC')

    def test_gap_excludes_the_pair(self):
        # Record 5 hours later is missing a clean run -- as_of is 6h later instead of 5h.
        rows = [record('KRW-BTC', 0, 100.), record('KRW-BTC', 6, 110.)]
        db = build_analog_db(rows, '1h', horizon=5)
        self.assertEqual(db, [])

    def test_pools_across_symbols(self):
        rows = [record('KRW-BTC', 0, 100.), record('KRW-BTC', 5, 105.),
                record('KRW-ETH', 0, 50.), record('KRW-ETH', 5, 45.)]
        db = build_analog_db(rows, '1h', horizon=5)
        self.assertEqual({row['symbol'] for row in db}, {'KRW-BTC', 'KRW-ETH'})

    def test_different_interval_rows_ignored(self):
        rows = [record('KRW-BTC', 0, 100., interval='1m'), record('KRW-BTC', 5, 110., interval='1m')]
        self.assertEqual(build_analog_db(rows, '1h', horizon=5), [])

    def test_unsupported_interval_raises(self):
        with self.assertRaises(ValueError):
            build_analog_db([], '2h', horizon=5)

    def test_unparseable_row_skipped_not_fatal(self):
        broken = record('KRW-BTC', 0, 100.); del broken['technical_features']
        rows = [broken, record('KRW-BTC', 5, 110.)]
        self.assertEqual(build_analog_db(rows, '1h', horizon=5), [])


class NearestAnalogsTests(unittest.TestCase):
    def db(self):
        # A controlled db with a known realized_return per RSI level (first feature dimension),
        # everything else held at 0 so distance is driven purely by RSI.
        return [
            dict(vector=[30.]+[0.]*(len(FEATURE_NAMES)-1), realized_return=-.02, symbol='A', as_of='x'),
            dict(vector=[50.]+[0.]*(len(FEATURE_NAMES)-1), realized_return=.01, symbol='A', as_of='x'),
            dict(vector=[70.]+[0.]*(len(FEATURE_NAMES)-1), realized_return=.03, symbol='A', as_of='x'),
            dict(vector=[90.]+[0.]*(len(FEATURE_NAMES)-1), realized_return=-.10, symbol='A', as_of='x'),
        ]

    def test_returns_closest_first(self):
        db = self.db()
        current = [65.]+[0.]*(len(FEATURE_NAMES)-1)  # nearest raw RSI distance: 70 (5), then 50 (15)
        neighbors = nearest_analogs(current, db, k=2)
        self.assertEqual([n['realized_return'] for n in neighbors], [.03, .01])

    def test_empty_db_or_none_vector(self):
        self.assertEqual(nearest_analogs(None, self.db()), [])
        self.assertEqual(nearest_analogs([1.], []), [])

    def test_mismatched_dimension_returns_empty(self):
        self.assertEqual(nearest_analogs([1., 2.], self.db()), [])


class AnalogSignalTests(unittest.TestCase):
    def test_aggregates_win_rate_and_mean(self):
        db = [dict(vector=[0.], realized_return=r, symbol='A', as_of='x') for r in (.05, -.02, .03, .01)]
        sig = analog_signal([0.], db, k=4)
        self.assertEqual(sig['n'], 4)
        self.assertAlmostEqual(sig['win_rate'], .75)
        self.assertAlmostEqual(sig['mean_return'], (.05-.02+.03+.01)/4)

    def test_empty_neighbors_gives_none_fields(self):
        sig = analog_signal(None, [])
        self.assertEqual(sig, dict(n=0, win_rate=None, mean_return=None, median_return=None))


class ConfirmsBuyTests(unittest.TestCase):
    def test_refuses_on_thin_sample(self):
        ok, reason = confirms_buy(dict(n=5, win_rate=.9, mean_return=.05), min_n=15)
        self.assertFalse(ok)
        self.assertIn('부족', reason)

    def test_refuses_on_low_win_rate(self):
        ok, reason = confirms_buy(dict(n=20, win_rate=.4, mean_return=.05), min_win_rate=.55)
        self.assertFalse(ok)
        self.assertIn('승률', reason)

    def test_refuses_on_low_mean_return(self):
        ok, reason = confirms_buy(dict(n=20, win_rate=.9, mean_return=-.01), min_mean_return=0.)
        self.assertFalse(ok)
        self.assertIn('평균', reason)

    def test_confirms_when_all_thresholds_met(self):
        ok, reason = confirms_buy(dict(n=20, win_rate=.7, mean_return=.02))
        self.assertTrue(ok)
        self.assertIn('20개', reason)


class TooUncertainTests(unittest.TestCase):
    def test_unset_threshold_never_blocks(self):
        blocked, reason = too_uncertain(dict(forecast_uncertainty=.5), None)
        self.assertFalse(blocked)
        self.assertIsNone(reason)

    def test_missing_uncertainty_never_blocks(self):
        blocked, reason = too_uncertain({}, .05)
        self.assertFalse(blocked)
        self.assertIsNone(reason)

    def test_non_finite_uncertainty_never_blocks(self):
        blocked, reason = too_uncertain(dict(forecast_uncertainty=float('nan')), .05)
        self.assertFalse(blocked)

    def test_blocks_when_over_threshold(self):
        blocked, reason = too_uncertain(dict(forecast_uncertainty=.08), .05)
        self.assertTrue(blocked)
        self.assertIn('불확실성', reason)

    def test_allows_when_under_threshold(self):
        blocked, reason = too_uncertain(dict(forecast_uncertainty=.02), .05)
        self.assertFalse(blocked)
        self.assertIsNone(reason)


if __name__ == '__main__':
    unittest.main()

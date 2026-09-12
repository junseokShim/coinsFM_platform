import unittest
import numpy as np

from prop_trader.downstream import RidgeHead, head_features, uncertainty_targets, volatility_targets
from prop_trader.technical import FEATURES


def tech(**overrides):
    values = dict(rsi14=.5, macd_hist=.0, atr14=.01, bb_position=.5, ema_gap=.01, adx14=.2,
                  cmf20=.0, volume_ratio=1., return1=.0, return5=.0, return20=.0, range_fraction=.02)
    values.update(overrides)
    return {k: values[k] for k in FEATURES}


class HeadFeaturesTests(unittest.TestCase):
    def test_shape_is_raw_path_plus_technical_plus_trend_agree(self):
        x = head_features([0., .001, .002, .003, .004], tech())
        self.assertEqual(x.shape, (5 + len(FEATURES) + 1,))

    def test_trend_agree_positive_when_ema_and_forecast_direction_match(self):
        x = head_features([0., .001, .002, .003, .01], tech(ema_gap=.02))
        self.assertGreater(x[-1], 0)

    def test_trend_agree_negative_when_forecast_reverses_the_trend(self):
        x = head_features([0., -.001, -.002, -.003, -.01], tech(ema_gap=.02))
        self.assertLess(x[-1], 0)

    def test_rejects_wrong_path_length(self):
        with self.assertRaises(ValueError):
            head_features([0., .001, .002], tech())


class RidgeHeadTests(unittest.TestCase):
    def test_fit_predict_recovers_a_linear_signal(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(200, 6))
        y = 2. * x[:, 0] - x[:, 1] + 3.
        head = RidgeHead(alpha=.01).fit(x, y)
        pred = head.predict(x[:5])
        self.assertTrue(np.allclose(pred, y[:5], atol=.5))

    def test_to_dict_from_dict_roundtrip_matches_predictions(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(50, 4))
        y = rng.normal(size=50)
        head = RidgeHead().fit(x, y)
        restored = RidgeHead.from_dict(head.to_dict())
        self.assertTrue(np.allclose(head.predict(x), restored.predict(x)))

    def test_rejects_too_few_rows(self):
        with self.assertRaises(ValueError):
            RidgeHead().fit(np.zeros((5, 3)), np.zeros(5))

    def test_rejects_mismatched_feature_width_at_predict(self):
        head = RidgeHead().fit(np.random.default_rng(2).normal(size=(40, 3)), np.zeros(40))
        with self.assertRaises(ValueError):
            head.predict(np.zeros((1, 5)))


class TargetsTests(unittest.TestCase):
    def test_uncertainty_targets_is_mean_abs_error_per_window(self):
        pred = np.array([[0., .01, .02, .03, .04]])
        actual = np.array([[0., .01, .02, .03, .06]])
        self.assertAlmostEqual(uncertainty_targets(pred, actual)[0], .02 / 5)

    def test_volatility_targets_is_worst_low_vs_anchor(self):
        class Bar:
            def __init__(self, low):
                self.low = low
        bars = [[Bar(98.), Bar(95.), Bar(97.), Bar(99.), Bar(100.)]]
        out = volatility_targets([100.], bars)
        self.assertAlmostEqual(out[0], .05)

    def test_volatility_targets_floors_at_zero_when_price_only_rises(self):
        class Bar:
            def __init__(self, low):
                self.low = low
        bars = [[Bar(101.), Bar(102.), Bar(103.)]]
        out = volatility_targets([100.], bars)
        self.assertEqual(out[0], 0.)


if __name__ == '__main__':
    unittest.main()

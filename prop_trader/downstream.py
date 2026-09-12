"""Two learned downstream heads on TimesFM's raw forecast + technical features, trained the
same way as technical.HybridCalibrator (ridge, validation-split-only, hash-locked to one
adapter/data snapshot) -- but predicting different targets. HybridCalibrator corrects the price
path itself (a residual added to the base forecast); these two heads instead size conviction and
risk, so they're a direct regression y = f(x) rather than a residual-on-base correction.

  UncertaintyHead  - predicts this trade's expected forecast error (mean |actual-predicted| log
                     return across the 5-step path). Used to shrink ranking conviction: a
                     positive expected edge from a forecast the model tends to get wrong a lot
                     is worth less than the same edge from a reliable one.
  VolatilityHead   - predicts this trade's expected realized max adverse excursion (how far price
                     moves against a long entry before the horizon closes, as a fraction of
                     entry price). Used to size the stop-loss/take-profit distance per trade
                     instead of a fixed 2.5% for every trade regardless of regime.

Both take the same 18-dim input: TimesFM's own raw 5-step log-return path, the 12
technical.FEATURES, and trend_agree -- +1 when the recent trend (ema_gap sign) and the model's
own predicted direction agree (a trend-following bet), -1 when they oppose (a rebound/reversal
bet), 0 if either is exactly flat. This promotes the ad hoc check multiframe.decide() already did
(`raw[-1] < bid`) into a real feature instead of a hardcoded strength penalty.
"""
import numpy as np

from .technical import FEATURES

HEAD_DIM = len(FEATURES) + 5 + 1  # raw 5-step path + 12 technical features + trend_agree


def head_features(raw_log_returns, technical_features):
    """raw_log_returns: (5,) TimesFM's own predicted cumulative log-return path (pre-calibration,
    i.e. `after` in timesfm_predict.predict_symbol / `val_raw`/`test_raw` in hybrid_research).
    technical_features: dict of the 12 technical.FEATURES for the same origin bar."""
    raw = np.asarray(raw_log_returns, dtype=float)
    if raw.shape != (5,):
        raise ValueError('raw_log_returns must have exactly 5 steps')
    tech = np.array([technical_features[k] for k in FEATURES], dtype=float)
    trend_sign = np.sign(technical_features['ema_gap'])
    pred_sign = np.sign(raw[-1])
    return np.concatenate([raw, tech, [trend_sign * pred_sign]])


class RidgeHead:
    """Direct ridge regression y = f(x): standardized features, closed-form solve, centered on
    the training target's mean (not on a base forecast -- these targets are magnitudes, not a
    price-path correction)."""
    def __init__(self, alpha=10.0):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError('alpha must be positive')
        self.alpha = alpha

    def fit(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if len(x) < 30 or len(x) != len(y) or x.ndim != 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Invalid head training data')
        self.mean = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0), 1e-8)
        z = np.clip((x - self.mean) / self.scale, -5, 5)
        self.y_mean = float(y.mean())
        self.coef = np.linalg.solve(z.T @ z + self.alpha * np.eye(z.shape[1]), z.T @ (y - self.y_mean))
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError('Invalid head features')
        z = np.clip((x - self.mean) / self.scale, -5, 5)
        return z @ self.coef + self.y_mean

    def to_dict(self):
        return dict(alpha=self.alpha, mean=self.mean.tolist(), scale=self.scale.tolist(),
                    coef=self.coef.tolist(), y_mean=self.y_mean)

    @classmethod
    def from_dict(cls, data):
        obj = cls(data['alpha'])
        obj.mean = np.asarray(data['mean'], dtype=float)
        obj.scale = np.asarray(data['scale'], dtype=float)
        obj.coef = np.asarray(data['coef'], dtype=float)
        obj.y_mean = float(data['y_mean'])
        return obj


def uncertainty_targets(pred, actual):
    """One scalar per window: mean absolute error across the 5-step path."""
    return np.abs(np.asarray(actual) - np.asarray(pred)).mean(axis=1)


def volatility_targets(anchors, window_bars):
    """One scalar per window: realized max adverse excursion for a long entry at `anchor` --
    (anchor - worst intrabar low across the 5 holding bars) / anchor, floored at 0. `window_bars`
    is a list (one per training window) of that window's 5 actual holding-period Bar objects,
    already the exact ones windows()/hybrid_research.evaluate() require to exist."""
    out = []
    for anchor, bars in zip(anchors, window_bars):
        worst = min(b.low for b in bars)
        out.append(max(0.0, (anchor - worst) / anchor))
    return np.array(out)

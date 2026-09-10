"""Causal OHLCV features from the project's reference ta library."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Use the checked-in reference implementation (ta 0.11.0).
_reference = Path(__file__).resolve().parents[1] / 'ta'
if str(_reference) not in sys.path:
    sys.path.insert(0, str(_reference))
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator, EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import ChaikinMoneyFlowIndicator

FEATURES = ['rsi14', 'macd_hist', 'atr14', 'bb_position', 'ema_gap', 'adx14',
            'cmf20', 'volume_ratio', 'return1', 'return5', 'return20', 'range_fraction']


def feature_frame(bars):
    if len(bars) < 60:
        raise ValueError('At least 60 completed candles required for technical features')
    df = pd.DataFrame([dict(time=b.timestamp, close=b.close, high=b.high, low=b.low,
                            volume=b.volume) for b in bars]).set_index('time')
    if not df.index.is_unique or not df.index.is_monotonic_increasing:
        raise ValueError('Technical data must be unique and chronological')
    c, h, l, v = (df[k] for k in ('close', 'high', 'low', 'volume'))
    out = pd.DataFrame(index=df.index)
    out['rsi14'] = RSIIndicator(c, window=14, fillna=False).rsi() / 100
    out['macd_hist'] = MACD(c, fillna=False).macd_diff() / c
    out['atr14'] = AverageTrueRange(h, l, c, window=14, fillna=False).average_true_range() / c
    bands = BollingerBands(c, window=20, fillna=False)
    out['bb_position'] = bands.bollinger_pband()
    # Flat observed windows have zero band width: define neutral position, no future fill.
    flat = c.rolling(20, min_periods=20).std(ddof=0).eq(0)
    out.loc[flat, 'bb_position'] = .5
    out['ema_gap'] = (EMAIndicator(c, 20).ema_indicator() - EMAIndicator(c, 50).ema_indicator()) / c
    out['adx14'] = ADXIndicator(h, l, c, window=14, fillna=False).adx() / 100
    out['cmf20'] = ChaikinMoneyFlowIndicator(h, l, c, v, window=20, fillna=False).chaikin_money_flow()
    out['volume_ratio'] = v / v.rolling(20, min_periods=20).mean()
    for n in (1, 5, 20):
        out[f'return{n}'] = c.pct_change(n, fill_method=None)
    out['range_fraction'] = (h - l) / c
    # Warm-up/zero-liquidity rows remain missing, never backfilled from the future.
    return out[FEATURES].replace([np.inf, -np.inf], np.nan)


def latest_features(bars):
    row = feature_frame(bars).iloc[-1]
    if not np.isfinite(row.to_numpy()).all():
        raise ValueError('Undefined indicators: insufficient history or zero liquidity')
    return {k: float(row[k]) for k in FEATURES}


class HybridCalibrator:
    """Ridge residual correction using TimesFM path + technical features.

    Fit only on calibration labels, after the underlying model's training cutoff.
    This is an explicit fusion layer; it does not masquerade as TimesFM covariates.
    """
    def __init__(self, alpha=10.0):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError('alpha must be positive')
        self.alpha = alpha

    def fit(self, base, features, targets):
        base, features, targets = map(np.asarray, (base, features, targets))
        x = np.concatenate([base, features], axis=1)
        if len(x) < 30 or targets.shape != base.shape or not np.isfinite(x).all() or not np.isfinite(targets).all():
            raise ValueError('Invalid calibration data')
        self.mean = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0), 1e-8)
        z = np.clip((x - self.mean) / self.scale, -5, 5)
        residual = targets - base
        self.intercept = residual.mean(axis=0)
        self.coef = np.linalg.solve(z.T @ z + self.alpha * np.eye(z.shape[1]),
                                    z.T @ (residual - self.intercept))
        return self

    def predict(self, base, features):
        base = np.asarray(base)
        x = np.concatenate([base, np.asarray(features)], axis=1)
        if x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError('Invalid hybrid features')
        return base + np.clip((x - self.mean) / self.scale, -5, 5) @ self.coef + self.intercept

    def to_dict(self):
        return dict(alpha=self.alpha, feature_names=FEATURES,
                    **{k: getattr(self, k).tolist() for k in ('mean', 'scale', 'intercept', 'coef')})

    @classmethod
    def from_dict(cls, data):
        if data['feature_names'] != FEATURES:
            raise ValueError('Feature schema mismatch')
        obj = cls(data['alpha'])
        for k in ('mean', 'scale', 'intercept', 'coef'):
            setattr(obj, k, np.asarray(data[k], dtype=float))
        return obj

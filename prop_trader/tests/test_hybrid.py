import unittest
from datetime import datetime,timedelta,timezone
import numpy as np
from prop_trader.engine import Bar
from prop_trader.technical import feature_frame,latest_features,HybridCalibrator


def bars(n=150):
    start=datetime(2026,1,1,tzinfo=timezone.utc)
    return [Bar((start+i*timedelta(hours=1)).isoformat(),'A',100+i*.1,
        102+i*.1,98+i*.1,100+i*.1+np.sin(i),1000+i) for i in range(n)]


class HybridTests(unittest.TestCase):
    def test_future_data_cannot_change_past_features(self):
        history=bars();prefix=feature_frame(history[:100]);full=feature_frame(history)
        np.testing.assert_allclose(prefix.iloc[-1].values,full.loc[history[99].timestamp].values)

    def test_flat_band_is_neutral_without_backward_fill(self):
        history=[Bar(str(i).zfill(3),'A',100,101,99,100,1000) for i in range(70)]
        self.assertEqual(latest_features(history)['bb_position'],.5)
        self.assertTrue(feature_frame(history).iloc[0].isna().any())

    def test_serialized_calibration_reproduces_forecasts(self):
        rng=np.random.default_rng(42)
        raw=rng.normal(size=(50,5));features=rng.normal(size=(50,12));y=raw+features[:,:1]*.01
        model=HybridCalibrator().fit(raw,features,y)
        loaded=HybridCalibrator.from_dict(model.to_dict())
        np.testing.assert_allclose(model.predict(raw,features),loaded.predict(raw,features))

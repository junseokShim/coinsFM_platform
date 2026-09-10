from datetime import datetime
import unittest
from prop_trader.forecast_time import forecast_points,SECONDS


class ForecastTimeTests(unittest.TestCase):
    def test_five_future_close_boundaries_for_each_interval(self):
        last='2026-09-09T00:00:00+00:00'
        for interval,seconds in SECONDS.items():
            rows=forecast_points(last,interval,[1,2,3,4,5])
            self.assertEqual(len(rows),5)
            origin=datetime.fromisoformat(rows[0]['candle_open'])
            end=datetime.fromisoformat(rows[-1]['candle_close'])
            self.assertEqual((end-origin).total_seconds(),5*seconds)
            self.assertEqual(rows[-1]['predicted_close'],5)

    def test_timezone_required(self):
        with self.assertRaises(ValueError):forecast_points('2026-09-09','1h',[1]*5)

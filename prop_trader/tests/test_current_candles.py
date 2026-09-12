import unittest
from unittest.mock import patch

from prop_trader.current_candles import fetch_series

HOUR = 3_600_000  # ms


def kline(open_ms, close_offset=HOUR - 1):
    # Only fields [0] open_time and [6] close_time matter to fetch_series/the caller.
    return [open_ms, '0', '0', '0', '0', '0', open_ms + close_offset]


class FetchSeriesTests(unittest.TestCase):
    def test_single_page_matches_original_behavior(self):
        cutoff = 100 * HOUR
        bars = [kline(n * HOUR) for n in range(90, 100)]  # 10 candles up to just under cutoff

        def fake_request(endpoint, params):
            self.assertEqual(params['limit'], 11)
            return bars, 'url'

        with patch('prop_trader.current_candles.request', side_effect=fake_request):
            raw, meta = fetch_series('DOGEUSDT', '1h', count=10, pages=1, cutoff=cutoff, expected=HOUR)
        self.assertEqual([r[0] for r in raw], [n * HOUR for n in range(90, 100)])
        self.assertEqual(meta['symbol'], 'DOGEUSDT')

    def test_multi_page_stitches_without_gap_or_duplicate(self):
        # 25 continuous hourly candles total, forced into 9+9+7 pages via a small page_cap, to
        # exercise the page-boundary stitching without a 999-row fixture.
        cutoff = 100 * HOUR
        all_bars = [kline(n * HOUR) for n in range(70, 100)]  # opens 70..99

        def fake_request(endpoint, params):
            end = params['endTime']
            limit = params['limit']
            page = [b for b in all_bars if b[0] <= end][-limit:]
            return page, 'url'

        with patch('prop_trader.current_candles.request', side_effect=fake_request):
            raw, meta = fetch_series('DOGEUSDT', '1h', count=25, pages=4, cutoff=cutoff, expected=HOUR, page_cap=9)
        opens = [r[0] for r in raw]
        self.assertEqual(opens, sorted(opens))
        self.assertEqual(len(opens), len(set(opens)), 'no duplicate candles across page boundaries')
        self.assertTrue(all(b - a == HOUR for a, b in zip(opens, opens[1:])), 'no gap across page boundaries')
        self.assertEqual(opens, [n * HOUR for n in range(75, 100)])

    def test_incomplete_current_candle_dropped_only_from_first_page(self):
        cutoff = 100 * HOUR + 30 * 60 * 1000  # mid-candle 100
        bars = [kline(n * HOUR) for n in range(90, 100)] + [kline(100 * HOUR)]  # last one still open

        def fake_request(endpoint, params):
            return bars, 'url'

        with patch('prop_trader.current_candles.request', side_effect=fake_request):
            raw, meta = fetch_series('DOGEUSDT', '1h', count=10, pages=1, cutoff=cutoff, expected=HOUR)
        self.assertNotIn(100 * HOUR, [r[0] for r in raw])

    def test_insufficient_history_raises(self):
        cutoff = 100 * HOUR
        bars = [kline(n * HOUR) for n in range(95, 100)]  # only 5 available

        def fake_request(endpoint, params):
            return bars, 'url'

        with patch('prop_trader.current_candles.request', side_effect=fake_request):
            with self.assertRaises(ValueError):
                fetch_series('DOGEUSDT', '1h', count=10, pages=1, cutoff=cutoff, expected=HOUR)

    def test_noncontinuous_candles_rejected(self):
        cutoff = 100 * HOUR
        bars = [kline(n * HOUR) for n in range(90, 95)] + [kline(n * HOUR) for n in range(96, 101)]

        def fake_request(endpoint, params):
            return bars[:10], 'url'

        with patch('prop_trader.current_candles.request', side_effect=fake_request):
            with self.assertRaises(ValueError):
                fetch_series('DOGEUSDT', '1h', count=10, pages=1, cutoff=cutoff, expected=HOUR)


if __name__ == '__main__':
    unittest.main()

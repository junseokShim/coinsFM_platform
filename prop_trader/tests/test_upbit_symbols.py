import unittest

from prop_trader.upbit_symbols import annotate, to_upbit_market


class SymbolMappingTests(unittest.TestCase):
    def test_maps_usdt_pair_to_krw_market(self):
        self.assertEqual(to_upbit_market('DOGEUSDT'), 'KRW-DOGE')

    def test_non_usdt_pair_has_no_mapping(self):
        self.assertIsNone(to_upbit_market('BTCETH'))

    def test_too_short_symbol_has_no_mapping(self):
        self.assertIsNone(to_upbit_market('USDT'))

    def test_annotate_marks_tradable_and_untradable(self):
        entries = [dict(symbol='DOGEUSDT'), dict(symbol='BONKUSDT')]
        annotate(entries, {'KRW-DOGE', 'KRW-BTC'})
        self.assertTrue(entries[0]['upbit_tradable'])
        self.assertEqual(entries[0]['upbit_market'], 'KRW-DOGE')
        self.assertFalse(entries[1]['upbit_tradable'])


if __name__ == '__main__':
    unittest.main()

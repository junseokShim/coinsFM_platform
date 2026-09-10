"""Map Binance-style USDT symbols to Upbit KRW markets; never drops data, only annotates."""
import time

from .upbit_broker import UpbitClient

_CACHE = {'markets': None, 'at': 0.}
_TTL = 300.


def krw_markets(client=None, force=False):
    """Cached set of active Upbit KRW-quoted market codes, e.g. {'KRW-BTC', ...}."""
    now = time.time()
    if force or _CACHE['markets'] is None or now - _CACHE['at'] > _TTL:
        client = client or UpbitClient()
        meta = client.request('GET', '/v1/market/all', dict(is_details='true'), private=False)
        _CACHE['markets'] = {m['market'] for m in meta if m['market'].startswith('KRW-')}
        _CACHE['at'] = now
    return _CACHE['markets']


def to_upbit_market(symbol):
    """'DOGEUSDT' -> 'KRW-DOGE'. Only USDT-quoted symbols have a defined mapping."""
    if not symbol.endswith('USDT') or len(symbol) <= 4:
        return None
    return 'KRW-' + symbol[:-4]


def annotate(entries, markets):
    """Attach upbit_market / upbit_tradable to each entry in place; returns entries."""
    for e in entries:
        market = to_upbit_market(e['symbol'])
        e['upbit_market'] = market
        e['upbit_tradable'] = bool(market and market in markets)
    return entries

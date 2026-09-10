"""Pure execution gates shared by live trading and regression tests."""
import math
from datetime import datetime
from decimal import Decimal, ROUND_DOWN

from .forecast_time import SECONDS
from .rank_coins import buy_vwap


def timestamp(value):
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError('Timezone required')
    return dt.timestamp()


def fresh_forecast(row, interval, now):
    try:
        origin = timestamp(row['as_of'])
        prices = row['forecast_prices']
        return (row['interval'] == interval and row['horizon'] == 5
                and origin <= now < origin + SECONDS[interval]
                and len(prices) == 5
                and all(math.isfinite(float(p)) and float(p) > 0 for p in prices))
    except (KeyError, ValueError, TypeError, OverflowError):
        return False


def reprice(row, book, interval, now, budget, fee=.001, slippage=.001):
    if not fresh_forecast(row, interval, now):
        raise ValueError('예측 만료 또는 형식 오류')
    age = now - float(book['timestamp']) / 1000
    if not math.isfinite(age) or not 0 <= age <= 10:
        raise ValueError('호가 만료')
    entry = buy_vwap(book, budget / (1 + fee))
    net = float(row['forecast_prices'][-1]) * (1-slippage) * (1-fee) / (entry*(1+fee)) - 1
    return dict(row, stale=False, expected_net_return=net, positive_after_costs=net > 0,
                estimated_entry_vwap=entry, valid_until=datetime.fromtimestamp(
                    min(now + 10, timestamp(row['as_of']) + SECONDS[interval]),
                    datetime.fromisoformat(row['as_of']).tzinfo).isoformat())


def sell_quantity(owned, available):
    return Decimal(str(min(owned, available))).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)


def minimum_entry(minimum=5000., stop=.025, fee=.001, slippage=.001):
    # Includes room for the protective stop, execution costs, and a 1% sizing buffer.
    # Gaps can still create dust; this is not a guarantee of exit liquidity.
    return minimum * (1+fee) / ((1-stop)*(1-slippage)) * 1.01


def fills(order):
    qty = float(order.get('executed_volume') or 0)
    fee = float(order.get('paid_fee') or 0)
    trades = order.get('trades') or []
    volume = sum(float(t['volume']) for t in trades)
    funds = sum(float(t['funds']) for t in trades)
    if not all(math.isfinite(v) and v >= 0 for v in (qty, fee, volume, funds)):
        raise ValueError('Invalid fill')
    if qty and (abs(qty-volume) > max(1e-8, qty*1e-8) or funds <= 0):
        raise ValueError('체결 상세 대기')
    return qty, funds, fee

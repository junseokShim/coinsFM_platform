"""Three distinct forecast horizons: daily regime, hourly edge, minute entry timing."""
from .execution_rules import reprice
import math

FRAMES = ('1m', '1h', '1d')


def decide(rows, book, now, budget, threshold=.002, held=False):
    try:
        priced = {iv: reprice(rows[iv], book, iv, now, budget) for iv in FRAMES}
    except (KeyError, TypeError, ValueError, OverflowError):
        return dict(action='STAY', reason='1분·1시간·1일 예측 또는 실시간 호가 갱신 대기', complete=False)
    net = {iv: r['expected_net_return'] for iv,r in priced.items()}
    try:
        bid = max(float(u['bid_price']) for u in book['orderbook_units'])
        if not math.isfinite(bid) or bid <= 0:
            raise ValueError('Invalid bid')
    except (ValueError, TypeError, KeyError):
        return dict(action='STAY', reason='매도 호가 검증 실패', complete=False)
    # For holding, entry costs are sunk: compare future price with liquidation now.
    remaining = {iv: float(rows[iv]['forecast_prices'][-1])/bid-1 for iv in FRAMES}
    action = 'STAY'
    reason = '주기별 방향 불일치 또는 남은 순수익 부족'
    if held:
        if remaining['1d'] < -threshold or remaining['1h'] < -threshold or (
                remaining['1m'] < -threshold and remaining['1h'] <= 0):
            action, reason = 'SELL', '다중 주기 예측의 보유 조건 이탈'
        else:
            reason = '다중 주기 보유 조건 유지'
    elif net['1d'] > 0 and net['1h'] > threshold and net['1m'] > 0:
        action, reason = 'BUY', '일봉 방향·시간봉 순수익·분봉 진입 조건 충족'
    atr = float((rows['1h'].get('technical_features') or {}).get('atr14', .01))
    if not math.isfinite(atr) or atr < 0:
        return dict(action='STAY', reason='변동성 지표 검증 실패', complete=False)
    volatility = max(.001, atr * math.sqrt(5))
    strength = max(0., min(1., net['1h']/volatility))
    # A transparent signal score, NOT a calibrated probability of profit.
    raw = rows['1h'].get('timesfm_prices')
    if raw and float(raw[-1]) < bid:
        strength *= .5
    sell_fraction = 1. if remaining['1d'] < -threshold or remaining['1h'] < -2*threshold else .5
    exit_strength = max(0., min(1., -remaining['1h']/volatility))
    return dict(action=action, reason=reason, complete=True, returns=net,
                remaining_returns=remaining, priced=priced, score=net['1h'],
                signal_strength=strength, exit_strength=exit_strength, sell_fraction=sell_fraction,
                forecast_origins={iv: rows[iv]['as_of'] for iv in FRAMES})


def hourly_only(rows, book, now, budget, threshold=.002):
    """Partial-conviction reading using ONLY the hourly frame -- unlike decide(), does not
    require 1m to produce anything. Many smaller Upbit markets simply don't trade every single
    minute, so a gap-free 1m candle series is often structurally unavailable for them; that
    should not make a solid hourly edge invisible to the whole decision pipeline. Only ever
    feeds the smaller partial_budget sizing downstream, never the full-conviction one -- a
    missing 1m/1d view is exactly the lower-confidence case that smaller size is for."""
    try:
        priced_1h = reprice(rows['1h'], book, '1h', now, budget)
        bid = max(float(u['bid_price']) for u in book['orderbook_units'])
        if not math.isfinite(bid) or bid <= 0:
            raise ValueError('Invalid bid')
        atr = float((rows['1h'].get('technical_features') or {}).get('atr14', .01))
        if not math.isfinite(atr) or atr < 0:
            raise ValueError('Invalid volatility')
    except (KeyError, TypeError, ValueError, OverflowError):
        return dict(action='STAY', reason='시간봉 예측 또는 실시간 호가 갱신 대기', complete=False)
    net_1h = priced_1h['expected_net_return']
    volatility = max(.001, atr * math.sqrt(5))
    strength = max(0., min(1., net_1h / volatility))
    return dict(action='STAY', reason='시간봉 단독 판단(1분봉 데이터 없음)', complete=True,
                returns={'1h': net_1h}, priced={'1h': priced_1h}, score=net_1h,
                signal_strength=strength, forecast_origins={'1h': rows['1h']['as_of']})


def buy_budget(decision, cash, cap):
    from .execution_rules import minimum_entry
    strength = decision.get('signal_strength', 0.)
    if not decision.get('complete') or decision.get('action') != 'BUY' or not math.isfinite(strength):
        return 0.
    amount = min(cash*.9, cap*(.35+.65*max(0., min(1., strength))))
    return amount if amount >= minimum_entry() else 0.


def partial_qualifies(decision, threshold):
    """True when the hourly edge alone is solidly positive even though the strict multiframe
    gate declined the full BUY (1d/1m disagreement). `decide()` already reports this as
    `returns['1h']` -- the same number the Upbit ranking table shows as a positive expected
    return -- so an outright STAY on it discards a real, if lower-confidence, signal."""
    if not decision.get('complete') or decision.get('action') == 'BUY':
        return False
    hourly = decision.get('returns', {}).get('1h')
    return hourly is not None and math.isfinite(hourly) and hourly > threshold


def partial_budget(decision, cash, cap, threshold=.002):
    """Lower-conviction sizing: takes half of whatever headroom exists between the exchange's
    practical minimum notional and what a confirmed three-frame signal at the same strength
    would get (buy_budget's own amount), so it always sizes strictly below a full-conviction
    trade yet still grades with strength -- never a flat cap*(.35+.65*strength)*.5 ceiling, which
    tops out at cap*0.5 regardless of strength and silently pins EVERY partial trade to the exact
    minimum whenever cap*0.5 <= floor (true for any cap below ~2x the floor, e.g. an 8000 cap
    against a ~5190 floor: cap*0.5=4000 < floor always, so the old formula's max(floor,
    half_ceiling) was floor unconditionally -- reported live 2026-09-11, every partial fill sized
    identically regardless of signal strength).
    """
    from .execution_rules import minimum_entry
    if not partial_qualifies(decision, threshold):
        return 0.
    strength = decision.get('signal_strength', 0.)
    if not math.isfinite(strength):
        return 0.
    floor = minimum_entry()
    full = cap*(.35+.65*max(0., min(1., strength)))  # what buy_budget would size at this strength
    headroom = max(0., full-floor)
    amount = min(cash*.9, cap, floor+headroom*.5)
    return amount if amount >= floor else 0.


def exit_fraction(requested, quantity, bid, minimum=5000.):
    # Do not strand a tradable position as unsellable dust by splitting it.
    fraction = max(0., min(1., requested))
    value = quantity*bid*.999
    return fraction if value*fraction >= minimum and value*(1-fraction) >= minimum else 1.


def candidate_score(ticker, momentum_weight=5.):
    """Rank a market for forecast/orderbook attention: liquidity, boosted by today's upward
    move. A pure 24h-notional cut systematically misses a coin early in a pump (its volume
    hasn't caught up yet) -- since the WebSocket ticker already carries signed_change_rate for
    every KRW market at no extra request cost, folding it in surfaces movers, not just majors.
    Downward moves get no boost (this is a discovery score, not a short signal)."""
    try:
        volume = float(ticker.get('acc_trade_price_24h', 0) or 0)
        change = float(ticker.get('signed_change_rate', 0) or 0)
    except (TypeError, ValueError):
        return 0.
    if not math.isfinite(volume) or not math.isfinite(change) or volume < 0:
        return 0.
    return volume * (1. + max(0., change) * momentum_weight)

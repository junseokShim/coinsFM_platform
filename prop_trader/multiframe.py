"""Three distinct forecast horizons: daily regime, hourly edge, minute entry timing."""
from .execution_rules import reprice
import math

FRAMES = ('1m', '1h', '1d')
# How hard downstream.UncertaintyHead's predicted error penalizes ranking score. Unvalidated by
# backtest (exit_sweep.py's search was on stop/target parameters, not this) -- starts at 1.0 (one
# unit of expected error fully offsets one unit of expected return) and should be revisited once
# there's enough live/paper history with the head actually trained to check it against outcomes.
UNCERTAINTY_PENALTY = 1.0


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
    # Conviction-adjusted ranking: a positive edge from a forecast downstream.UncertaintyHead
    # expects to be wrong a lot is worth less than the same edge from a reliable one. None (head
    # not trained yet, or this snapshot's calibrator/head pair was rejected as stale) leaves
    # score unchanged -- same fail-open convention as too_uncertain/analog gates.
    predicted_uncertainty = rows['1h'].get('predicted_uncertainty')
    score = net['1h']
    conviction = None
    if predicted_uncertainty is not None and math.isfinite(predicted_uncertainty):
        score -= UNCERTAINTY_PENALTY * predicted_uncertainty
        if predicted_uncertainty > 0:
            # Expected edge per unit of the head's own expected error -- used for position
            # sizing (buy_budget/partial_budget), not just ranking: a trade the head expects to
            # be reliable gets sized up, a shaky one gets sized down, instead of every trade
            # sizing off the same ATR-ratio regardless of how much the model trusts itself.
            conviction = max(0., net['1h']) / predicted_uncertainty
    return dict(action=action, reason=reason, complete=True, returns=net,
                remaining_returns=remaining, priced=priced, score=score, conviction=conviction,
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
    predicted_uncertainty = rows['1h'].get('predicted_uncertainty')
    conviction = None
    if predicted_uncertainty is not None and math.isfinite(predicted_uncertainty) and predicted_uncertainty > 0:
        conviction = max(0., net_1h) / predicted_uncertainty
    return dict(action='STAY', reason='시간봉 단독 판단(1분봉 데이터 없음)', complete=True,
                returns={'1h': net_1h}, priced={'1h': priced_1h}, score=net_1h, conviction=conviction,
                signal_strength=strength, forecast_origins={'1h': rows['1h']['as_of']})


CONVICTION_REFERENCE = 3.0  # conviction (expected return / downstream.UncertaintyHead's own
# expected error) that maps to full position size. Unvalidated -- no backtest has calibrated
# this yet; revisit once there's enough live/paper history with the head actually trained to
# check realized outcomes against it.


def _sizing_scale(decision):
    """0..1 position-sizing scale. Prefers conviction (expected edge per unit of the head's own
    expected error, see decide()/hourly_only()) over the older ATR-ratio signal_strength when
    available, so a trade the model expects to be reliable is sized up and a shaky one sized
    down -- not every trade sizing off the same volatility ratio regardless of confidence.
    Returns None (caller must reject the trade) on a non-finite value, same as the original
    signal_strength guard this replaces -- garbage in should block sizing, not silently zero it."""
    conviction = decision.get('conviction')
    if conviction is not None:
        return max(0., min(1., conviction / CONVICTION_REFERENCE)) if math.isfinite(conviction) else None
    strength = decision.get('signal_strength', 0.)
    return max(0., min(1., strength)) if math.isfinite(strength) else None


def buy_budget(decision, cash, cap):
    from .execution_rules import minimum_entry
    if not decision.get('complete') or decision.get('action') != 'BUY':
        return 0.
    scale = _sizing_scale(decision)
    if scale is None:
        return 0.
    amount = min(cash*.9, cap*(.35+.65*scale))
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
    scale = _sizing_scale(decision)
    if scale is None:
        return 0.
    floor = minimum_entry()
    full = cap*(.35+.65*scale)  # what buy_budget would size at this same scale
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


def _log_returns(closes):
    closes = [float(c) for c in closes]
    return [math.log(b / a) for a, b in zip(closes, closes[1:]) if a > 0 and b > 0]


def _correlation(a, b):
    n = min(len(a), len(b))
    if n < 2:
        return None
    a, b = a[-n:], b[-n:]
    mean_a, mean_b = sum(a) / n, sum(b) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    if var_a <= 0 or var_b <= 0:
        return None
    return cov / math.sqrt(var_a * var_b)


def too_correlated(candidate_closes, held_closes_list, max_corr=.85, min_n=10):
    """True when the candidate's recent return series moves too much like an already-held
    position's -- concentration risk a pure liquidity/momentum ranking never sees (several
    correlated memecoins filling every open slot at once looks fine position-by-position).
    Downgrade-only gate, same convention as too_uncertain/confirms_buy: only ever turns a BUY
    into STAY, never blocks (returns False) on too little shared history to judge."""
    candidate = _log_returns(candidate_closes)
    if len(candidate) < min_n or not held_closes_list:
        return False, None
    worst = None
    for held_closes in held_closes_list:
        corr = _correlation(candidate, _log_returns(held_closes))
        if corr is not None and (worst is None or corr > worst):
            worst = corr
    if worst is not None and worst > max_corr:
        return True, f'보유 종목과 최근 수익률 상관계수 과다({worst:.2f} > 기준 {max_corr:.2f})'
    return False, None

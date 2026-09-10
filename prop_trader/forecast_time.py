"""Explicit candle-open and close boundaries for five-step forecasts."""
from datetime import datetime,timedelta

# Supported timeframes for forecasting. 1m/1h/1d have trained adapters today;
# the rest are wired up so a new timeframe only needs data + a trained adapter,
# no code changes.
SECONDS={'1m':60,'3m':180,'5m':300,'15m':900,'30m':1800,'1h':3600,'4h':14400,'1d':86400}


def forecast_points(last_open,interval,prices):
    if interval not in SECONDS:
        raise ValueError(f'Unsupported timeframe: {interval} (known: {", ".join(SECONDS)})')
    last=datetime.fromisoformat(last_open)
    if last.tzinfo is None:raise ValueError('Timezone required')
    delta=timedelta(seconds=SECONDS[interval])
    return [dict(step=k+1,candle_open=(last+(k+1)*delta).isoformat(),
        candle_close=(last+(k+2)*delta).isoformat(),predicted_close=price) for k,price in enumerate(prices)]


def path_stats(current_close,prices):
    """Return-space and path-shape stats for a predicted close trajectory.

    Used for long/short ranking: two forecasts can share the same t+5 return
    but very different paths (steady climb vs. dip-then-recover), so callers
    should look at more than the endpoint return.
    """
    if current_close<=0:raise ValueError('current_close must be positive')
    if not prices:raise ValueError('prices must be non-empty')
    returns=[p/current_close-1 for p in prices]
    diffs=[b-a for a,b in zip(prices,prices[1:])]
    trend=1 if prices[-1]>=prices[0] else -1
    consistency=sum(1 for d in diffs if (d>=0)==(trend>=0))/len(diffs) if diffs else 1.0
    slope=(prices[-1]-prices[0])/(len(prices)-1) if len(prices)>1 else 0.0
    return dict(return_by_step=returns,max_predicted_return=max(returns),min_predicted_return=min(returns),
        forecast_slope=slope,forecast_consistency=consistency)


def decide_signal(stats,threshold=0.01,min_consistency=0.5):
    """Research-only long/short/hold label. No order is ever placed from this.

    Endpoint return alone can't tell a steady climb from a dip-then-recover
    with the same t+5 return, so also require the path to be directionally
    consistent and not have moved too far against the trade before recovering.
    """
    final_return=stats['return_by_step'][-1]
    if final_return>threshold and stats['forecast_consistency']>=min_consistency and stats['min_predicted_return']>-threshold:
        return 'LONG'
    if final_return<-threshold and stats['forecast_consistency']>=min_consistency and stats['max_predicted_return']<threshold:
        return 'SHORT'
    return 'HOLD'

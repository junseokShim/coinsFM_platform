"""Long-only trend/pullback recovery with prior-bar universe breadth filter."""
from .engine import Desk


class PullbackDesk(Desk):
    def step(self, bars):
        # Snapshot before any symbol history is updated: symbol-order independent.
        ready = [self.history[b.symbol] for b in bars
                 if len(self.history[b.symbol]) >= self.cfg.lookback]
        rising = sum(p[-1].close > sum(b.close for b in p)/len(p) for p in ready)
        self.market_ok = len(ready) == len(bars) and bool(ready) and rising / len(ready) >= .5
        super().step(bars)

    def signal(self, symbol, bar, past):
        if not self.market_ok or symbol in self.positions:
            return None
        fast = sum(b.close for b in past[-24:]) / 24
        slow = sum(b.close for b in past) / len(past)
        trend = past[-1].close / past[0].close - 1
        volume = sum(b.volume for b in past[-24:]) / 24
        # No giant breakout chase: shallow pullback followed by recovery above prior high.
        if (fast > slow and trend > 0 and past[-1].close <= fast
                and bar.close > max(fast, past[-1].high)
                and bar.close <= fast * 1.03
                and min(b.low for b in past[-6:]) >= fast * .95
                and volume > 0 and bar.volume >= volume * .8):
            return dict(side=1, score=trend, reason='trend_pullback_recovery')

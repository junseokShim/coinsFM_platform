"""Bar-close decisions, next-bar execution, and conservative paper accounting."""
from dataclasses import dataclass, asdict
from collections import defaultdict
import math


@dataclass(frozen=True)
class Config:
    capital: float = 10_000
    lookback: int = 20
    volume_multiple: float = 2.0
    risk_fraction: float = .005
    max_allocation: float = .2
    max_positions: int = 3
    daily_loss_limit: float = .03
    stop_fraction: float = .025
    reward_multiple: float = 2
    fee: float = .001
    slippage: float = .001
    allow_short: bool = False

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Configuration must be finite')
        if self.capital <= 0 or self.lookback < 2 or self.max_positions < 1:
            raise ValueError('Invalid capital, lookback, or position limit')
        for key in ('risk_fraction', 'max_allocation', 'daily_loss_limit', 'stop_fraction'):
            if not 0 < getattr(self, key) < 1:
                raise ValueError(f'Invalid {key}')
        if self.reward_multiple <= 0 or self.volume_multiple <= 1:
            raise ValueError('Invalid signal or reward threshold')
        if not 0 <= self.fee < 1 or not 0 <= self.slippage < 1:
            raise ValueError('Invalid costs')


@dataclass(frozen=True)
class Bar:
    timestamp: str
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self):
        values = (self.open, self.high, self.low, self.close, self.volume)
        if not all(math.isfinite(x) for x in values):
            raise ValueError('Non-finite market data')
        if min(values[:4]) <= 0 or self.volume < 0:
            raise ValueError('Invalid price or volume')
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError('Inconsistent OHLC')


class Desk:
    def __init__(self, config=Config()):
        self.cfg = config
        self.cash = config.capital
        self.history = defaultdict(list)
        self.positions = {}
        self.pending = {}
        self.marks = {}
        self.events = []
        self.curve = []
        self.day = None
        self.day_equity = config.capital
        self.halted = False
        self.last_timestamp = None

    @property
    def equity(self):
        return self.cash + sum(p['side'] * p['qty'] * (self.marks[s] - p['entry'])
                               for s, p in self.positions.items())

    def emit(self, kind, timestamp, symbol='', **details):
        self.events.append(dict(kind=kind, timestamp=timestamp, symbol=symbol, **details))

    def close(self, symbol, raw_price, timestamp, reason):
        p = self.positions.pop(symbol)
        price = raw_price * (1 - p['side'] * self.cfg.slippage)
        fee = price * p['qty'] * self.cfg.fee
        pnl = p['side'] * p['qty'] * (price - p['entry']) - fee
        self.cash += pnl
        self.emit('exit', timestamp, symbol, price=price, reason=reason,
                  pnl=pnl - p['entry_fee'], side=p['side'], qty=p['qty'])

    def enter(self, symbol, bar, signal):
        c = self.cfg
        side = signal['side']
        price = bar.open * (1 + side * c.slippage)
        equity = self.equity
        gross = sum(p['qty'] * self.marks[s] for s, p in self.positions.items())
        # Include estimated round-trip costs in the risk budget; gaps can exceed it.
        unit_risk = price * (c.stop_fraction + 2 * (c.fee + c.slippage))
        qty = min(max(0, equity) * c.risk_fraction / unit_risk,
                  max(0, equity) * c.max_allocation / price,
                  max(0, equity - gross) / (price * (1 + c.fee)))
        if qty <= 0:
            return
        fee = qty * price * c.fee
        self.cash -= fee
        self.positions[symbol] = dict(side=side, qty=qty, entry=price, entry_fee=fee,
            stop=price * (1 - side * c.stop_fraction),
            target=price * (1 + side * c.stop_fraction * c.reward_multiple))
        self.emit('entry', bar.timestamp, symbol, price=price, qty=qty, **signal)

    def step(self, bars):
        """Process one complete, UTC timestamp batch in deterministic symbol order."""
        if not bars:
            return
        timestamp = bars[0].timestamp
        if any(b.timestamp != timestamp for b in bars) or len({b.symbol for b in bars}) != len(bars):
            raise ValueError('Expected unique symbols at one timestamp')
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError('Timestamps must be strictly increasing')
        by_symbol = {b.symbol: b for b in bars}
        if not self.positions.keys() <= by_symbol.keys():
            raise ValueError('Missing bar for an open position')
        self.last_timestamp = timestamp
        if timestamp[:10] != self.day:
            self.day, self.day_equity, self.halted = timestamp[:10], self.equity, False
        self.marks.update({s: b.open for s, b in by_symbol.items()})
        self.check_loss(timestamp)
        # Yesterday's close signal is executed at the next available bar open.
        pending, self.pending = self.pending, {}
        for s, signal in sorted(pending.items(), key=lambda item: (-item[1]['score'], item[0])):
            if s not in by_symbol or self.halted:
                continue
            if s in self.positions and self.positions[s]['side'] != signal['side']:
                self.close(s, by_symbol[s].open, timestamp, 'reversal')
            if s not in self.positions and len(self.positions) < self.cfg.max_positions:
                self.enter(s, by_symbol[s], signal)
        for s in sorted(list(self.positions)):
            b, p = by_symbol[s], self.positions[s]
            long = p['side'] == 1
            stopped = b.low <= p['stop'] if long else b.high >= p['stop']
            target = b.high >= p['target'] if long else b.low <= p['target']
            if stopped:
                # Stop wins when both levels occur in an OHLC bar; gaps fill at open.
                price = min(b.open, p['stop']) if long else max(b.open, p['stop'])
                self.close(s, price, timestamp, 'stop')
            elif target:
                self.close(s, p['target'], timestamp, 'target')
        self.marks.update({s: b.close for s, b in by_symbol.items()})
        self.check_loss(timestamp)
        for s, b in sorted(by_symbol.items()):
            past = self.history[s]
            if len(past) >= self.cfg.lookback and not self.halted:
                signal = self.signal(s, b, past)
                if signal:
                    self.pending[s] = signal
                    self.emit('signal', timestamp, s, **signal)
            past.append(b)
            del past[:-self.cfg.lookback]
        self.curve.append(dict(timestamp=timestamp, equity=self.equity,
                               positions=len(self.positions), halted=self.halted))

    def signal(self, symbol, bar, past):
        avg = sum(x.volume for x in past) / len(past)
        ratio = bar.volume / avg if avg > 0 else 0
        side = 1 if bar.close > max(x.high for x in past) else (
            -1 if self.cfg.allow_short and bar.close < min(x.low for x in past) else 0)
        if side and ratio >= self.cfg.volume_multiple:
            return dict(side=side, score=ratio, reason='volume_breakout')

    def check_loss(self, timestamp):
        if self.equity <= self.day_equity * (1 - self.cfg.daily_loss_limit):
            if not self.halted:
                self.emit('halt', timestamp, reason='daily_loss_limit')
            self.halted = True
            self.pending.clear()
            for s in sorted(list(self.positions)):
                self.close(s, self.marks[s], timestamp, 'daily_loss_limit')

    def finish(self):
        if self.last_timestamp is None:
            raise ValueError('No bars provided')
        for s in sorted(list(self.positions)):
            self.close(s, self.marks[s], self.last_timestamp, 'end_of_data')
        self.pending.clear()
        self.curve.append(dict(timestamp=self.last_timestamp, equity=self.equity,
                               positions=0, halted=self.halted))
        peak, drawdown = self.cfg.capital, 0
        for row in self.curve:
            peak = max(peak, row['equity'])
            drawdown = max(drawdown, (peak - row['equity']) / peak)
        trades = [e for e in self.events if e['kind'] == 'exit']
        return dict(initial_equity=self.cfg.capital, final_equity=self.equity,
                    return_fraction=self.equity / self.cfg.capital - 1,
                    max_drawdown=drawdown, trades=len(trades),
                    wins=sum(t['pnl'] > 0 for t in trades))

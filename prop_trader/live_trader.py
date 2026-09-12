"""WebSocket quotes, persistent completed-bar forecasting, 10-second MTF decisions.

CLI/start uses 1m/1h/1d jointly. Legacy single-frame tick remains for offline
regression compatibility. Live orders require explicit --mode live --confirm-live.
"""
import argparse
import json
import math
import sqlite3
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from .upbit_broker import UpbitClient, BrokerError, order_payload, submit, OrderJournal
from .forecast_time import SECONDS
from .execution_rules import timestamp, fresh_forecast, reprice, sell_quantity, minimum_entry, fills
from .multiframe import (decide as decide_multiframe, buy_budget, exit_fraction, candidate_score,
                          partial_qualifies, partial_budget, hourly_only, too_correlated)
from .analog_signal import feature_vector, load_history, build_analog_db, analog_signal, confirms_buy, too_uncertain

LIVE_ROOT = Path('runs/live')
RANKINGS_DIR = Path('runs/rankings')


def mode_paths(mode):
    """paper/test/live track completely different capital (virtual vs real); never share
    one equity/decisions/state file across modes, or a mode switch corrupts the return curve
    with an unrelated capital base (a virtual-capital drop right next to a real-capital jump)."""
    root = LIVE_ROOT / mode
    return dict(state=root / 'state.json', equity=root / 'equity.csv', decisions=root / 'decisions.jsonl')

# Fixed top-10-by-global-market-cap universe (stablecoins excluded), matching
# download.SYMBOLS -- see that module's comment for how/when this list was derived. Replaces
# the old dynamic top-N-by-24h-volume selection: concentrating the resident worker's few-shot
# budget on 10 large, stable coins instead of spreading it across dozens of thin small-caps
# (many of which never even had usable 1m data -- see the 2026-09-12 monitoring notes).
TOP10_MARKETS = frozenset({'KRW-BTC', 'KRW-ETH', 'KRW-XRP', 'KRW-SOL', 'KRW-TRX',
                            'KRW-DOGE', 'KRW-LINK', 'KRW-ADA', 'KRW-XLM', 'KRW-BCH'})
ENTRY_THRESHOLD = 0.002   # matches the documented "0.2% after cost" entry gate
DAILY_MDD_LIMIT = 0.03    # UTC-day drawdown from day-start equity that halts new entries
STOP_FRACTION = 0.025     # matches engine.Config's backtested stop-loss
# REWARD_MULTIPLE=2 (5% take-profit) was inherited from engine.py's old breakout-strategy
# backtest, never re-validated against the live multiframe system's actual holding period.
# Checked live 2026-09-11: of 29 closed positions that reached horizon timeout (68% of all
# trades), the realized entry->exit move ranged -1.5%..+3.38%, mean ~0% -- not one came close
# to 5%. The old target was effectively decorative; 1 (2.5% take-profit, symmetric with the
# stop) is reachable given this system's real volatility instead of only ever exiting on time.
REWARD_MULTIPLE = 1
STOP_FRACTION_BOUNDS = (0.01, 0.05)  # clip range for downstream.VolatilityHead's per-trade stop


def _stop_fraction(predicted=None):
    """Per-trade stop distance from downstream.VolatilityHead's predicted realized excursion,
    clipped to a sane range; falls back to the fixed STOP_FRACTION when the head hasn't produced
    a value yet (untrained, stale snapshot, or this position predates the head existing) --
    exit_sweep.py's own walk-forward search never found a clearly better fixed constant, so the
    fallback stays exactly what live already used rather than a second guess."""
    if predicted is not None and math.isfinite(predicted) and predicted > 0:
        return min(STOP_FRACTION_BOUNDS[1], max(STOP_FRACTION_BOUNDS[0], predicted))
    return STOP_FRACTION
FEE = 0.001
SLIPPAGE = 0.001
MIN_ORDER_KRW = 5000.     # Upbit's exchange-wide minimum notional for KRW markets
DUST_RESCUE_BUDGET = MIN_ORDER_KRW * 1.15  # top-up size for a stuck sub-minimum position --
                                            # comfortable margin above the floor after fees/slippage
DUST_RESCUE_COOLDOWN = 600.  # don't keep buying into a still-falling price every retry
REFRESH_COOLDOWN = 60.    # minimum gap between on-demand refresh requests, so a still-stale
                          # dashboard polling every 15s can't repeatedly re-trigger the same run
ANALOG_DB_REFRESH_SECONDS = 600.  # forecast_history.jsonl only grows meaningfully on this
                                   # timescale; rebuilding it every 10s tick would be pure waste
PRIORITY_REFRESH_SECONDS = 60.    # how often the tracked top-N candidate set re-ranks; see
                                   # _realtime_tick for why this can't just be every tick


@dataclass
class LiveConfig:
    interval: str = '1h'
    mode: str = 'paper'                 # paper | test | live
    tick_seconds: int = 10
    refresh_seconds: int = 300
    max_positions: int = 3
    max_krw_per_trade: float = 10_000.
    paper_capital: float = 1_000_000.
    entry_threshold: float = ENTRY_THRESHOLD
    # Partial-conviction (hourly_only fallback) entries need a much higher bar than the full
    # 3-frame gate: live data (2026-09-10, n=23) showed full-conviction buys at 66.7% win rate
    # (+665 krw) vs. partial-conviction at 29.4% (-729 krw) -- the missing 1m/1d confirmation is
    # a real accuracy gap, not just noise, so it needs its own stricter threshold rather than
    # sharing entry_threshold.
    partial_entry_threshold: float = 0.01
    universe_top: int = 12               # how many top-24h-volume KRW markets to rank each refresh
    retrain_seconds: int = 86400         # how often the backbone LoRA adapter retrains; see _retrain_once
    analog_gate_enabled: bool = False    # off by default. Once on, it self-activates: a no-op
                                          # until the archive has accumulated analog_min_n
                                          # same-symbol horizon-apart pairs, then starts actually
                                          # filtering BUYs -- see the _realtime_tick integration
                                          # and analog_signal.py.
    analog_min_n: int = 15
    analog_min_win_rate: float = 0.55
    analog_k: int = 30
    # TimesFM 2.5's own native quantile spread ((q90-q10)/anchor at the forecast horizon) --
    # unlike the analog gate, this needs no accumulated history, so it's independent of
    # analog_gate_enabled. None (default) leaves it purely observational until enough real
    # forecast_uncertainty values have been seen live to pick an informed cutoff -- same
    # observe-before-enforce philosophy as the analog gate's own self-activation.
    analog_max_uncertainty: float = None

    def __post_init__(self):
        if self.mode not in ('paper', 'test', 'live'):
            raise ValueError('mode must be paper, test, or live')
        if self.interval not in SECONDS:
            raise ValueError('Unsupported interval')
        if not 5 <= self.tick_seconds <= 300:
            raise ValueError('tick_seconds out of sane range (5..300)')
        if self.refresh_seconds < 60:
            raise ValueError('refresh_seconds must be >=60 to keep model inference load sane')
        if (not all(math.isfinite(v) for v in (self.max_krw_per_trade, self.paper_capital, self.entry_threshold))
                or self.max_positions < 1 or self.max_krw_per_trade <= 0 or self.paper_capital <= 0):
            raise ValueError('Invalid sizing configuration')
        if self.entry_threshold < 0:
            raise ValueError('entry_threshold must be >=0')
        if not math.isfinite(self.partial_entry_threshold) or self.partial_entry_threshold < 0:
            raise ValueError('partial_entry_threshold must be >=0')
        if not 1 <= self.universe_top <= 100:
            # The resident-worker/WebSocket path (_realtime_tick) has no dependency on
            # upbit_data.py's own 30-market cap -- that only bound the legacy subprocess-based
            # _refresh_once path, which start() no longer even schedules. 100 is a deliberate
            # sanity ceiling for the resident worker's real inference throughput, not a leftover.
            raise ValueError('universe_top must be 1..100')
        if self.retrain_seconds < 3600:
            raise ValueError('retrain_seconds must be >=3600 -- LoRA fine-tuning is too heavy to run more often')
        if self.analog_min_n < 1 or self.analog_k < 1:
            raise ValueError('analog_min_n and analog_k must be >=1')
        if not 0 <= self.analog_min_win_rate <= 1:
            raise ValueError('analog_min_win_rate must be 0..1')
        if self.analog_max_uncertainty is not None and (
                not math.isfinite(self.analog_max_uncertainty) or self.analog_max_uncertainty <= 0):
            raise ValueError('analog_max_uncertainty must be >0 when set')


def evaluate_position(position, price, now_ts):
    """Pure stop/target/horizon rule for one open position. No network, no side effects."""
    if price <= position['stop']:
        return 'SELL', '손절(stop-loss)'
    if price >= position['target']:
        return 'SELL', '익절(take-profit)'
    if now_ts >= position['horizon_expires']:
        return 'SELL', '예측 만료 청산(horizon)'
    return 'STAY', '보유 유지'


def evaluate_entry(rank_row, cfg, now_ts=None):
    """Pure entry rule for one ranking row. No network, no side effects."""
    if rank_row is None:
        return 'STAY', '후보 없음'
    try:
        expired = (time.time() if now_ts is None else now_ts) >= timestamp(rank_row['valid_until'])
        finite = math.isfinite(float(rank_row['expected_net_return']))
    except (KeyError, TypeError, ValueError, OverflowError):
        return 'STAY', '예측 유효기간 또는 수익률 누락'
    if not finite or expired or rank_row.get('stale'):
        return 'STAY', '순위 데이터 오래됨(재계산 대기)'
    if not rank_row.get('positive_after_costs'):
        return 'STAY', '비용 차감 후 기대수익이 음수'
    if rank_row['expected_net_return'] <= cfg.entry_threshold:
        return 'STAY', f"기대수익 {rank_row['expected_net_return']:.2%} <= 임계값 {cfg.entry_threshold:.2%}"
    return 'BUY', f"기대수익 {rank_row['expected_net_return']:.2%}, 순위 {rank_row['rank']}"


def account_balance(client):
    """Real KRW cash + coin holdings valued at current price. Read-only; never places orders."""
    accounts = client.request('GET', '/v1/accounts', private=True)
    krw = next((a for a in accounts if a['currency'] == 'KRW'), None)
    krw_balance = float(krw['balance']) if krw else 0.
    krw_locked = float(krw['locked']) if krw else 0.
    holdings = [a for a in accounts if a['currency'] != 'KRW' and (float(a['balance']) + float(a['locked'])) > 1e-12]
    markets = ['KRW-' + h['currency'] for h in holdings]
    prices = {}
    if markets:
        try:
            tickers = client.request('GET', '/v1/ticker', dict(markets=','.join(markets)), private=False)
            prices = {t['market']: float(t['trade_price']) for t in tickers}
        except BrokerError:
            # One delisted/forked holding (e.g. a KRW market with no active ticker) 404s the
            # whole batch; fall back to per-market lookups so the rest still price correctly.
            # Paced -- an unpaced burst of per-market calls right after a failed batch call
            # trips Upbit's per-second ticker rate limit almost every time, turning one bad
            # request into many.
            for i, market in enumerate(markets):
                if i:
                    time.sleep(.12)
                try:
                    ticker = client.request('GET', '/v1/ticker', dict(markets=market), private=False)
                    prices[market] = float(ticker[0]['trade_price'])
                except BrokerError:
                    pass
    rows, coin_value = [], 0.
    for h in holdings:
        market = 'KRW-' + h['currency']
        qty = float(h['balance']) + float(h['locked'])
        price = prices.get(market)
        value = qty * price if price is not None else None
        if value is not None:
            coin_value += value
        rows.append(dict(currency=h['currency'], market=market, qty=qty,
                          avg_buy_price=float(h.get('avg_buy_price') or 0), price=price, value_krw=value))
    rows.sort(key=lambda r: -(r['value_krw'] or 0))
    total = krw_balance + krw_locked + coin_value
    return dict(krw_balance=krw_balance, krw_locked=krw_locked, holdings=rows,
                total_krw=total, fetched_at=datetime.now(timezone.utc).isoformat())


def net_external_flow(client, since_iso):
    """Net KRW moved into the account by deposits/withdrawals (not trading) since `since_iso`.

    Total equity naturally jumps when the user deposits or withdraws real KRW -- that is not
    trading profit and must not be counted as return. Deposits count once ACCEPTED; withdrawals
    count (amount + fee, what actually leaves the balance) once DONE."""
    since = datetime.fromisoformat(since_iso)
    total = 0.
    for d in client.request('GET', '/v1/deposits', dict(currency='KRW'), private=True):
        if d.get('state') == 'ACCEPTED' and d.get('done_at') and datetime.fromisoformat(d['done_at']) > since:
            total += float(d['amount'])
    for w in client.request('GET', '/v1/withdraws', dict(currency='KRW'), private=True):
        if w.get('state') == 'DONE' and w.get('done_at') and datetime.fromisoformat(w['done_at']) > since:
            total -= float(w['amount']) + float(w.get('fee') or 0)
    return total


class LiveTrader:
    def __init__(self, cfg, client=None):
        self.cfg = cfg
        self.client = client or UpbitClient()
        paths = mode_paths(cfg.mode)
        self.state_path, self.equity_path, self.decisions_path = paths['state'], paths['equity'], paths['decisions']
        self.positions = {}          # symbol -> position dict
        self.pending = {}
        self.runtime = None
        self.multiframe_decisions = {}
        self._market_cache = (0., [])
        self._priority_cache = (0., [])
        self._analog_db_cache = (0., [])
        self.last_tick_duration = None
        self.cooldowns = {}
        self.retry_after = {}
        self.journal_path = Path('runs/upbit/orders.sqlite3')
        self.last_prices = {}        # symbol -> last ticker price seen by a tick, for display only
        self.cash = cfg.paper_capital
        self.baseline_equity = None
        self.baseline_set_at = None
        self.day_anchor_equity = None
        self.day_anchor_date = None
        self.daily_return = None
        self._external_flow_cache = {'value': 0., 'at': 0.}
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.last_tick_at = None
        self.last_refresh_at = None
        self.last_retrain_at = None
        self.last_error = None
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._force_refresh = threading.Event()
        self._last_refresh_started = 0.
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_state()
        if not self.equity_path.exists():
            self.equity_path.write_text('timestamp,equity_krw,return_pct\n')
        if self.baseline_equity is None and cfg.mode != 'paper':
            # Seed the baseline from real balance *before* any tick executes a trade, so the
            # return curve reflects "since auto-trading started," not a mid-order settlement dip.
            try:
                self.baseline_equity = account_balance(self.client)['total_krw']
                self.baseline_set_at = datetime.now(timezone.utc).isoformat()
            except BrokerError:
                pass
        if self.baseline_set_at is None:
            # Loaded an older state file that predates baseline_set_at tracking: the baseline
            # was in fact captured at trader startup, so treat any deposit after that as external.
            self.baseline_set_at = self.started_at

    # ---- persistence ----
    def _load_state(self):
        if self.state_path.exists():
            try:
                saved = json.loads(self.state_path.read_text())
                if (saved.get('positions') or saved.get('pending')) and saved.get('interval') != self.cfg.interval:
                    raise ValueError('보유/대기 주문이 있는 상태에서 봉 주기 변경 불가')
                if saved.get('mode') == self.cfg.mode and saved.get('interval') == self.cfg.interval:
                    self.positions = saved.get('positions', {})
                    if self.cfg.mode == 'paper':
                        for position in self.positions.values():
                            position.setdefault('cost_krw', position['qty']*position['entry_price']/(1-FEE))
                    self.pending = saved.get('pending', {})
                    self.cooldowns = saved.get('cooldowns', {})
                    self.retry_after = saved.get('retry_after', {})
                    self.cash = saved.get('cash', self.cfg.paper_capital)
                    self.baseline_equity = saved.get('baseline_equity')
                    self.baseline_set_at = saved.get('baseline_set_at')
                    self.day_anchor_equity = saved.get('day_anchor_equity')
                    self.day_anchor_date = saved.get('day_anchor_date')
                    self.started_at = saved.get('started_at', self.started_at)
                    self.last_retrain_at = saved.get('last_retrain_at')
            except (json.JSONDecodeError, OSError):
                raise ValueError('거래 상태 파일 손상: 복구 전 실행 불가') from None

    def _save_state(self):
        target = self.state_path.with_suffix('.tmp')
        with target.open('w') as f:
            json.dump(dict(
                mode=self.cfg.mode, interval=self.cfg.interval, positions=self.positions,
                pending=self.pending, cooldowns=self.cooldowns, retry_after=self.retry_after,
                cash=self.cash, baseline_equity=self.baseline_equity, baseline_set_at=self.baseline_set_at,
                day_anchor_equity=self.day_anchor_equity, day_anchor_date=self.day_anchor_date,
                daily_return=self.daily_return,
                started_at=self.started_at, last_tick_at=self.last_tick_at, last_refresh_at=self.last_refresh_at,
                last_retrain_at=self.last_retrain_at, last_error=self.last_error), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        target.replace(self.state_path)

    def _log_decision(self, symbol, action, reason, extra=None):
        entry = dict(ts=datetime.now(timezone.utc).isoformat(), symbol=symbol, action=action,
                     reason=reason, mode=self.cfg.mode, **(extra or {}))
        with self.decisions_path.open('a') as f:
            f.write(json.dumps(entry) + '\n')

    def _cached_external_flow(self):
        """net_external_flow is a couple of extra private API calls; cache it briefly so a 30s
        tick loop doesn't re-fetch full deposit/withdraw history every single tick."""
        now = time.time()
        if now - self._external_flow_cache['at'] < REFRESH_COOLDOWN:
            return self._external_flow_cache['value']
        try:
            value = net_external_flow(self.client, self.baseline_set_at)
        except BrokerError:
            return self._external_flow_cache['value']
        self._external_flow_cache = {'value': value, 'at': now}
        return value

    def _append_equity(self, equity):
        if self.baseline_equity is None:
            self.baseline_equity = equity
            self.baseline_set_at = datetime.now(timezone.utc).isoformat()
        # Deposits/withdrawals move total equity without being trading profit -- exclude them so
        # "adding more capital" never shows up as a fake return spike (or a fake loss on withdrawal).
        external = self._cached_external_flow() if self.cfg.mode != 'paper' else 0.
        adjusted = equity - external
        ret = adjusted / self.baseline_equity - 1 if self.baseline_equity else 0.
        today = datetime.now(timezone.utc).date().isoformat()
        if self.day_anchor_date != today:
            # New UTC day (or first observation ever): the Daily MDD breaker measures drawdown
            # from *today's* starting equity, not the all-time baseline above.
            self.day_anchor_equity = adjusted
            self.day_anchor_date = today
        self.daily_return = adjusted / self.day_anchor_equity - 1 if self.day_anchor_equity else 0.
        with self.equity_path.open('a') as f:
            f.write(f'{datetime.now(timezone.utc).isoformat()},{equity:.2f},{ret:.6f}\n')
        return ret

    # ---- data access ----
    def _analog_db(self):
        """Cached historical-analog database for cfg.interval, rebuilt from the resident
        worker's own forecast_history.jsonl archive at most every ANALOG_DB_REFRESH_SECONDS."""
        now = time.time()
        built_at, db = self._analog_db_cache
        if now - built_at < ANALOG_DB_REFRESH_SECONDS:
            return db
        archive = self.state_path.parent / 'realtime' / 'forecast_history.jsonl'
        try:
            db = build_analog_db(load_history(archive), self.cfg.interval)
        except ValueError:
            db = []
        self._analog_db_cache = (now, db)
        return db

    def latest_ranking(self):
        path = RANKINGS_DIR / f'{self.cfg.interval}.json'
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None

    def _tickers(self, markets):
        if not markets:
            return {}
        raw = self.client.request('GET', '/v1/ticker', dict(markets=','.join(sorted(set(markets)))), private=False)
        return {t['market']: float(t['trade_price']) for t in raw}

    # ---- execution ----
    def _identifier(self, symbol, side):
        return 'auto-' + side + '-' + uuid.uuid4().hex[:28]

    def _block(self, market, reason, seconds=300):
        self.last_error = reason
        self.retry_after[market] = time.time() + seconds
        self._log_decision(market, 'STAY', reason)

    def _position(self, market, qty, funds, fee, intent):
        fill = funds / qty
        frac = _stop_fraction(intent.get('predicted_stop_fraction'))
        return dict(symbol=market, qty=qty, entry_price=fill, cost_krw=funds+fee,
                    entry_time=intent['created_at'], stop=fill*(1-frac),
                    target=fill*(1+frac*REWARD_MULTIPLE),
                    horizon_expires=intent['horizon_expires'], identifier=intent['identifier'],
                    forecast_as_of=intent.get('forecast_as_of'),
                    predicted_stop_fraction=intent.get('predicted_stop_fraction'), reconciled=True)

    def _apply_buy_fill(self, market, qty, funds, fee, intent):
        """A second buy fill for an already-held market (e.g. a dust-rescue top-up, see
        _rescue_dust) accumulates into the existing position -- fee-inclusive weighted-average
        entry, stop/target rebased on it -- instead of overwriting it and silently losing track
        of the coins already held. Keeps the original entry_time/horizon_expires/forecast_as_of
        and predicted_stop_fraction: a top-up doesn't reset how long the original bet has been
        running or restart risk sizing from a different forecast's estimate."""
        existing = self.positions.get(market)
        if not existing:
            self.positions[market] = self._position(market, qty, funds, fee, intent)
            return
        total_qty = existing['qty'] + qty
        total_cost = existing['cost_krw'] + funds + fee
        avg_price = total_cost / total_qty
        frac = _stop_fraction(existing.get('predicted_stop_fraction'))
        self.positions[market] = dict(existing, qty=total_qty, cost_krw=total_cost, entry_price=avg_price,
                                       stop=avg_price*(1-frac),
                                       target=avg_price*(1+frac*REWARD_MULTIPLE), reconciled=True)

    def _reconcile(self):
        """Pending orders block new submissions until terminal fills are durably applied.

        A lost POST response (or a crash before POST) is never retried as a new order.
        A 404 remains blocked for operator reconciliation, not assumed rejected.
        """
        for identifier, intent in list(self.pending.items()):
            order = self.client.request('GET', '/v1/order', dict(identifier=identifier))
            if order.get('state') not in ('done', 'cancel'):
                continue
            qty, funds, fee = fills(order)
            market = intent['market']
            if intent['side'] == 'buy' and qty:
                self._apply_buy_fill(market, qty, funds, fee, intent)
            elif intent['side'] == 'sell' and qty:
                position = self.positions[market]
                if qty > position['qty'] + 1e-8:
                    raise ValueError('체결 수량이 봇 보유량 초과')
                fraction = min(1., qty / position['qty'])
                cost = position['cost_krw'] * fraction
                position['qty'] = max(0., position['qty'] - qty)
                position['cost_krw'] -= cost
                if position['qty'] < 1e-8:
                    del self.positions[market]
                intent['pnl_krw'] = funds - fee - cost
                self.retry_after[market] = time.time()+SECONDS['1m']
            del self.pending[identifier]
            # State is authoritative; saving before the audit log makes fill application
            # idempotent on restart even if the journal/log write is interrupted.
            self._save_state()
            journal = OrderJournal(self.journal_path)
            try:
                journal.update(identifier, order['state'].upper(), order)
            finally:
                journal.db.close()
            self._log_decision(market, ('BUY' if intent['side']=='buy' else 'SELL') if qty else 'ORDER_CANCELED',
                               '체결 확인' if qty else '체결 없이 종료',
                               dict(identifier=identifier, qty=qty, funds_krw=funds,
                                    fee_krw=fee, pnl_krw=intent.get('pnl_krw'), state=order['state']))
        return not self.pending

    def _migrate_positions(self, accounts):
        """Verify legacy estimated positions before trading. Never infer an order fill."""
        for market, position in self.positions.items():
            if position.get('reconciled'):
                continue
            order = self.client.request('GET', '/v1/order', dict(identifier=position['identifier']))
            if order.get('state') not in ('done', 'cancel'):
                raise ValueError('기존 매수 주문 체결 확인 대기')
            qty, funds, fee = fills(order)
            held = next((a for a in accounts if a['currency'] == market[4:]), {})
            available = float(held.get('balance', 0)) + float(held.get('locked', 0))
            if qty <= 0 or available + 1e-8 < qty:
                raise ValueError('기존 포지션과 실잔고 불일치: 주문 내역 확인 필요')
            position.update(qty=qty, entry_price=funds/qty, cost_krw=funds+fee,
                            stop=funds/qty*(1-STOP_FRACTION),
                            target=funds/qty*(1+STOP_FRACTION*REWARD_MULTIPLE), reconciled=True)
        self._save_state()

    def _send(self, market, side, amount, intent, owned_volume=None):
        if self._stop.is_set():
            return
        identifier = intent['identifier']
        if self.cfg.mode == 'live':
            self.pending[identifier] = intent
            self._save_state()  # durable intent BEFORE any POST
        try:
            result = submit(self.client, order_payload(market, side, amount, identifier),
                            mode=self.cfg.mode, journal_path=self.journal_path,
                            max_krw=self.cfg.max_krw_per_trade, owned_volume=owned_volume,
                            deadline=intent.get('quote_expires'))
        except BrokerError as e:
            # Only pre-submission validation failures can safely clear an intent.
            journal = OrderJournal(self.journal_path)
            try:
                recorded = journal.db.execute('SELECT status FROM orders WHERE identifier=?',
                                              (identifier,)).fetchone()
            finally:
                journal.db.close()
            if not recorded:
                self.pending.pop(identifier, None)
            self._block(market, str(e))
            self._save_state()
            return
        self._log_decision(market, side.upper()+'_SUBMITTED', intent['reason'],
                           dict(identifier=identifier))
        if self.cfg.mode == 'test':
            self._block(market, '테스트 주문 확인: 실제 보유량 변경 없음')

    def _buy(self, market, price, rank_row, reason, budget=None):
        budget = min(self.cfg.max_krw_per_trade, self.cash, budget if budget is not None else self.cfg.max_krw_per_trade)
        if budget < minimum_entry():
            self._block(market, '손절 후 최소 주문금액을 확보할 가용 자금 부족')
            return
        identifier = self._identifier(market, 'buy')
        intent = dict(identifier=identifier, market=market, side='buy', reason=reason,
                      created_at=self.started_tick_ts, forecast_as_of=rank_row['as_of'],
                      quote_expires=timestamp(rank_row['valid_until']),
                      horizon_expires=timestamp(rank_row['as_of']) + SECONDS[self.cfg.interval]*5,
                      predicted_stop_fraction=rank_row.get('predicted_stop_fraction'))
        self._log_decision(market, 'BUY_INTENT', reason,
                           dict(identifier=identifier, forecast=rank_row, budget_krw=budget))
        if self.cfg.mode == 'paper':
            fill = rank_row['estimated_entry_vwap']
            principal = budget / (1+FEE)
            qty = principal / fill
            self.cash -= budget
            self.positions[market] = self._position(market, qty, principal, budget-principal, intent)
            self._log_decision(market, 'BUY', reason, dict(price=fill, qty=qty, notional_krw=budget))
        else:
            # Budget includes fees, unlike Upbit's price parameter (principal only).
            self._send(market, 'buy', format(budget/(1+FEE), '.8f'), intent)

    def _sell(self, market, price, reason, fraction=1.):
        position = self.positions.get(market)
        if position is None:
            return
        fraction = exit_fraction(fraction, position['qty'], price)
        sell_qty = position['qty']*fraction
        self.cooldowns[market] = dict(until=self.started_tick_ts+SECONDS[self.cfg.interval],
                                     as_of=position.get('forecast_as_of'))
        if self.cfg.mode == 'paper':
            fill = price*(1-SLIPPAGE)
            proceeds = sell_qty*fill*(1-FEE)
            if sell_qty*fill < MIN_ORDER_KRW:
                self._block(market, '청산 최소 금액 미달: 소액 잔량 대기', 60)
                return
            self.cash += proceeds
            cost = position['cost_krw']*fraction
            position['qty'] -= sell_qty
            position['cost_krw'] -= cost
            if position['qty'] < 1e-8:
                del self.positions[market]
            else:
                self.retry_after[market] = self.started_tick_ts+SECONDS['1m']
            self._log_decision(market, 'SELL', reason,
                               dict(price=fill, qty=sell_qty, pnl_krw=proceeds-cost))
            return
        accounts = self.client.request('GET', '/v1/accounts', private=True)
        held = next((a for a in accounts if a['currency']==market[4:]), {})
        volume = sell_quantity(sell_qty, float(held.get('balance', 0)))
        if float(volume)*price*(1-SLIPPAGE) < MIN_ORDER_KRW:
            # Upbit rejects a sell below its own minimum notional outright -- no amount of
            # retrying changes that. The only way out is a small top-up buy of the same coin so
            # the combined holding clears the floor; the normal SELL path above (re-checked
            # every tick) then closes the whole thing out once it does.
            self._rescue_dust(market, position)
            self._block(market, '청산 가능 수량/최소 금액 부족: 최소금액 확보용 추가매수 시도', 60)
            return
        intent = dict(identifier=self._identifier(market, 'sell'), market=market, side='sell',
                      reason=reason, created_at=self.started_tick_ts)
        self._send(market, 'sell', str(volume), intent, owned_volume=str(position['qty']))

    def _rescue_dust(self, market, position):
        """Top up a position stuck below Upbit's sell minimum so the combined holding clears
        it, then let the normal SELL path (re-checked every tick, see the caller) close the
        whole thing out on a later tick. Real money, so: live/test only (paper isn't actually
        capped by an exchange minimum the same way), one attempt per cooldown window so a
        still-falling price can't trigger repeat rescue buys chasing it down, and skipped
        outright if it would leave too little cash to do anything else."""
        if self.cfg.mode == 'paper':
            return
        if time.time() - position.get('rescue_attempted_at', 0) < DUST_RESCUE_COOLDOWN:
            return
        if self.cash < DUST_RESCUE_BUDGET:
            self._log_decision(market, 'STAY', '먼지 정리용 추가 매수 자금 부족')
            return
        position['rescue_attempted_at'] = time.time()
        identifier = self._identifier(market, 'buy')
        intent = dict(identifier=identifier, market=market, side='buy',
                      reason='최소 매도금액 확보용 추가 매수(먼지 정리)', created_at=self.started_tick_ts,
                      forecast_as_of=position.get('forecast_as_of'), quote_expires=self.started_tick_ts+30,
                      horizon_expires=position.get('horizon_expires', self.started_tick_ts))
        self._log_decision(market, 'BUY_INTENT', '먼지 정리용 추가 매수',
                           dict(identifier=identifier, budget_krw=DUST_RESCUE_BUDGET))
        self._send(market, 'buy', format(DUST_RESCUE_BUDGET/(1+FEE), '.8f'), intent)

    # ---- one tick ----
    def tick(self):
        self.started_tick_ts = time.time()
        now = self.started_tick_ts
        self.last_error = None
        if self.cfg.mode == 'live' and not self._reconcile():
            self.last_error = '주문 체결 확인 대기: 추가 주문 보류'
            self._save_state()
            return
        if self.runtime is not None:
            return self._realtime_tick()
        ranking = self.latest_ranking() or {}
        by_symbol = {r['symbol']: r for r in ranking.get('rankings', [])}
        accounts = []
        if self.cfg.mode != 'paper':
            accounts = self.client.request('GET', '/v1/accounts', private=True)
            if self.cfg.mode == 'live':
                self._migrate_positions(accounts)
            self.cash = sum(float(a['balance']) for a in accounts if a['currency']=='KRW')
        markets = set(self.positions) | set(by_symbol)
        markets.update('KRW-'+a['currency'] for a in accounts
                       if a['currency']!='KRW' and float(a['balance'])+float(a['locked'])>0)
        # Refresh tradable universe every tick; unlisted holdings must not poison a batch.
        metadata = self.client.request('GET', '/v1/market/all', dict(is_details='true'), private=False)
        listed = {r['market'] for r in metadata if r['market'].startswith('KRW-')}
        eligible = {r['market'] for r in metadata if r['market'] in listed
                    and r.get('market_warning') != 'CAUTION'
                    and not r.get('market_event', {}).get('warning', False)}
        prices = self._tickers(markets & listed)
        self.last_prices = prices
        if self.cfg.mode != 'paper':
            equity = 0.
            reliable = True
            for account in accounts:
                qty = float(account['balance'])+float(account['locked'])
                if account['currency']=='KRW':
                    equity += qty
                elif qty>0:
                    mark = prices.get('KRW-'+account['currency'])
                    if mark is None:
                        reliable = False
                    else:
                        equity += qty*mark
            if reliable:
                self._append_equity(equity)
            else:
                self.last_error = '평가 시세 누락: 수익률 기록 생략'
        exited = set()
        for market, position in list(self.positions.items()):
            if now < self.retry_after.get(market, 0):
                continue
            price = prices.get(market)
            if price is None:
                continue
            action, reason = evaluate_position(position, price, now)
            row = by_symbol.get(market)
            if action == 'STAY' and row and fresh_forecast(row, self.cfg.interval, now):
                # Compare future versus immediate liquidation; entry fee is sunk.
                if float(row['forecast_prices'][-1])/price-1 < -self.cfg.entry_threshold:
                    action, reason = 'SELL', '최신 예측의 잔여 상승 여력 소멸'
            if action == 'SELL':
                # A stopped position cannot re-enter on this tick or this forecast.
                position['forecast_as_of'] = row.get('as_of') if row else position.get('forecast_as_of')
                exited.add(market)
                self._sell(market, price, reason)
                if self.pending:
                    break
            else:
                self._log_decision(market, 'STAY', reason, dict(price=price))
        candidates = []
        if not self.pending and self.cash >= minimum_entry():
            for market, row in by_symbol.items():
                cooldown = self.cooldowns.get(market, {})
                if (market not in eligible or market not in prices or market in self.positions or market in exited
                        or now < self.retry_after.get(market, 0)
                        or now < cooldown.get('until', 0)
                        or row.get('as_of') == cooldown.get('as_of')
                        or not fresh_forecast(row, self.cfg.interval, now)):
                    continue
                candidates.append(row)
            if candidates and len(self.positions) < self.cfg.max_positions:
                raw = self.client.request('GET', '/v1/orderbook',
                                          dict(markets=','.join(r['symbol'] for r in candidates)), private=False)
                books = {b['market']: b for b in raw}
                repriced = []
                for row in candidates:
                    try:
                        repriced.append(reprice(row, books[row['symbol']], self.cfg.interval,
                                                time.time(), min(self.cash, self.cfg.max_krw_per_trade)))
                    except (KeyError, ValueError, TypeError):
                        continue
                repriced.sort(key=lambda r: -r['expected_net_return'])
                for rank, row in enumerate(repriced, 1):
                    if self.pending or len(self.positions)>=self.cfg.max_positions:
                        break
                    row['rank'] = rank
                    action, reason = evaluate_entry(row, self.cfg)
                    if action == 'BUY':
                        self._buy(row['symbol'], prices[row['symbol']], row, reason)
        if self.cfg.mode == 'paper':
            equity = self.cash + sum(prices.get(m, p['entry_price'])*p['qty'] for m,p in self.positions.items())
            self._append_equity(equity)
        if self.pending:
            self.last_error = '주문 체결 확인 대기: 추가 주문 보류'
        self.last_tick_at = datetime.now(timezone.utc).isoformat()
        self._save_state()

    def realtime_snapshot(self):
        if self.runtime is None:
            return dict(enabled=False)
        feed = self.runtime.stream.snapshot()
        worker = self.runtime.forecasts()
        return dict(enabled=True, connected=feed['connected'], error=feed['error'],
                    fresh_tickers=len(feed['tickers']), fresh_books=len(feed['books']),
                    model_loaded=worker.get('model_loaded', False), busy=worker.get('busy'),
                    last_job_seconds=worker.get('last_job_seconds'),
                    updated_at=worker.get('updated_at'), errors=worker.get('errors', {}))

    def _realtime_tick(self):
        now = time.time()
        accounts = []
        if self.cfg.mode != 'paper':
            accounts = self.client.request('GET', '/v1/accounts', private=True)
            if self.cfg.mode == 'live':
                self._migrate_positions(accounts)
            self.cash = sum(float(a['balance']) for a in accounts if a['currency']=='KRW')
        if now-self._market_cache[0] > 300:
            metadata = self.client.request('GET', '/v1/market/all', dict(is_details='true'), private=False)
            self._market_cache = (now, metadata)
        metadata = self._market_cache[1]
        listed = {r['market'] for r in metadata if r['market'].startswith('KRW-')}
        eligible = {r['market'] for r in metadata if r['market'] in listed
                    and r.get('market_warning') != 'CAUTION'
                    and not r.get('market_event', {}).get('warning', False)}
        feed = self.runtime.stream.snapshot()
        if now-self._priority_cache[0] > PRIORITY_REFRESH_SECONDS or not self._priority_cache[1]:
            # Universe is the fixed TOP10_MARKETS set, not a dynamic top-N-by-volume ranking
            # (--universe-top no longer controls universe size, only compatibility). Still
            # re-rank by candidate_score at most once a minute -- with only 10 symbols this
            # mostly just decides refresh order, not which markets are ever considered at all.
            liquid = sorted((r for m,r in feed['tickers'].items() if m in eligible and m in TOP10_MARKETS),
                            key=lambda r: -candidate_score(r))
            self._priority_cache = (now, [r['market'] for r in liquid])
        selected = self._priority_cache[1]
        # Held coins first, then candidate_score rank (selected is already sorted by it).
        # Previously re-sorted by last-known signal_strength, defaulting an unevaluated market
        # to 0 -- self-reinforcing: whichever markets got forecast first stayed permanently
        # ahead of a wider universe's brand-new candidates, which always sorted last and so
        # never got their first evaluation. candidate_score's own order already reflects
        # liquidity + momentum; no separate freshness bias is needed on top of it.
        priority = list(dict.fromkeys([m for m in self.positions if m in listed]+selected))
        self.runtime.stream.subscribe(listed, priority)
        self.runtime.request(priority)
        worker = self.runtime.forecasts()
        records = worker.get('records', {})
        generated = [r['generated_at'] for group in records.values() for r in group.values() if r.get('generated_at')]
        if generated:
            self.last_refresh_at = max(generated)
        prices = {m: float(r['trade_price']) for m,r in feed['tickers'].items()}
        missing = (set(self.positions)&listed)-set(prices)
        if missing:
            # Disconnection blocks entries, but protective exits still get REST prices.
            prices.update(self._tickers(missing))
        self.last_prices = prices
        self.multiframe_decisions = {}
        exited = set()
        buy_candidates = []
        # Daily MDD circuit breaker: once today's (UTC) equity has fallen DAILY_MDD_LIMIT from
        # its start-of-day anchor, stop opening new positions for the rest of the day. Existing
        # positions still get their normal protective exits (stop/target/horizon/multiframe) --
        # this only ever suppresses new entries, never blocks an exit.
        halted = self.daily_return is not None and self.daily_return <= -DAILY_MDD_LIMIT
        for market in priority:
            held = market in self.positions
            book = feed['books'].get(market)
            try:
                decision = decide_multiframe(records.get(market, {}), book, time.time(),
                            max(minimum_entry(), min(self.cash, self.cfg.max_krw_per_trade)),
                            self.cfg.entry_threshold, held=held)
            except (ValueError, TypeError, KeyError, ZeroDivisionError):
                decision = dict(action='STAY', reason='예측/호가 검증 실패', complete=False)
            if not held and not decision.get('complete'):
                # Many smaller Upbit markets can't sustain a gap-free 1m candle series (they
                # simply don't trade every minute), so the full three-frame gate can never even
                # evaluate them. Fall back to an hourly-only reading -- it can only ever qualify
                # for the smaller partial_budget below, never the full-conviction one.
                try:
                    fallback = hourly_only(records.get(market, {}), book, time.time(),
                                max(minimum_entry(), min(self.cash, self.cfg.max_krw_per_trade)),
                                self.cfg.partial_entry_threshold)
                except (ValueError, TypeError, KeyError, ZeroDivisionError):
                    fallback = None
                if fallback and fallback.get('complete'):
                    decision = fallback
            budget = buy_budget(decision, self.cash, self.cfg.max_krw_per_trade) if not held else 0.
            public = {k:v for k,v in decision.items() if k != 'priced'}
            public['budget_krw'] = budget
            public['signal_action'] = decision['action']
            if not held and decision['action']=='BUY' and not budget:
                public.update(action='STAY', reason='신호 강도에 따른 예산이 최소 주문금액 미달')
            self.multiframe_decisions[market] = public
            price = prices.get(market)
            if held and price is not None:
                position = self.positions[market]
                action, reason = evaluate_position(position, price, now)
                fraction = 1.
                if action == 'STAY' and decision['action']=='SELL':
                    if now < self.retry_after.get(market, 0):
                        continue
                    action, reason = 'SELL', decision['reason']
                    fraction = decision.get('sell_fraction', 1.)
                if action=='SELL':
                    # Minimum-notional retries still have a cooldown; risk exit always full.
                    if now < self.retry_after.get(market, 0) and position['qty']*price < MIN_ORDER_KRW:
                        continue
                    row = records.get(market, {}).get(self.cfg.interval, {})
                    position['forecast_as_of'] = row.get('as_of', position.get('forecast_as_of'))
                    exited.add(market)
                    self._sell(market, price, reason, fraction)
                    public.update(action='SELL', reason=reason, requested_sell_fraction=fraction)
                    if self.pending:
                        break
                else:
                    self._log_decision(market, 'STAY', decision['reason'], dict(multiframe=public))
            elif halted and not held:
                public.update(action='STAY', reason=f'Daily MDD 서킷브레이커: 당일 {self.daily_return:.2%} 손실, 신규 매수 중단')
                self.multiframe_decisions[market] = public
                self._log_decision(market, 'STAY', public['reason'], dict(multiframe=public))
            elif decision['action']=='BUY' and budget:
                buy_candidates.append((market, decision, False))
            elif not held and partial_budget(decision, self.cash, self.cfg.max_krw_per_trade, self.cfg.partial_entry_threshold):
                # The strict three-frame gate declined, but the hourly edge alone is still
                # solidly positive -- the ranking table already reports this return as positive.
                # Take it, just smaller (see multiframe.partial_budget).
                public.update(action='PARTIAL_CANDIDATE', reason='시간봉 단독 근거 부분 진입 후보: '+decision['reason'])
                self.multiframe_decisions[market] = public
                buy_candidates.append((market, decision, True))
            else:
                self._log_decision(market, 'STAY', decision['reason'], dict(multiframe=public))
        # Full-conviction candidates (partial=False sorts first) always outrank partial ones.
        buy_candidates.sort(key=lambda item: (item[2], -item[1]['signal_strength'], -item[1]['score']))
        for rank, (market, decision, partial) in enumerate(buy_candidates, 1):
            if self.pending or len(self.positions)>=self.cfg.max_positions:
                self.multiframe_decisions[market].update(action='STAY', reason='대기 주문 또는 보유 종목 수 한도')
                continue
            cooldown = self.cooldowns.get(market, {})
            base_row = records.get(market, {}).get(self.cfg.interval, {})
            if (market in exited or market in self.positions or market not in eligible
                    or now < self.retry_after.get(market, 0) or now < cooldown.get('until', 0)
                    or cooldown.get('as_of') == base_row.get('as_of')):
                self.multiframe_decisions[market].update(action='STAY', reason='청산 후 재진입/재시도 대기')
                continue
            # Recheck WebSocket freshness and all three forecasts at the actual submit point.
            current = self.runtime.stream.snapshot()
            book = current['books'].get(market)
            def size(d, partial_size):
                return partial_budget(d, self.cash, self.cfg.max_krw_per_trade, self.cfg.partial_entry_threshold) \
                    if partial_size else buy_budget(d, self.cash, self.cfg.max_krw_per_trade)
            budget = size(decision, partial)
            if not budget:
                self.multiframe_decisions[market].update(action='STAY', reason='가용 매수 예산 부족', budget_krw=0)
                continue
            decision = decide_multiframe(records.get(market, {}), book, time.time(), budget,
                                         self.cfg.entry_threshold)
            if not decision.get('complete') and partial:
                # Same fallback as the initial scan: this market may simply never produce a
                # complete three-frame reading (no gap-free 1m data), so re-verifying on the
                # full decide() alone would always fail it here even though nothing about its
                # hourly edge has changed.
                try:
                    fallback = hourly_only(records.get(market, {}), book, time.time(), budget,
                                           self.cfg.partial_entry_threshold)
                except (ValueError, TypeError, KeyError, ZeroDivisionError):
                    fallback = None
                if fallback and fallback.get('complete'):
                    decision = fallback
            if decision['action'] == 'BUY':
                partial = False  # reconfirmed at full conviction -- size and log it as such
            elif not (partial and partial_qualifies(decision, self.cfg.partial_entry_threshold)):
                self.multiframe_decisions[market].update(action='STAY', reason=decision['reason'])
                continue
            budget = min(budget, size(decision, partial))
            if not budget:
                continue
            record = records.get(market, {}).get(self.cfg.interval, {})
            self.multiframe_decisions[market]['forecast_uncertainty'] = record.get('forecast_uncertainty')
            blocked, uncertainty_reason = too_uncertain(record, self.cfg.analog_max_uncertainty)
            if blocked:
                self.multiframe_decisions[market].update(action='STAY', reason=uncertainty_reason)
                continue
            held_histories = [
                [h['c'] for h in records.get(held, {}).get(self.cfg.interval, {}).get('history', [])]
                for held in self.positions]
            correlated, correlation_reason = too_correlated([h['c'] for h in record.get('history', [])],
                                                             held_histories)
            if correlated:
                self.multiframe_decisions[market].update(action='STAY', reason=correlation_reason)
                continue
            if self.cfg.analog_gate_enabled:
                # Final confirmation: only downgrades a BUY the rules already approved to STAY,
                # never the reverse. Uses the same technical-features record the multiframe
                # decision just priced -- not a fresh fetch -- so the two never disagree about
                # what "now" looked like.
                # Self-activating: below analog_min_n the archive simply doesn't have an opinion
                # yet, so the gate stays a no-op (visible in multiframe_decisions, never blocking)
                # until enough same-symbol horizon-apart pairs have accumulated on their own --
                # no restart needed once the analog database is finally worth trusting.
                vector = feature_vector(record)
                signal = analog_signal(vector, self._analog_db(), k=self.cfg.analog_k)
                self.multiframe_decisions[market]['analog_signal'] = signal
                if signal['n'] >= self.cfg.analog_min_n:
                    ok, analog_reason = confirms_buy(signal, min_win_rate=self.cfg.analog_min_win_rate,
                                                     min_n=self.cfg.analog_min_n)
                    if not ok:
                        self.multiframe_decisions[market].update(
                            action='STAY', reason='과거 유사 패턴 백테스트 미충족: '+analog_reason)
                        continue
            row = dict(decision['priced'][self.cfg.interval], rank=rank,
                       multiframe={k:v for k,v in decision.items() if k!='priced'})
            row['valid_until'] = min(r['valid_until'] for r in decision['priced'].values())
            self.multiframe_decisions[market]['budget_krw'] = budget
            reason = ('부분 진입(시간봉 단독 근거): ' if partial else '') + decision['reason']
            self._buy(market, row['estimated_entry_vwap'], row, reason, budget=budget)
        if self.cfg.mode == 'paper':
            self._append_equity(self.cash+sum(prices.get(m,p['entry_price'])*p['qty'] for m,p in self.positions.items()))
        else:
            total, complete = 0., True
            for a in accounts:
                qty = float(a['balance'])+float(a['locked'])
                if a['currency']=='KRW':
                    total += qty
                elif qty:
                    price = prices.get('KRW-'+a['currency'])
                    if price is None:
                        # Untradeable dust with no active market (e.g. ETHW/ETHF/STRK, sitting
                        # in this account since before this bot ever ran) can NEVER get a price
                        # -- blocking the whole snapshot on it froze equity logging entirely for
                        # a full day while real trades kept happening. Only block on a holding
                        # this bot is actually tracking: that's a real accounting gap, not dust.
                        if 'KRW-'+a['currency'] in self.positions:
                            complete = False
                    else:
                        total += qty*price
            if complete:
                self._append_equity(total)
        if not feed['connected']:
            self.last_error = 'WebSocket 연결 대기: 신규 진입 보류, 보호 청산은 REST 시세 사용'
        elif self.pending:
            self.last_error = '주문 체결 확인 대기'
        self.last_tick_at = datetime.now(timezone.utc).isoformat()
        self._save_state()

    def snapshot(self):
        with self.lock:
            positions = []
            for p in self.positions.values():
                mark = self.last_prices.get(p['symbol'])
                enriched = dict(p, mark_price=mark,
                                 unrealized_pct=(mark / p['entry_price'] - 1) if mark else None,
                                 value_krw=(mark * p['qty']) if mark else None)
                positions.append(enriched)
            return dict(mode=self.cfg.mode, interval=self.cfg.interval, tick_seconds=self.cfg.tick_seconds,
                        refresh_seconds=self.cfg.refresh_seconds, max_positions=self.cfg.max_positions,
                        max_krw_per_trade=self.cfg.max_krw_per_trade, cash=self.cash,
                        positions=positions, pending=list(self.pending.values()),
                        realtime=self.realtime_snapshot(), multiframe_decisions=self.multiframe_decisions,
                        last_tick_duration=self.last_tick_duration,
                        cooldowns=self.cooldowns, retry_after=self.retry_after, started_at=self.started_at,
                        last_tick_at=self.last_tick_at, last_refresh_at=self.last_refresh_at,
                        last_retrain_at=self.last_retrain_at, last_error=self.last_error,
                        daily_return=self.daily_return, daily_mdd_limit=DAILY_MDD_LIMIT,
                        daily_halted=bool(self.daily_return is not None and self.daily_return <= -DAILY_MDD_LIMIT))

    # ---- background loops ----
    def _tick_loop(self):
        while not self._stop.is_set():
            cycle_start = time.monotonic()
            with self.lock:
                try:
                    self.tick()
                except Exception as e:
                    self.last_error = str(e) if isinstance(e, (BrokerError, ValueError)) else '매매 처리 오류: 추가 주문 보류'
                    self._save_state()
            self.last_tick_duration = time.monotonic()-cycle_start
            self._stop.wait(max(0.1, self.cfg.tick_seconds-self.last_tick_duration))

    def request_refresh(self):
        """Ask the background refresh loop to run now instead of waiting out refresh_seconds.
        Rate-limited so a dashboard that keeps seeing 'stale' can't spam Upbit with reruns."""
        if self.runtime is not None:
            self.runtime.request(list(dict.fromkeys(list(self.positions)+list(self.multiframe_decisions))))
            return True
        if time.time() - self._last_refresh_started < REFRESH_COOLDOWN:
            return False
        self._force_refresh.set()
        return True

    def _refresh_once(self):
        self._last_refresh_started = time.time()
        venv_python = Path('.venv-timesfm/bin/python')
        env = dict(os.environ, HF_HOME=str(Path('.cache/huggingface').resolve()), HF_HUB_OFFLINE='1')
        try:
            subprocess.run([sys.executable, '-m', 'prop_trader.upbit_data', '--interval', self.cfg.interval,
                             '--top', str(self.cfg.universe_top)], check=True, timeout=120, env=env)
            if venv_python.exists():
                subprocess.run([str(venv_python), '-m', 'prop_trader.rank_coins', '--interval', self.cfg.interval,
                                 '--notional', str(self.cfg.max_krw_per_trade), '--refresh-orderbooks'],
                                check=True, timeout=600, env=env)
            self.last_refresh_at = datetime.now(timezone.utc).isoformat()
            self.last_error = None
        except (subprocess.SubprocessError, OSError) as e:
            self.last_error = f'예측 갱신 실패: {e}'
        with self.lock:
            self._save_state()

    def _refresh_loop(self):
        self._refresh_once()
        while not self._stop.is_set():
            triggered = self._force_refresh.wait(timeout=self.cfg.refresh_seconds)
            if self._stop.is_set():
                break
            self._force_refresh.clear()
            self._refresh_once()

    def _retrain_once(self):
        """Daily LoRA fine-tune of the TimesFM backbone for all three frames (1m/1h/1d) on the
        latest Binance candles. Live decisions use all three frames jointly (see
        multiframe.decide) -- retraining only self.cfg.interval would leave two of three
        forecasts permanently stale even though the backbone is nominally "kept fresh".

        Each interval trains into its own scratch directory and only swaps into
        runs/forecast5/{interval} -- what live inference actually reads -- after training
        succeeds; one interval's failure does not block the others. The technical calibrator is
        regenerated once for all three intervals together afterward (hybrid_research defaults to
        all three); if that fails, every interval trained this cycle is rolled back together,
        since a calibrator must match its adapter's data hash or predict_symbol refuses to run.
        """
        venv_python = Path('.venv-timesfm/bin/python')
        if not venv_python.exists():
            self.last_error = 'LoRA 재학습 불가: .venv-timesfm 없음'
            with self.lock:
                self._save_state()
            return
        env = dict(os.environ, HF_HOME=str(Path('.cache/huggingface').resolve()), HF_HUB_OFFLINE='1')
        errors = []
        try:
            subprocess.run([sys.executable, '-m', 'prop_trader.current_candles', '--out', 'data/current'],
                           check=True, timeout=300, env=env)
        except (subprocess.SubprocessError, OSError) as e:
            self.last_error = f'LoRA 재학습 실패, 이전 모델 유지: 캔들 갱신 실패: {e}'
            with self.lock:
                self._save_state()
            return
        trained = []
        for interval in ('1m', '1h', '1d'):
            live_dir = Path('runs/forecast5') / interval
            tmp_dir = Path('runs/forecast5') / f'{interval}_retrain_tmp'
            backup_dir = Path('runs/forecast5') / f'{interval}_prev'
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            try:
                subprocess.run([str(venv_python), '-m', 'prop_trader.timesfm_run',
                                 '--csv', f'data/current/{interval}.csv', '--out', str(tmp_dir),
                                 '--interval', interval, '--rolling-split', '--horizon', '5',
                                 '--device', 'cpu'],  # keep the GPU free for live MPS inference during retrain
                                check=True, timeout=10800, env=env)
                if not (tmp_dir / 'adapter' / 'adapter_model.safetensors').exists():
                    raise RuntimeError('training finished without an adapter checkpoint')
                json.loads((tmp_dir / 'protocol.json').read_text())  # sanity: must parse
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                if live_dir.exists():
                    live_dir.rename(backup_dir)
                tmp_dir.rename(live_dir)
                trained.append(interval)
            except (subprocess.SubprocessError, OSError, RuntimeError, json.JSONDecodeError) as e:
                errors.append(f'{interval}: {e}')
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
        if trained:
            # Calibrator is keyed to each adapter's data hash; without this, rank_coins and
            # timesfm_predict both refuse to run ("Stale calibrator from another training snapshot").
            try:
                subprocess.run([str(venv_python), '-m', 'prop_trader.hybrid_research'],
                                check=True, timeout=10800, env=env)
                for interval in trained:
                    backup_dir = Path('runs/forecast5') / f'{interval}_prev'
                    if backup_dir.exists():
                        shutil.rmtree(backup_dir)
            except (subprocess.SubprocessError, OSError) as e:
                errors.append(f'calibrator: {e}')
                for interval in trained:
                    live_dir = Path('runs/forecast5') / interval
                    backup_dir = Path('runs/forecast5') / f'{interval}_prev'
                    if backup_dir.exists():
                        if live_dir.exists():
                            shutil.rmtree(live_dir)
                        backup_dir.rename(live_dir)
                trained = []
        self.last_error = ('LoRA 재학습 일부 실패, 해당 주기는 이전 모델 유지: ' + '; '.join(errors)) if errors else None
        if trained:
            self.last_retrain_at = datetime.now(timezone.utc).isoformat()
        with self.lock:
            self._save_state()

    def _retrain_loop(self):
        # Wait a full cycle before the first run -- retraining is heavy (hours of CPU time is
        # plausible), so starting one the instant the server boots would surprise whoever just
        # launched it. Disabled entirely when retrain_seconds is falsy-checked by the caller.
        while not self._stop.wait(self.cfg.retrain_seconds):
            self._retrain_once()

    def start(self, retrain=True):
        import fcntl
        self._instance_file = (self.state_path.parent / 'process.lock').open('a')
        try:
            fcntl.flock(self._instance_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._instance_file.close()
            raise ValueError('동일 모드 자동매매가 이미 실행 중입니다') from None
        from .realtime_runtime import RealtimeRuntime
        self.runtime = RealtimeRuntime(self.state_path.parent)
        self.runtime.start()
        self._tick_thread = threading.Thread(target=self._tick_loop, daemon=True, name='live-tick')
        self._tick_thread.start()
        # Resident inference no longer pins a fixed adapter/calibrator pair for the whole
        # session: predict_symbol() reloads the LoRA adapter from disk on every call, and
        # resident_forecasts.run() now re-reads protocol.json per job too (see that module),
        # so a background retrain's atomic adapter+calibrator swap is picked up automatically
        # on the next prediction for each market/interval -- no separate "activate" step needed.
        if retrain:
            self._retrain_thread = threading.Thread(target=self._retrain_loop, daemon=True, name='live-retrain')
            self._retrain_thread.start()

    def stop(self):
        self._stop.set()
        if hasattr(self, '_tick_thread'):
            self._tick_thread.join(timeout=45)
        if hasattr(self, '_retrain_thread'):
            # Bounded like the tick thread's join: if a retrain subprocess is mid-flight (up to
            # the 3h timeout in _retrain_once), this does not wait for it -- it's a daemon
            # thread, so process exit does not hang on it either way.
            self._retrain_thread.join(timeout=45)
        if self.runtime:
            self.runtime.stop()


def main():
    import importlib.util
    import sys
    import signal
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    if importlib.util.find_spec('websocket') is None:
        python = Path('.venv-timesfm/bin/python').absolute()
        if not python.exists() or Path(sys.prefix).absolute() == python.parent.parent:
            raise SystemExit('websocket-client 설치 필요: .venv-timesfm/bin/python -m pip install websocket-client==1.8.0')
        os.execv(str(python), [str(python), '-m', 'prop_trader.live_trader', *sys.argv[1:]])
    p = argparse.ArgumentParser(description='Upbit auto-trader: buy/stay/sell every tick from cached rankings')
    p.add_argument('--interval', choices=['1m', '1h', '1d'], default='1h', help='Position expiry timeframe; all three frames always participate in decisions')
    p.add_argument('--mode', choices=['paper', 'test', 'live'], default='paper')
    p.add_argument('--confirm-live', action='store_true', help='Required in addition to --mode live to place real orders')
    p.add_argument('--tick-seconds', type=int, default=10)
    p.add_argument('--refresh-seconds', type=int, default=300, help='Legacy option; resident forecasts now refresh on completed candles')
    p.add_argument('--max-positions', type=int, default=3)
    p.add_argument('--max-krw', type=float, default=float(os.environ.get('UPBIT_MAX_ORDER_KRW', 10000)))
    p.add_argument('--paper-capital', type=float, default=1_000_000.)
    p.add_argument('--entry-threshold', type=float, default=ENTRY_THRESHOLD, help='Minimum expected net return to enter (0 = any post-cost-positive edge)')
    p.add_argument('--partial-entry-threshold', type=float, default=0.01, help='Minimum hourly-only expected return required for a partial-conviction (missing 1m/1d confirmation) entry; kept much stricter than --entry-threshold since that path historically has a materially lower win rate')
    p.add_argument('--universe-top', type=int, default=12, help='How many top-24h-volume KRW markets to rank each refresh (max 30)')
    p.add_argument('--retrain-seconds', type=int, default=86400, help='Legacy option; backbone retraining is offline while resident inference runs')
    p.add_argument('--no-daily-retrain', action='store_true', help='Compatibility option; concurrent backbone retraining is disabled for resident mode')
    p.add_argument('--analog-gate', action='store_true', help='Require historical-analog backtest confirmation before every BUY (off by default; needs days of accumulated forecast_history.jsonl to have enough same-symbol pairs)')
    p.add_argument('--analog-min-n', type=int, default=15, help='Minimum historical analogs required to trust the analog signal')
    p.add_argument('--analog-min-win-rate', type=float, default=0.55, help='Minimum win rate among historical analogs to confirm a BUY')
    p.add_argument('--analog-k', type=int, default=30, help='How many nearest historical analogs to average')
    p.add_argument('--analog-max-uncertainty', type=float, default=None, help='Block a BUY when TimesFM native quantile spread ((q90-q10)/anchor at horizon) exceeds this; unset (default) leaves forecast_uncertainty purely observational, independent of --analog-gate')
    args = p.parse_args()
    if args.mode == 'live' and not args.confirm_live:
        p.error('실거래 주문을 실행하려면 --mode live 와 함께 --confirm-live 를 명시해야 합니다.')
    cfg = LiveConfig(interval=args.interval, mode=args.mode, tick_seconds=args.tick_seconds,
                      refresh_seconds=args.refresh_seconds, max_positions=args.max_positions,
                      max_krw_per_trade=args.max_krw, paper_capital=args.paper_capital,
                      entry_threshold=args.entry_threshold, partial_entry_threshold=args.partial_entry_threshold,
                      universe_top=args.universe_top,
                      retrain_seconds=args.retrain_seconds, analog_gate_enabled=args.analog_gate,
                      analog_min_n=args.analog_min_n, analog_min_win_rate=args.analog_min_win_rate,
                      analog_k=args.analog_k, analog_max_uncertainty=args.analog_max_uncertainty)
    trader = LiveTrader(cfg)
    trader.start(retrain=not args.no_daily_retrain)
    print(f'live-trader started: mode={cfg.mode} interval={cfg.interval} tick={cfg.tick_seconds}s refresh={cfg.refresh_seconds}s')
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        trader.stop()


if __name__ == '__main__':
    main()

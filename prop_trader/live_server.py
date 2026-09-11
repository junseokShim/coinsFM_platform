"""Local HTTP server: real-time Korean/Upbit-styled dashboard + read-only balance/trader APIs.

Binds to 127.0.0.1 by default. Upbit keys never leave this process -- the browser only ever
receives already-fetched JSON, never the Authorization header or the keys themselves.
"""
import argparse
import csv
import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .live_dashboard import PAGE
from .live_trader import LiveTrader, LiveConfig, account_balance, RANKINGS_DIR
from .upbit_broker import BrokerError
from .upbit_symbols import krw_markets, annotate

FORECAST_ROOT = Path('runs/forecast5')
LIVE_DIR = Path('data/live')
INTERVALS = ('1m', '1h', '1d')
SIGNAL_THRESHOLD = {'1m': 0.001, '1h': 0.005, '1d': 0.005}


def _load_history(csv_path, symbol, n=60):
    rows = []
    if not csv_path.exists():
        return rows
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row['symbol'] == symbol:
                rows.append(dict(t=row['timestamp'], o=float(row['open']), h=float(row['high']),
                                  l=float(row['low']), c=float(row['close'])))
    return rows[-n:]


def build_realtime_payload(trader):
    from .execution_rules import reprice
    import time
    worker = trader.runtime.forecasts()
    feed = trader.runtime.stream.snapshot()
    forecasts, rankings = {}, {}
    for iv in INTERVALS:
        entries, rows = [], []
        for market, group in worker.get('records', {}).items():
            rec = group.get(iv)
            if not rec:
                continue
            current = feed['tickers'].get(market, {}).get('trade_price', rec['last_close'])
            returns = [p/current-1 for p in rec['forecast_prices']]
            entries.append(dict(symbol=market, upbit_market=market, last_close=current,
                as_of=rec['as_of'], forecast=rec['forecast'], returns=returns, signal='HOLD',
                history=rec.get('history', []), consistency=rec['forecast_consistency']))
            try:
                r = reprice(rec, feed['books'][market], iv, time.time(), trader.cfg.max_krw_per_trade)
                r.update(notional_krw=trader.cfg.max_krw_per_trade,
                         expected_pnl_krw=trader.cfg.max_krw_per_trade*r['expected_net_return'])
                rows.append(r)
            except (KeyError, ValueError, TypeError):
                continue
        rows.sort(key=lambda r: -r['expected_net_return'])
        for i,r in enumerate(rows, 1): r['rank'] = i
        forecasts[iv] = dict(entries=entries, excluded=[])
        rankings[iv] = dict(rankings=rows, generated_at=datetime.now(timezone.utc).isoformat())
    return dict(forecasts=forecasts, rankings=rankings, generated_at=datetime.now(timezone.utc).isoformat())


def build_dashboard_payload(trader=None):
    if trader is not None and trader.runtime is not None:
        return build_realtime_payload(trader)
    from .forecast_time import decide_signal
    try:
        markets = krw_markets()
    except BrokerError:
        markets = set()
    forecasts, rankings = {}, {}
    for interval in INTERVALS:
        preds_path = FORECAST_ROOT / interval / 'recent_predictions.json'
        entries = []
        if preds_path.exists():
            preds = json.loads(preds_path.read_text())
            csv_path = LIVE_DIR / f'{interval}.csv'
            for rec in preds:
                stats = dict(return_by_step=[rec[f'return_t{i}'] for i in range(1, 6)],
                             max_predicted_return=rec['max_predicted_return'],
                             min_predicted_return=rec['min_predicted_return'],
                             forecast_consistency=rec['forecast_consistency'])
                signal = decide_signal(stats, threshold=SIGNAL_THRESHOLD[interval], min_consistency=0.5)
                entries.append(dict(symbol=rec['symbol'], as_of=rec['as_of'], last_close=rec['last_close'],
                                     forecast=rec['forecast'], returns=stats['return_by_step'],
                                     max_return=rec['max_predicted_return'], min_return=rec['min_predicted_return'],
                                     consistency=rec['forecast_consistency'], signal=signal,
                                     history=_load_history(csv_path, rec['symbol'])))
            annotate(entries, markets)
        tradable = [e for e in entries if e['upbit_tradable']]
        excluded = [e['symbol'] for e in entries if not e['upbit_tradable']]
        tradable.sort(key=lambda e: (-e['returns'][-1], e['symbol']))
        forecasts[interval] = dict(threshold=SIGNAL_THRESHOLD[interval], entries=tradable, excluded=excluded)
        rank_path = RANKINGS_DIR / f'{interval}.json'
        if rank_path.exists():
            try:
                rankings[interval] = json.loads(rank_path.read_text())
            except json.JSONDecodeError:
                pass
    return dict(generated_at=datetime.now(timezone.utc).isoformat(), forecasts=forecasts, rankings=rankings)


def tail_jsonl(path, n=50):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines()[-n:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    out.reverse()
    return out


def read_equity(path, n=500):
    if not path.exists():
        return []
    rows = list(csv.DictReader(path.open()))[-n:]
    return [dict(t=r['timestamp'], equity=float(r['equity_krw']), return_pct=float(r['return_pct'])) for r in rows]


def make_handler(trader):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'PropTraderLive/1.0'

        def log_message(self, fmt, *args):
            pass

        def _json(self, obj, status=200):
            body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split('?', 1)[0]
            if path == '/':
                body = PAGE.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == '/api/dashboard':
                self._json(build_dashboard_payload(trader))
            elif path == '/api/balance':
                try:
                    self._json(account_balance(trader.client))
                except BrokerError as e:
                    self._json(dict(error=str(e)))
            elif path == '/api/equity':
                self._json(dict(points=read_equity(trader.equity_path), baseline=trader.baseline_equity, mode=trader.cfg.mode))
            elif path == '/api/trader':
                snap = trader.snapshot()
                snap['decisions'] = tail_jsonl(trader.decisions_path, 50)
                self._json(snap)
            elif path == '/api/health':
                self._json(dict(ok=True, time=datetime.now(timezone.utc).isoformat()))
            else:
                self._json(dict(error='not found'), status=404)

        def do_POST(self):
            path = self.path.split('?', 1)[0]
            if path == '/api/refresh':
                self._json(dict(requested=trader.request_refresh()))
            else:
                self._json(dict(error='not found'), status=404)
    return Handler


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
        os.execv(str(python), [str(python), '-m', 'prop_trader.live_server', *sys.argv[1:]])
    p = argparse.ArgumentParser(description='Local real-time dashboard + auto-trader server (binds 127.0.0.1 by default)')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8899)
    p.add_argument('--interval', choices=['1m', '1h', '1d'], default='1h', help='Position expiry timeframe; all three frames always participate in decisions')
    p.add_argument('--mode', choices=['paper', 'test', 'live'], default='paper')
    p.add_argument('--confirm-live', action='store_true', help='Required in addition to --mode live to place real orders')
    p.add_argument('--tick-seconds', type=int, default=10)
    p.add_argument('--refresh-seconds', type=int, default=300, help='Legacy option; resident forecasts now refresh on completed candles')
    p.add_argument('--max-positions', type=int, default=3)
    p.add_argument('--max-krw', type=float, default=float(os.environ.get('UPBIT_MAX_ORDER_KRW', 10000)))
    p.add_argument('--paper-capital', type=float, default=1_000_000.)
    p.add_argument('--entry-threshold', type=float, default=0.002, help='Minimum expected net return to enter (0 = any post-cost-positive edge)')
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
    if args.host not in ('127.0.0.1', 'localhost'):
        print('경고: 인증 없이 잔고/자동매매 상태를 노출합니다. 로컬 호스트 외 바인딩은 권장하지 않습니다.', flush=True)
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
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(trader))
    print(f'http://{args.host}:{args.port} 에서 대시보드 실행 중 (mode={cfg.mode}, interval={cfg.interval})', flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        trader.stop()
        httpd.shutdown()


if __name__ == '__main__':
    main()

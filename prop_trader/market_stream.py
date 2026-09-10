"""Public Upbit WebSocket cache. Trading never consumes disconnected/stale quotes."""
import copy
import json
import math
import threading
import time
import uuid


class MarketStream:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.changed = threading.Event()
        self.codes = ((), ())
        self.tickers, self.books = {}, {}
        self.connected = False
        self.error = None
        self.socket = None

    def subscribe(self, markets, books):
        codes = (tuple(sorted(markets)), tuple(sorted(books)))
        with self.lock:
            if codes != self.codes:
                self.codes = codes
                self.changed.set()

    def accept(self, message, received=None):
        now = time.time() if received is None else received
        kind, market = message.get('type'), message.get('code')
        if kind not in ('ticker', 'orderbook') or not market:
            return
        stamp = float(message.get('timestamp', 0))/1000
        if not math.isfinite(stamp) or not -2 <= now-stamp <= 10:
            return
        try:
            if kind == 'ticker':
                values = [float(message['trade_price'])]
            else:
                values = [float(u[k]) for u in message['orderbook_units'] for k in ('ask_price','bid_price')]
                if not values:
                    return
            if not all(math.isfinite(v) and v > 0 for v in values):
                return
        except (KeyError, ValueError, TypeError):
            return
        with self.lock:
            target = self.tickers if kind == 'ticker' else self.books
            old = target.get(market)
            if old and old[1]['timestamp'] > message['timestamp']:
                return
            target[market] = (now, dict(message, market=market))

    def snapshot(self, max_age=10, now=None):
        now = time.time() if now is None else now
        with self.lock:
            def fresh(cache):
                return {m: copy.deepcopy(row) for m,(received,row) in cache.items()
                        if self.connected and 0 <= now-received <= max_age
                        and -2 <= now-float(row['timestamp'])/1000 <= max_age}
            return dict(connected=self.connected, error=self.error,
                        tickers=fresh(self.tickers), books=fresh(self.books))

    def _run(self):
        import websocket
        delay = 1
        while not self.stop_event.is_set():
            if not self.codes[0]:
                self.stop_event.wait(.2)
                continue
            ws = None
            try:
                ws = websocket.create_connection('wss://api.upbit.com/websocket/v1', timeout=2)
                self.socket = ws
                with self.lock:
                    self.tickers.clear(); self.books.clear()
                    self.connected = True
                    self.error = None
                self.changed.set()
                last_ping = time.monotonic()
                while not self.stop_event.is_set():
                    if self.changed.is_set():
                        with self.lock:
                            tickers, books = self.codes
                            self.changed.clear()
                        request = [dict(ticket=str(uuid.uuid4())), dict(type='ticker',codes=list(tickers))]
                        if books:
                            request.append(dict(type='orderbook',codes=list(books)))
                        ws.send(json.dumps(request))
                    if time.monotonic()-last_ping > 20:
                        ws.ping(); last_ping = time.monotonic()
                    try:
                        raw = ws.recv()
                    except websocket.WebSocketTimeoutException:
                        continue
                    if not raw:
                        raise ConnectionError('Disconnected')
                    message = json.loads(raw)
                    if 'error' in message:
                        raise ValueError('Subscription rejected')
                    self.accept(message)
                    delay = 1
            except Exception:
                self.error = 'WebSocket 연결 대기 · 신규 진입 보류'
            finally:
                with self.lock:
                    self.connected = False
                if ws:
                    ws.close()
                self.socket = None
            self.stop_event.wait(delay)
            delay = min(delay*2, 30)

    def start(self):
        import websocket  # dependency failure is explicit before any worker starts
        self.thread = threading.Thread(target=self._run, daemon=True, name='upbit-websocket')
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.socket:
            self.socket.close()
        if hasattr(self, 'thread'):
            self.thread.join(timeout=3)

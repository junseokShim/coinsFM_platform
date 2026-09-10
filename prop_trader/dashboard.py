"""Self-contained web dashboard: signal board + forecast explorer + evaluation +
backtest ledger, in one static HTML file.

Pure-stdlib generator, same convention as forecast_report.py: reads the already
-produced runs/forecast5/{interval}/{recent_predictions.json,eval_full.json,
backtest/*/summary.json} and data/live/{interval}.csv, and writes one HTML file
with the data embedded inline. No server, no external network calls at view
time, no model inference here -- rerun the forecast/evaluator/backtest_forecast
pipeline first, then this just renders what's already on disk.
"""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from .forecast_time import decide_signal

INTERVALS = ('1m', '1h', '1d')
MODEL_ORDER = ['naive', 'momentum', 'zero_shot', 'lora', 'lora_fewshot']
MODEL_LABEL = dict(naive='Naive', momentum='Momentum', zero_shot='Zero-shot',
                    lora='LoRA', lora_fewshot='LoRA + Recent')
# Matches the thresholds already validated in BACKTEST_REPORT.md: 1m moves are
# far smaller than 1h/1d, so a single global threshold would never fire on 1m.
SIGNAL_THRESHOLD = {'1m': 0.001, '1h': 0.005, '1d': 0.005}
MIN_CONSISTENCY = 0.5


def load_history(csv_path, symbol, n=60):
    rows = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row['symbol'] == symbol:
                rows.append(dict(t=row['timestamp'], o=float(row['open']), h=float(row['high']),
                                  l=float(row['low']), c=float(row['close'])))
    return rows[-n:]


def build_payload(root=Path('runs/forecast5'), live_dir=Path('data/live')):
    forecasts, evaluations, backtests = {}, {}, {}
    for interval in INTERVALS:
        run_dir = root / interval
        preds = json.loads((run_dir / 'recent_predictions.json').read_text())
        csv_path = live_dir / f'{interval}.csv'
        entries = []
        for rec in preds:
            stats = dict(return_by_step=[rec[f'return_t{i}'] for i in range(1, 6)],
                         max_predicted_return=rec['max_predicted_return'],
                         min_predicted_return=rec['min_predicted_return'],
                         forecast_consistency=rec['forecast_consistency'])
            signal = decide_signal(stats, threshold=SIGNAL_THRESHOLD[interval], min_consistency=MIN_CONSISTENCY)
            entries.append(dict(symbol=rec['symbol'], as_of=rec['as_of'], generated_at=rec['generated_at'],
                                 last_close=rec['last_close'], forecast=rec['forecast'],
                                 returns=stats['return_by_step'], max_return=rec['max_predicted_return'],
                                 min_return=rec['min_predicted_return'], consistency=rec['forecast_consistency'],
                                 slope=rec['forecast_slope'], signal=signal,
                                 history=load_history(csv_path, rec['symbol'])))
        entries.sort(key=lambda e: (-e['returns'][-1], e['symbol']))
        forecasts[interval] = dict(threshold=SIGNAL_THRESHOLD[interval], min_consistency=MIN_CONSISTENCY, entries=entries)

        eval_full = json.loads((run_dir / 'eval_full.json').read_text())
        evaluations[interval] = eval_full['models']

        bt = {}
        for model in MODEL_ORDER:
            summary_path = run_dir / 'backtest' / model / 'summary.json'
            if summary_path.exists():
                bt[model] = json.loads(summary_path.read_text())
        backtests[interval] = bt
    rankings={}
    for interval in INTERVALS:
        path=Path('runs/rankings')/f'{interval}.json'
        if path.exists():rankings[interval]=json.loads(path.read_text())
    return dict(generated_at=datetime.now(timezone.utc).isoformat(), forecasts=forecasts,rankings=rankings,
                evaluations=evaluations, backtests=backtests, model_order=MODEL_ORDER, model_label=MODEL_LABEL)


PAGE = r"""<title>TimesFM Forecast Desk</title>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet" />
<style>
  :root {
    color-scheme: light;
    --bg: #f4f5f7;
    --surface: #ffffff;
    --surface-2: #eef0f3;
    --border: rgba(16,19,26,0.10);
    --text: #10131a;
    --text-2: #52586a;
    --text-3: #878d9c;
    --accent: #2a78d6;
    --accent-soft: rgba(42,120,214,0.12);
    --good: #0ca30c;
    --good-soft: rgba(12,163,12,0.12);
    --critical: #d03b3b;
    --critical-soft: rgba(208,59,59,0.12);
    --warn: #b8790a;
    --warn-soft: rgba(250,178,25,0.18);
    --chip-bg: #eceef2;
    --shadow: 0 1px 2px rgba(16,19,26,0.06), 0 8px 24px -12px rgba(16,19,26,0.18);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --bg: #0b0d11;
      --surface: #12151b;
      --surface-2: #171b22;
      --border: rgba(255,255,255,0.09);
      --text: #eef1f5;
      --text-2: #a7b0bd;
      --text-3: #6b7280;
      --accent: #4f97ea;
      --accent-soft: rgba(79,151,234,0.16);
      --good: #0ca30c;
      --good-soft: rgba(12,163,12,0.16);
      --critical: #d03b3b;
      --critical-soft: rgba(208,59,59,0.16);
      --warn: #fab219;
      --warn-soft: rgba(250,178,25,0.14);
      --chip-bg: #1c212a;
      --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 12px 32px -16px rgba(0,0,0,0.6);
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --bg: #0b0d11;
    --surface: #12151b;
    --surface-2: #171b22;
    --border: rgba(255,255,255,0.09);
    --text: #eef1f5;
    --text-2: #a7b0bd;
    --text-3: #6b7280;
    --accent: #4f97ea;
    --accent-soft: rgba(79,151,234,0.16);
    --good: #0ca30c;
    --good-soft: rgba(12,163,12,0.16);
    --critical: #d03b3b;
    --critical-soft: rgba(208,59,59,0.16);
    --warn: #fab219;
    --warn-soft: rgba(250,178,25,0.14);
    --chip-bg: #1c212a;
    --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 12px 32px -16px rgba(0,0,0,0.6);
  }

  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font: 400 14px/1.5 "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .mono { font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; font-variant-numeric: tabular-nums; }
  a { color: var(--accent); }

  .shell { max-width: 1180px; margin: 0 auto; padding: 28px 20px 64px; }

  .topbar {
    display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between;
    gap: 12px 24px; padding-bottom: 20px; margin-bottom: 24px; border-bottom: 1px solid var(--border);
  }
  .brand { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
  .brand h1 { font-size: 20px; font-weight: 700; margin: 0; letter-spacing: -0.01em; }
  .brand p { margin: 0; color: var(--text-2); font-size: 13px; }
  .badge {
    display: inline-flex; align-items: center; gap: 6px; font-family: "IBM Plex Mono", monospace;
    font-size: 11px; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
    padding: 5px 10px; border-radius: 999px; background: var(--warn-soft); color: var(--warn); white-space: nowrap;
  }
  .badge::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
  .meta { text-align: right; color: var(--text-3); font-size: 12px; }
  .meta .mono { color: var(--text-2); }

  .controls { display: flex; align-items: center; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }
  .segmented { display: inline-flex; background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px; padding: 3px; gap: 2px; }
  .segmented button {
    font: 600 13px/1 "IBM Plex Mono", monospace; padding: 8px 16px; border-radius: 8px; border: none;
    background: transparent; color: var(--text-2); cursor: pointer; transition: background .15s, color .15s;
  }
  .segmented button:hover { color: var(--text); }
  .segmented button[aria-pressed="true"] { background: var(--accent); color: #fff; }
  .controls .hint { color: var(--text-3); font-size: 12px; }

  section { margin-bottom: 36px; }
  .section-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; }
  .section-head h2 { font-size: 15px; font-weight: 700; margin: 0; }
  .section-head .sub { color: var(--text-3); font-size: 12.5px; }

  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; box-shadow: var(--shadow); }

  .board { display: grid; grid-template-columns: repeat(auto-fill, minmax(178px, 1fr)); gap: 10px; }
  .sigcard {
    text-align: left; cursor: pointer; padding: 14px; display: flex; flex-direction: column; gap: 8px;
    border: 1px solid var(--border); background: var(--surface);
  }
  .sigcard:hover { border-color: var(--accent); }
  .sigcard[aria-pressed="true"] { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent) inset; }
  .sigcard .row1 { display: flex; align-items: center; justify-content: space-between; }
  .sigcard .sym { font-weight: 700; font-size: 13.5px; letter-spacing: -0.01em; }
  .pill {
    font-family: "IBM Plex Mono", monospace; font-size: 10.5px; font-weight: 700; letter-spacing: 0.03em;
    padding: 3px 7px; border-radius: 6px;
  }
  .pill.LONG { background: var(--good-soft); color: var(--good); }
  .pill.SHORT { background: var(--critical-soft); color: var(--critical); }
  .pill.HOLD { background: var(--chip-bg); color: var(--text-3); }
  .sigcard .price { font-family: "IBM Plex Mono", monospace; font-size: 15px; font-weight: 500; }
  .sigcard .ret { font-family: "IBM Plex Mono", monospace; font-size: 12.5px; }
  .up { color: var(--good); } .down { color: var(--critical); } .flat { color: var(--text-3); }
  .spark { width: 100%; height: 28px; display: block; }

  .detail { display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(220px, 0.9fr); gap: 16px; }
  @media (max-width: 780px) { .detail { grid-template-columns: 1fr; } }
  .detail .panel { padding: 18px; }
  .detail-head { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px; gap: 10px; }
  .detail-head h3 { margin: 0; font-size: 16px; }
  .detail-head .as-of { color: var(--text-3); font-size: 11.5px; }
  .chartwrap { position: relative; }
  svg.chart { width: 100%; height: 300px; display: block; overflow: visible; }
  .tooltip {
    position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px 10px; font-size: 11.5px; box-shadow: var(--shadow); opacity: 0;
    transition: opacity .1s; white-space: nowrap; z-index: 5;
  }
  .tooltip .t1 { color: var(--text-3); margin-bottom: 2px; }
  .tooltip .t2 { font-family: "IBM Plex Mono", monospace; font-weight: 600; }
  .legend { display: flex; gap: 16px; margin-top: 6px; font-size: 11.5px; color: var(--text-2); }
  .legend span { display: inline-flex; align-items: center; gap: 5px; }
  .legend i { width: 12px; height: 2px; display: inline-block; background: var(--text-2); }
  .legend i.fc { background: var(--accent); border-top: 2px dashed var(--accent); height: 0; }

  .statgrid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 14px; margin-top: 14px; }
  .stat .k { color: var(--text-3); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }
  .stat .v { font-family: "IBM Plex Mono", monospace; font-size: 14px; font-weight: 600; margin-top: 2px; }

  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  th, td { text-align: right; padding: 9px 12px; white-space: nowrap; }
  th:first-child, td:first-child { text-align: left; }
  thead th { color: var(--text-3); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.03em; border-bottom: 1px solid var(--border); }
  tbody td { font-family: "IBM Plex Mono", monospace; border-bottom: 1px solid var(--border); }
  tbody td:first-child { font-family: "IBM Plex Sans", sans-serif; font-weight: 600; }
  tbody tr:last-child td { border-bottom: none; }
  tbody tr:hover td { background: var(--surface-2); }
  .tablewrap { overflow-x: auto; }
  .bar-cell { display: flex; align-items: center; gap: 8px; justify-content: flex-end; }
  .bar-track { width: 64px; height: 6px; border-radius: 3px; background: var(--chip-bg); overflow: hidden; }
  .bar-fill { height: 100%; background: var(--accent); border-radius: 3px; }
  .na { color: var(--text-3); }

  .forecast-table { margin-top: 14px; }

  footer { border-top: 1px solid var(--border); padding-top: 18px; color: var(--text-3); font-size: 12px; line-height: 1.7; }
  footer code { font-family: "IBM Plex Mono", monospace; background: var(--surface-2); padding: 1px 5px; border-radius: 4px; }
  footer .repro { margin-top: 8px; background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; font-family: "IBM Plex Mono", monospace; font-size: 11.5px; overflow-x: auto; }
  footer .repro div { white-space: pre; }
</style>

<div class="shell">
  <div class="topbar">
    <div class="brand">
      <h1>TimesFM Forecast Desk</h1>
      <p>Historical LoRA + recent few-shot &rarr; next 5-candle close forecast</p>
    </div>
    <div style="display:flex; align-items:center; gap:14px;">
      <span class="badge">Paper only &middot; no live orders</span>
      <span class="meta">generated<br /><span class="mono" id="generatedAt"></span></span>
    </div>
  </div>

  <div class="controls">
    <div class="segmented" id="intervalSwitch" role="tablist" aria-label="Timeframe"></div>
    <span class="hint" id="thresholdHint"></span>
  </div>

  <section id="board-section">
    <div class="section-head">
      <h2>Signal board</h2>
      <span class="sub">예상 t+5 수익률 높은 순 · 연구 신호 · click a symbol for detail</span>
    </div>
    <div class="board" id="board"></div>
  </section>

  <section>
    <div class="section-head"><h2>업비트 예상 순수익률 순위</h2><span class="sub">TimesFM + 기술적 지표 · 동일 주기 5개 봉 · KRW 현물</span></div>
    <p id="rankMeta" class="sub"></p>
    <table><thead><tr><th>순위</th><th>종목</th><th>예상 순수익률</th><th>가정 투자금</th><th>예상 손익</th><th>상태</th></tr></thead><tbody id="rankRows"></tbody></table>
    <p class="sub">모델 추정이며 수익 보장이 아닙니다. 비용 가정과 현재 호가를 반영하지만 업비트 이전 성능은 미검증입니다. 이 정적 화면에서 실제 주문은 전송하지 않습니다.</p>
  </section>

  <section id="detail-section">
    <div class="section-head">
      <h2>Forecast explorer</h2>
      <span class="sub">Solid = observed close &middot; dashed = predicted close (close only, no synthetic future OHLC)</span>
    </div>
    <div class="detail">
      <div class="card panel">
        <div class="detail-head">
          <h3 id="detailSymbol">&mdash;</h3>
          <span class="as-of mono" id="detailAsOf"></span>
        </div>
        <div class="chartwrap">
          <svg class="chart" id="chart" viewBox="0 0 900 300" preserveAspectRatio="none"></svg>
          <div class="tooltip" id="tooltip"><div class="t1" id="ttT1"></div><div class="t2" id="ttT2"></div></div>
        </div>
        <div class="legend">
          <span><i></i> Observed close</span>
          <span><i class="fc"></i> Predicted close (t+1..t+5)</span>
        </div>
      </div>
      <div class="card panel">
        <div id="detailSignal"></div>
        <div class="statgrid" id="detailStats"></div>
        <div class="tablewrap forecast-table">
          <table>
            <thead><tr><th>Step</th><th>Close</th><th>Return</th></tr></thead>
            <tbody id="forecastRows"></tbody>
          </table>
        </div>
      </div>
    </div>
  </section>

  <section id="eval-section">
    <div class="section-head">
      <h2>Forecast accuracy</h2>
      <span class="sub">Walk-forward test split, 96 windows / timeframe, 6 symbols &middot; DA = direction accuracy</span>
    </div>
    <div class="card tablewrap">
      <table>
        <thead><tr>
          <th>Model</th><th>MAE</th><th>RMSE</th><th>MAPE t+5</th><th>DA@1</th><th>DA@5</th><th>Top-K precision</th>
        </tr></thead>
        <tbody id="evalRows"></tbody>
      </table>
    </div>
  </section>

  <section id="backtest-section">
    <div class="section-head">
      <h2>Signal backtest</h2>
      <span class="sub" id="backtestSub"></span>
    </div>
    <div class="card tablewrap">
      <table>
        <thead><tr>
          <th>Model</th><th>Trades</th><th>Win rate</th><th>Avg net / trade</th><th>Total return</th><th>Max DD</th><th>Buy&amp;hold avg</th>
        </tr></thead>
        <tbody id="btRows"></tbody>
      </table>
    </div>
  </section>

  <footer>
    Close-only forecasts (no synthetic OHLC), 96-window / 6-symbol walk-forward pilot &mdash; not a
    generalization or profitability claim. Backtest ignores intrabar stop-loss/take-profit (only
    close-path forecasts available), uses fixed notional with no compounding or position limits, and
    applies a 0.4% round-trip cost. Signal thresholds are fixed per timeframe, not optimized.
    See <code>prop_trader/FORECASTING.md</code> for full methodology and limitations.
    <div class="repro"><div>python3 -m prop_trader.current_candles --out data/live
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 .venv-timesfm/bin/python -m prop_trader.timesfm_predict --run runs/forecast5/1h --csv data/live/1h.csv
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 .venv-timesfm/bin/python -m prop_trader.evaluator --csv-dir data/live
python3 -m prop_trader.backtest_forecast --threshold 0.005 --interval-threshold 1m=0.001
python3 -m prop_trader.dashboard</div></div>
  </footer>
</div>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
(function () {
  var DATA = JSON.parse(document.getElementById('payload').textContent);
  var INTERVALS = ['1m', '1h', '1d'];
  var state = { interval: '1h', symbol: null };

  document.getElementById('generatedAt').textContent = DATA.generated_at.replace('T', ' ').slice(0, 19) + ' UTC';

  function fmtPct(v, digits) {
    if (v === null || v === undefined) return '<span class="na">N/A</span>';
    digits = digits === undefined ? 2 : digits;
    var cls = v > 0.0001 ? 'up' : v < -0.0001 ? 'down' : 'flat';
    var sign = v > 0 ? '+' : '';
    return '<span class="' + cls + '">' + sign + (v * 100).toFixed(digits) + '%</span>';
  }
  function fmtPlainPct(v, digits) {
    if (v === null || v === undefined) return 'N/A';
    digits = digits === undefined ? 2 : digits;
    var sign = v > 0 ? '+' : '';
    return sign + (v * 100).toFixed(digits) + '%';
  }
  function fmtPrice(v) {
    if (v === undefined || v === null) return '';
    var abs = Math.abs(v);
    var digits = abs >= 100 ? 2 : abs >= 1 ? 4 : abs >= 0.01 ? 6 : 9;
    return v.toFixed(digits);
  }
  function fmtDate(iso) {
    return iso ? iso.replace('T', ' ').slice(0, 16) + 'Z' : '';
  }

  function buildIntervalSwitch() {
    var el = document.getElementById('intervalSwitch');
    el.innerHTML = '';
    INTERVALS.forEach(function (iv) {
      var b = document.createElement('button');
      b.textContent = iv;
      b.setAttribute('role', 'tab');
      b.setAttribute('aria-pressed', iv === state.interval ? 'true' : 'false');
      b.onclick = function () { state.interval = iv; state.symbol = null; render(); };
      el.appendChild(b);
    });
  }

  function sparkPath(history, w, h) {
    if (!history.length) return '';
    var closes = history.map(function (b) { return b.c; });
    var lo = Math.min.apply(null, closes), hi = Math.max.apply(null, closes);
    var span = Math.max(hi - lo, hi * 0.0005);
    var pts = closes.map(function (c, i) {
      var x = (i / (closes.length - 1)) * w;
      var y = h - ((c - lo) / span) * h;
      return x.toFixed(1) + ',' + y.toFixed(1);
    });
    return pts.join(' ');
  }

  function buildBoard() {
    var board = document.getElementById('board');
    board.innerHTML = '';
    var fc = DATA.forecasts[state.interval];
    fc.entries.forEach(function (e) {
      if (!state.symbol) state.symbol = e.symbol;
      var t5 = e.returns[4];
      var card = document.createElement('button');
      card.className = 'sigcard';
      card.setAttribute('aria-pressed', e.symbol === state.symbol ? 'true' : 'false');
      var pts = sparkPath(e.history, 160, 26);
      card.innerHTML =
        '<div class="row1"><span class="sym">' + e.symbol.replace('USDT', '') + '</span>' +
        '<span class="pill ' + e.signal + '">' + e.signal + '</span></div>' +
        '<svg class="spark" viewBox="0 0 160 26" preserveAspectRatio="none">' +
        '<polyline points="' + pts + '" fill="none" stroke="var(--text-3)" stroke-width="1.5" /></svg>' +
        '<div class="price mono">' + fmtPrice(e.last_close) + '</div>' +
        '<div class="ret">t+5 ' + fmtPct(t5) + '</div>';
      card.onclick = function () { state.symbol = e.symbol; renderDetail(); highlightBoard(); };
      board.appendChild(card);
    });
  }

  function highlightBoard() {
    var cards = document.querySelectorAll('#board .sigcard');
    var fc = DATA.forecasts[state.interval];
    cards.forEach(function (c, i) {
      c.setAttribute('aria-pressed', fc.entries[i].symbol === state.symbol ? 'true' : 'false');
    });
  }

  function drawChart(entry) {
    var svg = document.getElementById('chart');
    var W = 900, H = 300, padL = 8, padR = 8, padT = 14, padB = 22;
    var hist = entry.history;
    var closes = hist.map(function (b) { return b.c; });
    var fcCloses = entry.forecast.map(function (p) { return p.predicted_close; });
    var all = closes.concat(fcCloses);
    var lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
    var span = Math.max(hi - lo, hi * 0.0008);
    var n = hist.length + entry.forecast.length;
    var stepX = (W - padL - padR) / (n - 1);
    function y(v) { return H - padB - ((v - lo) / span) * (H - padT - padB); }
    function x(i) { return padL + i * stepX; }

    var svgns = 'http://www.w3.org/2000/svg';
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    for (var g = 0; g < 4; g++) {
      var gv = lo + (span * g) / 3;
      var line = document.createElementNS(svgns, 'line');
      line.setAttribute('x1', padL); line.setAttribute('x2', W - padR);
      line.setAttribute('y1', y(gv)); line.setAttribute('y2', y(gv));
      line.setAttribute('stroke', 'var(--border)'); line.setAttribute('stroke-width', '1');
      svg.appendChild(line);
    }

    var histPts = hist.map(function (b, i) { return x(i) + ',' + y(b.c).toFixed(1); }).join(' ');
    var histLine = document.createElementNS(svgns, 'polyline');
    histLine.setAttribute('points', histPts);
    histLine.setAttribute('fill', 'none'); histLine.setAttribute('stroke', 'var(--text-2)');
    histLine.setAttribute('stroke-width', '1.75');
    svg.appendChild(histLine);

    var fcPts = [x(hist.length - 1) + ',' + y(hist[hist.length - 1].c).toFixed(1)];
    entry.forecast.forEach(function (p, i) { fcPts.push(x(hist.length + i) + ',' + y(p.predicted_close).toFixed(1)); });
    var fcLine = document.createElementNS(svgns, 'polyline');
    fcLine.setAttribute('points', fcPts.join(' '));
    fcLine.setAttribute('fill', 'none'); fcLine.setAttribute('stroke', 'var(--accent)');
    fcLine.setAttribute('stroke-width', '2'); fcLine.setAttribute('stroke-dasharray', '5 4');
    svg.appendChild(fcLine);

    var boundary = document.createElementNS(svgns, 'line');
    boundary.setAttribute('x1', x(hist.length - 1)); boundary.setAttribute('x2', x(hist.length - 1));
    boundary.setAttribute('y1', padT); boundary.setAttribute('y2', H - padB);
    boundary.setAttribute('stroke', 'var(--border)'); boundary.setAttribute('stroke-width', '1');
    boundary.setAttribute('stroke-dasharray', '2 3');
    svg.appendChild(boundary);

    var tooltip = document.getElementById('tooltip');
    var t1 = document.getElementById('ttT1'), t2 = document.getElementById('ttT2');
    function showTip(evt, label, value) {
      t1.textContent = label; t2.textContent = value;
      var rect = svg.getBoundingClientRect();
      var wrap = svg.parentElement.getBoundingClientRect();
      tooltip.style.left = (evt.clientX - wrap.left + 12) + 'px';
      tooltip.style.top = (evt.clientY - wrap.top - 36) + 'px';
      tooltip.style.opacity = 1;
    }
    function hideTip() { tooltip.style.opacity = 0; }

    hist.forEach(function (b, i) {
      var c = document.createElementNS(svgns, 'circle');
      c.setAttribute('cx', x(i)); c.setAttribute('cy', y(b.c)); c.setAttribute('r', 7);
      c.setAttribute('fill', 'transparent');
      c.addEventListener('mousemove', function (evt) { showTip(evt, fmtDate(b.t), fmtPrice(b.c)); });
      c.addEventListener('mouseleave', hideTip);
      svg.appendChild(c);
    });
    entry.forecast.forEach(function (p, i) {
      var c = document.createElementNS(svgns, 'circle');
      c.setAttribute('cx', x(hist.length + i)); c.setAttribute('cy', y(p.predicted_close)); c.setAttribute('r', 4);
      c.setAttribute('fill', 'var(--accent)');
      var hit = document.createElementNS(svgns, 'circle');
      hit.setAttribute('cx', x(hist.length + i)); hit.setAttribute('cy', y(p.predicted_close)); hit.setAttribute('r', 9);
      hit.setAttribute('fill', 'transparent');
      hit.addEventListener('mousemove', function (evt) {
        showTip(evt, 'T+' + p.horizon + ' &middot; ' + fmtDate(p.timestamp), fmtPrice(p.predicted_close) + '  (' + fmtPlainPct(entry.returns[i]) + ')');
      });
      hit.addEventListener('mouseleave', hideTip);
      svg.appendChild(c); svg.appendChild(hit);
    });
  }

  function renderDetail() {
    var fc = DATA.forecasts[state.interval];
    var entry = fc.entries.filter(function (e) { return e.symbol === state.symbol; })[0];
    if (!entry) return;
    document.getElementById('detailSymbol').textContent = entry.symbol + ' · ' + state.interval;
    document.getElementById('detailAsOf').textContent = 'as-of ' + fmtDate(entry.as_of);
    drawChart(entry);

    var t5 = entry.returns[4];
    document.getElementById('detailSignal').innerHTML =
      '<div class="stat"><div class="k">Signal</div><div class="v"><span class="pill ' + entry.signal + '" style="font-size:12px;padding:5px 10px;">' + entry.signal + '</span></div></div>';

    document.getElementById('detailStats').innerHTML =
      '<div class="stat"><div class="k">Last close</div><div class="v">' + fmtPrice(entry.last_close) + '</div></div>' +
      '<div class="stat"><div class="k">t+5 return</div><div class="v">' + fmtPct(t5) + '</div></div>' +
      '<div class="stat"><div class="k">Path min / max</div><div class="v">' + fmtPlainPct(entry.min_return) + ' / ' + fmtPlainPct(entry.max_return) + '</div></div>' +
      '<div class="stat"><div class="k">Consistency</div><div class="v">' + Math.round(entry.consistency * 100) + '%</div></div>';

    var rows = entry.forecast.map(function (p, i) {
      return '<tr><td>T+' + p.horizon + '</td><td class="mono">' + fmtPrice(p.predicted_close) + '</td><td>' + fmtPct(entry.returns[i]) + '</td></tr>';
    }).join('');
    document.getElementById('forecastRows').innerHTML = rows;
  }

  function daBar(v) {
    if (v === null || v === undefined) return '<span class="na">N/A</span>';
    var pct = Math.max(0, Math.min(100, v * 100));
    return '<div class="bar-cell"><div class="bar-track"><div class="bar-fill" style="width:' + pct + '%"></div></div><span>' + pct.toFixed(1) + '%</span></div>';
  }

  function renderEval() {
    var models = DATA.evaluations[state.interval];
    var rows = DATA.model_order.map(function (m) {
      var r = models[m];
      if (!r) return '';
      var byH = {}; r.per_horizon.forEach(function (h) { byH[h.horizon] = h; });
      return '<tr><td>' + DATA.model_label[m] + '</td>' +
        '<td class="mono">' + r.mae.toFixed(6) + '</td>' +
        '<td class="mono">' + r.rmse.toFixed(6) + '</td>' +
        '<td class="mono">' + byH[5].mape_pct.toFixed(2) + '%</td>' +
        '<td>' + daBar(byH[1].direction_accuracy) + '</td>' +
        '<td>' + daBar(byH[5].direction_accuracy) + '</td>' +
        '<td>' + daBar(r.top_k_directional_precision) + '</td></tr>';
    }).join('');
    document.getElementById('evalRows').innerHTML = rows;
  }

  function renderBacktest() {
    var bt = DATA.backtests[state.interval];
    var threshold = DATA.forecasts[state.interval].threshold;
    document.getElementById('backtestSub').textContent =
      'Signal threshold ' + (threshold * 100).toFixed(2) + '% · fixed 1000 USDT/trade, no compounding, exit at actual t+5 close, 0.4% round-trip cost';
    var rows = DATA.model_order.map(function (m) {
      var s = bt[m];
      if (!s) return '';
      var winRate = s.win_rate === null ? '<span class="na">N/A</span>' : (s.win_rate * 100).toFixed(1) + '%';
      return '<tr><td>' + DATA.model_label[m] + '</td>' +
        '<td class="mono">' + s.trades + '</td>' +
        '<td class="mono">' + winRate + '</td>' +
        '<td>' + fmtPct(s.avg_net_return_pct === null ? null : s.avg_net_return_pct / 100) + '</td>' +
        '<td>' + fmtPct(s.total_return_pct / 100) + '</td>' +
        '<td class="down">' + s.max_drawdown_pct.toFixed(2) + '%</td>' +
        '<td class="mono">' + s.buy_and_hold.avg_return_pct.toFixed(3) + '%</td></tr>';
    }).join('');
    document.getElementById('btRows').innerHTML = rows;
  }

  function render() {
    buildIntervalSwitch();
    document.getElementById('thresholdHint').textContent =
      'signal threshold ' + (DATA.forecasts[state.interval].threshold * 100).toFixed(2) + '% · min path consistency ' + (DATA.forecasts[state.interval].min_consistency * 100).toFixed(0) + '%';
    buildBoard();
    renderDetail();
    renderEval();
    renderBacktest();
    var ranking = DATA.rankings && DATA.rankings[state.interval];
    document.getElementById('rankMeta').textContent = ranking ? '생성 시각 ' + ranking.generated_at : '이 주기의 순위 파일이 아직 없습니다.';
    document.getElementById('rankRows').innerHTML = ranking ? ranking.rankings.map(function(r) {
      var expired = r.stale || Date.now() >= (r.valid_until ? Date.parse(r.valid_until) : Date.parse(r.as_of) + ({'1m':60000,'1h':3600000,'1d':86400000}[r.interval]));
      return '<tr><td>' + r.rank + '</td><td>' + r.symbol + '</td><td>' + fmtPct(r.expected_net_return) + '</td><td>' + r.notional_krw.toLocaleString() + ' KRW</td><td>' + r.expected_pnl_krw.toFixed(0) + ' KRW</td><td>' + (expired ? '갱신 필요' : '연구용·미검증') + '</td></tr>';
    }).join('') : '';
  }

  render();
})();
</script>
"""


def render(payload):
    body = json.dumps(payload).replace('</', '<\\/')
    return PAGE.replace('__PAYLOAD__', body)


def main():
    payload = build_payload()
    out = Path('runs/forecast5/dashboard.html')
    out.write_text(render(payload))
    print('Wrote', out)


if __name__ == '__main__':
    main()

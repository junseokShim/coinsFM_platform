"""Korean, "Aureum" (emerald + gold) real-time dashboard page. Pure template string; no data
baked in -- everything is fetched client-side from /api/* so the page reflects live state on
every poll. Visual language ported from a design-canvas exploration the user picked (Option C:
deep emerald + gold, Spectral serif numbers, Work Sans UI) onto the real, data-bound dashboard.
"""

PAGE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<title>Aureum · 업비트 자동매매</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,500;0,600;0,700;1,500&family=Work+Sans:wght@400;500;600;700&display=swap" rel="stylesheet" />
<style>
  :root{
    color-scheme: dark;
    --bg: oklch(0.175 0.025 155);
    --bg2: oklch(0.215 0.028 155);
    --bg3: oklch(0.258 0.030 155);
    --line: oklch(1 0 0 / 0.08);
    --line2: oklch(1 0 0 / 0.13);
    --text: oklch(0.960 0.010 120);
    --text2: oklch(0.740 0.020 140);
    --text3: oklch(0.530 0.020 140);
    --gold: oklch(0.780 0.110 85);
    --gold-soft: oklch(0.780 0.110 85 / 0.15);
    --gold-dim: oklch(0.780 0.110 85 / 0.55);
    --up: oklch(0.640 0.190 25);
    --up-soft: oklch(0.640 0.190 25 / 0.14);
    --down: oklch(0.620 0.150 250);
    --down-soft: oklch(0.620 0.150 250 / 0.14);
    --warn: oklch(0.780 0.140 80);
    --warn-soft: oklch(0.780 0.140 80 / 0.16);
  }
  *{box-sizing:border-box;}
  html,body{margin:0;padding:0;}
  body{
    background:
      radial-gradient(1100px 480px at 82% -10%, var(--gold-soft) 0%, transparent 60%),
      var(--bg);
    color:var(--text);font-family:'Work Sans',system-ui,-apple-system,sans-serif;
    -webkit-font-smoothing:antialiased;
  }
  a{color:var(--gold);} a:hover{color:oklch(0.86 0.09 85);}
  .display{font-family:'Spectral',Georgia,serif;font-variant-numeric:tabular-nums;}
  .label{font-family:'Spectral',Georgia,serif;font-style:italic;font-weight:500;text-transform:uppercase;letter-spacing:0.06em;}
  .up{color:var(--up);} .down{color:var(--down);} .flat{color:var(--text3);}
  .na{color:var(--text3);}
  svg.icon{width:15px;height:15px;stroke:currentColor;fill:none;stroke-width:1.6;stroke-linecap:round;stroke-linejoin:round;flex:none;}

  .shell{max-width:1400px;margin:0 auto;padding:26px 24px 56px;}

  .header{display:flex;align-items:center;justify-content:space-between;padding-bottom:16px;margin-bottom:20px;border-bottom:1px solid var(--gold-soft);}
  .brand{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;}
  .brand .mark{font-family:'Spectral',serif;font-weight:600;font-size:25px;letter-spacing:0.05em;color:var(--gold);}
  .brand .tag{font-size:12.5px;color:var(--text3);}
  .headRight{display:flex;align-items:center;gap:12px;flex-wrap:wrap;}
  .dot{width:7px;height:7px;border-radius:50%;background:var(--up);box-shadow:0 0 0 3px oklch(0.640 0.190 25 / 0.22);flex:none;}
  .dot.bad{background:var(--text3);box-shadow:none;}
  .modePill{display:inline-flex;align-items:center;gap:7px;padding:7px 13px;border-radius:999px;font-size:12px;font-weight:600;letter-spacing:0.03em;}
  .modePill.live{background:var(--up-soft);color:var(--up);}
  .modePill.test{background:var(--warn-soft);color:var(--warn);}
  .modePill.paper{background:oklch(1 0 0 / 0.08);color:var(--text2);}
  .clock{font-size:12.5px;color:var(--text3);font-variant-numeric:tabular-nums;}

  .hero{display:grid;grid-template-columns:1.1fr 1fr;gap:16px;margin-bottom:16px;}
  @media (max-width:840px){ .hero{grid-template-columns:1fr;} }
  .heroCard{background:var(--bg2);border:1px solid var(--line);border-radius:20px;padding:22px 26px;position:relative;overflow:hidden;}
  .heroCard .rule{position:absolute;top:0;left:26px;right:26px;height:1px;background:linear-gradient(90deg,transparent,var(--gold-dim),transparent);}
  .heroLabel{font-size:12px;color:var(--text3);margin-bottom:9px;}
  .heroBig{font-weight:600;font-size:48px;line-height:1;letter-spacing:-0.005em;}
  .heroBig.gold{color:var(--gold);text-shadow:0 0 40px var(--gold-soft);}
  .heroDelta{margin-top:10px;display:flex;align-items:center;gap:7px;font-size:13.5px;flex-wrap:wrap;}
  .heroDelta .sub{color:var(--text3);font-size:12px;}
  .heroSpark{width:100%;height:46px;margin-top:14px;display:block;}

  .statRow{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:16px;}
  @media (max-width:840px){ .statRow{grid-template-columns:repeat(2,1fr);} }
  .statTile{background:var(--bg2);border:1px solid var(--line);border-radius:16px;padding:15px 18px;}
  .statTile .k{color:var(--text3);font-size:11.5px;margin-bottom:8px;}
  .statTile .v{font-weight:600;font-size:21px;}
  .statTile .v2{font-size:11.5px;color:var(--text3);margin-top:3px;}

  section{margin-bottom:28px;}
  .section-head{display:flex;align-items:baseline;justify-content:space-between;gap:12px;margin-bottom:12px;flex-wrap:wrap;}
  .section-head h2{margin:0;font-family:'Spectral',serif;font-weight:600;font-size:16px;display:flex;align-items:center;gap:9px;}
  .section-head .sub{color:var(--text3);font-size:12px;}
  .card{background:var(--bg2);border:1px solid var(--line);border-radius:20px;}

  .mainGrid{display:grid;grid-template-columns:1.5fr 1fr;gap:16px;}
  @media (max-width:960px){ .mainGrid{grid-template-columns:1fr;} }
  .panel{background:var(--bg2);border:1px solid var(--line);border-radius:20px;padding:18px 20px;display:flex;flex-direction:column;min-height:0;}
  .col{display:flex;flex-direction:column;gap:16px;min-height:0;}

  .aiRow{display:grid;grid-template-columns:24px 1fr 90px 70px 58px;align-items:center;gap:12px;padding:10px 4px;border-bottom:1px solid var(--line);}
  .aiRow:last-child{border-bottom:none;}
  .rankDot{width:22px;height:22px;border-radius:50%;background:var(--bg3);border:1px solid var(--gold-soft);display:flex;align-items:center;justify-content:center;font-family:'Spectral',serif;font-size:11px;font-weight:600;color:var(--gold);}
  .aiSym{display:flex;flex-direction:column;min-width:0;}
  .aiSym .s{font-weight:700;font-size:13.5px;}
  .aiSym .n{font-size:10.5px;color:var(--text3);}
  .aiRet{font-family:'Spectral',serif;font-weight:700;font-size:17px;text-align:right;}
  .sig{font-size:10.5px;font-weight:600;letter-spacing:0.02em;padding:4px 10px;border-radius:999px;text-align:center;white-space:nowrap;}
  .sig.buy{background:var(--up-soft);color:var(--up);}
  .sig.hold{background:oklch(1 0 0 / 0.07);color:var(--text2);}

  table{width:100%;border-collapse:collapse;font-size:12.5px;}
  th,td{text-align:right;padding:8px 6px;white-space:nowrap;}
  th:first-child,td:first-child{text-align:left;}
  thead th{color:var(--text3);font-weight:500;font-size:10.5px;text-transform:uppercase;letter-spacing:0.04em;border-bottom:1px solid var(--line);}
  tbody td{font-variant-numeric:tabular-nums;border-bottom:1px solid var(--line);}
  tbody tr:last-child td{border-bottom:none;}
  tbody td:first-child{font-weight:700;}
  tbody tr:hover td{background:var(--bg3);}
  .tablewrap{overflow-x:auto;}
  .empty{color:var(--text3);padding:20px;text-align:center;font-size:12.5px;}

  .logList{display:flex;flex-direction:column;margin-top:4px;max-height:172px;overflow-y:auto;}
  .logRow{display:flex;align-items:baseline;gap:10px;padding:7px 0;border-bottom:1px solid var(--line);font-size:12px;}
  .logRow:last-child{border-bottom:none;}
  .logRow .t{color:var(--text3);font-size:10.5px;min-width:50px;font-variant-numeric:tabular-nums;}
  .logRow .tag{font-weight:600;font-size:10px;padding:2px 8px;border-radius:999px;white-space:nowrap;}
  .logRow .tag.BUY,.logRow .tag.BUY_SUBMITTED{background:var(--up-soft);color:var(--up);}
  .logRow .tag.SELL{background:var(--down-soft);color:var(--down);}
  .logRow .tag.STAY{background:oklch(1 0 0 / 0.07);color:var(--text2);}
  .logRow .tag.BUY_FAILED,.logRow .tag.SELL_FAILED,.logRow .tag.SELL_SKIPPED{background:var(--warn-soft);color:var(--warn);}
  .logRow .m{color:var(--text2);}

  .board{display:grid;grid-template-columns:repeat(auto-fill,minmax(168px,1fr));gap:10px;}
  .sigcard{padding:14px;display:flex;flex-direction:column;gap:8px;border:1px solid var(--line);background:var(--bg2);border-radius:16px;}
  .sigcard .row1{display:flex;align-items:center;justify-content:space-between;}
  .sigcard .sym{font-weight:700;font-size:13.5px;}
  .sigcard .price{font-family:'Spectral',serif;font-size:15px;font-weight:500;}
  .sigcard .ret{font-size:12.5px;}
  .spark{width:100%;height:26px;display:block;}
  .pill{font-size:10.5px;font-weight:700;letter-spacing:0.03em;padding:3px 8px;border-radius:999px;}
  .pill.LONG{background:var(--up-soft);color:var(--up);}
  .pill.SHORT{background:var(--down-soft);color:var(--down);}
  .pill.HOLD{background:oklch(1 0 0 / 0.07);color:var(--text3);}

  .ivSwitch{display:inline-flex;gap:4px;}
  .ivSwitch button{font:600 12px 'Work Sans',sans-serif;padding:6px 13px;border-radius:8px;border:1px solid var(--line2);background:var(--bg2);color:var(--text2);cursor:pointer;}
  .ivSwitch button[aria-pressed="true"]{background:var(--gold);color:oklch(0.2 0.02 85);border-color:var(--gold);}

  .footnote{color:var(--text3);font-size:11.5px;margin-top:8px;line-height:1.7;}
  footer{border-top:1px solid var(--line);padding-top:16px;color:var(--text3);font-size:11.5px;line-height:1.7;}
</style>
</head>
<body>
<div class="shell">
  <div class="header">
    <div class="brand">
      <span class="mark">Aureum</span>
      <span class="tag">AI 자동매매 · 실시간 자산 데스크</span>
    </div>
    <div class="headRight">
      <span class="dot" id="connDot"></span>
      <span class="modePill" id="modeBadge">-</span>
      <span class="clock" id="liveClock"></span>
    </div>
  </div>

  <div class="hero">
    <div class="heroCard">
      <div class="rule"></div>
      <div class="heroLabel label">총 자산</div>
      <div class="heroBig display" id="heroTotal">-</div>
      <div class="heroDelta"><span class="sub" id="heroTotalSub">잔액 조회 중…</span></div>
    </div>
    <div class="heroCard">
      <div class="rule"></div>
      <div class="heroLabel label">자동매매 누적 수익률</div>
      <div class="heroBig display gold" id="heroReturn">-</div>
      <div class="heroDelta"><span class="sub" id="heroReturnSub"></span></div>
      <svg class="heroSpark" id="heroSpark" viewBox="0 0 400 46" preserveAspectRatio="none"></svg>
    </div>
  </div>

  <div class="statRow">
    <div class="statTile"><div class="k">KRW 잔액</div><div class="v display" id="tileKrw">-</div><div class="v2" id="tileKrwLocked"></div></div>
    <div class="statTile"><div class="k">코인 평가금액</div><div class="v display" id="tileCoin">-</div><div class="v2" id="tileCoinCount"></div></div>
    <div class="statTile"><div class="k">보유 포지션</div><div class="v display" id="tilePos">-</div><div class="v2" id="tilePosSub"></div></div>
    <div class="statTile"><div class="k">마지막 판단</div><div class="v display" id="tileTick">-</div><div class="v2" id="tileTickSub"></div></div>
  </div>

  <div class="mainGrid">
    <div class="panel">
      <div class="section-head" style="margin-bottom:10px;">
        <h2><svg class="icon" viewBox="0 0 24 24" style="stroke:var(--gold)"><path d="M12 2l2.4 6.6L21 11l-6.6 2.4L12 20l-2.4-6.6L3 11l6.6-2.4L12 2z"/></svg>AI 실시간 추천</h2>
        <span class="sub" id="aiSub">업비트 KRW · 5개 봉 예측</span>
      </div>
      <div id="aiRows"></div>
      <div class="empty" id="aiEmpty" style="display:none;">아직 순위 데이터가 없습니다</div>
    </div>

    <div class="col">
      <div class="panel">
        <div class="section-head" style="margin-bottom:10px;">
          <h2>보유 포지션</h2><span class="sub" id="posSub"></span>
        </div>
        <div class="tablewrap">
          <table><thead><tr><th>종목</th><th>투자금액</th><th>진입가</th><th>현재가</th><th>평가액</th><th>평가손익</th></tr></thead><tbody id="posRows"></tbody></table>
        </div>
        <div class="empty" id="posEmpty" style="display:none;">보유 중인 포지션이 없습니다</div>
      </div>
      <div class="panel">
        <div class="section-head" style="margin-bottom:6px;"><h2>최근 판단</h2></div>
        <p id="executionStatus" role="status" class="sub"></p>
        <div class="logList" id="logRows"></div>
      </div>
    </div>
  </div>

  <section>
    <div class="section-head"><h2>다중 주기 매매 판단</h2><span class="sub">1분봉 → 5분 · 1시간봉 → 5시간 · 일봉 → 5일</span></div>
    <p id="realtimeStatus" class="sub"></p>
    <div class="card tablewrap"><table><thead><tr><th>종목</th><th>분봉 순수익</th><th>시간봉 순수익</th><th>일봉 순수익</th><th>신호 강도*</th><th>매수 예산</th><th>판단</th><th>이유</th></tr></thead><tbody id="multiRows"></tbody></table></div>
    <p class="footnote">* 신호 강도는 변동성 대비 예측 여력 점수이며 수익 확률이 아닙니다. 주문 상한·가용 잔액·최소 주문액에 따라 실제 주문이 제한됩니다.</p>
  </section>
  <section>
    <div class="section-head"><h2>업비트 예상 순수익률 순위</h2><span class="sub" id="rankSub">TimesFM + 기술적 지표 · 동일 주기 5개 봉 · KRW 현물</span></div>
    <div class="card tablewrap">
      <table><thead><tr><th>순위</th><th>종목</th><th>예상 순수익률</th><th>가정 투자금</th><th>예상 손익</th><th>상태</th></tr></thead><tbody id="rankRows"></tbody></table>
    </div>
    <p class="footnote">모델 추정치이며 수익을 보장하지 않습니다. 비용 가정과 현재 호가를 반영했지만 업비트 실거래 성능은 검증되지 않았습니다.</p>
  </section>

  <section>
    <div class="section-head">
      <h2>리서치 신호 보드</h2>
      <span class="ivSwitch" id="ivSwitch"></span>
    </div>
    <p class="sub" id="boardSub"></p>
    <div class="board" id="board"></div>
    <p class="footnote" id="excludedNote"></p>
  </section>

  <footer>
    이 화면은 로컬(127.0.0.1)에서만 서비스됩니다. Upbit API 키는 서버 프로세스에서만 사용되며 브라우저로 전송되지 않습니다.
    모델 예측은 점 추정치이며 통계적으로 교정된 기대소득이 아닙니다. paper/test 모드는 실주문을 전송하지 않고,
    live 모드만 실제 자산을 매매합니다.
  </footer>
</div>

<script>
(function () {
  var state = { interval: '1h', dash: null, traderInterval: null, startedAt: null };
  var KOREAN_NAME = {
    BTC:'비트코인', ETH:'이더리움', XRP:'리플', SOL:'솔라나', DOGE:'도지코인', USDT:'테더',
    ADA:'에이다', TRX:'트론', SHIB:'시바이누', AVAX:'아발란체', DOT:'폴카닷', MATIC:'폴리곤',
    LINK:'체인링크', ATOM:'코스모스', ETC:'이더리움클래식', BCH:'비트코인캐시', LTC:'라이트코인',
    UNI:'유니스왑', NEAR:'니어프로토콜', APT:'앱토스', ARB:'아비트럼', OP:'옵티미즘', SUI:'수이',
    PEPE:'페페', WIF:'도그위프햇', BONK:'봉크', FLOKI:'플로키'
  };

  function fmtKrw(v) { if (v === null || v === undefined) return '-'; return '₩' + Math.round(v).toLocaleString(); }
  function fmtPct(v, digits) {
    if (v === null || v === undefined) return '<span class="na">N/A</span>';
    digits = digits === undefined ? 2 : digits;
    var cls = v > 0.0001 ? 'up' : v < -0.0001 ? 'down' : 'flat';
    var sign = v > 0 ? '+' : '';
    return '<span class="' + cls + '">' + sign + (v * 100).toFixed(digits) + '%</span>';
  }
  function fmtPrice(v) {
    if (v === undefined || v === null) return '-';
    var abs = Math.abs(v);
    var digits = abs >= 100 ? 0 : abs >= 1 ? 2 : abs >= 0.01 ? 4 : 8;
    return v.toLocaleString(undefined, {maximumFractionDigits: digits});
  }
  var kstFormatter = new Intl.DateTimeFormat('sv-SE', {
    timeZone: 'Asia/Seoul', year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false
  });
  function fmtTime(iso) {
    if (!iso) return '-';
    return kstFormatter.format(new Date(iso)) + ' KST';
  }
  function ago(iso) {
    if (!iso) return '-';
    var s = Math.max(0, Math.round((Date.now() - Date.parse(iso)) / 1000));
    if (s < 60) return s + '초 전';
    if (s < 3600) return Math.floor(s / 60) + '분 전';
    return Math.floor(s / 3600) + '시간 전';
  }
  function daysSince(iso) {
    if (!iso) return null;
    return Math.max(0, Math.floor((Date.now() - Date.parse(iso)) / 86400000));
  }
  function baseSymbol(market) { return market.startsWith('KRW-') ? market.slice(4) : market; }

  var ok = true;
  function setConn(good) {
    ok = good;
    document.getElementById('connDot').className = 'dot' + (good ? '' : ' bad');
  }

  function tickClock() {
    document.getElementById('liveClock').textContent = kstFormatter.format(new Date());
  }
  tickClock();
  setInterval(tickClock, 1000);

  function loadBalance() {
    fetch('/api/balance').then(function (r) { return r.json(); }).then(function (b) {
      setConn(true);
      if (b.error) {
        document.getElementById('heroTotal').textContent = '-';
        document.getElementById('heroTotalSub').textContent = 'API 키 필요 (' + b.error + ')';
        document.getElementById('tileKrw').textContent = 'API 키 필요';
        document.getElementById('tileCoin').textContent = '-';
        return;
      }
      document.getElementById('heroTotal').textContent = fmtKrw(b.total_krw);
      document.getElementById('heroTotalSub').textContent = '조회 ' + ago(b.fetched_at);
      document.getElementById('tileKrw').textContent = fmtKrw(b.krw_balance);
      document.getElementById('tileKrwLocked').textContent = b.krw_locked > 0 ? '주문중 ' + fmtKrw(b.krw_locked) : '';
      var coinValue = b.holdings.reduce(function (s, h) { return s + (h.value_krw || 0); }, 0);
      document.getElementById('tileCoin').textContent = fmtKrw(coinValue);
      document.getElementById('tileCoinCount').textContent = b.holdings.length + '개 종목 보유';
    }).catch(function () { setConn(false); });
  }

  function drawHeroSpark(points) {
    var svg = document.getElementById('heroSpark');
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    if (points.length < 2) return;
    var W = 400, H = 46;
    var vals = points.map(function (p) { return p.equity; });
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    var span = Math.max(hi - lo, hi * 0.001, 1);
    var stepX = W / Math.max(1, points.length - 1);
    function y(v) { return H - ((v - lo) / span) * H; }
    var last = points[points.length - 1];
    var color = last.return_pct >= 0 ? 'var(--up)' : 'var(--down)';
    var pts = points.map(function (p, i) { return (i * stepX).toFixed(1) + ',' + y(p.equity).toFixed(1); }).join(' ');
    var poly = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    poly.setAttribute('points', pts); poly.setAttribute('fill', 'none');
    poly.setAttribute('stroke', color); poly.setAttribute('stroke-width', '2');
    svg.appendChild(poly);
  }

  function loadEquity() {
    fetch('/api/equity').then(function (r) { return r.json(); }).then(function (d) {
      setConn(true);
      drawHeroSpark(d.points);
      var last = d.points[d.points.length - 1];
      document.getElementById('heroReturn').innerHTML = last ? fmtPct(last.return_pct) : '-';
      var days = daysSince(state.startedAt);
      var startTxt = state.startedAt ? fmtTime(state.startedAt).slice(0, 10) + ' 시작' : '';
      document.getElementById('heroReturnSub').textContent =
        (startTxt ? startTxt + (days !== null ? ' · ' + days + '일째 운용' : '') : '') +
        (d.mode === 'paper' ? ' (모의투자 기준)' : ' (실계좌 기준)');
    }).catch(function () { setConn(false); });
  }

  function renderPositions(positions, maxPositions, maxKrw) {
    document.getElementById('posSub').textContent = positions.length + ' / ' + maxPositions + '개 · 종목당 최대 ' + Math.round(maxKrw).toLocaleString() + '원';
    var rows = document.getElementById('posRows'), empty = document.getElementById('posEmpty');
    if (!positions.length) { rows.innerHTML = ''; empty.style.display = 'block'; return; }
    empty.style.display = 'none';
    rows.innerHTML = positions.map(function (p) {
      var sym = baseSymbol(p.symbol);
      var pnl = p.unrealized_pct === null || p.unrealized_pct === undefined ? '<span class="na">-</span>' : fmtPct(p.unrealized_pct);
      var mark = p.mark_price === null || p.mark_price === undefined ? '<span class="na">-</span>' : fmtPrice(p.mark_price);
      var invested = p.cost_krw === null || p.cost_krw === undefined ? '<span class="na">-</span>' : fmtKrw(p.cost_krw);
      var value = p.value_krw === null || p.value_krw === undefined ? '<span class="na">-</span>' : fmtKrw(p.value_krw);
      return '<tr><td>' + sym + '</td><td>' + invested + '</td><td>' + fmtPrice(p.entry_price) + '</td><td>' + mark + '</td><td>' + value + '</td><td>' + pnl + '</td></tr>';
    }).join('');
    document.getElementById('tilePos').textContent = positions.length + ' / ' + maxPositions;
  }

  function renderMulti(t) {
    var rt = t.realtime || {};
    document.getElementById('realtimeStatus').textContent = (rt.connected ? 'WebSocket 연결됨' : 'WebSocket 대기') + ' · 판단 주기 ' + t.tick_seconds + '초 · ' +
      (rt.model_loaded ? '모델 상주' : '모델 로딩 중') + (rt.busy ? ' · ' + rt.busy.symbol + ' ' + rt.busy.interval + ' 갱신 중' : '') +
      (rt.last_job_seconds ? ' · 최근 추론 ' + rt.last_job_seconds.toFixed(2) + '초' : '');
    var body = document.getElementById('multiRows'); body.innerHTML = '';
    Object.keys(t.multiframe_decisions || {}).forEach(function (market) {
      var d = t.multiframe_decisions[market], ret = d.returns || {}, tr = document.createElement('tr');
      var cells = [market, ret['1m'] == null ? '갱신 대기' : (ret['1m']*100).toFixed(2)+'%',
        ret['1h'] == null ? '갱신 대기' : (ret['1h']*100).toFixed(2)+'%', ret['1d'] == null ? '갱신 대기' : (ret['1d']*100).toFixed(2)+'%',
        d.signal_strength == null ? '—' : Math.round((d.action === 'SELL' ? d.exit_strength || 0 : d.signal_strength)*100)+'/100', Math.round(d.budget_krw || 0).toLocaleString()+'원',
        {BUY:'매수',STAY:'유지/대기',SELL:'매도',PARTIAL_CANDIDATE:'부분진입(시간봉)'}[d.action] || '대기', d.reason];
      cells.forEach(function (value) { var td = document.createElement('td'); td.textContent = value; tr.appendChild(td); }); body.appendChild(tr);
    });
  }

  function loadTrader() {
    fetch('/api/trader').then(function (r) { return r.json(); }).then(function (t) {
      setConn(true);
      state.traderInterval = t.interval;
      state.multi = t.multiframe_decisions || {};
      renderMulti(t);
      state.startedAt = t.started_at;
      var badge = document.getElementById('modeBadge');
      badge.textContent = { paper: '모의투자', test: '주문 검증(체결 없음)', live: '실거래 · LIVE' }[t.mode] || t.mode;
      badge.className = 'modePill ' + t.mode;
      document.getElementById('tileTick').textContent = ago(t.last_tick_at);
      document.getElementById('tileTickSub').textContent = 'AI 갱신 ' + ago(t.last_refresh_at) + (t.last_retrain_at ? ' · 모델 재학습 ' + ago(t.last_retrain_at) : ' · 모델 재학습 대기 중');
      document.getElementById('tilePosSub').textContent = '최대 ' + Math.round(t.max_krw_per_trade).toLocaleString() + '원/종목';

      renderPositions(t.positions, t.max_positions, t.max_krw_per_trade);
      document.getElementById('executionStatus').textContent = t.last_error || ((t.pending || []).length ? '주문 체결 확인 중 · 추가 주문 대기' : '실제 체결 확인 및 재진입 제한 적용');

      document.getElementById('logRows').innerHTML = t.decisions.map(function (d) {
        return '<div class="logRow"><span class="t">' + fmtTime(d.ts).slice(11, 19) + '</span><span class="tag ' + d.action + '">' + d.action + '</span><span class="m">' + (d.symbol ? baseSymbol(d.symbol) + ' · ' : '') + (d.reason || '') + '</span></div>';
      }).join('') || '<div class="empty">아직 판단 기록이 없습니다</div>';

      if (state.dash) renderAiPanel();
    }).catch(function () { setConn(false); });
  }

  function sparkPath(history, w, h) {
    if (!history || !history.length) return '';
    var closes = history.map(function (b) { return b.c; });
    var lo = Math.min.apply(null, closes), hi = Math.max.apply(null, closes);
    var span = Math.max(hi - lo, hi * 0.0005);
    return closes.map(function (c, i) {
      var x = (i / Math.max(1, closes.length - 1)) * w;
      var y = h - ((c - lo) / span) * h;
      return x.toFixed(1) + ',' + y.toFixed(1);
    }).join(' ');
  }

  function renderAiPanel() {
    var iv = state.traderInterval || '1h';
    document.getElementById('aiSub').textContent = '업비트 KRW · 1분·1시간·1일 종합 판단';
    var rk = state.dash.rankings[iv];
    var rows = document.getElementById('aiRows'), empty = document.getElementById('aiEmpty');
    if (!rk || !rk.rankings.length) { rows.innerHTML = ''; empty.style.display = 'block'; return; }
    empty.style.display = 'none';
    var historyBySymbol = {};
    var fc = state.dash.forecasts[iv];
    if (fc) fc.entries.forEach(function (e) { if (e.upbit_market) historyBySymbol[e.upbit_market] = e.history; });
    rows.innerHTML = rk.rankings.slice(0, 5).map(function (r) {
      var sym = baseSymbol(r.symbol);
      var name = KOREAN_NAME[sym] || '';
      var decision = (state.multi || {})[r.symbol] || {};
      var buy = decision.action === 'BUY' && decision.budget_krw > 0 && r.valid_until && Date.now() < Date.parse(r.valid_until);
      var pts = sparkPath(historyBySymbol[r.symbol], 90, 22);
      var spark = pts ? '<svg viewBox="0 0 90 22" width="90" height="22"><polyline points="' + pts + '" fill="none" stroke="' + (r.expected_net_return >= 0 ? 'var(--up)' : 'var(--down)') + '" stroke-width="1.5"/></svg>' : '<span></span>';
      return '<div class="aiRow"><div class="rankDot">' + r.rank + '</div><div class="aiSym"><span class="s">' + sym + '</span><span class="n">' + name + '</span></div>' +
        spark + '<div class="aiRet ' + (r.expected_net_return >= 0 ? 'up' : 'down') + '">' + (r.expected_net_return >= 0 ? '+' : '') + (r.expected_net_return * 100).toFixed(2) + '%</div>' +
        '<span class="sig ' + (buy ? 'buy' : 'hold') + '">' + (buy ? '매수' : '관망') + '</span></div>';
    }).join('');
  }

  function buildIvSwitch() {
    var el = document.getElementById('ivSwitch'); el.innerHTML = '';
    ['1m', '1h', '1d'].forEach(function (iv) {
      var b = document.createElement('button');
      b.textContent = iv;
      b.setAttribute('aria-pressed', iv === state.interval ? 'true' : 'false');
      b.onclick = function () { state.interval = iv; renderDash(); };
      el.appendChild(b);
    });
  }

  var lastRefreshRequestAt = 0;
  function requestRefresh() {
    var now = Date.now();
    if (now - lastRefreshRequestAt < 60000) return false;
    lastRefreshRequestAt = now;
    fetch('/api/refresh', { method: 'POST' }).catch(function () {});
    return true;
  }

  function renderDash() {
    if (!state.dash) return;
    buildIvSwitch();
    var fc = state.dash.forecasts[state.interval] || { entries: [], excluded: [] };
    document.getElementById('boardSub').textContent = fc.entries.length + '개 업비트 상장 종목 · 예상 t+5 수익률 높은 순';
    document.getElementById('excludedNote').textContent = fc.excluded.length ? ('업비트 미상장으로 제외됨: ' + fc.excluded.join(', ')) : '';
    var board = document.getElementById('board'); board.innerHTML = '';
    fc.entries.forEach(function (e) {
      var t5 = e.returns[4];
      var card = document.createElement('div'); card.className = 'sigcard';
      var pts = sparkPath(e.history, 148, 26);
      card.innerHTML = '<div class="row1"><span class="sym">' + e.symbol.replace('USDT', '') + '</span><span class="pill ' + e.signal + '">' + e.signal + '</span></div>' +
        '<svg class="spark" viewBox="0 0 148 26" preserveAspectRatio="none"><polyline points="' + pts + '" fill="none" stroke="var(--text3)" stroke-width="1.5" /></svg>' +
        '<div class="price display">' + fmtPrice(e.last_close) + '</div><div class="ret">t+5 ' + fmtPct(t5) + '</div>';
      board.appendChild(card);
    });

    var rk = state.dash.rankings[state.interval];
    var rankRows = document.getElementById('rankRows');
    if (!rk) { rankRows.innerHTML = '<tr><td colspan="6" class="empty">이 주기의 순위 데이터가 아직 없습니다</td></tr>'; }
    else {
      var anyExpired = false;
      rankRows.innerHTML = rk.rankings.map(function (r) {
        var horizonMs = { '1m': 60000, '1h': 3600000, '1d': 86400000 }[r.interval];
        var expired = r.stale || Date.now() >= (r.valid_until ? Date.parse(r.valid_until) : Date.parse(r.as_of) + horizonMs);
        if (expired) anyExpired = true;
        return '<tr><td>' + r.rank + '</td><td>' + r.symbol + '</td><td>' + fmtPct(r.expected_net_return) + '</td><td>' +
          Math.round(r.notional_krw).toLocaleString() + ' 원</td><td>' + Math.round(r.expected_pnl_krw).toLocaleString() + ' 원</td><td>' +
          (expired ? '갱신 필요' : '연구용·미검증') + '</td></tr>';
      }).join('');
      var sub = '생성 시각 ' + fmtTime(rk.generated_at) + ' (' + ago(rk.generated_at) + ')';
      if (anyExpired && state.interval === state.traderInterval) {
        sub += requestRefresh() ? ' · 갱신 요청함' : ' · 갱신 진행 중';
      }
      document.getElementById('rankSub').textContent = sub;
    }
    renderAiPanel();
  }

  function loadDashboard() {
    fetch('/api/dashboard').then(function (r) { return r.json(); }).then(function (d) {
      setConn(true); state.dash = d; renderDash();
    }).catch(function () { setConn(false); });
  }

  loadBalance(); loadEquity(); loadTrader(); loadDashboard();
  setInterval(loadTrader, 5000);
  setInterval(loadBalance, 15000);
  setInterval(loadEquity, 15000);
  setInterval(loadDashboard, 5000);
})();
</script>
</body>
</html>
"""

"""Standalone observed-candle / forecast-close viewer with no external services."""
import json
from pathlib import Path
from .__main__ import load


def main():
    root=Path('runs/forecast5')
    charts=[]
    for interval in ('1m','1h','1d'):
        candles=load(Path(f'data/live/{interval}.csv'))
        forecasts=json.loads((root/interval/'recent_predictions.json').read_text())
        for forecast in forecasts:
            history=[dict(time=b.timestamp,open=b.open,high=b.high,low=b.low,close=b.close)
                     for b in candles if b.symbol==forecast['symbol']][-60:]
            charts.append(dict(**forecast,history=history))
    payload=json.dumps(charts).replace('</','<\\/')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>다음 5개 봉 종가 예측</title><style>
body{font:16px system-ui;margin:32px;background:#101722;color:#e8edf4}main{max-width:1150px;margin:auto}select{padding:10px;background:#233043;color:white;border:1px solid #4a5b72;border-radius:6px;margin:8px}svg{width:100%;background:#141f2e;border-radius:12px}td,th{padding:10px 20px;text-align:right;border-bottom:1px solid #334155}table{border-collapse:collapse}p{color:#aab8ca;line-height:1.6}.up{fill:#30c69a}.down{fill:#f27982}</style>
<main><h1>다음 5개 봉 종가 예측</h1><label>종목 <select id="symbol"></select></label><label>봉 주기 <select id="interval"><option>1m</option><option>1h</option><option>1d</option></select></label>
<p id="meta"></p><svg id="chart" viewBox="0 0 1100 460" role="img" aria-label="과거 캔들과 다음 다섯 종가 예측"></svg>
<p>초록·분홍: 관측 OHLC 캔들 · 노랑 점선: 미래 종가 예측. 미래 시가·고가·저가는 생성하지 않습니다. 저장 시점의 예측이며 실시간 자동 갱신 화면은 아닙니다.</p>
<table><thead><tr><th>미래 봉</th><th>종료 시각 (UTC)</th><th>예측 종가 (USDT)</th></tr></thead><tbody id="rows"></tbody></table>
<p>TimesFM 2.5 + 주기별 LoRA + 최근 완료 사례 8개로 추가 적응. 검증된 수익 또는 확률 보장이 없는 연구 결과입니다.</p></main><script>
const data=PAYLOAD;const symbols=[...new Set(data.map(x=>x.symbol))];
symbol.innerHTML=symbols.map(s=>`<option>${s}</option>`).join('');
function render(){const d=data.find(d=>d.symbol===symbol.value&&d.interval===interval.value);const h=d.history,p=d.points;
const lo=Math.min(...h.map(x=>x.low),...p.map(x=>x.predicted_close)),hi=Math.max(...h.map(x=>x.high),...p.map(x=>x.predicted_close));
const span=Math.max(hi-lo,hi*.001),y=v=>390-(v-lo)/span*320,x=i=>75+i*14;
let svg='';for(let k=0;k<5;k++){let v=lo+span*k/4;svg+=`<line x1="65" x2="1035" y1="${y(v)}" y2="${y(v)}" stroke="#304057"/><text x="5" y="${y(v)}" fill="#aab8ca" font-size="11">${v.toPrecision(5)}</text>`;}
h.forEach((b,i)=>{let c=b.close>=b.open?'#30c69a':'#f27982';svg+=`<g><title>${b.time} O:${b.open} H:${b.high} L:${b.low} C:${b.close}</title><line x1="${x(i)}" x2="${x(i)}" y1="${y(b.high)}" y2="${y(b.low)}" stroke="${c}"/><rect x="${x(i)-4}" y="${Math.min(y(b.open),y(b.close))}" width="8" height="${Math.max(1,Math.abs(y(b.open)-y(b.close)))}" fill="${c}"/></g>`;});
const pts=[[x(h.length-1),y(h.at(-1).close)],...p.map((b,i)=>[x(h.length+i),y(b.predicted_close)])];
svg+=`<polyline points="${pts.map(v=>v.join(',')).join(' ')}" fill="none" stroke="#f5ca58" stroke-width="2" stroke-dasharray="5 4"/>`;
p.forEach((b,i)=>{svg+=`<circle cx="${x(h.length+i)}" cy="${y(b.predicted_close)}" r="4" fill="#f5ca58"><title>${b.candle_close} ${b.predicted_close}</title></circle>`;});
svg+=`<text x="75" y="435" fill="#aab8ca" font-size="12">${h[0].time}</text><text x="760" y="435" fill="#aab8ca" font-size="12">예측 기준 ${d.as_of}</text>`;
chart.innerHTML=svg;meta.textContent=`${d.symbol} · ${d.interval} · 완료 봉 기준 ${d.as_of} · 생성 ${d.generated_at}`;
rows.innerHTML=p.map(b=>`<tr><td>+${b.step}</td><td>${b.candle_close}</td><td>${b.predicted_close.toPrecision(8)}</td></tr>`).join('');}
symbol.onchange=interval.onchange=render;render();</script></html>'''.replace('PAYLOAD',payload)
    (root/'forecast.html').write_text(page)
    (root/'forecasts.json').write_text(json.dumps(charts,indent=2))
    lines=['# 다음 5개 캔들 종가 예측','',
        '- 주기: 1m→5분, 1h→5시간, 1d→5일. 출력은 OHLC 전체가 아닌 각 미래 봉의 종가.',
        '- TimesFM 2.5에 주기별 LoRA를 2 epoch 학습하고 최신 완료 사례 8개로 종목별 추가 적응.',
        '- 각 주기의 완료 봉 800개/종목에서 시간순 80/10/10% 분할. 6개 종목에서 96개 학습 창을 표본 추출한 소규모 실행.',
        '- 과거/정답 경계를 넘는 학습 창 제외. 원점마다 128개 최근 관측 입력, 5개 정답이 이미 확정된 8개 사례로 추가 적응.',
        '- 원본 모델의 사전학습 데이터 중복 가능성은 배제하지 못함. 작은 평가 표본과 종목 선택 편향이 있음.',
        '- 차트: [forecast.html](forecast.html), 전체 예측: [forecasts.json](forecasts.json).', '',
        '| 봉 주기 | 모델 | 평가 창 | 평균 절대 로그가격 오차 | 방향 정확도 |','|---|---|---:|---:|---:|']
    for interval in ('1m','1h','1d'):
        path=root/interval/'metrics_combined.json'
        if not path.exists():path=root/interval/'metrics.json'
        for name,r in json.loads(path.read_text()).items():
            acc='N/A' if name=='last_price' else f'{r["direction_accuracy"]:.1%}'
            lines.append(f'| {interval} | {name} | {r["windows"]} | {r["path_mae"]:.6f} | {acc} |')
    lines+=['','오차는 5개 예측 종가의 로그가격 오차 평균입니다. 수익률이나 거래 성과가 아닙니다. 최신 few-shot 결과는 미래 정답이 아직 없어 평가하지 않습니다. LoRA 성능이 기본 모델이나 마지막 가격 유지보다 나쁘면 그대로 표기하며 자동 채택하지 않습니다.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()

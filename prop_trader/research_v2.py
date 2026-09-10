"""Freeze an alternative on old data, then evaluate new 2026 data once."""
import csv
import json
from dataclasses import asdict, replace
from itertools import groupby, product
from pathlib import Path
from datetime import datetime
from .__main__ import load
from .engine import Config, Desk
from .pullback import PullbackDesk
from . import research

OUT = Path('runs/pullback')


def batches(path):
    result = [(ts,list(rows)) for ts, rows in groupby(load(path), key=lambda b:b.timestamp)]
    universe = sorted(b.symbol for b in result[0][1])
    previous = None
    for ts, rows in result:
        now = datetime.fromisoformat(ts)
        if sorted(b.symbol for b in rows) != universe or len(set(universe)) != len(universe):
            raise ValueError(f'Incomplete universe at {ts}')
        if previous and (now-previous).total_seconds() != 3600:
            raise ValueError(f'Hourly gap at {ts}')
        previous = now
    return result


def select():
    OUT.mkdir(parents=True,exist_ok=True)
    data = batches(Path('data/binance/candles.csv'))
    configs = [Config(lookback=l,stop_fraction=s) for l,s in product((168,336),(.025,.05))]
    protocol = dict(strategy='trend_pullback_recovery', candidates=[asdict(c) for c in configs],
        selection='Require positive 2024 and 2025 return, >=30 trades per period; rank by worst period return. If none pass deploy cash and evaluate best candidate diagnostically.',
        new_holdout='2026-01-01 through 2026-08-31, subject to archive availability',
        caveat='2024-2025 already inspected; development data only. No new-data tuning.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    for i,c in enumerate(configs):
        for name,start,end in [('2024','2024-04-01','2025-01-01'),('2025','2025-01-01','2026-01-01')]:
            r=research.run(data,c,start,end,desk_class=PullbackDesk)
            rows.append(dict(candidate=i,period=name,**r))
            print(i,name,round(r['return_fraction']*100,2),r['trades'],flush=True)
    research.write_csv(OUT/'development.csv',rows)
    eligible=[]
    for i in range(len(configs)):
        subset=[r for r in rows if r['candidate']==i]
        if all(r['return_fraction']>0 and r['trades']>=30 for r in subset):
            eligible.append(i)
    best=max(eligible or list(range(len(configs))),key=lambda i:min(r['return_fraction'] for r in rows if r['candidate']==i))
    frozen=dict(config=asdict(configs[best]),candidate=best,deploy=bool(eligible),
                status='research_candidate' if eligible else 'rejected_use_cash')
    (OUT/'frozen.json').write_text(json.dumps(frozen,indent=2))
    print(json.dumps(frozen,indent=2))


def evaluate():
    frozen=json.loads((OUT/'frozen.json').read_text())
    config=Config(**frozen['config'])
    old=batches(Path('data/binance/candles.csv'))
    new=batches(Path('data/binance/candles_2026.csv'))
    data=old+new
    results=[]
    research.OUT=OUT
    for name,c,desk in [('old_breakout',Config(lookback=48,volume_multiple=3,stop_fraction=.05),Desk),
                        ('pullback',config,PullbackDesk),
                        ('pullback_stress',replace(config,slippage=.003),PullbackDesk)]:
        r=research.run(data,c,'2026-01-01','2026-09-01',artifacts=name,desk_class=desk)
        results.append(dict(strategy=name,**r))
    research.write_csv(OUT/'new_holdout.csv',results)
    lines=['# 추격 매수에서 추세·눌림 회복으로 변경','',
        '- 과거 봉으로 계산한 상승 종목 비율이 50% 이상일 때만 거래.',
        '- 24시간 평균 > 장기 평균, 장기 가격 상승, 평균선 아래 눌림 후 전봉 고가와 평균선 회복.',
        '- 평균선보다 3% 넘게 오른 가격은 추격하지 않음. 다음 봉 시가 진입, 기존 비용·위험 관리 유지.',
        '- 2024~2025년은 이미 본 개발 데이터. 4개 후보를 비교한 후 설정을 동결하고 2026년 평가.',
        f'- 개발 단계 판정: `{frozen["status"]}`. 탈락 시 현금 유지가 운영 선택이며 후보 성과는 진단용.',
        f'- 새 평가 데이터: {new[0][0]} ~ {new[-1][0]}.', '',
        '| 전략 | 순수익률 | 최대 낙폭 | 거래 수 |','|---|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["strategy"]} | {r["return_fraction"]:.2%} | {r["max_drawdown"]:.2%} | {r["trades"]} |')
    lines += ['| 현금 | 0% | 0% | 0 |','','선택 편향이 있는 6개 현물 종목 표본이며 미래 성과를 보장하지 않습니다. 호가·시장충격·거래량 제한은 모델링하지 않습니다. 숏 실험은 포함하지 않습니다. 원본 출처와 SHA256은 data/binance/manifest_2026.json에 있습니다.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['select','evaluate'])
    select() if p.parse_args().phase=='select' else evaluate()

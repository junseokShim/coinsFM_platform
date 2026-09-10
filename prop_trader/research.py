"""Predefined chronological train/validation/holdout experiment."""
import csv
import hashlib
import json
from dataclasses import asdict, replace
from itertools import groupby, product
from pathlib import Path
from .__main__ import load
from .engine import Config, Desk

OUT = Path('runs/backtest')


def run(batches, config, start, end, artifacts=None, desk_class=Desk):
    d = desk_class(config)
    warmup = [rows for ts, rows in batches if ts < start][-config.lookback:]
    for rows in warmup:
        for b in rows:
            d.history[b.symbol].append(b)
    for ts, rows in batches:
        if start <= ts < end:
            d.step(rows)
    result = d.finish()
    trades = [e for e in d.events if e['kind'] == 'exit']
    gains = sum(max(0, t['pnl']) for t in trades)
    losses = -sum(min(0, t['pnl']) for t in trades)
    result.update(profit_factor=gains / losses if losses else None,
                  win_rate=result['wins'] / len(trades) if trades else 0,
                  net_pnl=sum(t['pnl'] for t in trades),
                  halts=sum(e['kind'] == 'halt' for e in d.events))
    if abs(result['net_pnl'] - (result['final_equity'] - config.capital)) > 1e-6:
        raise AssertionError('PnL reconciliation failed')
    if artifacts:
        (OUT / f'{artifacts}_events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in d.events))
        write_csv(OUT / f'{artifacts}_equity.csv', d.curve)
    return result


def write_csv(path, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = Path('data/binance/candles.csv')
    bars = load(source)
    batches = [(ts, list(rows)) for ts, rows in groupby(bars, key=lambda b: b.timestamp)]
    universe = sorted({b.symbol for b in bars})
    from datetime import datetime
    previous = None
    for ts, rows in batches:
        if sorted(b.symbol for b in rows) != universe:
            raise ValueError(f'Incomplete/duplicate universe at {ts}')
        now = datetime.fromisoformat(ts)
        if previous and (now - previous).total_seconds() != 3600:
            raise ValueError(f'Hourly gap at {ts}')
        previous = now
    periods = dict(train=('2024-04-01', '2025-01-01'),
                   validation=('2025-01-01', '2025-07-01'),
                   holdout=('2025-07-01', '2026-01-01'))
    grid = [Config(lookback=l, volume_multiple=v, stop_fraction=s)
            for l, v, s in product((20, 48, 96), (1.5, 2., 3.), (.025, .05))]
    protocol = dict(periods=periods, grid=[asdict(c) for c in grid],
                    selection='Top three train net returns with >=30 trades; choose best validation net return. Cash if none eligible.',
                    data_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    symbols=universe, bars=len(bars), interval='1h',
                    costs='fee 0.1% + slippage 0.1% per side; stress slippage 0.3%, 0.5%',
                    holdout_policy='Holdout not used for parameter selection; no post-result tuning')
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    train = []
    for i, c in enumerate(grid):
        result = run(batches, c, *periods['train'])
        train.append(dict(candidate=i, **asdict(c), **result))
        print(f'Train {i+1}/{len(grid)}: {result["return_fraction"]:.2%}', flush=True)
    write_csv(OUT / 'train_grid.csv', train)
    top = sorted((r for r in train if r['trades'] >= 30), key=lambda r: (-r['return_fraction'], r['candidate']))[:3]
    if not top:
        raise ValueError('No eligible candidate: remain in cash')
    validation = [dict(candidate=r['candidate'], **run(batches, grid[r['candidate']], *periods['validation'])) for r in top]
    write_csv(OUT / 'validation.csv', validation)
    winner = max(validation, key=lambda r: (r['return_fraction'], -r['candidate']))
    selected = grid[winner['candidate']]
    results = []
    for name, period in periods.items():
        for label, config in [('default', Config()), ('selected', selected)]:
            results.append(dict(period=name, strategy=label, **run(batches, config, *period, artifacts=f'{name}_{label}')))
    for slip in (.003, .005):
        results.append(dict(period='holdout', strategy=f'slippage_{slip}',
            **run(batches, replace(selected, slippage=slip), *periods['holdout'])))
    # Diagnostic only: spot-price short simulation does not establish futures performance.
    results.append(dict(period='holdout', strategy='synthetic_long_short',
        **run(batches, replace(selected, allow_short=True), *periods['holdout'])))
    write_csv(OUT / 'results.csv', results)
    per_symbol = []
    for symbol in universe:
        single = [(ts, [b for b in rows if b.symbol == symbol]) for ts, rows in batches]
        per_symbol.append(dict(symbol=symbol, **run(single, selected, *periods['holdout'])))
    write_csv(OUT / 'holdout_symbols.csv', per_symbol)
    benchmarks = []
    for name, (start, end) in periods.items():
        rows = [r for ts, r in batches if start <= ts < end]
        first, last = {b.symbol:b for b in rows[0]}, {b.symbol:b for b in rows[-1]}
        net = sum(last[s].close * (1-selected.slippage) * (1-selected.fee) /
                  (first[s].open * (1+selected.slippage) * (1+selected.fee)) - 1 for s in universe) / len(universe)
        benchmarks.append(dict(period=name, cash=0, equal_weight_buy_hold=net, buy_hold_60pct=net*.6))
    write_csv(OUT / 'benchmarks.csv', benchmarks)
    lines = ['# 실제 시장 데이터 백테스트', '',
        f'- 기간: {batches[0][0]} ~ {batches[-1][0]}',
        f'- 종목: {", ".join(universe)} / 시간봉 {len(bars):,}개',
        '- 출처: [Binance 공개 데이터](https://github.com/binance/binance-public-data). 원본 ZIP과 SHA256 체크섬 보관.',
        '- 학습: 2024년 4~12월 / 검증: 2025년 1~6월 / 최종 평가: 2025년 7~12월.',
        '- 18개 설정 중 학습 거래 30회 이상인 수익률 상위 3개를 검증 구간에서 비교하여 하나 선택.',
        f'- 선택 설정: `{json.dumps(asdict(selected))}`', '',
        '| 구간 | 전략 | 순수익률 | 최대 낙폭 | 거래 수 | 승률 | Profit factor |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in results:
        pf = f'{r["profit_factor"]:.2f}' if r['profit_factor'] is not None else 'N/A'
        lines.append(f'| {r["period"]} | {r["strategy"]} | {r["return_fraction"]:.2%} | {r["max_drawdown"]:.2%} | {r["trades"]} | {r["win_rate"]:.2%} | {pf} |')
    lines += ['', '## 비교 기준', '', '| 구간 | 현금 | 동일 비중 매수 보유 | 60% 투자·40% 현금 |', '|---|---:|---:|---:|']
    for b in benchmarks:
        lines.append(f'| {b["period"]} | 0% | {b["equal_weight_buy_hold"]:.2%} | {b["buy_hold_60pct"]:.2%} |')
    final = next(r for r in results if r['period'] == 'holdout' and r['strategy'] == 'selected')
    lines += ['', '## 판단', '',
        f'선택 전략의 최종 평가 순수익률은 {final["return_fraction"]:.2%}, 최대 낙폭은 {final["max_drawdown"]:.2%}입니다.',
        '현재 결과로 실거래 투입을 정당화하지 않습니다. 추가 가설은 이번 최종 평가 구간과 분리한 새 데이터에서 검증해야 합니다.',
        '', '![최종 평가 자산과 낙폭](holdout.png)', '', '## 해석의 한계', '',
        '- 사후 선택한 밈코인 6개이며 소형 시가총액 전체나 상장폐지 종목을 대표하지 않습니다. 생존·선택 편향이 있습니다.',
        '- 기본/선택 전략은 현물 롱 전용입니다. synthetic_long_short는 현물 가격상의 가상 숏으로 펀딩·차입·청산 비용이 없습니다.',
        '- 각 구간은 초기 자금 10,000으로 독립 시작합니다. 이전 구간 데이터는 지표 준비에만 사용하며 포지션을 승계하지 않습니다.',
        '- 같은 봉에 손절·익절이 닿으면 손절 우선. 봉 내부 체결 순서, 호가 깊이, 시장 충격과 거래량 제한은 모델링하지 않습니다.',
        '- 비용 스트레스에서는 비용을 반영한 주문 수량도 달라집니다. 순수한 고정 체결 비용 비교가 아닙니다.',
        '- 수익률 순위는 통계적 유의성이나 미래 수익을 보장하지 않습니다. 최종 평가 결과로 다시 설정을 선택하지 않았습니다.',
        '- 매수 보유 비교는 동일 비중 초기 매수 후 유지하며 진입·청산 비용 포함. 노출과 위험은 전략과 다릅니다.',
        '- 개별 종목 실험은 각자 자금 10,000으로 별도 실행되므로 포트폴리오 손익 기여도와 다릅니다.',
        '', '## 재현', '', '```sh', 'python3 -m prop_trader.download', 'python3 -m prop_trader.research', '```']
    (OUT / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(selected=asdict(selected), results=results), indent=2))


if __name__ == '__main__':
    main()

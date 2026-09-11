"""Decompose realized P&L from decisions.jsonl into: fee+slippage cost, and forecast-error
residual bucketed by which exit mechanism closed the trade (stop-loss / horizon-timeout /
other signal exit / take-profit). Every trade's true fee+slippage drag is already known
(fee_krw is logged on both the buy and sell fill) rather than assumed, so "gross P&L" here
means the realized P&L with that known cost added back -- what the price move alone did,
independent of the ~0.1%/side fee (execution_rules.FEE) or execution slippage.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def bucket(reason):
    if '손절' in reason:
        return 'stop_loss'
    if '익절' in reason:
        return 'take_profit'
    if '만료' in reason:
        return 'horizon_timeout'
    return 'signal_exit'


def trades(decisions_path):
    """Pair each BUY fill with the SELL fill that closed it (FIFO per symbol) and yield one
    record per closed round-trip: realized pnl, both fill fees, and the exit bucket.

    The fill-confirmation entries (action BUY/SELL) always log reason='체결 확인' -- the real
    reason (stop-loss / take-profit / horizon timeout / multiframe signal) is only on the
    preceding *_SUBMITTED entry (live_trader._send), matched by identifier."""
    open_buys = defaultdict(list)
    submitted_reason = {}
    with Path(decisions_path).open() as f:
        lines = f.readlines()
    for line in lines:
        d = json.loads(line)
        symbol, action = d.get('symbol'), d.get('action')
        if action in ('BUY_SUBMITTED', 'SELL_SUBMITTED') and d.get('identifier'):
            submitted_reason[d['identifier']] = d['reason']
        elif action == 'BUY' and d.get('reason') == '체결 확인' and 'fee_krw' in d:
            open_buys[symbol].append(d)
        elif action == 'SELL' and d.get('reason') == '체결 확인' and d.get('pnl_krw') is not None:
            buy = open_buys[symbol].pop(0) if open_buys[symbol] else None
            exit_reason = submitted_reason.get(d.get('identifier'), '')
            yield dict(symbol=symbol, ts=d['ts'], pnl_krw=d['pnl_krw'], sell_fee_krw=d.get('fee_krw', 0.),
                       buy_fee_krw=(buy or {}).get('fee_krw', 0.), bucket=bucket(exit_reason))


def report(decisions_path):
    rows = list(trades(decisions_path))
    totals = defaultdict(lambda: dict(n=0, pnl=0., fee=0., gross=0.))
    for r in rows:
        fee = r['buy_fee_krw']+r['sell_fee_krw']
        gross = r['pnl_krw']+fee
        b = totals[r['bucket']]
        b['n'] += 1; b['pnl'] += r['pnl_krw']; b['fee'] += fee; b['gross'] += gross
    return rows, totals


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--decisions', type=Path, default=Path('runs/live/live/decisions.jsonl'))
    args = p.parse_args()
    rows, totals = report(args.decisions)
    total_pnl = sum(r['pnl_krw'] for r in rows)
    total_fee = sum(r['buy_fee_krw']+r['sell_fee_krw'] for r in rows)
    print(f'{len(rows)} closed trades, realized P&L {total_pnl:.1f} KRW, total fees {total_fee:.1f} KRW\n')
    print(f'{"bucket":<16}{"n":>4}{"realized_pnl":>16}{"fees":>12}{"gross(pre-fee)":>16}')
    for name, b in sorted(totals.items(), key=lambda kv: kv[1]['pnl']):
        print(f'{name:<16}{b["n"]:>4}{b["pnl"]:>16.1f}{b["fee"]:>12.1f}{b["gross"]:>16.1f}')


if __name__ == '__main__':
    main()

"""One long-lived CPU model; completed candle keyed Few-shot + TA forecasts.

Public data only. No private API credentials or execution capability is needed.
The UI process exchanges atomic JSON files with this dedicated ML process.
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from .engine import Bar
from .forecast_time import SECONDS
from .upbit_broker import UpbitClient

FRAMES = ('1d', '1h', '1m')


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def boundary(interval, now):
    return int(now)//SECONDS[interval]*SECONDS[interval]


def completed_bars(client, symbol, interval, cutoff):
    step = SECONDS[interval]
    endpoint = '/v1/candles/days' if interval == '1d' else f'/v1/candles/minutes/{step//60}'
    raw = client.request('GET', endpoint, dict(market=symbol, count=200,
                         to=datetime.fromtimestamp(cutoff, timezone.utc).isoformat()), private=False)
    bars = []
    for r in raw:
        stamp = datetime.fromisoformat(r['candle_date_time_utc']).replace(tzinfo=timezone.utc)
        if stamp.timestamp()+step > cutoff:
            continue
        bars.append(Bar(stamp.isoformat(), symbol, float(r['opening_price']),
                        float(r['high_price']), float(r['low_price']), float(r['trade_price']),
                        float(r['candle_acc_trade_volume'])))
    bars.sort(key=lambda b: b.timestamp)
    if len(bars)<168 or datetime.fromisoformat(bars[-1].timestamp).timestamp()+step != cutoff:
        raise ValueError('완성 봉 부족 또는 최근 봉 누락')
    if any(datetime.fromisoformat(b.timestamp).timestamp()-datetime.fromisoformat(a.timestamp).timestamp()!=step
           for a,b in zip(bars,bars[1:])):
        raise ValueError('무거래 봉 누락: 예측 보류')
    return bars


def next_job(markets, records, attempted, now):
    # Per symbol long frames first, then minute; held markets arrive first.
    # Each key is attempted once per completed bar, even when data is missing.
    for market in markets:
        for interval in FRAMES:
            cutoff = boundary(interval, now)
            key = (market, interval, cutoff)
            if key in attempted:
                continue
            row = records.get(market, {}).get(interval)
            if row and datetime.fromisoformat(row['as_of']).timestamp()==cutoff:
                continue
            return key
    return None


def run(request_path, output_path, stop_path, max_jobs=None, parent_pid=None):
    import torch
    from transformers import TimesFm2_5ModelForPrediction
    from .timesfm_predict import predict_symbol
    # Use the whole machine: with 20 candidate markets needing 1m/1h/1d forecasts kept fresh,
    # a fixed 4 threads on an 8-core box left most candidates permanently stuck at "waiting for
    # forecast" -- never even reaching a BUY/SELL decision, not because the strategy is
    # cautious but because inference throughput couldn't keep up with the universe size.
    torch.set_num_threads(max(1, os.cpu_count() or 4))
    torch.manual_seed(42)
    protocols = {iv: read_json(Path('runs/forecast5')/iv/'protocol.json', {}) for iv in FRAMES}
    if len({p['model'] for p in protocols.values()}) != 1:
        raise ValueError('상주 모델의 기본 체크포인트가 주기별로 다릅니다')
    # shots_sweep.py measures walk-forward accuracy per candidate few-shot count and writes the
    # best one here; fall back to 8 (the original fixed pilot value) when no sweep has run yet.
    shots = {iv: read_json(Path('runs/forecast5')/iv/'best_shots.json', {}).get('best_shots', 8) for iv in FRAMES}
    for iv, k in shots.items():
        if not isinstance(k, int) or not 1 <= k <= 64:
            shots[iv] = 8
    started = time.monotonic()
    base = TimesFm2_5ModelForPrediction.from_pretrained(protocols['1h']['model'],
                dtype=torch.float32, local_files_only=True).to('cpu')
    load_seconds = time.monotonic()-started
    records, errors, attempted, retry = {}, {}, set(), {}
    completed = 0
    client = UpbitClient(access='', secret='')
    status = dict(records=records, errors=errors, model_loaded=True, load_seconds=load_seconds)
    while not Path(stop_path).exists():
        if parent_pid and os.getppid() != parent_pid:
            break
        retry = {k:t for k,t in retry.items() if k[2]>=boundary(k[1],time.time())}
        request = read_json(request_path, {})
        markets = request.get('markets', [])
        blocked = attempted | {k for k,t in retry.items() if t > time.time()}
        job = next_job(markets, records, blocked, time.time())
        if job is None:
            status.update(updated_at=time.time(), records=records, errors=errors, busy=None)
            atomic_json(output_path, status)
            time.sleep(.5)
            continue
        market, interval, cutoff = job
        attempted = {k for k in attempted if k[2]>=boundary(k[1],time.time())}
        start = time.monotonic()
        status.update(updated_at=time.time(), busy=dict(symbol=market, interval=interval))
        atomic_json(output_path, status)
        try:
            bars = completed_bars(client, market, interval, cutoff)
            attempted.add(job)  # Few-shot is invoked only once for this completed bar.
            row, base = predict_symbol(base, Path('runs/forecast5')/interval, protocols[interval],
                                       bars, bars[-1].timestamp, shots[interval], 'cpu', save_adapter=False)
            row['history'] = [dict(t=b.timestamp,o=b.open,h=b.high,l=b.low,c=b.close) for b in bars[-60:]]
            records.setdefault(market, {})[interval] = row
            errors.pop(market+':'+interval, None)
            # Every successful forecast is archived for later time-ordered evaluation.
            archive = Path(output_path).parent/'forecast_history.jsonl'
            with archive.open('a') as f:
                f.write(json.dumps(dict(row, history=None))+'\n')
        except Exception as e:
            retry[job] = time.time()+30
            errors[market+':'+interval] = str(e) if isinstance(e, ValueError) else '예측 작업 실패'
            # predict_symbol may have attached adapters before failing. Remove them so
            # another coin never inherits its in-progress Few-shot weights.
            if hasattr(base, 'unload'):
                base = base.unload()
            else:
                from peft.tuners.tuners_utils import BaseTunerLayer
                # The wrapped base can contain tuner layers despite not exposing unload.
                # Exit for a clean process restart instead of trusting mutated state.
                if any(isinstance(m, BaseTunerLayer) for m in base.modules()):
                    status.update(errors=errors, fatal='어댑터 복구를 위한 재시작 필요')
                    atomic_json(output_path, status)
                    raise
        completed += 1
        status.update(records=records, errors=errors, busy=None, completed=completed,
                      last_job_seconds=time.monotonic()-start, updated_at=time.time())
        atomic_json(output_path, status)
        if max_jobs is not None and completed >= max_jobs:
            break


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stop', type=Path, required=True)
    p.add_argument('--max-jobs', type=int)
    p.add_argument('--parent-pid', type=int)
    a=p.parse_args()
    run(a.request, a.output, a.stop, a.max_jobs, a.parent_pid)

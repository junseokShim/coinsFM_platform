"""Public quote stream and a single persistent inference subprocess per trader."""
import os
import subprocess
import time
from pathlib import Path

from .market_stream import MarketStream
from .resident_forecasts import atomic_json, read_json


class RealtimeRuntime:
    def __init__(self, root):
        self.root = Path(root)/'realtime'
        self.root.mkdir(parents=True, exist_ok=True)
        self.stream = MarketStream()
        self.process = None
        self.log = None
        self.last_launch = 0.

    def start(self):
        self.stream.start()
        (self.root/'stop').unlink(missing_ok=True)
        (self.root/'forecasts.json').unlink(missing_ok=True)
        self.ensure_worker()

    def ensure_worker(self):
        if self.process and self.process.poll() is None:
            return
        if time.monotonic()-self.last_launch < 30:
            return
        python = Path('.venv-timesfm/bin/python').absolute()
        if not python.exists():
            raise ValueError('예측 가상환경 .venv-timesfm이 없습니다')
        self.last_launch = time.monotonic()
        env = dict(os.environ, HF_HOME=str(Path('.cache/huggingface').resolve()), HF_HUB_OFFLINE='1')
        for key in ('UPBIT_ACCESS_KEY','UPBIT_SECRET_KEY'):
            env.pop(key, None)
        if self.log:
            self.log.close()
        self.log = (self.root/'worker.log').open('a')
        self.process = subprocess.Popen([str(python), '-m', 'prop_trader.resident_forecasts',
                 '--request', str(self.root/'request.json'), '--output', str(self.root/'forecasts.json'),
                 '--stop', str(self.root/'stop'), '--parent-pid', str(os.getpid())],
                 env=env, stdout=self.log, stderr=self.log)

    def request(self, markets):
        atomic_json(self.root/'request.json', dict(markets=list(dict.fromkeys(markets))))
        self.ensure_worker()

    def forecasts(self):
        result = read_json(self.root/'forecasts.json', dict(records={}))
        if self.process is not None and self.process.poll() is not None:
            result.update(model_loaded=False, fatal='예측 프로세스 재시작 대기')
        return result

    def stop(self):
        (self.root/'stop').touch()
        self.stream.stop()
        if self.process:
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill(); self.process.wait()
        if self.log:
            self.log.close()

# coinsFM

Upbit KRW 마켓을 대상으로 한 TimesFM 기반 예측 + 멀티프레임(1분/1시간/1일) 합의 전략 자동매매 시스템입니다. 실시간 WebSocket 시세 수집, 상주 추론 워커, 페이퍼/라이브 두 모드를 지원하는 거래 엔진, 대시보드를 포함합니다.

모든 코드는 `prop_trader/` 패키지에 있습니다. 세부 문서는 다음을 참고하세요.

- [prop_trader/README.md](prop_trader/README.md) — 연구용 돌파 전략 baseline, 실행/테스트 방법
- [prop_trader/FORECASTING.md](prop_trader/FORECASTING.md) — TimesFM 2.5 + LoRA + few-shot 예측 파이프라인
- [prop_trader/TRADING_INTEGRATION.md](prop_trader/TRADING_INTEGRATION.md) — 기술적 지표 결합, 신뢰성 검증, 종목 순위, 업비트 주문 연결

## 설치

두 가지 파이썬 환경이 필요합니다.

1. **경량 런타임** (실시간 거래 서버, 대시보드, 업비트 API 연동, 기술적 지표) — 워크스페이스 루트에서:

   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **TimesFM 예측 환경** (모델 파인튜닝/few-shot 추론, 무거운 ML 의존성) — 별도 venv 권장:

   ```sh
   python3 -m venv .venv-timesfm
   .venv-timesfm/bin/pip install -r prop_trader/requirements-timesfm.lock.txt
   ```

   `HF_HOME`을 프로젝트 로컬 캐시(`.cache/huggingface`)로 지정해 모델을 내려받고, 이후에는 `HF_HUB_OFFLINE=1`로 오프라인 실행합니다. 자세한 절차는 [FORECASTING.md](prop_trader/FORECASTING.md)를 참고하세요.

## API 키 설정

실거래/계좌 조회 기능은 업비트 Open API 키가 필요합니다. **API 키를 코드나 커밋에 절대 넣지 마세요.**

```sh
cp .env.example .env
# .env를 열어 UPBIT_ACCESS_KEY / UPBIT_SECRET_KEY를 채우세요.
```

`.env`는 `.gitignore`에 등록되어 있어 커밋되지 않습니다. 공개 시세 조회(순위, 백테스트, 연구용 실행)는 키 없이도 동작합니다.

## 실시간 거래 서버 실행

```sh
# 페이퍼 트레이딩(모의 매매) — 기본값
python3 -m prop_trader.live_server --mode paper

# 실거래 — 반드시 --confirm-live를 함께 지정해야 실주문이 나갑니다.
python3 -m prop_trader.live_server --mode live --confirm-live
```

대시보드는 서버 실행 후 안내되는 로컬 포트로 접속하면 됩니다. 주요 CLI 플래그(`--entry-threshold`, `--universe-top`, `--analog-gate` 등)는 `prop_trader/live_trader.py`의 `main()`을 참고하세요.

## 테스트

```sh
python3 -m unittest discover -s prop_trader/tests -v
```

TimesFM/torch에 의존하는 일부 테스트(`test_shots_sweep.py` 등)는 `.venv-timesfm/bin/python -m unittest ...`로 실행해야 합니다.

## 면책

이 저장소는 연구/실험 목적입니다. 실거래 모드는 실제 자금으로 주문을 제출합니다. 사용에 따른 손익은 전적으로 사용자 책임입니다.

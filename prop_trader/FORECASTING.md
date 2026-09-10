# 다음 5개 봉 종가 예측

기술적 분석 결합·업비트 순위·비용 포함 신뢰성 검증·업비트 주문 연결은 [TRADING_INTEGRATION.md](TRADING_INTEGRATION.md)를 참고하세요. 아래의 종가만 사용한다는 설명은 기존 TimesFM 입력을 뜻하며, 새 보정 계층은 완료된 OHLCV 기술적 지표를 별도로 사용합니다.

입력 주기와 예측 주기는 같습니다. 1분봉은 5분, 1시간봉은 5시간, 1일봉은 5일 뒤까지 **각 봉의 종가 5개**를 출력합니다. 미래 OHLC 전체를 생성하는 모델은 아닙니다.

## 실제 구현

1. 공개 Binance API에서 완료된 캔들만 수집합니다. 거래소 서버 시각으로 아직 열려 있는 봉을 제외합니다.
2. TimesFM 2.5 200M 기반에 주기별 LoRA 어댑터를 학습합니다. 학습되는 파라미터는 1,382,912개입니다.
3. 시간순 80% 학습·10% 검증·10% 평가로 분리하고, 정답이 분할 경계를 넘는 창을 제외합니다. 검증 손실로 epoch를 선택합니다.
4. 예측 시 LoRA 어댑터를 불러오고 최근 8개 완료 사례(context 128봉 + 정답 5봉)로 두 번 추가 업데이트합니다. 정답 구간이 중복되지 않으며 모두 예측 기준 이전에 완료됩니다.
5. 최신 128봉을 넣고 다음 5개 종가를 예측합니다. 종목마다 원래 LoRA로 초기화해 적응 결과가 다른 종목에 섞이지 않게 합니다.

여기서 few-shot은 최근 8개 지도학습 사례를 이용한 온라인 적응입니다. TimesFM이 LLM처럼 예시 문장을 프롬프트로 받는다는 의미는 아닙니다. 최신 관측 입력 갱신과 최근 사례 기반 가중치 갱신을 모두 수행합니다.

## 재현

워크스페이스 루트에서 실행합니다. 전용 `.venv-timesfm` 환경과 프로젝트 `.cache/huggingface`의 기본 모델을 사용합니다. 설치 패키지 목록은 `requirements-timesfm.lock.txt`입니다. 기본 모델 리비전은 `5a9806b9b291fad9233b5249d88263f1846304d3`으로 이번 실행에서 확인했습니다.

```sh
python3 -m prop_trader.current_candles --out data/current
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 .venv-timesfm/bin/python -m prop_trader.timesfm_run --csv data/current/1h.csv --out runs/forecast5/1h --interval 1h --rolling-split --horizon 5
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 .venv-timesfm/bin/python -m prop_trader.timesfm_fewshot_eval --run runs/forecast5/1h --csv data/current/1h.csv --rolling-split
python3 -m prop_trader.current_candles --out data/live
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 .venv-timesfm/bin/python -m prop_trader.timesfm_predict --run runs/forecast5/1h --csv data/live/1h.csv
```

1m/1h/1d에 각각 학습·평가·예측을 실행한 뒤 `python3 -m prop_trader.forecast_report`로 통합 차트를 생성합니다. 다운로드 파일과 결과 파일은 같은 경로에서 재실행하면 갱신됩니다. 데이터 원본 해시와 수집 기준은 manifest.json에 기록됩니다.

## 단일 종목 CLI

배치 예측(`recent_predictions.json`) 없이 종목 하나만 바로 보고 싶을 때 사용합니다. `data/live/{interval}.csv`가 있어야 하며 내부적으로 위 학습·few-shot 적응 로직(`timesfm_predict.predict_symbol`)을 그대로 재사용합니다.

```sh
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 .venv-timesfm/bin/python -m prop_trader.forecast --symbol DOGEUSDT --interval 1h
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 .venv-timesfm/bin/python -m prop_trader.forecast --symbol DOGEUSDT --interval 1m --plot
```

`--plot`은 관측 종가 실선 + 예측 5개 점선(과거 캔들을 미래로 이어붙이지 않음) + 예측 경계선 PNG를 `runs/forecast5/{interval}/{symbol}_forecast.png`에 저장합니다. LONG/SHORT/HOLD 신호는 t+5 수익률이 `--threshold`(기본 1%)를 넘고, 경로가 방향과 반대로 임계치 이상 움직이지 않으며(`min_predicted_return`/`max_predicted_return`), 경로 일관성(`forecast_consistency`)이 낮지 않을 때만 표시합니다. 즉 t+5 수익률이 같아도 중간에 반대로 크게 흔들린 경로는 HOLD로 내려갑니다. 연구용 신호이며 이 CLI는 주문을 전송하지 않습니다.

`runs/forecast5/{interval}/recent_predictions.json`의 각 레코드에는 `last_close`, `forecast`(horizon/timestamp/predicted_close), `return_t1..return_t5`, `max_predicted_return`, `min_predicted_return`, `forecast_slope`, `forecast_consistency`가 들어 있어 그대로 횡단면(cross-sectional) 롱/숏 랭킹에 쓸 수 있습니다. 기존 `points` 필드는 하위호환을 위해 유지됩니다.

## 시간봉 확장

지원 주기는 `forecast_time.SECONDS`에 있습니다: 현재 학습된 어댑터는 1m/1h/1d뿐이지만 3m/5m/15m/30m/4h도 이미 매핑돼 있어 `timesfm_run`으로 학습만 하면 나머지 파이프라인(예측/평가/CLI) 코드 변경 없이 그대로 동작합니다. `current_candles`가 실제로 내려받는 주기는 `current_candles.SUPPORTED`로 별도 관리합니다. 학습되지 않은 주기로 `forecast` CLI를 호출하면 "No trained adapter" 오류로 명확히 실패합니다.

## 평가와 baseline 비교

`timesfm_run`/`timesfm_fewshot_eval`이 만드는 `metrics_combined.json`은 5번째 봉 방향 정확도와 5개 종가 평균 로그오차만 봅니다. `evaluator.py`는 같은 walk-forward test split에서 더 엄격하게 비교합니다: naive(마지막 가격 유지), momentum(최근 10개 봉 평균 로그수익률 선형 외삽), zero-shot(어댑터 비활성 pretrained TimesFM), LoRA, LoRA+recent few-shot 다섯 모델 각각에 대해 수평선별 MAE/RMSE/MAPE/DA@1..DA@5와 Top-K(상위 20% 확신 예측) 방향 정밀도를 계산합니다. naive는 방향을 정의할 수 없어(항상 flat) DA/Top-K가 N/A로 표시됩니다(기존 `metrics_combined.json`의 `last_price` 동일 규칙).

```sh
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 .venv-timesfm/bin/python -m prop_trader.evaluator --csv-dir data/live
```

결과: 주기별 `runs/forecast5/{interval}/eval_full.json`, 통합 표 `runs/forecast5/EVAL_REPORT.md`. 학습은 재실행하지 않고 이미 저장된 adapter로만 추론합니다. `evaluator.py`는 원시 예측값도 `runs/forecast5/{interval}/eval_predictions.csv`에 저장합니다(모델별 window마다 예측 5개 로그수익률 + 실제 5개 로그수익률 + anchor 가격).

## 신호 기반 매매 백테스트

방향 정확도가 높다고 실제로 돈을 벌었는지는 별개입니다. `backtest_forecast.py`는 `eval_predictions.csv`(위 evaluator 결과, 모델 재추론 없음)를 읽어 각 window의 `decide_signal` 결과(LONG/SHORT/HOLD)를 실제 t+5 종가로 체결하는 단순 시뮬레이션을 돌립니다. 수수료+슬리피지 왕복 0.4%(`engine.Config` 기본값), 고정 명목 1000 USDT, 복리 없음, 동시 포지션 한도 없음 — 포트폴리오 시뮬레이션이 아니라 "이 신호가 비용을 넘는 엣지가 있는가"만 봅니다. 봉 내부 경로(고가/저가)를 모르므로 손절/익절은 시뮬레이션하지 않고 t+5 실제 종가로만 청산합니다.

```sh
python3 -m prop_trader.evaluator --csv-dir data/live   # eval_predictions.csv 생성(필요 시 재사용)
python3 -m prop_trader.backtest_forecast --threshold 0.005 --interval-threshold 1m=0.001
```

주기마다 예측 변동폭이 서로 다르므로(`1m` 예측 t+5 수익률은 최대 ±0.27%) `--interval-threshold`로 주기별 신호 임계값을 따로 줄 수 있습니다. 결과: `runs/forecast5/BACKTEST_REPORT.md`, 주기·모델별 `runs/forecast5/{interval}/backtest/{model}/{trades.csv,equity.csv,summary.json}`.

이번 실행 요약(threshold 1m 0.1%, 1h/1d 0.5%, 96 test window/주기, 6종목): **1h만 비용 차감 후 순이익**(LoRA/LoRA+few-shot 승률 64%, 총수익 +0.21%, MDD -1.59% — momentum -5.05%, zero-shot +0.13% 대비 우위). **1m은 모든 모델이 비용 차감 후 손실**(승률 11~12%대, 거래당 순수익 -0.28~-0.42%) — 애초에 신호 없음이 확인된 주기이므로 예상된 결과. **1d는 DA@1이 65%로 높은데도 모든 모델이 순손실**(LoRA -11.57%, momentum -6.63%, zero-shot -9.98%, MDD 최대 -16.9%) — t+1 방향은 맞아도 t+5 청산 시점 변동성과 거래당 0.4% 비용이 방향 정확도 우위를 잠식합니다. 방향 정확도가 높다고 자동으로 수익성을 뜻하지 않는다는 사례로 그대로 기록합니다.

## Web 대시보드

`dashboard.py`는 위에서 생성된 결과 파일만 읽어(모델 추론 없음) 신호 보드 + forecast explorer + 정확도 표 + 백테스트 표를 하나의 정적 HTML로 만듭니다. 서버 없이 로컬에서 열거나 그대로 공유할 수 있습니다.

```sh
python3 -m prop_trader.dashboard
```

결과: `runs/forecast5/dashboard.html`. 주기(1m/1h/1d) 전환, 종목 클릭 시 상세 차트(관측 종가 실선 + 예측 종가 점선, hover 툴팁)와 LONG/SHORT/HOLD 배지가 갱신됩니다. 재현하려면 위 forecast/evaluator/backtest_forecast 단계를 먼저 실행해 `recent_predictions.json`, `eval_full.json`, `backtest/{model}/summary.json`이 있어야 합니다.

## 출력과 한계

- `runs/forecast5/forecast.html`: 종목·봉 주기 선택, 과거 캔들 및 미래 종가 점선.
- `runs/forecast5/forecasts.json`: 18개 조합의 기준 시각·생성 시각·5개 봉 종료 시각과 예측 종가.
- 주기별 `adapter/`: 검증으로 선택한 LoRA, `recent_adapters/`: 종목별 최근 사례로 추가 적응한 LoRA.
- 주기별 `metrics_combined.json`: 기본 모델, LoRA, LoRA+few-shot, 짧은 입력, 마지막 가격 유지 비교(5번째 봉 기준, 요약용).
- `runs/forecast5/EVAL_REPORT.md`, 주기별 `eval_full.json`: naive/momentum/zero-shot/LoRA/LoRA+few-shot 5개 모델의 수평선별(DA@1..DA@5) 비교표(정밀 평가용).

첫 실행은 주기별 종목당 800봉, 학습 창 96개, 2 epoch인 소규모 파일럿입니다. 충분한 학습량·종목 일반화·실거래 성능을 검증했다는 뜻이 아닙니다. 방향 정확도는 마지막 5번째 봉의 변화 방향, 오차는 다섯 종가의 로그가격 오차입니다. 오차와 수익률은 다릅니다.

예측 기준 `as_of`는 마지막 완료 봉의 종료 경계입니다. 원본 CSV timestamp는 봉 시작 시각입니다. `generated_at`과 `as_of`를 구분하며 저장된 화면은 실시간 갱신되지 않습니다. 1분봉은 시간이 지나면 예측이 만료되므로 최신 데이터를 다시 수집하여 예측해야 합니다.

최근 입력/학습에 사용하는 값은 종가의 마지막 관측값 기준 로그 비율입니다. 거래량이나 고가·저가는 모델 특징으로 사용하지 않습니다. 추정 분위수를 보정된 확률로 표시하지 않으며, 가격 예측의 성과가 좋지 않아도 결과를 그대로 기록합니다. `EVAL_REPORT.md`의 1d 결과처럼 momentum이 LoRA+few-shot보다 DA@5가 높게 나오는 경우도 있으며, 그런 경우도 그대로 보고합니다 — LoRA/few-shot이 항상 이긴다고 가정하지 않습니다.

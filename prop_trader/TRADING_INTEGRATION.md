# 기술적 분석·신뢰성 검증·종목 순위·업비트 주문

## 기술적 분석 결합

`technical.py`는 참고용으로 놓아둔 **로컬 ta/ta 라이브러리**를 실제로 호출합니다. RSI14, MACD histogram, ATR14, Bollinger 위치, EMA20/50 차이, ADX14, CMF20, 거래량 비율, 1/5/20봉 수익률과 봉 범위를 계산합니다. 모든 특징은 예측 시점까지의 완료된 OHLCV에서만 계산하며 미래 데이터로 결측치를 채우지 않습니다. 가격이 일정해 Bollinger 폭이 0인 구간은 관측된 상수 창에 한해 중립값 0.5로 정의합니다.

TimesFM + LoRA + 최근 8개 사례 적응을 유지하고, 그 5개 종가 예측과 기술적 특징을 ridge 잔차 보정 모델에 함께 넣습니다. 보정은 기존 validation 구간에서만 학습합니다. 기술적 지표를 TimesFM 내부의 native covariate로 넣는 구조는 아닙니다.

`financial_pjt`의 옵션·선물 가격 수식과 ETF 자산배분 실습은 단기 코인 기술적 지표가 아니며, 앞서 확인된 수식 오류도 있어 예측 특징으로 재사용하지 않습니다.

주기별 `runs/hybrid/{interval}/calibrator.json`이 존재하면 기존 `forecast` CLI와 `timesfm_predict` 배치 예측은 자동으로 기술적 보정을 사용합니다. 원본 TimesFM 가격은 `timesfm_prices`, 사용한 지표는 `technical_features`, 적용 모델은 `predictor`에 함께 남습니다. 모델/보정 데이터 해시와 시점을 검증하여 다른 학습 스냅샷 또는 미래에 학습한 보정기를 거부합니다.

## 신뢰성 검증

```sh
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 .venv-timesfm/bin/python -m prop_trader.hybrid_research
```

- 입력은 학습 당시 해시와 일치하는 `data/current/{1m,1h,1d}.csv`. 바뀐 `data/live`로 경계를 재계산하지 않습니다.
- TimesFM+few-shot, 기술적 지표만, 결합 모델, momentum, 마지막 가격 유지를 동일한 예측 시점에서 비교합니다.
- 봉 종가에서 예측을 생성하고 실제 다음 봉 시가로 진입, 5번째 봉 실제 종가로 청산합니다. 업비트와 일치하게 롱 현물만 사용합니다.
- 초기 자금 10,000 USDT, 최대 3종목, 주문당 현재 자산의 20%, 사용 가능한 현금 이내로 제한합니다. 예상 순수익률이 비용 차감 후 0.2%를 초과해야 진입합니다.
- 기본 편도 수수료·슬리피지는 각각 0.1%. 스트레스에서 슬리피지 0.3%. 거래마다 손익과 현금을 대사합니다.
- 신뢰구간은 같은 시점의 종목들을 묶고 연속 2개 시점 블록을 재표본화합니다. 표본은 주기당 16개 원점 그룹으로 작습니다.

결과는 `runs/hybrid/REPORT.md`. 이 기간은 기존 연구에서 이미 확인했으므로 **새로운 미관측 holdout이 아닙니다**. 기술적 지표의 추가가 성능 향상을 보장하지 않습니다. Binance 데이터의 결과를 업비트 실거래 검증으로 주장하지 않습니다. MDD는 봉 시가 평가 기준이며, 봉 내부 손절·호가 충격·펀딩비는 시뮬레이션하지 않습니다. 기존 Claude 구현의 `backtest_forecast.py`는 신호별 단순 손익 분석으로 그대로 보존합니다.

## 청산 규칙 탐색 (익절/손절)

`live_trader.py`의 `STOP_FRACTION=2.5%, REWARD_MULTIPLE=1`(2.5% 익절)은 라이브 거래 29건의 실현폭을 한 번 보고 정한 값이며 백테스트로 검증되지 않았다. `exit_sweep.py`는 `eval_predictions.csv`(배포 모델 `lora_fewshot`)를 실제 캔들 고가/저가와 결합해 동일 봉 내 손절 우선·갭 시가 체결 규칙(`engine.Desk.step`과 동일)으로 실제 익절 도달 여부를 시뮬레이션하고, 고정 stop/target 그리드와 CoinsFM 예측경로 기반 적응형 목표가(예측 최고가 × shrink)를 확장형 walk-forward 교차검증으로 비교한다.

```sh
python3 -m prop_trader.exit_sweep
```

결과: `runs/forecast5/EXIT_SWEEP_REPORT.md`, 주기별 `runs/forecast5/{interval}/exit_sweep.json`. 진입 게이트를 통과하는 표본이 주기당 한 자릿수~10여 건으로 작아 결과는 참고용이다. 1m은 진입 게이트를 통과하는 구간이 사실상 없다(`BACKTEST_REPORT.md`의 1m 무신호 결과와 일치). 1d 표본외 결과에서는 어떤 고정 익절 값(현재 라이브 값 포함)도 "익절 없음, 손절+만료만"보다 못했다 — 익절 자체가 현재 형태로는 도움이 안 될 수 있다는 신호이며, 값 하나를 바꾼다고 해결되지 않을 수 있다. 여기서 나온 어떤 설정도 `live_server.py --mode paper`로 먼저 모의 검증한 뒤에만 실거래 상수를 바꿔야 한다.

## 예상 순수익률 순위

```sh
python3 -m prop_trader.upbit_data --interval 1h --top 12
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 .venv-timesfm/bin/python -m prop_trader.rank_coins --interval 1h --notional 10000 --refresh-orderbooks
python3 -m prop_trader.dashboard
```

공개 API만 사용합니다. 업비트 KRW 마켓 중 24시간 거래대금 상위 12개에서 유의 종목, 누락/무거래 봉이 있는 시계열을 제외합니다. 순위의 모집단은 이 표본이며 전체 코인에서 고르는 것이 아닙니다. `--interval 1m/1h/1d` 각각 지원합니다.

순위 = `예상 5번째 종가 × (1−매도 슬리피지) × (1−매도 수수료) ÷ (현재 매수 VWAP × (1+매수 수수료)) − 1` 내림차순입니다. 매수 VWAP은 현재 호가 깊이와 수수료 포함 총예산으로 계산하며, 보이는 호가에 충분한 물량이 없으면 거부합니다. 주기가 다른 예상 수익률을 섞지 않습니다.

`runs/rankings/{interval}.json/.csv/.md`에 예상 순수익률, 가정 투자금, 예상 손익, 원본/결합 예측, 지표, 기준 시각, 호가 만료 여부가 저장됩니다. 대시보드에는 업비트 순위 표가 추가됩니다. 예측은 모델의 점 추정치이며 통계적으로 교정된 기대소득이 아닙니다. Binance→Upbit, USDT→KRW, 학습에 없던 종목의 도메인 변경을 `research_unverified_cross_exchange`로 표시합니다. 실제 업비트 계정 수수료는 주문 가능 정보에서 확인하며, 순위의 기본 수수료는 보수적 실험 가정입니다.

## 업비트 API 설정

키는 채팅·소스·CLI 인자에 넣지 말고 로컬 환경에 설정합니다. `.env.example`은 이름만 정의한 빈 템플릿이며 실제 값은 없습니다. `.env`와 주문 일지는 `.gitignore`에 포함됩니다. `.env`는 Python이 자동으로 읽지 않습니다. 사용한다면 로컬에서 편집한 뒤 아래처럼 환경에 로드합니다.

```sh
cp .env.example .env
# 로컬 편집기로 .env에 본인 키 입력
set -a
source .env
set +a
```

공식 문서상 잔고·주문 조회 및 주문 권한과 허용 IP 설정이 필요합니다. 출금 기능은 구현하지 않았습니다. 인증은 HS512 JWT와 SHA512 query hash를 사용하며, 키나 Authorization 헤더를 결과에 기록하지 않습니다.

## 매수·매도 사용법

기본 동작은 **네트워크 요청 없는 미리보기**입니다.

```sh
python3 -m prop_trader.upbit_broker buy --market KRW-BTC --krw 10000 --identifier my-buy-001
python3 -m prop_trader.upbit_broker sell --market KRW-BTC --volume 0.0001 --identifier my-sell-001
```

키 설정 후 실제 주문을 만들지 않는 거래소 검증:

```sh
python3 -m prop_trader.upbit_broker buy --market KRW-BTC --krw 10000 --identifier my-buy-001 --mode test
```

실제 주문을 원할 때만 `--mode live`를 지정합니다. 아래 명령은 실제 자산을 매수·매도하므로 사용자가 시장·금액·수량을 정한 뒤 실행하는 명령입니다.

```sh
python3 -m prop_trader.upbit_broker buy --market KRW-BTC --krw 10000 --identifier my-buy-001 --mode live
python3 -m prop_trader.upbit_broker sell --market KRW-BTC --volume 0.0001 --identifier my-sell-001 --mode live
python3 -m prop_trader.upbit_broker status --identifier my-buy-001
python3 -m prop_trader.upbit_broker cancel --identifier my-buy-001 --mode live
python3 -m prop_trader.upbit_broker balances
```

- 시장가 매수는 KRW 총액, 시장가 매도는 보유 코인 수량입니다. `BTCUSDT` 같은 타 거래소 코드를 자동 변환하지 않습니다.
- 기본 주문 상한은 10,000 KRW입니다. `UPBIT_MAX_ORDER_KRW` 또는 `--max-krw`로 설정합니다. 시장가 매도 상한은 제출 전 현재가 기준 추정치이며 실제 체결 금액을 보장하지 않습니다.
- 잔고, 수수료, 종목 활성 상태, 지원 주문 유형, 최소/최대 금액을 조회하여 확인합니다.
- `identifier`는 주문별로 고유하게 정하고 응답 유실 시 같은 값을 유지합니다. 동일 주문 재전송은 SQLite 일지로 차단합니다. 불명확한 결과는 `UNKNOWN`으로 기록하고 `status`로 조회합니다. POST 자동 재시도는 없습니다.
- API가 주문을 접수한 상태와 실제 체결 완료를 구분합니다. 부분 체결 수량은 조회 결과에서 확인합니다. 이 CLI는 개별 주문 도구이며, 자동매매는 아래 live_server가 담당합니다.

공식 근거: [인증](https://docs.upbit.com/kr/reference/auth), [주문 가능 정보](https://docs.upbit.com/kr/reference/available-order-information), [주문 생성](https://docs.upbit.com/kr/reference/new-order), [주문 테스트](https://docs.upbit.com/kr/reference/order-test).

## 실시간 대시보드 및 자동매매

정적 HTML 대신 로컬 서버(`prop_trader/live_server.py`)가 30~60초마다 판단하는 자동매매 루프(`prop_trader/live_trader.py`)를 백그라운드로 돌리고, 브라우저는 `/api/*`를 5~15초마다 폴링해 화면을 갱신합니다. 서버는 기본적으로 `127.0.0.1`에만 바인딩하며, API 키와 Authorization 헤더는 서버 프로세스 밖으로 나가지 않습니다(브라우저는 이미 조회된 JSON만 받습니다).

```sh
set -a; source .env; set +a
python3 -m prop_trader.live_server --interval 1h --mode paper
# 브라우저에서 http://127.0.0.1:8899 접속
```

- **판단 주기 분리**: TimesFM 추론은 30초~1분 주기로 매번 재실행하기엔 너무 느립니다. 그래서 예측/순위(`runs/rankings/{interval}.json`)는 별도 스레드가 `--refresh-seconds`(기본 300초, 최소 60초)마다 기존 `upbit_data` → `rank_coins` 파이프라인을 서브프로세스로 재실행해 갱신하고, `--tick-seconds`(기본 30초) 루프는 그 캐시된 순위와 실시간 공개 시세만으로 매수·보유·매도를 판단합니다. 순위가 오래되면(`stale`) 신규 진입을 보류합니다.
- **판단 규칙**: 미보유 종목은 순위가 신선하고 비용 차감 후 기대수익이 임계값(기본 0.2%)을 초과할 때만 매수합니다. 보유 종목은 진입가 대비 2.5% 손절, 5% 익절(`engine.py` 백테스트와 동일한 기본값), 또는 예측 호라이즌(5개 봉) 경과 시 매도합니다. 그 외에는 보유 유지(STAY)입니다.
- **모드**: `paper`(가상 자금, 실주문 없음, 기본값) · `test`(`/v1/orders/test`로 계좌 검증만, 실주문 없음) · `live`(실제 주문). `live`는 `--mode live`와 `--confirm-live`를 모두 지정해야 시작됩니다. 종목당 주문 한도는 `--max-krw`(기본 `UPBIT_MAX_ORDER_KRW`, 10,000 KRW), 동시 보유는 `--max-positions`(기본 3)로 제한합니다.
- **잔액**: `/api/balance`가 실제 계좌의 KRW 잔액과 코인 평가금액을 조회합니다(읽기 전용, 주문 없음). **API 키에 접속 허용 IP가 설정돼 있으면 현재 서버를 실행하는 머신의 IP를 Upbit Open API 관리 페이지에 등록해야 401(`no_authorization_ip`) 없이 조회됩니다.**
- **수익률 시각화**: `runs/live/{mode}/equity.csv`에 틱마다 평가자산을 기록합니다. paper 모드는 가상 자금(`--paper-capital`, 기본 1,000,000 KRW) 기준, test/live 모드는 실제 계좌 평가자산 기준이며, 시작 시점 대비 누적 수익률을 카드와 라인 차트로 보여줍니다.
- **업비트 미상장 종목 정리**: 리서치 신호 보드는 Binance 학습 심볼(예: BONKUSDT, FLOKIUSDT) 중 대응하는 `KRW-*` 마켓이 없는 종목을 제외하고, 제외된 종목명은 화면 하단에 표기합니다(`prop_trader/upbit_symbols.py`).
- 자동매매 판단/체결 로그는 `runs/live/{mode}/decisions.jsonl`, 상태는 `runs/live/{mode}/state.json`에 남습니다.

## 검증

```sh
.venv-timesfm/bin/python -m unittest discover -s prop_trader/tests -v
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 .venv-timesfm/bin/python -m unittest discover -s prop_trader/ml_tests -v
```

2026-09-10 수정은 모의 클라이언트로 검증했으며 실제 주문은 실행하지 않았습니다.


## 2026-09-10 실행 안정성 수정

- 매수 금액은 수수료 포함 예산이다. 기본 비용/손절 가정상 최소 진입 예산은 약 5,190원이며, 5,000원 설정이면 신규 매수를 건너뛴다. 가격 급락으로 생긴 최소 주문액 미만 잔량은 삭제하거나 추가 매수하지 않고 대기 상태로 둔다.
- 봇 청산에는 매수 원화 한도를 적용하지 않는다. 실제 봇 소유 수량, 사용 가능 잔고와 거래소 한도를 모두 확인한다. CLI 개별 매도는 기존 금액 상한을 유지한다.
- 매수/매도 접수 시 영속적인 pending 상태를 먼저 저장한다. done/cancel 상태와 체결 상세를 확인한 후 수량·수수료·원가·손익을 반영한다. 부분 체결 뒤 취소된 잔량도 유지한다. 대기 또는 결과 불명확 주문이 있으면 추가 주문을 보류한다.
- 재시작 시 pending 주문을 identifier로 조회한다. 404나 통신 장애를 주문 실패로 단정해 새 주문을 넣지 않는다. 상태 파일을 삭제해서 우회하면 안 된다. 기존 버전의 추정 포지션은 원래 매수 주문의 실제 체결 내역과 잔고가 맞을 때만 마이그레이션한다. 불일치는 화면에 표시하고 실행을 보류한다.
- 같은 tick 및 같은 예측의 재진입을 차단한다. 청산 후 최소 한 봉 동안 신규 진입을 보류한다.
- 예측 기준 시각이 현재 봉에 속하는지 확인한 뒤 10초 이내 주문장으로 진입 순수익과 순위를 다시 계산한다. 화면의 저장된 순위와 실제 주문 순간 순위는 다를 수 있다.
- 고정 손절/익절에 더해, 신선한 예측의 마지막 종가가 현재가보다 진입 임계값 이상 낮으면 청산한다. 예측 만료는 매수 시각이 아닌 원래 예측의 마지막 목표 시각이다. 변동성 추적 손절과 확률 기반 순위는 이번 수정에 포함하지 않았다.
- API 요청을 프로세스 내에서 간격 제한하고 HTTP 429 이후 60초간 대기한다. POST를 자동 재전송하지 않는다. 동일 모드의 중복 실행은 프로세스 잠금으로 차단한다.
- BUY_INTENT에 당시 예측과 재계산 가격을 기록한다. BUY/SELL 체결 기록에는 실제 수량·금액·수수료가 포함된다. 계좌 전체 수익률은 외부 입출금/기존 보유자산의 영향과 별개로 해석해야 한다.

적용하려면 기존 서버를 Ctrl+C로 종료하고 **기존 모드·봉 주기·금액·포지션 수 설정을 유지하여 재시작**한다. 이 작업에서는 실행 중인 서버를 재시작하거나 실제 주문을 보내지 않았다. 모의 검증 실행 예:

```sh
python3 -m prop_trader.live_server --mode paper --interval 1h --max-krw 8000 --no-daily-retrain
```

회귀 테스트: `python3 -m unittest discover -s prop_trader/tests -q`.
체결 처리 테스트 통과는 수익성 검증이 아니다. 새 청산 규칙은 실거래 확대 전 별도 시간순 모의 검증이 필요하다.

## WebSocket · 상주 모델 · 다중 주기 · 유동 수량 (2026-09-10)

이번 변경은 `live_server`와 `live_trader`의 `start()` 경로에 기본 적용된다. 기본 판단 주기는 10초이며 작업 시간을 제외한 나머지만 기다린다. 처리 시간이 10초를 초과하면 다음 판단이 지연될 수 있으며 중첩 실행하지 않는다. 대시보드에 최근 추론 시간과 각 주기의 예측 상태를 표시한다.

- **시세와 호가:** 공개 WebSocket의 ticker와 orderbook을 구독한다. 연결 해제/10초 이상 지난 데이터는 신규 진입에 사용하지 않는다. 연결 장애에도 기존 포지션의 보호 청산은 REST 현재가로 판단한다. 재연결은 지수 대기하며 구독과 스냅샷을 다시 받는다.
- **추론:** 별도 프로세스가 TimesFM 기본 모델 하나를 메모리에 유지한다. 각 주기의 기존 LoRA를 초기 상태로 적용하고 최신 8개 완성 예제로 Few-shot을 수행한다. 모델 입력에 미완성 봉을 끼워 넣지 않는다. 현재가 변화는 각 예측까지의 남은 수익 계산에 즉시 반영한다.
- **갱신:** `(종목, 주기, 마지막 완성 봉)`으로 중복 Few-shot을 막는다. 새 봉이 생길 때 해당 작업만 수행한다. 보유 종목을 우선하며 다음으로 거래대금 상위 후보 안에서 직전 신호가 강한 종목부터 처리한다. 초기 로딩/시장 급변 시 전 종목 갱신이 10초 내 완료된다는 보장은 없다. 느린 예측 때문에 주문 판단 스레드는 대기하지 않는다.
- **세 주기 판단:** 일봉 5일 순수익이 양수이고, 시간봉 5시간 순수익이 진입 임계값(기본 0.2%)을 넘고, 분봉 5분 순수익도 양수일 때 진입 후보가 된다. 세 주기의 수익률을 평균내지 않는다. 세 예측 중 하나라도 빠지거나 오래됐으면 신규 매수하지 않는다. 보유 중에는 일봉/시간봉 하락 또는 시간봉 여력이 없으면서 분봉 하락일 때 청산 후보가 된다. 고정 손절·익절·예측 만료 청산은 별도로 유지한다.
- **유동 매수 수량:** 시간봉 순수익을 `ATR × sqrt(5)`로 나눈 값을 0~1로 제한한 신호 강도를 사용한다. 원시 TimesFM 방향이 반대이면 강도를 절반으로 낮춘다. 목표 매수 예산은 `주문 상한 × (0.35 + 0.65 × 강도)`이며 가용 KRW의 90% 이내로 제한한다. 최소 안전 진입 금액에 못 미치면 매수하지 않는다. 이 강도는 수익 확률이 아니며 통계적으로 보정된 신뢰도가 아니다.
- **유동 매도 수량:** 약한 시간봉 악화는 50%, 강한 악화나 일봉 하락은 100% 청산 후보로 삼는다. 보호 손절·기존 목표가·만료 청산은 전량이다. 부분 매도 후 매도분과 잔여분이 모두 최소 주문금액을 충족하지 못하면 전량으로 처리한다. 따라서 8,000원 규모 포지션은 대개 분할 매도가 불가능하다. 실제 보유 수량과 부분 체결 잔량을 기준으로 처리한다.
- **화면:** 각 종목의 분봉/시간봉/일봉 순수익, 신호 강도, 매수 예산, 종합 판단과 이유를 표시한다. 화면 예측도 기존 정적 파일 대신 상주 예측 프로세스에서 읽는다. WebSocket 상태와 추론 상태도 표시한다.

기존 `--interval`은 포지션 만료/재진입 제한 주기 호환용이다. 어떤 값을 주어도 매매 판단은 세 주기를 모두 사용한다. `--refresh-seconds`는 호환용이며 상주 모드에서는 완성 봉으로 갱신을 결정한다. 상주 추론 도중 전체 LoRA/보정기를 바꾸는 기존 일일 재학습은 실행하지 않는다. 전체 모델 재학습은 별도 오프라인 작업 후 서버를 재시작한다.

기존 서버를 종료한 뒤 실행한다. 아래는 **실제 자동매매 재개** 명령이며, 종목당 상한 8,000원·최대 6개·후보 20개 설정 예다. 주문 상한은 강도가 높아도 초과하지 않는다.

```sh
cd /Users/junseokshim/workspace/auto
set -a
source .env
set +a
.venv-timesfm/bin/python -m prop_trader.live_server \
  --mode live --confirm-live --interval 1h --tick-seconds 10 \
  --max-krw 8000 --max-positions 6 --universe-top 20
```

모의 실행은 `--mode paper`로 바꾸고 `--confirm-live`를 제거한다. 기존 `python3 -m prop_trader.live_server`도 WebSocket 패키지가 없으면 프로젝트 가상환경으로 전환한다. 의존성은 `requirements-timesfm.lock.txt`에 추가했다.

예측 기록은 `runs/live/{mode}/realtime/forecast_history.jsonl`, 상주 프로세스 로그는 같은 폴더의 `worker.log`에 남는다. 현재 상태 파일을 삭제하지 말고 기존 서버를 먼저 종료한다. 정상 SIGTERM/Ctrl+C는 예측 프로세스도 종료하며, 부모 프로세스가 사라지면 작업자도 현재 작업 후 종료한다.

검증: 단위/회귀 테스트, 실제 공개 WebSocket ticker/orderbook 수신, BTC 3개 주기 실제 모델 예측, 임시 모의매매 상태의 수신·추론·판단 연결을 확인했다. 실제 계좌 주문은 실행하지 않았다. 새 복합 규칙과 신호 강도별 자금 배분의 수익성은 아직 시간순 실거래 재생으로 검증하지 않았다.

공식 프로토콜: [WebSocket 안내](https://docs.upbit.com/kr/reference/websocket-guide), [현재가](https://docs.upbit.com/kr/reference/websocket-ticker), [호가](https://docs.upbit.com/kr/reference/websocket-orderbook).

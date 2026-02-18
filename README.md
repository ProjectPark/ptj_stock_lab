# PTJ Stock Lab

PTJ 매매법 실험/시뮬레이션 전용 저장소.

백테스트, 하이퍼파라미터 최적화, 전략 실험을 이 repo에서 수행하고,
검증이 완료된 시그널 엔진은 **ptj_stock** repo로 프로모션하여 실서비스에 배포한다.

> 서빙/배포 코드는 [ptj_stock](https://github.com/your-org/ptj_stock) repo에서 관리합니다.

---

## 프로젝트 구조

```
ptj_stock_lab/
├── strategies/          # 시그널 엔진 (v1~v5)
│   ├── signals.py           # v1 — 기본 5개 규칙
│   ├── signals_v2.py        # v2 — 파라미터화, 임계값 조정
│   ├── signals_v3.py        # v3 — Optuna 최적화 반영
│   ├── signals_v4.py        # v4 — 스윙 이벤트, 서킷브레이커 감지
│   └── signals_v5.py        # v5 — 횡보장 감지, 진입 시간 제한, 리스크 제어
│
├── backtests/           # 백테스트 스크립트
│   ├── backtest.py          # v1 백테스트
│   ├── backtest_v2.py       # v2 백테스트
│   ├── backtest_v3.py       # v3 백테스트
│   ├── backtest_v4.py       # v4 백테스트
│   ├── backtest_v5.py       # v5 백테스트
│   ├── backtest_common.py   # 공통 백테스트 유틸
│   ├── backtest_twin_gap.py # 쌍둥이 갭 전용 백테스트
│   └── backtest_5min_signals.py  # 5분봉 시그널 백테스트
│
├── optimizers/          # Optuna 기반 하이퍼파라미터 최적화
│   ├── optimize_v2_optuna.py
│   ├── optimize_v3_optuna.py
│   ├── optimize_v3_train_test.py  # Train/Test 분할 검증
│   ├── optimize_v4_optuna.py
│   ├── optimize_v5_optuna.py
│   ├── optimize_stoploss.py       # 손절 라인 최적화
│   └── optimize_with_fees.py      # 수수료 포함 최적화
│
├── experiments/         # 실험, 분석, compliance 평가
│   ├── experiment_v4_weights.py   # v4 가중치 실험
│   ├── experiment_v5_weights.py   # v5 가중치 실험
│   ├── evaluate_compliance.py     # 규칙 준수율 평가
│   ├── analyze_trial_79.py        # 특정 trial 분석
│   ├── run_trial_79_backtest.py   # 특정 trial 백테스트
│   ├── check_study.py             # Optuna study 확인
│   ├── monitor_train_test.py      # Train/Test 모니터링
│   └── test_v2_params.py          # v2 파라미터 검증
│
├── fetchers/            # 데이터 수집 (KIS API, yfinance, Polygon)
│   ├── fetcher.py               # yfinance 기반 fetcher
│   ├── fetcher_kis.py           # 한국투자증권 API fetcher
│   ├── kis_client.py            # KIS API 클라이언트
│   ├── fetch_data.py            # 데이터 수집 진입점
│   ├── collect_poly_history.py  # Polymarket 히스토리 수집
│   └── collect_poly_history_async.py  # 비동기 Polymarket 수집
│
├── polymarket/          # Polymarket 연동 모듈
│   ├── poly_config.py       # Polymarket 설정
│   ├── poly_fetcher.py      # Polymarket 데이터 수집
│   ├── poly_signals.py      # Polymarket 시그널 생성
│   ├── poly_history.py      # Polymarket 히스토리
│   └── poly_history_async.py  # 비동기 히스토리
│
├── history/             # 거래내역 (증권사 거래명세)
│   ├── 2024/                # 2024 거래내역 CSV, 분석 결과
│   ├── 2025/                # 2025 거래내역 CSV
│   └── tools/               # PDF -> CSV 변환 도구
│
├── docs/                # 리포트, 트레이딩 규칙서
│   ├── trading_rules_v1~v5.md   # 버전별 매매 규칙서
│   ├── ARCHITECTURE.md          # 시스템 아키텍처 문서
│   ├── simulation_architecture.md  # 시뮬레이션 아키텍처
│   ├── *_report.md              # 백테스트/최적화 결과 리포트
│   └── optuna_best_practices.md # Optuna 최적화 가이드
│
├── scripts/             # 유틸리티 스크립트
│   ├── generate_charts.py       # 차트 생성
│   ├── analyze_with_fees.py     # 수수료 포함 분석
│   ├── hold_time_analysis.py    # 보유 시간 분석
│   ├── download_daily.py        # 일봉 데이터 다운로드
│   └── check_v45_parity.sh      # v4/v5 패리티 검증
│
├── data/                # 데이터 디렉토리 (gitignore 대상)
│   ├── parquet/             # 종목별 시세 데이터 (.parquet)
│   ├── optuna/              # Optuna 최적화 DB (.db)
│   └── results/             # 백테스트 결과 JSON
│
├── config.py            # 공통 설정 (종목, 페어, 매매 파라미터)
├── run.py               # CLI 실행 진입점
├── dashboard.py         # Streamlit 대시보드
├── dashboard_html.py    # HTML 대시보드 생성기
├── app.py               # Legacy Streamlit app
└── requirements.txt     # Python 의존성
```

---

## 전략 버전 히스토리

| 버전 | 파일 | 핵심 변경 |
|---|---|---|
| v1 | `signals.py` | 기본 5개 규칙 (금 시황, 쌍둥이 갭, 조건부 매매, 손절, 하락장) |
| v2 | `signals_v2.py` | 파라미터화, 임계값 조정, Polymarket 강세장 모드 도입 |
| v3 | `signals_v3.py` | Optuna 최적화 반영, 선별 매매형 (갭/트리거 기준 상향) |
| v4 | `signals_v4.py` | 스윙 이벤트, 서킷브레이커(CB) 감지, 급락 역매수, CONL 조건부 |
| v5 | `signals_v5.py` | 횡보장 감지, 진입 시간 제한, 레버리지 배수별 손절, VIX 방어모드 |

각 버전의 상세 매매 규칙은 `docs/trading_rules_v1.md` ~ `docs/trading_rules_v5.md`에 정의되어 있다.

---

## 엔진 프로모션 워크플로우

이 repo에서 검증된 전략을 실서비스(ptj_stock)에 반영하는 절차:

```
ptj_stock_lab (실험)                        ptj_stock (배포)
─────────────────────                       ─────────────────
1. 새 전략 개발 & 백테스트
2. Optuna 최적화로 파라미터 확정
3. strategies/signals_vN.py 정리
                          ──── 복사 ────>   backend/app/core/signals.py
                                            4. make dev 로컬 테스트
                                            5. git push -> 배포
```

---

## 실행 방법

### 환경 설정

```bash
# pyenv 환경 활성화 (market 환경 사용)
pyenv shell market

# 의존성 설치
pip install -r requirements.txt
```

추가로 Optuna 최적화를 실행하려면 `optuna` 패키지가 필요하다:
```bash
pip install optuna
```

### 백테스트

```bash
# v5 백테스트 실행
pyenv shell market && python backtests/backtest_v5.py

# v4 백테스트 실행
pyenv shell market && python backtests/backtest_v4.py

# 쌍둥이 갭 전용 백테스트
pyenv shell market && python backtests/backtest_twin_gap.py
```

### Optuna 최적화

```bash
# v5 하이퍼파라미터 최적화
pyenv shell market && python optimizers/optimize_v5_optuna.py

# 손절 라인 최적화
pyenv shell market && python optimizers/optimize_stoploss.py

# 수수료 포함 최적화
pyenv shell market && python optimizers/optimize_with_fees.py
```

### 실험/분석

```bash
# 규칙 준수율 평가
pyenv shell market && python experiments/evaluate_compliance.py

# 가중치 실험
pyenv shell market && python experiments/experiment_v5_weights.py
```

### 대시보드

```bash
# Streamlit 대시보드
pyenv shell market && streamlit run dashboard.py

# CLI 대시보드 (콘솔 출력 + HTML 생성)
pyenv shell market && python run.py
```

---

## 데이터 관리

`data/` 디렉토리는 `.gitignore` 대상이다. 아래 파일들은 로컬에서만 관리한다:

- `data/parquet/` -- 종목별 시세 데이터 (yfinance, KIS API에서 수집)
- `data/optuna/` -- Optuna 최적화 DB (`.db` 파일)
- `data/results/` -- 백테스트 결과 JSON

데이터 수집:
```bash
# yfinance를 통한 시세 데이터 수집
pyenv shell market && python fetchers/fetch_data.py

# Polymarket 히스토리 수집
pyenv shell market && python fetchers/collect_poly_history.py
```

---

## 거래내역 관리

`history/` 디렉토리에 증권사 거래명세를 연도별로 보관한다.

- `history/2024/` -- 2024년 거래내역 CSV 및 분석 결과
- `history/2025/` -- 2025년 거래내역 CSV
- `history/tools/` -- PDF에서 CSV로 변환하는 도구 (`pdf_to_csv.py`)

> PDF, 이미지 원본은 `.gitignore` 대상이며, 변환된 CSV만 추적한다.

---

## 주요 의존성

- Python 3.10+
- yfinance -- 미국 주식 시세 데이터
- pandas / pyarrow -- 데이터 처리 및 parquet I/O
- streamlit -- 대시보드 UI
- optuna -- 하이퍼파라미터 최적화 (최적화 실행 시)

---

## 관련 저장소

| 저장소 | 용도 |
|---|---|
| **ptj_stock** | 배포 서버용 (FastAPI backend, React frontend, Docker) |
| **ptj_stock_lab** (여기) | 실험용 (백테스트, 최적화, 시그널 연구) |

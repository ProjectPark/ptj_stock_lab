# 데이터 관리 가이드 — ptj_stock_lab

이 문서는 Claude와 소통할 때 데이터 구조, 수집 방법, 사용법을 참조하기 위한 가이드입니다.

## 데이터 디렉토리 구조

```
data/
├── market/                    # 시장 데이터 (외부 API에서 수집)
│   ├── cache/                 # 종목별 일봉 캐시 (자동 갱신, 1시간 만료)
│   │   └── {TICKER}.parquet   # 예: GLD.parquet, BITU.parquet (16개)
│   ├── ohlcv/                 # 백테스트용 합산 분봉 (전 종목 포함)
│   │   ├── backtest_1min.parquet      # v1 1분봉 (146만행)
│   │   ├── backtest_1min_v2.parquet   # v2 1분봉 (149만행)
│   │   └── backtest_5min.parquet      # 5분봉 (30만행)
│   ├── daily/                 # 일봉 데이터
│   │   ├── market_daily.parquet       # 32종목 일봉 (MultiIndex)
│   │   ├── soxx_iren_daily.parquet    # SOXX/IREN 일봉
│   │   └── history.parquet            # yfinance 3개월 히스토리 캐시
│   └── fx/                    # 환율
│       └── usdkrw_hourly.parquet      # USD/KRW 시간봉
│
├── polymarket/                # Polymarket 확률 데이터 (연도별)
│   ├── 2024/                  # 335개 JSON
│   ├── 2025/                  # 365개 JSON
│   ├── 2026/                  # 51개+ JSON
│   └── _cache/                # 이벤트 캐시
│       └── events_cache.json
│
├── optuna/                    # Optuna 최적화 DB
│   ├── optuna_v2.db
│   ├── optuna_v3.db
│   ├── optuna_v3_phase2.db
│   ├── optuna_v3_train_test.db
│   ├── optuna_v3_train_test_v2.db
│   └── optuna_v3_train_test_wide.db
│
├── results/                   # 실행 결과 출력
│   ├── backtests/             # 백테스트 트레이드 로그 CSV
│   ├── optimization/          # 최적화 결과 CSV
│   ├── analysis/              # 분석/compliance CSV
│   ├── baselines/             # 기준 성과 JSON (v3, v4, v5)
│   └── events/                # 이벤트 로그 JSONL (CB, Swing)
│
└── meta/                      # 메타데이터
    ├── ticker_mapping.json    # 티커 심볼 매핑
    └── fee_analysis_data.json # 수수료 분석 데이터
```

## config.py 경로 상수

모든 코드는 아래 상수를 사용합니다. 하드코딩된 경로를 사용하지 마세요.

| 상수 | 경로 | 용도 |
|---|---|---|
| `config.DATA_DIR` | `data/` | 데이터 루트 |
| `config.CACHE_DIR` | `data/market/cache/` | 종목별 캐시 |
| `config.OHLCV_DIR` | `data/market/ohlcv/` | 합산 분봉 |
| `config.DAILY_DIR` | `data/market/daily/` | 일봉 |
| `config.FX_DIR` | `data/market/fx/` | 환율 |
| `config.POLY_DATA_DIR` | `data/polymarket/` | Polymarket |
| `config.OPTUNA_DIR` | `data/optuna/` | Optuna DB |
| `config.RESULTS_DIR` | `data/results/` | 결과 출력 |
| `config.META_DIR` | `data/meta/` | 메타데이터 |

## 데이터 수집 방법

### 1. 종목별 일봉 캐시 (자동)

```bash
# fetchers/fetch_data.py — yfinance로 16종목 일봉 수집
# 캐시 만료: 1시간 (config.CACHE_EXPIRE_HOURS)
# 저장: data/market/cache/{TICKER}.parquet
pyenv shell ptj_stock_lab && python fetchers/fetch_data.py
```

- `fetch_ticker("GLD")` — 단일 종목
- `fetch_all()` — 전체 종목

### 2. 장중 실시간 데이터

```bash
# fetchers/fetcher.py — yfinance 1분봉 (장중)
# fetchers/fetcher_kis.py — 한국투자증권 API (실시간)
# 저장: 메모리 (fetcher_kis) 또는 data/market/daily/history.parquet (fetcher)
```

- `fetcher.fetch_intraday()` — yfinance 장중 1분봉
- `fetcher_kis.fetch_intraday()` — KIS API 장중 1분봉
- `fetcher.fetch_history()` — 3개월 일봉 히스토리

### 3. 백테스트용 합산 분봉 (수동)

```bash
# backtests/backtest.py 내부 함수로 수집
# Alpaca API 사용 (ALPACA_API_KEY, ALPACA_SECRET_KEY 필요)
# 저장: data/market/ohlcv/backtest_*.parquet
pyenv shell ptj_stock_lab && python -c "
from backtests.backtest import fetch_1min_data_v2
fetch_1min_data_v2(use_cache=False)
"
```

- `fetch_backtest_data()` — 5분봉 (전 종목)
- `fetch_1min_data()` — 1분봉 v1
- `fetch_1min_data_v2()` — 1분봉 v2 (현재 사용)
- `fetch_usdkrw_hourly()` — USD/KRW 환율

### 4. 시장 일봉 (수동)

```bash
# yfinance 32종목 일봉
pyenv shell ptj_stock_lab && python scripts/download_daily.py

# Alpaca SOXX/IREN 일봉
pyenv shell ptj_stock_lab && python scripts/download_alpaca_soxx_iren.py
```

### 5. Polymarket 확률 (수동)

```bash
# 단일 날짜
pyenv shell ptj_stock_lab && python fetchers/collect_poly_history.py --date 2026-02-19 --fidelity 1

# 날짜 범위 (비동기, 빠름)
pyenv shell ptj_stock_lab && python fetchers/collect_poly_history_async.py --start 2026-02-01 --end 2026-02-19

# 저장된 파일 목록
pyenv shell ptj_stock_lab && python fetchers/collect_poly_history.py --list
```

- 저장: `data/polymarket/{연도}/{날짜}_{fidelity}m.json`
- 연도별 하위 폴더에 자동 저장

## 데이터 사용 패턴

### 백테스트 실행

```bash
# v5 백테스트 (가장 최신)
pyenv shell ptj_stock_lab && python backtests/backtest_v5.py
```

읽는 데이터:
- `config.OHLCV_DIR / "backtest_1min_v2.parquet"` — 전 종목 1분봉
- `config.POLY_DATA_DIR` — Polymarket 확률 (rglob으로 연도별 탐색)

출력:
- `config.RESULTS_DIR / "backtests" / "backtest_v5_trades.csv"`

### Optuna 최적화

```bash
pyenv shell ptj_stock_lab && python optimizers/optimize_v5_optuna.py
```

읽는 데이터: 백테스트와 동일
출력:
- `config.OPTUNA_DIR / "optuna_v5.db"` — Optuna study DB
- `config.RESULTS_DIR / "baselines" / "v5_baseline_result.json"` — 기준 성과

### 대시보드

```bash
pyenv shell ptj_stock_lab && streamlit run dashboard.py
```

읽는 데이터:
- `config.CACHE_DIR / "{TICKER}.parquet"` — 종목별 캐시
- `config.DAILY_DIR / "history.parquet"` — 히스토리

## 데이터 갱신 주기

| 데이터 | 갱신 방법 | 주기 |
|---|---|---|
| 종목별 캐시 (`cache/`) | 자동 (1시간 만료) | 매 실행 시 |
| 히스토리 (`daily/history.parquet`) | 자동 (1시간 만료) | 매 실행 시 |
| 합산 분봉 (`ohlcv/`) | 수동 재수집 | 필요 시 |
| 시장 일봉 (`daily/market_daily.parquet`) | 수동 스크립트 | 필요 시 |
| Polymarket (`polymarket/`) | 수동 스크립트 | 매일 또는 필요 시 |
| Optuna DB (`optuna/`) | 최적화 실행 시 | 전략 변경 시 |
| 결과 (`results/`) | 백테스트/분석 실행 시 | 매 실행 시 |

## 주의사항

- **합산 분봉 파일은 종목 분리하지 않는다** — 백테스트 엔진이 전 종목을 한번에 로드해야 성능이 나옴
- **Polymarket JSON은 연도별 폴더에 저장** — `backtest_common.py`가 `rglob("*.json")`으로 재귀 탐색
- **새 데이터 파일 추가 시** 반드시 `config.py`에 경로 상수 추가
- **`.gitignore`에 `data/` 하위 전체가 포함됨** — 데이터 파일은 git에 커밋하지 않음
- **API 키 필요**: KIS(`.env`), Alpaca(`.env`), Polymarket(키 불요)

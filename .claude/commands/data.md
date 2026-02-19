# /data — 시뮬레이션 데이터 관리

시뮬레이션에 사용되는 데이터의 상태 조회, 수집/갱신, 검증, 정리를 수행합니다.

## 사용자 인자

$ARGUMENTS

## 인자 파싱 규칙

| 인자 패턴 | 액션 |
|-----------|------|
| (빈 값) | 전체 상태 요약 출력 |
| `status` | 상세 상태 출력 (`python scripts/data_manager.py status`) |
| `status alpaca` | Alpaca 캐시만 상태 출력 |
| `status fx` | FX 캐시만 상태 출력 |
| `status polymarket` | Polymarket만 상태 출력 |
| `fetch 1min-v1` | 1분봉 v1 수집: `from fetchers import fetch_1min_v1; fetch_1min_v1(use_cache=False)` |
| `fetch 1min-v2` | 1분봉 v2 수집: `from fetchers import fetch_1min_v2; fetch_1min_v2(use_cache=False)` |
| `fetch 5min` | 5분봉 수집: `from fetchers import fetch_5min_v1; fetch_5min_v1(use_cache=False)` |
| `fetch fx` | 환율 수집: `from fetchers import fetch_usdkrw_hourly; fetch_usdkrw_hourly(use_cache=False)` |
| `fetch daily [TICKERS]` | 일봉 수집: `from fetchers import fetch_daily; fetch_daily([...])` |
| `fetch polymarket [DATE]` | Polymarket 히스토리 수집 |
| `fetch all` | 모든 핵심 캐시 갱신 (1min-v2, 5min, fx) |
| `validate` | 캐시 무결성 검증 (`python scripts/data_manager.py validate`) |
| `clean` | 정리 대상 확인 (`python scripts/data_manager.py clean --dry-run`) |
| `clean --force` | 실제 정리 실행 (`python scripts/data_manager.py clean`) |

## 실행 방법

### Python 환경
모든 Python 명령은 반드시 `pyenv shell ptj_stock_lab && python ...` 형태로 실행한다.

### 1. 상태 조회 (status)

```bash
pyenv shell ptj_stock_lab && python scripts/data_manager.py status
```

출력 결과를 사용자에게 보여주고, 누락된 캐시가 있으면 수집을 제안한다.

### 2. 데이터 수집 (fetch)

수집 전 **반드시 현재 캐시 상태를 먼저 확인**한다. 캐시가 이미 존재하면 사용자에게 덮어쓸지 확인한다.

#### Alpaca 분봉
```bash
pyenv shell ptj_stock_lab && python -c "
from fetchers import fetch_1min_v2
df = fetch_1min_v2(use_cache=False)
print(f'수집 완료: {len(df):,} rows, 종목: {sorted(df[\"symbol\"].unique())}')
"
```

#### 기간 지정 수집
사용자가 기간을 지정하면:
```bash
pyenv shell ptj_stock_lab && python -c "
from datetime import date
from fetchers import fetch_bars
import config
df = fetch_bars(
    list(config.TICKERS_V2.keys()), 1,
    date(2025, 6, 1), date(2025, 12, 31),
    cache_path=config.OHLCV_DIR / 'custom_1min.parquet',
    ticker_map=config.ALPACA_TICKER_MAP,
    ticker_reverse_map=config.ALPACA_TICKER_REVERSE,
)
"
```

#### Polymarket 히스토리
```bash
pyenv shell ptj_stock_lab && python fetchers/collect_poly_history.py --date 2026-02-18 --fidelity 5
```

범위 수집 (비동기 고속):
```bash
pyenv shell ptj_stock_lab && python -c "
import asyncio, logging
logging.basicConfig(level=logging.INFO)
from datetime import date
from polymarket.poly_history_async import collect_range_async
asyncio.run(collect_range_async(date(2026, 2, 1), date(2026, 2, 18), fidelity=5))
"
```

### 3. 검증 (validate)

```bash
pyenv shell ptj_stock_lab && python scripts/data_manager.py validate
```

문제가 발견되면 원인과 해결 방법을 설명한다.

### 4. 정리 (clean)

먼저 dry-run으로 대상 확인:
```bash
pyenv shell ptj_stock_lab && python scripts/data_manager.py clean --dry-run
```

사용자가 승인하면 실제 삭제:
```bash
pyenv shell ptj_stock_lab && python scripts/data_manager.py clean
```

## 데이터 소스 정리

| 소스 | 캐시 위치 | 형식 | 수집 방법 |
|------|----------|------|----------|
| Alpaca 1분봉 v1 | `data/market/ohlcv/backtest_1min.parquet` | parquet | `fetch_1min_v1()` |
| Alpaca 1분봉 v2 | `data/market/ohlcv/backtest_1min_v2.parquet` | parquet | `fetch_1min_v2()` |
| Alpaca 5분봉 | `data/market/ohlcv/backtest_5min.parquet` | parquet | `fetch_5min_v1()` |
| Alpaca 일봉 | `data/market/daily/*.parquet` | parquet | `fetch_daily()` |
| USD/KRW 환율 | `data/market/fx/usdkrw_hourly.parquet` | parquet | `fetch_usdkrw_hourly()` |
| Polymarket 확률 | `data/polymarket/*.json` | JSON | `collect_history_for_date()` |
| Polymarket 캐시 | `data/polymarket/_cache/` | JSON | 자동 생성 |

## 주의사항

- Alpaca API는 rate limit이 있으므로 1분봉 전체 갱신 시 수 분 소요
- Polymarket 비동기 수집은 수백 일도 ~15분 내 완료
- `fetch all`은 1min-v2 + 5min + fx만 갱신 (가장 자주 사용되는 핵심 캐시)
- 수집 중 오류 발생 시 부분 데이터가 캐시에 저장될 수 있으므로 validate로 확인

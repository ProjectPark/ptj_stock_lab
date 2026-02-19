# PTJ v3 Baseline 시뮬레이션 리포트

> 생성일: 2026-02-18 04:09

## 1. 개요

현재 `config.py`에 설정된 v3 파라미터로 백테스트를 실행한 결과입니다.

## 2. 핵심 지표

| 지표 | 값 |
|---|---|
| 총 수익률 | **-7.38%** |
| 최종 자산 | 18,524,821원 |
| MDD | -10.13% |
| Sharpe Ratio | -0.9150 |
| 총 손익 | -1,475,179원 |
| 총 수수료 | 5,774,855원 |

## 3. 매매 통계

| 지표 | 값 |
|---|---|
| 거래일 | 240일 |
| 매수 횟수 | 390회 |
| 매도 횟수 | 2255회 |
| 승/패 | 1296W / 959L |
| 승률 | 57.5% |
| 평균 수익 | +5,333원 |
| 평균 손실 | -8,746원 |

## 4. v3 선별 매매 효과

| 지표 | 값 |
|---|---|
| 횡보장 감지일 | 15일 / 240일 |
| 횡보장 차단 매수 | 0회 |
| 시간제한 차단 | 26906회 |
| 일일1회 차단 | 0회 |
| **총 차단 매수** | **26906회** |

## 5. 매도 사유별 성과

| 사유 | 횟수 | P&L | 승률 |
|---|---|---|---|
| carry_sell | 15회 | +1,011,165원 | 100.0% |
| coin_profit | 13회 | +1,252,229원 | 100.0% |
| conl_avg_drop | 10회 | -692,848원 | 20.0% |
| conl_profit | 24회 | +828,981원 | 100.0% |
| eod_close | 5회 | -24,256원 | 20.0% |
| staged_sell | 1981회 | +114,854원 | 58.7% |
| stop_loss | 72회 | -3,898,525원 | 11.1% |
| time_stop | 35회 | -561,997원 | 51.4% |
| twin_converge | 100회 | +495,218원 | 53.0% |

## 6. 시그널별 성과

| 시그널 | 횟수 | P&L | 승률 |
|---|---|---|---|
| bearish | 9회 | +96,289원 | 66.7% |
| conditional_coin | 49회 | +720,299원 | 65.3% |
| conditional_conl | 799회 | -896,961원 | 37.7% |
| twin | 1398회 | -1,394,806원 | 68.5% |

## 7. 현재 파라미터 (config.py)

### v3 고유

| 파라미터 | 값 |
|---|---|
| `V3_COIN_TRIGGER_PCT` | 4.50 |
| `V3_CONL_TRIGGER_PCT` | 4.50 |
| `V3_DCA_MAX_COUNT` | 4 |
| `V3_ENTRY_CUTOFF_HOUR` | 10 |
| `V3_ENTRY_CUTOFF_MINUTE` | 30 |
| `V3_MAX_PER_STOCK_KRW` | 7,000,000 |
| `V3_PAIR_GAP_ENTRY_THRESHOLD` | 2.20 |
| `V3_SIDEWAYS_GAP_FAIL_COUNT` | 2 |
| `V3_SIDEWAYS_GLD_THRESHOLD` | 0.30 |
| `V3_SIDEWAYS_INDEX_THRESHOLD` | 0.50 |
| `V3_SIDEWAYS_MIN_SIGNALS` | 3 |
| `V3_SIDEWAYS_POLY_HIGH` | 0.60 |
| `V3_SIDEWAYS_POLY_LOW` | 0.40 |
| `V3_SIDEWAYS_TRIGGER_FAIL_COUNT` | 2 |
| `V3_SPLIT_BUY_INTERVAL_MIN` | 20 |

### v2 공유

| 파라미터 | 값 |
|---|---|
| `COIN_SELL_BEARISH_PCT` | 0.30 |
| `COIN_SELL_PROFIT_PCT` | 3.00 |
| `CONL_SELL_AVG_PCT` | 1.00 |
| `CONL_SELL_PROFIT_PCT` | 2.80 |
| `DCA_BUY_KRW` | 1,000,000 |
| `DCA_DROP_PCT` | -0.50 |
| `INITIAL_BUY_KRW` | 3,000,000 |
| `MAX_HOLD_HOURS` | 5 |
| `PAIR_GAP_SELL_THRESHOLD_V2` | 0.90 |
| `PAIR_SELL_FIRST_PCT` | 0.80 |
| `STOP_LOSS_BULLISH_PCT` | -8.00 |
| `STOP_LOSS_PCT` | -3.00 |
| `TAKE_PROFIT_PCT` | 2.00 |

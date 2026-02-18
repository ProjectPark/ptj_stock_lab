# PTJ v5 Baseline 시뮬레이션 리포트

> 생성일: 2026-02-18 05:12

## 1. 개요

현재 `config.py`에 설정된 v5 파라미터로 백테스트를 실행한 결과입니다.

## 2. 핵심 지표

| 지표 | 값 |
|---|---|
| 총 수익률 | **-12.50%** |
| 최종 자산 | 17,499,162원 |
| MDD | -15.85% |
| Sharpe Ratio | -1.8351 |
| 총 손익 | -2,500,838원 |
| 총 수수료 | 3,828,104원 |

## 3. 매매 통계

| 지표 | 값 |
|---|---|
| 거래일 | 240일 |
| 매수 횟수 | 253회 |
| 매도 횟수 | 1491회 |
| 승/패 | 834W / 657L |
| 승률 | 55.9% |
| 평균 수익 | +5,065원 |
| 평균 손실 | -10,236원 |

## 4. v5 선별 매매 효과

| 지표 | 값 |
|---|---|
| 횡보장 감지일 | 14일 / 240일 |
| 횡보장 차단 매수 | 0회 |
| 시간제한 차단 | 15131회 |
| 일일1회 차단 | 0회 |
| **총 차단 매수** | **15131회** |

## 5. 매도 사유별 성과

| 사유 | 횟수 | P&L | 승률 |
|---|---|---|---|
| carry_sell | 10회 | +536,802원 | 100.0% |
| coin_profit | 9회 | +827,170원 | 100.0% |
| conl_avg_drop | 7회 | -394,423원 | 28.6% |
| conl_profit | 17회 | +484,318원 | 100.0% |
| eod_close | 5회 | -24,256원 | 20.0% |
| staged_sell | 1305회 | +87,218원 | 57.3% |
| stop_loss | 45회 | -3,177,170원 | 6.7% |
| time_stop | 25회 | -1,213,113원 | 44.0% |
| twin_converge | 68회 | +372,615원 | 48.5% |

## 6. 시그널별 성과

| 시그널 | 횟수 | P&L | 승률 |
|---|---|---|---|
| bearish | 6회 | +102,528원 | 83.3% |
| conditional_coin | 38회 | -741,671원 | 55.3% |
| conditional_conl | 630회 | -768,614원 | 40.5% |
| twin | 817회 | -1,093,081원 | 67.7% |

## 7. 현재 파라미터 (config.py)

### v5 고유

| 파라미터 | 값 |
|---|---|
| `V5_CB_BTC_CRASH_PCT` | -5.00 |
| `V5_CB_BTC_SURGE_PCT` | 5.00 |
| `V5_CB_GLD_COOLDOWN_DAYS` | 3 |
| `V5_CB_GLD_SPIKE_PCT` | 3.00 |
| `V5_COIN_TRIGGER_PCT` | 4.50 |
| `V5_CONL_TRIGGER_PCT` | 4.50 |
| `V5_DCA_MAX_COUNT` | 4 |
| `V5_ENTRY_CUTOFF_HOUR` | 10 |
| `V5_ENTRY_CUTOFF_MINUTE` | 0 |
| `V5_MAX_PER_STOCK_KRW` | 7,000,000 |
| `V5_PAIR_GAP_ENTRY_THRESHOLD` | 2.20 |
| `V5_SIDEWAYS_GAP_FAIL_COUNT` | 2 |
| `V5_SIDEWAYS_GLD_THRESHOLD` | 0.30 |
| `V5_SIDEWAYS_INDEX_THRESHOLD` | 0.50 |
| `V5_SIDEWAYS_MIN_SIGNALS` | 3 |
| `V5_SIDEWAYS_POLY_HIGH` | 0.60 |
| `V5_SIDEWAYS_POLY_LOW` | 0.40 |
| `V5_SIDEWAYS_TRIGGER_FAIL_COUNT` | 2 |
| `V5_SPLIT_BUY_INTERVAL_MIN` | 20 |

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

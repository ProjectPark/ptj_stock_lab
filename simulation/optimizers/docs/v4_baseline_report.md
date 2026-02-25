# PTJ v4 Baseline 시뮬레이션 리포트

> 생성일: 2026-02-23 23:39

## 1. 개요

현재 `config.py`에 설정된 v4 파라미터로 백테스트를 실행한 결과입니다.

## 2. 핵심 지표

| 지표 | 값 |
|---|---|
| 총 수익률 | **+25.92%** |
| 최종 자산 | 18,888원 |
| MDD | -9.34% |
| Sharpe Ratio | 1.3923 |
| 총 손익 | +3,888원 |
| 총 수수료 | 510원 |

## 3. 매매 통계

| 지표 | 값 |
|---|---|
| 거래일 | 240일 |
| 매수 횟수 | 9회 |
| 매도 횟수 | 41회 |
| 승/패 | 28W / 6L |
| 승률 | 68.3% |
| 평균 수익 | +217원 |
| 평균 손실 | -364원 |

## 4. v4 선별 매매 효과

| 지표 | 값 |
|---|---|
| 횡보장 감지일 | 122일 / 240일 |
| 횡보장 차단 매수 | 0회 |
| 시간제한 차단 | 586회 |
| 일일1회 차단 | 0회 |
| CB 차단 매수 | 592회 |
| **총 차단 매수** | **586회** |

## 5. 매도 사유별 성과

| 사유 | 횟수 | P&L | 승률 |
|---|---|---|---|
| coin_profit | 1회 | +88원 | 100.0% |
| conl_profit | 1회 | +50원 | 100.0% |
| eod_close | 2회 | +25원 | 50.0% |
| staged_sell | 31회 | +48원 | 71.0% |
| swing_stage1_atr_stop | 1회 | -737원 | 0.0% |
| swing_stage1_drawdown_stop | 1회 | -1,432원 | 0.0% |
| swing_stage2_maturity | 2회 | +5,825원 | 100.0% |
| twin_converge | 2회 | +22원 | 50.0% |

## 6. 시그널별 성과

| 시그널 | 횟수 | P&L | 승률 |
|---|---|---|---|
| conditional_coin | 3회 | +113원 | 66.7% |
| conditional_conl | 34회 | +120원 | 70.6% |
| swing_stage1 | 2회 | -2,169원 | 0.0% |
| swing_stage2 | 2회 | +5,825원 | 100.0% |

## 7. 현재 파라미터 (config.py)

### v4 고유

| 파라미터 | 값 |
|---|---|
| `V4_CB_BTC_CRASH_PCT` | -3.50 |
| `V4_CB_BTC_SURGE_PCT` | 8.50 |
| `V4_CB_GLD_COOLDOWN_DAYS` | 1 |
| `V4_CB_GLD_SPIKE_PCT` | 4.00 |
| `V4_CB_VIX_COOLDOWN_DAYS` | 13 |
| `V4_CB_VIX_SPIKE_PCT` | 3.00 |
| `V4_COIN_TRIGGER_PCT` | 3.00 |
| `V4_CONL_ADX_MIN` | 10.00 |
| `V4_CONL_EMA_SLOPE_MIN_PCT` | 0.50 |
| `V4_CONL_TRIGGER_PCT` | 3.00 |
| `V4_DCA_BUY` | 750 |
| `V4_DCA_MAX_COUNT` | 1 |
| `V4_ENTRY_CUTOFF_HOUR` | 9 |
| `V4_ENTRY_CUTOFF_MINUTE` | 0 |
| `V4_HIGH_VOL_HIT_COUNT` | 3 |
| `V4_HIGH_VOL_MOVE_PCT` | 11.00 |
| `V4_HIGH_VOL_STOP_LOSS_PCT` | -3.00 |
| `V4_INITIAL_BUY` | 3500 |
| `V4_MAX_PER_STOCK` | 8500 |
| `V4_PAIR_FIXED_TP_PCT` | 7.50 |
| `V4_PAIR_GAP_ENTRY_THRESHOLD` | 2.00 |
| `V4_PAIR_IMMEDIATE_SELL_PCT` | 0.20 |
| `V4_SIDEWAYS_ATR_DECLINE_PCT` | 10.00 |
| `V4_SIDEWAYS_EMA_SLOPE_MAX` | 0.05 |
| `V4_SIDEWAYS_GAP_FAIL_COUNT` | 2 |
| `V4_SIDEWAYS_GLD_THRESHOLD` | 0.50 |
| `V4_SIDEWAYS_INDEX_THRESHOLD` | 0.50 |
| `V4_SIDEWAYS_MIN_SIGNALS` | 3 |
| `V4_SIDEWAYS_POLY_HIGH` | 0.60 |
| `V4_SIDEWAYS_POLY_LOW` | 0.30 |
| `V4_SIDEWAYS_RANGE_MAX_PCT` | 3.00 |
| `V4_SIDEWAYS_RSI_HIGH` | 65.00 |
| `V4_SIDEWAYS_RSI_LOW` | 35.00 |
| `V4_SIDEWAYS_TRIGGER_FAIL_COUNT` | 3 |
| `V4_SIDEWAYS_VOLUME_DECLINE_PCT` | 15.00 |
| `V4_SPLIT_BUY_INTERVAL_MIN` | 10 |

### v2 공유

| 파라미터 | 값 |
|---|---|
| `COIN_SELL_BEARISH_PCT` | 0.70 |
| `COIN_SELL_PROFIT_PCT` | 2.50 |
| `CONL_SELL_AVG_PCT` | 0.75 |
| `CONL_SELL_PROFIT_PCT` | 4.50 |
| `DCA_DROP_PCT` | -1.35 |
| `MAX_HOLD_HOURS` | 6 |
| `PAIR_GAP_SELL_THRESHOLD_V2` | 6.60 |
| `PAIR_SELL_FIRST_PCT` | 0.70 |
| `STOP_LOSS_BULLISH_PCT` | -16.00 |
| `STOP_LOSS_PCT` | -4.25 |
| `TAKE_PROFIT_PCT` | 3.00 |

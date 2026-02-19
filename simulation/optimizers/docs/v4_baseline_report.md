# PTJ v4 Baseline 시뮬레이션 리포트

> 생성일: 2026-02-19 02:42

## 1. 개요

현재 `config.py`에 설정된 v4 파라미터로 백테스트를 실행한 결과입니다.

## 2. 핵심 지표

| 지표 | 값 |
|---|---|
| 총 수익률 | **+25.74%** |
| 최종 자산 | 18,860원 |
| MDD | -9.34% |
| Sharpe Ratio | 1.3774 |
| 총 손익 | +3,860원 |
| 총 수수료 | 676원 |

## 3. 매매 통계

| 지표 | 값 |
|---|---|
| 거래일 | 240일 |
| 매수 횟수 | 21회 |
| 매도 횟수 | 89회 |
| 승/패 | 36W / 37L |
| 승률 | 40.4% |
| 평균 수익 | +179원 |
| 평균 손실 | -70원 |

## 4. v4 선별 매매 효과

| 지표 | 값 |
|---|---|
| 횡보장 감지일 | 37일 / 240일 |
| 횡보장 차단 매수 | 0회 |
| 시간제한 차단 | 1811회 |
| 일일1회 차단 | 0회 |
| CB 차단 매수 | 1644회 |
| **총 차단 매수** | **1811회** |

## 5. 매도 사유별 성과

| 사유 | 횟수 | P&L | 승률 |
|---|---|---|---|
| carry_sell | 1회 | +36원 | 100.0% |
| coin_profit | 1회 | +68원 | 100.0% |
| conl_profit | 1회 | +27원 | 100.0% |
| eod_close | 4회 | -15원 | 0.0% |
| fixed_tp | 1회 | +87원 | 100.0% |
| staged_sell | 65회 | +41원 | 36.9% |
| stop_loss | 2회 | -127원 | 0.0% |
| swing_stage1_atr_stop | 1회 | -736원 | 0.0% |
| swing_stage1_drawdown_stop | 1회 | -1,454원 | 0.0% |
| swing_stage2_maturity | 2회 | +5,837원 | 100.0% |
| time_stop | 3회 | +18원 | 66.7% |
| twin_converge | 7회 | +80원 | 57.1% |

## 6. 시그널별 성과

| 시그널 | 횟수 | P&L | 승률 |
|---|---|---|---|
| conditional_coin | 3회 | +53원 | 33.3% |
| conditional_conl | 39회 | -158원 | 5.1% |
| swing_stage1 | 2회 | -2,190원 | 0.0% |
| swing_stage2 | 2회 | +5,837원 | 100.0% |
| twin | 43회 | +319원 | 72.1% |

## 7. 현재 파라미터 (config.py)

### v4 고유

| 파라미터 | 값 |
|---|---|
| `V4_CB_BTC_CRASH_PCT` | -5.00 |
| `V4_CB_BTC_SURGE_PCT` | 5.00 |
| `V4_CB_GLD_COOLDOWN_DAYS` | 3 |
| `V4_CB_GLD_SPIKE_PCT` | 3.00 |
| `V4_CB_VIX_COOLDOWN_DAYS` | 7 |
| `V4_CB_VIX_SPIKE_PCT` | 6.00 |
| `V4_COIN_TRIGGER_PCT` | 4.50 |
| `V4_CONL_ADX_MIN` | 18.00 |
| `V4_CONL_EMA_SLOPE_MIN_PCT` | 0.00 |
| `V4_CONL_TRIGGER_PCT` | 4.50 |
| `V4_DCA_BUY` | 750 |
| `V4_DCA_MAX_COUNT` | 4 |
| `V4_ENTRY_CUTOFF_HOUR` | 10 |
| `V4_ENTRY_CUTOFF_MINUTE` | 0 |
| `V4_HIGH_VOL_HIT_COUNT` | 2 |
| `V4_HIGH_VOL_MOVE_PCT` | 10.00 |
| `V4_HIGH_VOL_STOP_LOSS_PCT` | -4.00 |
| `V4_INITIAL_BUY` | 2250 |
| `V4_MAX_PER_STOCK` | 5250 |
| `V4_PAIR_FIXED_TP_PCT` | 5.00 |
| `V4_PAIR_GAP_ENTRY_THRESHOLD` | 2.20 |
| `V4_PAIR_IMMEDIATE_SELL_PCT` | 0.40 |
| `V4_SIDEWAYS_ATR_DECLINE_PCT` | 20.00 |
| `V4_SIDEWAYS_EMA_SLOPE_MAX` | 0.10 |
| `V4_SIDEWAYS_GAP_FAIL_COUNT` | 2 |
| `V4_SIDEWAYS_GLD_THRESHOLD` | 0.30 |
| `V4_SIDEWAYS_INDEX_THRESHOLD` | 0.50 |
| `V4_SIDEWAYS_MIN_SIGNALS` | 3 |
| `V4_SIDEWAYS_POLY_HIGH` | 0.60 |
| `V4_SIDEWAYS_POLY_LOW` | 0.40 |
| `V4_SIDEWAYS_RANGE_MAX_PCT` | 2.00 |
| `V4_SIDEWAYS_RSI_HIGH` | 55.00 |
| `V4_SIDEWAYS_RSI_LOW` | 45.00 |
| `V4_SIDEWAYS_TRIGGER_FAIL_COUNT` | 2 |
| `V4_SIDEWAYS_VOLUME_DECLINE_PCT` | 30.00 |
| `V4_SPLIT_BUY_INTERVAL_MIN` | 20 |

### v2 공유

| 파라미터 | 값 |
|---|---|
| `COIN_SELL_BEARISH_PCT` | 0.30 |
| `COIN_SELL_PROFIT_PCT` | 3.00 |
| `CONL_SELL_AVG_PCT` | 1.00 |
| `CONL_SELL_PROFIT_PCT` | 2.80 |
| `DCA_DROP_PCT` | -0.50 |
| `MAX_HOLD_HOURS` | 5 |
| `PAIR_GAP_SELL_THRESHOLD_V2` | 0.90 |
| `PAIR_SELL_FIRST_PCT` | 0.80 |
| `STOP_LOSS_BULLISH_PCT` | -8.00 |
| `STOP_LOSS_PCT` | -3.00 |
| `TAKE_PROFIT_PCT` | 2.00 |

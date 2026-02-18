# PTJ v4 Baseline 시뮬레이션 리포트

> 생성일: 2026-02-18 05:12

## 1. 개요

현재 `config.py`에 설정된 v4 파라미터로 백테스트를 실행한 결과입니다.

## 2. 핵심 지표

| 지표 | 값 |
|---|---|
| 총 수익률 | **-7.84%** |
| 최종 자산 | 18,432,292원 |
| MDD | -13.49% |
| Sharpe Ratio | -1.0137 |
| 총 손익 | -1,567,708원 |
| 총 수수료 | 3,831,408원 |

## 3. 매매 통계

| 지표 | 값 |
|---|---|
| 거래일 | 240일 |
| 매수 횟수 | 253회 |
| 매도 횟수 | 1491회 |
| 승/패 | 835W / 656L |
| 승률 | 56.0% |
| 평균 수익 | +5,959원 |
| 평균 손실 | -9,975원 |

## 4. v4 선별 매매 효과

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
| carry_sell | 11회 | +1,247,206원 | 100.0% |
| coin_profit | 9회 | +827,170원 | 100.0% |
| conl_avg_drop | 7회 | -394,423원 | 28.6% |
| conl_profit | 17회 | +484,318원 | 100.0% |
| eod_close | 5회 | -24,256원 | 20.0% |
| staged_sell | 1305회 | +87,238원 | 57.3% |
| stop_loss | 46회 | -3,366,756원 | 6.5% |
| time_stop | 23회 | -800,820원 | 47.8% |
| twin_converge | 68회 | +372,615원 | 48.5% |

## 6. 시그널별 성과

| 시그널 | 횟수 | P&L | 승률 |
|---|---|---|---|
| bearish | 6회 | +143,210원 | 83.3% |
| conditional_coin | 38회 | +236,205원 | 57.9% |
| conditional_conl | 630회 | -768,614원 | 40.5% |
| twin | 817회 | -1,178,509원 | 67.7% |

## 7. 현재 파라미터 (config.py)

### v4 고유

| 파라미터 | 값 |
|---|---|
| `V4_CB_BTC_CRASH_PCT` | -5.00 |
| `V4_CB_BTC_SURGE_PCT` | 5.00 |
| `V4_CB_GLD_COOLDOWN_DAYS` | 3 |
| `V4_CB_GLD_SPIKE_PCT` | 3.00 |
| `V4_COIN_TRIGGER_PCT` | 4.50 |
| `V4_CONL_TRIGGER_PCT` | 4.50 |
| `V4_DCA_MAX_COUNT` | 4 |
| `V4_ENTRY_CUTOFF_HOUR` | 10 |
| `V4_ENTRY_CUTOFF_MINUTE` | 0 |
| `V4_MAX_PER_STOCK_KRW` | 7,000,000 |
| `V4_PAIR_GAP_ENTRY_THRESHOLD` | 2.20 |
| `V4_SIDEWAYS_GAP_FAIL_COUNT` | 2 |
| `V4_SIDEWAYS_GLD_THRESHOLD` | 0.30 |
| `V4_SIDEWAYS_INDEX_THRESHOLD` | 0.50 |
| `V4_SIDEWAYS_MIN_SIGNALS` | 3 |
| `V4_SIDEWAYS_POLY_HIGH` | 0.60 |
| `V4_SIDEWAYS_POLY_LOW` | 0.40 |
| `V4_SIDEWAYS_TRIGGER_FAIL_COUNT` | 2 |
| `V4_SPLIT_BUY_INTERVAL_MIN` | 20 |

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

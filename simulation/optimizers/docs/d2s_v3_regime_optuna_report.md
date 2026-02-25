# D2S v3 Regime Optuna 최적화 리포트

> 생성일: 2026-02-25 00:58  
> IS: 2025-03-03 ~ 2025-09-30  |  OOS: 2025-10-01 ~ 2026-02-17  
> trial: 500  |  n_jobs: 20  
> 탐색: v1+v2 전체 + R19(BB하드) + R20/R21(레짐 TP/HD) + Polymarket 레짐  
> 스코어: IS_Shp×10.0 + OOS_Shp×20.0 - MDD×0.5  
> Journal: d2s_v3_regime_journal.log

## 1. Best Trial

| 항목 | IS | OOS |
|---|---|---|
| Trial # | 444 | — |
| Score | 61.7950 | — |
| 수익률 | +28.22% | +23.60% |
| MDD | -4.8% | -9.3% |
| Sharpe | 2.156 | 2.131 |

## 2. Top 10

| # | score | IS% | OOS% | IS_Shp | OOS_Shp |
|---|---|---|---|---|---|
| 444 | +61.7950 | +28.22% | +23.60% | 2.156 | 2.131 |
| 441 | +61.5050 | +28.92% | +23.18% | 2.189 | 2.098 |
| 483 | +61.3350 | +22.88% | +15.11% | 2.489 | 1.891 |
| 440 | +61.2950 | +28.22% | +23.24% | 2.156 | 2.106 |
| 476 | +60.0650 | +21.95% | +14.45% | 2.436 | 1.854 |
| 477 | +60.0650 | +21.95% | +14.45% | 2.436 | 1.854 |
| 479 | +60.0650 | +21.95% | +14.45% | 2.436 | 1.854 |
| 481 | +60.0650 | +21.95% | +14.45% | 2.436 | 1.854 |
| 484 | +60.0650 | +21.95% | +14.45% | 2.436 | 1.854 |
| 485 | +60.0650 | +21.95% | +14.45% | 2.436 | 1.854 |

## 3. Best 파라미터

| 파라미터 | 값 |
|---|---|
| `amdl_friday_contrarian_threshold` | -2.5 |
| `atr_high_quantile` | 0.75 |
| `bb_danger_zone` | 1.2 |
| `bb_entry_hard_filter` | True |
| `bb_entry_hard_max` | 0.30000000000000004 |
| `bb_entry_max` | 0.5 |
| `bear_hold_days_max` | 4 |
| `bear_take_profit_pct` | 6.0 |
| `btc_up_max` | 0.6 |
| `btc_up_min` | 0.5 |
| `bull_hold_days_max` | 5 |
| `bull_take_profit_pct` | 6.0 |
| `buy_size_large` | 0.25 |
| `buy_size_small` | 0.08 |
| `contrarian_entry_threshold` | -0.5 |
| `daily_new_entry_cap` | 0.4 |
| `dca_max_daily` | 5 |
| `dca_max_layers` | 2 |
| `early_stoploss_days` | 4 |
| `early_stoploss_recovery` | 4.0 |
| `gap_bank_conl_max` | 7.5 |
| `gld_suppress_threshold` | 2.0 |
| `market_score_entry_a` | 0.8 |
| `market_score_entry_b` | 0.7 |
| `market_score_suppress` | 0.5 |
| `regime_bear_spy_streak` | 4 |
| `regime_btc_bear_threshold` | 0.35 |
| `regime_btc_bull_threshold` | 0.5 |
| `regime_bull_spy_streak` | 4 |
| `regime_spy_sma_bear_pct` | -1.2 |
| `regime_spy_sma_bull_pct` | 0.8 |
| `regime_spy_sma_period` | 23 |
| `riskoff_consecutive_boost` | 4 |
| `riskoff_gld_optimal_min` | 1.1 |
| `riskoff_panic_size_factor` | 0.3 |
| `riskoff_spy_min_threshold` | -0.8 |
| `riskoff_spy_optimal_max` | -0.3999999999999999 |
| `robn_pct_max` | 3.0 |
| `rsi_danger_zone` | 80 |
| `rsi_entry_max` | 70 |
| `rsi_entry_min` | 37 |
| `spy_bearish_threshold` | -2.0 |
| `spy_streak_max` | 4 |
| `vbounce_bb_threshold` | 0.1 |
| `vbounce_crash_threshold` | -9.0 |
| `vbounce_score_threshold` | 0.75 |
| `vbounce_size_max` | 0.2 |
| `vbounce_size_multiplier` | 2.0 |
| `vol_entry_max` | 2.5 |
| `vol_entry_min` | 0.8 |
| `w_btc` | 0.15000000000000002 |
| `w_gld` | 0.25 |
| `w_riskoff` | 0.15 |
| `w_spy` | 0.15000000000000002 |
| `w_streak` | 0.25 |
| `w_vol` | 0.25 |

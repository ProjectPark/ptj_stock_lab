# D2S v3 Regime Optuna 최적화 리포트

> 생성일: 2026-02-25 17:38  
> IS: 2024-09-18 ~ 2025-05-31  |  OOS: 2025-06-01 ~ 2026-02-17  
> trial: 500  |  n_jobs: 20  
> 탐색: v1+v2 전체 + R19(BB하드) + R20/R21(레짐 TP/HD) + Polymarket 레짐  
> 스코어: IS_Shp×10.0 + OOS_Shp×20.0 - MDD×0.5  
> Journal: d2s_v3_r4_norobn_journal.log

## 1. Best Trial

| 항목 | IS | OOS |
|---|---|---|
| Trial # | 0 | — |
| Score | 44.0400 | — |
| 수익률 | +39.81% | +64.27% |
| MDD | -9.2% | -14.7% |
| Sharpe | 1.886 | 1.489 |

## 2. Top 10

| # | score | IS% | OOS% | IS_Shp | OOS_Shp |
|---|---|---|---|---|---|
| 0 | +44.0400 | +39.81% | +64.27% | 1.886 | 1.489 |
| 480 | +41.7250 | +59.17% | +61.48% | 1.756 | 1.480 |
| 486 | +41.7250 | +59.17% | +61.48% | 1.756 | 1.480 |
| 453 | +41.7150 | +60.31% | +60.77% | 1.779 | 1.468 |
| 493 | +41.5550 | +49.72% | +58.17% | 1.839 | 1.430 |
| 485 | +41.5150 | +57.78% | +62.09% | 1.727 | 1.484 |
| 495 | +41.2850 | +48.41% | +58.76% | 1.804 | 1.434 |
| 452 | +41.1450 | +28.91% | +59.63% | 1.736 | 1.461 |
| 430 | +41.0950 | +46.52% | +60.04% | 1.709 | 1.472 |
| 470 | +41.0250 | +55.10% | +62.09% | 1.678 | 1.484 |

## 3. Best 파라미터

| 파라미터 | 값 |
|---|---|
| `amdl_friday_contrarian_threshold` | -3.0 |
| `atr_high_quantile` | 0.65 |
| `bb_danger_zone` | 1.1 |
| `bb_entry_hard_filter` | True |
| `bb_entry_hard_max` | 0.3 |
| `bb_entry_max` | 0.4 |
| `bear_hold_days_max` | 8 |
| `bear_take_profit_pct` | 6.5 |
| `btc_up_max` | 0.7 |
| `btc_up_min` | 0.4 |
| `bull_hold_days_max` | 12 |
| `bull_take_profit_pct` | 5.0 |
| `buy_size_large` | 0.1 |
| `buy_size_small` | 0.05 |
| `contrarian_entry_threshold` | -0.5 |
| `daily_new_entry_cap` | 0.15 |
| `dca_max_daily` | 2 |
| `dca_max_layers` | 2 |
| `early_stoploss_days` | 2 |
| `early_stoploss_recovery` | 3.0 |
| `gap_bank_conl_max` | 5.0 |
| `gld_suppress_threshold` | 0.5 |
| `market_score_entry_a` | 0.8 |
| `market_score_entry_b` | 0.7 |
| `market_score_suppress` | 0.45 |
| `regime_bear_spy_streak` | 1 |
| `regime_btc_bear_threshold` | 0.35 |
| `regime_btc_bull_threshold` | 0.55 |
| `regime_bull_spy_streak` | 5 |
| `regime_spy_sma_bear_pct` | -1.5 |
| `regime_spy_sma_bull_pct` | 1.1 |
| `regime_spy_sma_period` | 12 |
| `riskoff_consecutive_boost` | 2 |
| `riskoff_gld_optimal_min` | 0.7 |
| `riskoff_panic_size_factor` | 0.6 |
| `riskoff_spy_min_threshold` | -1.6 |
| `riskoff_spy_optimal_max` | -0.6 |
| `robn_pct_max` | 2.5 |
| `rsi_danger_zone` | 76 |
| `rsi_entry_max` | 69 |
| `rsi_entry_min` | 37 |
| `spy_bearish_threshold` | -1.25 |
| `spy_streak_max` | 3 |
| `vbounce_bb_threshold` | 0.2 |
| `vbounce_crash_threshold` | -12.0 |
| `vbounce_score_threshold` | 0.85 |
| `vbounce_size_max` | 0.3 |
| `vbounce_size_multiplier` | 2.5 |
| `vol_entry_max` | 3.5 |
| `vol_entry_min` | 1.1 |
| `w_btc` | 0.0909 |
| `w_gld` | 0.2273 |
| `w_riskoff` | 0.1818 |
| `w_spy` | 0.1364 |
| `w_streak` | 0.1818 |
| `w_vol` | 0.1818 |

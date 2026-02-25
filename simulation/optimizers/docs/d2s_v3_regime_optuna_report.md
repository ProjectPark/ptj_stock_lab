# D2S v3 Regime Optuna 최적화 리포트

> 생성일: 2026-02-25 17:38  
> IS: 2025-03-03 ~ 2025-09-30  |  OOS: 2025-10-01 ~ 2026-02-17  
> trial: 500  |  n_jobs: 20  
> 탐색: v1+v2 전체 + R19(BB하드) + R20/R21(레짐 TP/HD) + Polymarket 레짐  
> 스코어: IS_Shp×10.0 + OOS_Shp×20.0 - MDD×0.5  
> Journal: d2s_v3_r4_1y_journal.log

## 1. Best Trial

| 항목 | IS | OOS |
|---|---|---|
| Trial # | 449 | — |
| Score | 65.4900 | — |
| 수익률 | +34.69% | +15.37% |
| MDD | -2.0% | -9.8% |
| Sharpe | 3.530 | 1.559 |

## 2. Top 10

| # | score | IS% | OOS% | IS_Shp | OOS_Shp |
|---|---|---|---|---|---|
| 449 | +65.4900 | +34.69% | +15.37% | 3.530 | 1.559 |
| 496 | +64.9900 | +34.51% | +15.18% | 3.516 | 1.541 |
| 456 | +64.3900 | +33.45% | +15.16% | 3.460 | 1.539 |
| 476 | +64.0100 | +35.24% | +14.24% | 3.580 | 1.460 |
| 452 | +63.6900 | +35.01% | +14.24% | 3.548 | 1.460 |
| 347 | +63.5200 | +34.72% | +14.24% | 3.524 | 1.460 |
| 470 | +63.5100 | +34.69% | +14.24% | 3.530 | 1.460 |
| 345 | +63.3200 | +34.24% | +14.24% | 3.504 | 1.460 |
| 346 | +62.4300 | +33.37% | +14.24% | 3.415 | 1.460 |
| 458 | +62.3100 | +32.89% | +14.24% | 3.410 | 1.460 |

## 3. Best 파라미터

| 파라미터 | 값 |
|---|---|
| `amdl_friday_contrarian_threshold` | -2.0 |
| `atr_high_quantile` | 0.7 |
| `bb_danger_zone` | 1.0 |
| `bb_entry_hard_filter` | True |
| `bb_entry_hard_max` | 0.30000000000000004 |
| `bb_entry_max` | 0.3 |
| `bear_hold_days_max` | 5 |
| `bear_take_profit_pct` | 7.0 |
| `btc_up_max` | 0.8 |
| `btc_up_min` | 0.55 |
| `bull_hold_days_max` | 6 |
| `bull_take_profit_pct` | 6.0 |
| `buy_size_large` | 0.1 |
| `buy_size_small` | 0.05 |
| `contrarian_entry_threshold` | -1.5 |
| `daily_new_entry_cap` | 0.2 |
| `dca_max_daily` | 2 |
| `dca_max_layers` | 2 |
| `early_stoploss_days` | 2 |
| `early_stoploss_recovery` | 4.5 |
| `gap_bank_conl_max` | 9.5 |
| `gld_suppress_threshold` | 0.5 |
| `market_score_entry_a` | 0.75 |
| `market_score_entry_b` | 0.65 |
| `market_score_suppress` | 0.5 |
| `regime_bear_spy_streak` | 4 |
| `regime_btc_bear_threshold` | 0.25 |
| `regime_btc_bull_threshold` | 0.5 |
| `regime_bull_spy_streak` | 3 |
| `regime_spy_sma_bear_pct` | -1.5 |
| `regime_spy_sma_bull_pct` | 0.7 |
| `regime_spy_sma_period` | 24 |
| `riskoff_consecutive_boost` | 2 |
| `riskoff_gld_optimal_min` | 0.5 |
| `riskoff_panic_size_factor` | 0.6000000000000001 |
| `riskoff_spy_min_threshold` | -1.4 |
| `riskoff_spy_optimal_max` | -0.3999999999999999 |
| `robn_pct_max` | 2.5 |
| `rsi_danger_zone` | 85 |
| `rsi_entry_max` | 66 |
| `rsi_entry_min` | 47 |
| `spy_bearish_threshold` | -1.5 |
| `spy_streak_max` | 3 |
| `vbounce_bb_threshold` | 0.25 |
| `vbounce_crash_threshold` | -10.0 |
| `vbounce_score_threshold` | 0.8 |
| `vbounce_size_max` | 0.4 |
| `vbounce_size_multiplier` | 2.5 |
| `vol_entry_max` | 2.0 |
| `vol_entry_min` | 0.9 |
| `w_btc` | 0.15000000000000002 |
| `w_gld` | 0.15000000000000002 |
| `w_riskoff` | 0.25 |
| `w_spy` | 0.2 |
| `w_streak` | 0.25 |
| `w_vol` | 0.15000000000000002 |

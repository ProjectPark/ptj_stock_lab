# D2S v3 Regime Optuna 최적화 리포트

> 생성일: 2026-02-25 01:29  
> IS: 2024-09-18 ~ 2025-05-31  |  OOS: 2025-06-01 ~ 2026-02-17  
> trial: 500  |  n_jobs: 20  
> 탐색: v1+v2 전체 + R19(BB하드) + R20/R21(레짐 TP/HD) + Polymarket 레짐  
> 스코어: IS_Shp×10.0 + OOS_Shp×20.0 - MDD×0.5  
> Journal: d2s_v3_regime_no_robn_journal.log

## 1. Best Trial

| 항목 | IS | OOS |
|---|---|---|
| Trial # | 449 | — |
| Score | 46.7950 | — |
| 수익률 | +33.68% | +62.42% |
| MDD | -9.2% | -16.0% |
| Sharpe | 2.220 | 1.461 |

## 2. Top 10

| # | score | IS% | OOS% | IS_Shp | OOS_Shp |
|---|---|---|---|---|---|
| 449 | +46.7950 | +33.68% | +62.42% | 2.220 | 1.461 |
| 491 | +46.7550 | +34.65% | +61.50% | 2.296 | 1.421 |
| 446 | +46.3250 | +36.50% | +57.92% | 2.379 | 1.358 |
| 384 | +46.1400 | +36.26% | +58.24% | 2.372 | 1.362 |
| 450 | +46.1050 | +38.62% | +96.80% | 2.479 | 1.297 |
| 439 | +46.0750 | +34.56% | +57.83% | 2.286 | 1.392 |
| 492 | +46.0450 | +32.66% | +62.32% | 2.199 | 1.434 |
| 403 | +45.9950 | +35.25% | +58.33% | 2.334 | 1.364 |
| 496 | +45.8550 | +36.44% | +101.86% | 2.388 | 1.330 |
| 445 | +45.7800 | +33.42% | +59.94% | 2.204 | 1.423 |

## 3. Best 파라미터

| 파라미터 | 값 |
|---|---|
| `amdl_friday_contrarian_threshold` | -3.0 |
| `atr_high_quantile` | 0.65 |
| `bb_danger_zone` | 1.1 |
| `bb_entry_hard_filter` | True |
| `bb_entry_hard_max` | 0.30000000000000004 |
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
| `riskoff_panic_size_factor` | 0.6000000000000001 |
| `riskoff_spy_min_threshold` | -2.8 |
| `riskoff_spy_optimal_max` | -0.6 |
| `robn_pct_max` | 2.5 |
| `rsi_danger_zone` | 76 |
| `rsi_entry_max` | 69 |
| `rsi_entry_min` | 37 |
| `spy_bearish_threshold` | -1.25 |
| `spy_streak_max` | 3 |
| `vbounce_bb_threshold` | 0.2 |
| `vbounce_crash_threshold` | -12.0 |
| `vbounce_score_threshold` | 0.9 |
| `vbounce_size_max` | 0.30000000000000004 |
| `vbounce_size_multiplier` | 2.5 |
| `vol_entry_max` | 3.5 |
| `vol_entry_min` | 1.1 |
| `w_btc` | 0.1 |
| `w_gld` | 0.25 |
| `w_riskoff` | 0.2 |
| `w_spy` | 0.15000000000000002 |
| `w_streak` | 0.2 |
| `w_vol` | 0.2 |

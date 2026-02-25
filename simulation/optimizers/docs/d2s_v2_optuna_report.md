# D2S v2 Optuna 최적화 리포트

> 생성일: 2026-02-24 23:32  
> IS 기간: 2025-03-03 ~ 2025-09-30  
> 총 trial: 200  |  n_jobs: 20  
> 탐색 공간: v1 전체 + R17/R18/DCA 레이어 + gap/robn 필터 + hold_days_min  
> 스코어: win_rate×0.40 + sharpe×15 - mdd×0.30  
> Storage: d2s_v2_journal.log

## 1. Best Trial

| 항목 | 값 |
|---|---|
| Trial # | 180 |
| Score | 88.5920 |
| 승률 | 78.8% |
| 수익률 | +76.41% |
| MDD | -6.51% |
| Sharpe | 3.935 |
| 매도 거래 | 80건 |

## 2. Top 5

| # | score | 승률 | 수익률 | MDD | Sharpe |
|---|---|---|---|---|---|
| 180 | +88.5920 | 78.8% | +76.41% | -6.51% | 3.935 |
| 182 | +88.5920 | 78.8% | +76.41% | -6.51% | 3.935 |
| 172 | +84.2620 | 77.8% | +70.95% | -6.51% | 3.673 |
| 189 | +84.1590 | 78.9% | +81.83% | -6.17% | 3.630 |
| 197 | +83.9940 | 78.9% | +81.46% | -6.17% | 3.619 |

## 3. Best 파라미터

| 파라미터 | 값 |
|---|---|
| `amdl_friday_contrarian_threshold` | -1.5 |
| `atr_high_quantile` | 0.7 |
| `bb_danger_zone` | 1.2 |
| `bb_entry_max` | 0.4 |
| `btc_up_max` | 0.85 |
| `btc_up_min` | 0.30000000000000004 |
| `buy_size_large` | 0.1 |
| `buy_size_small` | 0.06 |
| `contrarian_entry_threshold` | 0.0 |
| `daily_new_entry_cap` | 0.15 |
| `dca_max_daily` | 7 |
| `dca_max_layers` | 3 |
| `early_stoploss_days` | 5 |
| `early_stoploss_recovery` | 5.0 |
| `gap_bank_conl_max` | 7.5 |
| `gld_suppress_threshold` | 1.5 |
| `market_score_entry_a` | 0.75 |
| `market_score_entry_b` | 0.6000000000000001 |
| `market_score_suppress` | 0.5 |
| `optimal_hold_days_max` | 5 |
| `optimal_hold_days_min` | 3 |
| `riskoff_consecutive_boost` | 3 |
| `riskoff_gld_optimal_min` | 1.1 |
| `riskoff_panic_size_factor` | 0.5 |
| `riskoff_spy_min_threshold` | -2.4 |
| `riskoff_spy_optimal_max` | -0.5 |
| `robn_pct_max` | 3.0 |
| `rsi_danger_zone` | 90 |
| `rsi_entry_max` | 67 |
| `rsi_entry_min` | 48 |
| `spy_bearish_threshold` | -1.75 |
| `spy_streak_max` | 3 |
| `take_profit_pct` | 6.5 |
| `vbounce_bb_threshold` | 0.15000000000000002 |
| `vbounce_crash_threshold` | -14.0 |
| `vbounce_score_threshold` | 0.9 |
| `vbounce_size_max` | 0.30000000000000004 |
| `vbounce_size_multiplier` | 2.0 |
| `vol_entry_max` | 1.5 |
| `vol_entry_min` | 1.2000000000000002 |
| `w_btc` | 0.2 |
| `w_gld` | 0.30000000000000004 |
| `w_riskoff` | 0.2 |
| `w_spy` | 0.1 |
| `w_streak` | 0.15000000000000002 |
| `w_vol` | 0.1 |

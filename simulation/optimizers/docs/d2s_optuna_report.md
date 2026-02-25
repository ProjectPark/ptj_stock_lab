# D2S Optuna 최적화 리포트 (v2)

> 생성일: 2026-02-24 23:14  
> IS 기간: 2025-03-03 ~ 2025-09-30  
> 총 trial: 200  |  n_jobs: 20  
> 스코어: win_rate×0.40 + sharpe×15 - mdd×0.30  
> Storage: d2s_journal_v2.log

## 1. Best Trial

| 항목 | 값 |
|---|---|
| Trial # | 179 |
| Score | 90.1860 |
| 승률 | 87.9% |
| 수익률 | +63.32% |
| MDD | -5.43% |
| Sharpe | 3.777 |
| 매도 거래 | 33건 |

## 2. Top 5

| # | score | 승률 | 수익률 | MDD | Sharpe |
|---|---|---|---|---|---|
| 179 | +90.1860 | 87.9% | +63.32% | -5.43% | 3.777 |
| 182 | +90.1710 | 87.9% | +63.46% | -5.43% | 3.776 |
| 170 | +90.1410 | 87.9% | +63.28% | -5.43% | 3.774 |
| 173 | +89.9460 | 87.9% | +62.92% | -5.43% | 3.761 |
| 175 | +89.9460 | 87.9% | +62.92% | -5.43% | 3.761 |

## 3. Best 파라미터

| 파라미터 | 값 |
|---|---|
| `amdl_friday_contrarian_threshold` | -1.0 |
| `atr_high_quantile` | 0.8 |
| `bb_danger_zone` | 1.1 |
| `bb_entry_max` | 0.4 |
| `btc_up_max` | 0.9 |
| `btc_up_min` | 0.2 |
| `buy_size_large` | 0.15000000000000002 |
| `buy_size_small` | 0.07 |
| `contrarian_entry_threshold` | -0.5 |
| `daily_new_entry_cap` | 0.2 |
| `dca_max_daily` | 2 |
| `gld_suppress_threshold` | 0.75 |
| `market_score_entry_a` | 0.65 |
| `market_score_entry_b` | 0.55 |
| `market_score_suppress` | 0.45 |
| `optimal_hold_days_max` | 10 |
| `riskoff_consecutive_boost` | 3 |
| `riskoff_gld_optimal_min` | 0.7 |
| `riskoff_panic_size_factor` | 0.5 |
| `riskoff_spy_min_threshold` | -2.8 |
| `riskoff_spy_optimal_max` | -1.0 |
| `rsi_danger_zone` | 72 |
| `rsi_entry_max` | 67 |
| `rsi_entry_min` | 46 |
| `spy_bearish_threshold` | -1.75 |
| `spy_streak_max` | 5 |
| `take_profit_pct` | 8.5 |
| `vol_entry_max` | 3.5 |
| `vol_entry_min` | 0.8 |
| `w_btc` | 0.1 |
| `w_gld` | 0.2 |
| `w_riskoff` | 0.25 |
| `w_spy` | 0.05 |
| `w_streak` | 0.1 |
| `w_vol` | 0.25 |

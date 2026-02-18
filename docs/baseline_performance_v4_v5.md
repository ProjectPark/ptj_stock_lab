# PTJ v4/v5 Baseline Performance Report (Original Weights)

- Created at: 2026-02-18
- Baseline source:
  - `data/v4_baseline_result.json`
  - `data/v5_baseline_result.json`
- Run timestamps:
  - v4: 2026-02-18T05:12:05.187969
  - v5: 2026-02-18T05:12:48.152094
- Execution mode: `--stage 1` (baseline only, no Optuna search)
- Backtest window: 2025-02-18 to 2026-02-17 (engine default)

## 1) Headline Metrics

| Metric | v4 Baseline | v5 Baseline |
|---|---:|---:|
| Total return | -7.84% | -12.50% |
| Final equity | 18,432,292 KRW | 17,499,162 KRW |
| MDD | 13.49% | 15.85% |
| Sharpe | -1.0137 | -1.8351 |
| Win rate | 56.0% | 55.9% |
| Trades (buy/sell) | 253 / 1,491 | 253 / 1,491 |
| Total fees | 3,831,408 KRW | 3,828,104 KRW |
| Total PnL | -1,567,708 KRW | -2,500,838 KRW |

## 2) Risk/Control Signals

| Metric | v4 | v5 |
|---|---:|---:|
| Trading days | 240 | 240 |
| Sideways days | 14 | 14 |
| Sideways blocks | 0 | 0 |
| Entry cutoff blocks | 15,131 | 15,131 |
| Daily limit blocks | 0 | 0 |
| Stop-loss count | 46 | 45 |
| Time-stop count | 23 | 25 |
| EOD close count | 5 | 5 |
| CB buy blocks | 27,070 | 25,716 |
| CB sell halt bars | 18,290 | - |

## 3) Signal-type PnL Breakdown

| Signal type | v4 Count | v4 PnL (KRW) | v5 Count | v5 PnL (KRW) |
|---|---:|---:|---:|
| twin | 817 | -1,178,509 | 817 | -1,093,081 |
| conditional_conl | 630 | -768,614 | 630 | -768,614 |
| conditional_coin | 38 | +236,205 | 38 | -741,671 |
| bearish | 6 | +143,210 | 6 | +102,528 |

## 4) Exit-reason PnL Breakdown

| Exit reason | v4 Count | v4 PnL (KRW) | v5 Count | v5 PnL (KRW) |
|---|---:|---:|---:|
| twin_converge | 68 | +372,615 | 68 | +372,615 |
| staged_sell | 1,305 | +87,238 | 1,305 | +87,218 |
| stop_loss | 46 | -3,366,756 | 45 | -3,177,170 |
| conl_profit | 17 | +484,318 | 17 | +484,318 |
| coin_profit | 9 | +827,170 | 9 | +827,170 |
| conl_avg_drop | 7 | -394,423 | 7 | -394,423 |
| carry_sell | 11 | +1,247,206 | 10 | +536,802 |
| time_stop | 23 | -800,820 | 25 | -1,213,113 |
| eod_close | 5 | -24,256 | 5 | -24,256 |

## 5) Notes

- v4/v5 baseline results are now different.
- Implemented split point:
  - v4: circuit breaker can halt both buy and sell paths.
  - v5: circuit breaker blocks new buys only and allows sells.
- This document is intentionally separate from:
  - `docs/v4_baseline_report.md`
  - `docs/v5_baseline_report.md`
- Use this file as a single baseline reference before applying v4/v5 rule-specific logic changes and re-running Stage 1.

## 6) Reproduction

```bash
/Users/toddpk/.pyenv/versions/market/bin/python optimize_v4_optuna.py --stage 1
/Users/toddpk/.pyenv/versions/market/bin/python optimize_v5_optuna.py --stage 1
```

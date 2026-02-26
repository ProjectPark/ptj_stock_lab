# 세션 이어받기 노트 — 2026-02-26

## 지금까지 완료된 작업

### Phase 1: Look-ahead Bias 수정 ✅

**수정 내용 (커밋: `6df929b`)**

| 파일 | 변경 |
|------|------|
| `simulation/backtests/backtest_d2s.py` | `slippage_pct=0.05` 추가; `_execute_buy/sell` → `opens` + slippage; `run()` → `pending_signals` 패턴 |
| `simulation/backtests/backtest_d2s_v2.py` | `slippage_pct` 파라미터 노출; `run()` → pending_signals + R18 |
| `simulation/backtests/backtest_d2s_v3.py` | `slippage_pct` 파라미터 노출; `run()` → pending_signals + 레짐 + R20/R21 |

**핵심 변경 패턴:**
- 수정 전: T일 종가로 시그널 결정 + 즉시 체결 (look-ahead bias)
- 수정 후: T일 종가로 결정 → `pending_signals`에 저장 → T+1일 시가 + 슬리피지 0.05%로 체결

**발견된 버그 (수정 완료):**
- `generate_daily_signals(snap, positions, daily_buy_counts)` → `generate_daily_signals(snap, positions, {})` 로 수정
  - 이유: 오늘 체결 카운트를 내일 신호 생성에 전달하면 `dca_max_daily` 조건이 valid 진입을 차단

---

### Phase 2: Study 10~12 실행 + 결과 ✅

#### Study 10: Look-ahead Bias 수정 전후 비교

| 기간 | biased (종가) | corrected (시가+슬리피지) | Δ |
|------|--------------|--------------------------|---|
| IS   | +6.28% / Sharpe 1.305 / 19건 | +3.29% / Sharpe 0.983 / **7건** | -2.99%p |
| OOS  | +0.72% / Sharpe 0.148 / 31건 | **-8.41%** / Sharpe -0.838 / 27건 | -9.13%p |
| FULL | +8.51% / Sharpe 0.528 / 51건 | -4.21% / Sharpe -0.258 / 35건 | -12.72%p |

- OOS 변화량 **9.1%p < 10%p → Study 13 Optuna 재최적화 불필요**
- 결과 파일: `data/results/backtests/study_10_bias_corrected_20260226.json`

#### Study 11: 레짐 ablation (corrected)

| 방법 | IS | OOS | OOS Sharpe | OOS 순위 |
|------|-----|-----|-----------|---------|
| no_regime | +4.07% | -8.77% | -0.858 | 4위 |
| streak_only | +4.07% | -9.50% | -0.956 | 5위 |
| ma_cross | +4.07% | -10.56% | -1.074 | 6위 |
| full_3signal | +3.29% | **-8.41%** | **-0.838** | 1위(공동) |
| no_poly | +3.29% | -8.41% | -0.838 | 1위(공동) |
| v3_current | +3.29% | -8.41% | -0.838 | 1위(공동) |

- ⚠️ **no_regime 우위 역전** — full_3signal/no_poly/v3_current가 동률 최우수
- full_3signal/no_poly/v3_current 3개가 완전히 동일한 결과 → 실질적으로 레짐이 OOS 결과에 영향 없음
- 결과 파일: `data/results/backtests/study_11_corrected_regime_20260226.json`

#### Study 12: market_score 가중치 (corrected)

| 스킴 | IS | OOS | FULL | OOS 순위 |
|------|-----|-----|------|---------|
| **v3_current** | +3.29% | **-8.41%** | -4.21% | **1위** |
| equal_weight | +5.59% | -11.91% | -5.81% | 2위 |
| v3_no_gld | +5.59% | -12.16% | -6.08% | 3위 |
| v3_spy_only | +6.12% | -12.16% | -5.62% | 3위 |

- ✅ **v3_current OOS 1위 유지 → params_d2s.py weights 확정**
- 결과 파일: `data/results/backtests/study_12_corrected_weights_20260226.json`

---

## 핵심 시사점 (전략 재평가 필요)

1. **look-ahead bias 제거 후 OOS -8.41%** — 전략이 실전에서 손실 발생
2. **IS 거래 횟수 19 → 7건** — pending_signals 패턴으로 체결 기회가 대폭 감소
3. **레짐 방법이 OOS 결과에 실질적 영향 없음** — full_3signal/no_poly/v3_current 동률
4. **현재 파라미터는 biased 환경에서 최적화된 것** → corrected 환경 재최적화 가능성 검토 필요

---

## 남은 작업 (Phase 3: 문서화)

- [ ] `docs/reports/backtest/d2s_v3_study_report.md` — Study 10~12 결과 추가
- [ ] `docs/rules/line_c/trading_rules_attach_v3.md` — Appendix 업데이트
- [ ] `docs/architecture/trading_rules_lineage.md` — 동기화

## 추가 검토 가능 방향

- **전략 재최적화**: corrected 환경에서 Optuna 재실행 (Study 13 조건은 미충족이나 결과가 나쁨 → 별도 판단 필요)
- **IS 거래 부족 원인 분석**: biased 19건 vs corrected 7건 차이가 너무 큼 — 엔진 로직 재검토
- **OOS 기간 (2025-06 ~ 2026-02) 시장 환경**: 이 기간 SPY 실제 수익률 vs 전략 수익률 비교

---

## 파일 상태 요약

```
simulation/backtests/
  backtest_d2s.py      ← look-ahead bias 수정 완료 (커밋됨)
  backtest_d2s_v2.py   ← 수정 완료 (커밋됨)
  backtest_d2s_v3.py   ← 수정 완료 (커밋됨)

experiments/
  study_10_bias_corrected_v3.py       ← 완성 (커밋됨)
  study_11_corrected_regime_ablation.py ← 완성 (커밋됨)
  study_12_corrected_mscore_weights.py  ← 완성 (커밋됨)

data/results/backtests/
  study_10_bias_corrected_20260226.json  ← 결과 저장 (gitignore)
  study_11_corrected_regime_20260226.json
  study_12_corrected_weights_20260226.json
```

## 클러스터 상태

- SLURM 클러스터: `gigaflops-node54` (로그인), `giganode-54` (컴퓨트)
- 현재 실행 중인 Job: 100 (ptj-optimize_v4_study6), 101 (ptj-optimizer_v5) — giganode-51
- 컨테이너 이미지: `ptj_stock_lab.sqsh` — giganode-51 및 giganode-54에 설치 완료
- 최근 완료 Jobs: 153 (Study 11), 154 (Study 12), 156 (Study 10)

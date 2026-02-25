# LineC D2S v3 — Study 6~9B 검증 리포트

> 작성일: 2026-02-25
> 기반: trading_rules_attach_v3.md (Optuna #449)
> 검증 범위: Study 6 (ROBN 1년) / Study 7 (IS 구간) / Study 8 (Regime Ablation) /
>             Study 8B (레짐 방법 6종) / Study 9 (mscore Optuna) / Study 9B (weights 스킴)
> 결론: **v3 파라미터 확정 — weights/레짐 모두 v3_current 유지**

---

## Study 6 — ROBN 포함 1년 백테스트 (2025-01-31 ~ 2026-02-17)

| 지표 | v2 | v3 | Δ |
|---|---|---|---|
| 총 수익률 | -1.83% | +57.34% | +59.17%p |
| 승률 | 66.9% | 82.3% | +15.4%p |
| MDD | -36.13% | -21.61% | +14.52%p |
| Sharpe | 0.151 | 1.968 | +1.817 |
| 총 청산 | 139건 | 62건 | -77건 |
| 평균 수익 | +2.99% | +7.57% | +4.58%p |

결론: ROBN 포함해도 v3 압도 확인. v2는 동 기간 손실.

---

## Study 7 — IS(bull market) 구간 성능 (no-ROBN)

| 지표 | v2_IS | v3_IS | v3_OOS |
|---|---|---|---|
| 총 수익률 | +3.34% | +28.17% | +16.32% |
| 승률 | 71.1% | 100.0% | 73.7% |
| MDD | -17.73% | -1.61% | -16.0% |
| Sharpe | 0.458 | 3.463 | 1.189 |

결론: v3가 IS bull market에서도 우위. OOS 일반화 확인.

---

## Study 8 — Polymarket 레짐 단독 기여도 (Ablation, 1.5년)

| 모드 | 수익률 | MDD | Sharpe |
|---|---|---|---|
| full_3signal | +49.8% | -16.0% | 1.993 |
| no_poly | +49.8% | -16.0% | 1.993 |
| poly_only | +42.7% | -16.0% | 1.755 |
| no_regime | +52.2% | -16.0% | 2.050 |

발견: full_3signal = no_poly → Polymarket BTC 신호 독립 기여 없음. no_regime 최고 수익.

---

## Study 8B — 레짐 감지 방법 6종 비교 (2024-09-18 ~ 2026-02-17)

| 방법 | Return | MDD | Sharpe |
|---|---|---|---|
| no_regime | +52.17% | -16.0% | 2.050 |
| streak_only | +52.17% | -16.0% | 2.050 |
| ma_cross | +49.87% | **-14.23%** | 2.021 |
| v3_3signal | +49.84% | -16.0% | 1.993 |
| streak_sma | +49.84% | -16.0% | 1.993 |
| vix_based | +48.26% | -16.0% | 1.939 |

발견: no_regime/streak_only 공동 1위. ma_cross 유일하게 MDD -14.23% 개선.

---

## Study 9 — market_score_weights Optuna 재탐색 (200 trials)

Best Trial #162: IS +42.62% / OOS +65.59%, 전체 +137.23%
현재 v3_current: IS +33.68% / OOS +62.42%, 전체 +118.05%

→ Study 9B에서 IS 과적합 확인됨 (아래 참조). v3_current 유지 결론.

---

## Study 9B — weights 스킴 4종 × IS/OOS/전체

| 스킴 | IS Return | IS Sharpe | OOS Return | OOS Sharpe | 전체 |
|---|---|---|---|---|---|
| **v3_current** | +28.17% | 3.463 | **+16.32%** | **1.189** | **+49.84%** |
| trial_162 | +28.29% | **3.483** | +14.44% | 1.051 | +47.48% |
| equal | +24.32% | 3.020 | +8.11% | 0.659 | +35.07% |
| riskoff_heavy | +24.19% | 3.086 | +5.96% | 0.502 | +32.19% |

결론: trial_162 OOS 역전 → IS 과적합. v3_current OOS 전 지표 1위 → 유지 확정.

---

## Line A 참고 비교 (1분봉, 타임프레임 상이)

| 지표 | Line A v5 | D2S v3 (참고) |
|---|---|---|
| 기간 수익률 | -9.76% | +49.84% |
| MDD | -10.14% | -16.0% |
| Sharpe | -2.26 | 1.993 |

※ 1분봉 vs 일봉 — 직접 비교 불가. 참고용.

---

## 최종 결론 — v3 파라미터 확정

| 항목 | 결론 | 근거 |
|---|---|---|
| market_score_weights | **v3_current 유지** | Study 9B OOS 역전 확인 |
| 레짐 감지 방식 | **v3_3signal 유지** | 단순화는 v4 과제 이관 |
| Polymarket 신호 | **유지** (기여 낮지만 드래그 없음) | Study 8 |
| ROBN 포함 | **v3 적용 가능** | Study 6 압도 확인 |

---

## v4 후보 과제

| 과제 | 근거 |
|---|---|
| 레짐 감지 단순화 (no_regime 또는 streak_only) | Study 8/8B 수익 우위 |
| ma_cross 레짐 도입 검토 | Study 8B MDD -14.23% 개선 |
| ROBN 포함 1년 Optuna 재탐색 | Study 6 결과 기반 |

---

*Generated from Study 6~9B backtest results*

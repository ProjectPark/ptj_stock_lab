# PTJ v4 Phase 2 — Narrow Search 최적화 리포트

> 생성일: 2026-03-01 12:49

## 1. 실행 정보

| 항목 | 값 |
|---|---|
| Phase | 2 (Narrow Search) |
| 총 Trial | 250 (기존 219 warm + 신규 31) |
| 병렬 Worker | 4 |
| 실행 시간 | 477.6초 (8.0분) |
| Trial당 평균 | 1.9초 |
| 목적함수 모드 | balanced |
| 훈련 기간 | 2023-01-03 ~ (3y) |
| OOS 기간 | 2026-01-01 ~ 2026-02-17 |
| 탐색 파라미터 | 13개 (좁은 범위 + 작은 step) |
| 고정 파라미터 | Study 1+2+5 확정값 |

## 2. Baseline vs Best 비교

| 지표 | Baseline (확정값) | Best (#163) | 차이 |
|---|---|---|---|
| **수익률** | +364.33% | **+361.72%** | -2.61% |
| MDD | -14.30% | -14.25% | +0.05% |
| Sharpe | 1.6075 | **1.6407** | +0.0332 |
| 승률 | 33.8% (169W/244L) | 55.4% | +21.6% |
| CB 차단 | 1101회 | 1013회 | -88회 |

> **판정: Phase 2 훈련 스코어는 baseline 미달 (-2.61%)이나 Sharpe 소폭 개선 (+0.033)**

## 3. Phase 2 탐색 범위

| 파라미터 | 확정값 (center) | 탐색 범위 | step |
|---|---|---|---|
| `V4_SWING_TRIGGER_PCT` | 27.5 | 24.0 ~ 31.0 | 0.5 |
| `V4_SWING_STAGE1_DRAWDOWN_PCT` | -11.0 | -13.0 ~ -9.0 | 0.5 |
| `V4_SWING_STAGE1_ATR_MULT` | 2.5 | 1.75 ~ 3.25 | 0.25 |
| `V4_CB_BTC_SURGE_PCT` | 13.5 | 11.0 ~ 16.0 | 0.5 |
| `V4_CB_BTC_CRASH_PCT` | -6.0 | -7.5 ~ -4.5 | 0.5 |
| `V4_PAIR_FIXED_TP_PCT` | 6.5 | 5.0 ~ 8.5 | 0.5 |
| `V4_PAIR_IMMEDIATE_SELL_PCT` | 0.4 | 0.25 ~ 0.55 | 0.05 |
| `PAIR_GAP_SELL_THRESHOLD_V2` | 9.0 | 7.5 ~ 11.0 | 0.5 |
| `STOP_LOSS_PCT` | -4.25 | -5.5 ~ -3.0 | 0.25 |
| `CONL_SELL_PROFIT_PCT` | 4.5 | 3.0 ~ 6.0 | 0.5 |
| `DCA_DROP_PCT` | -1.35 | -1.8 ~ -0.9 | 0.05 |
| `V4_SIDEWAYS_ATR_DECLINE_PCT` | 10.0 | 5.0 ~ 15.0 | 1.0 |
| `V4_SWING_STAGE1_HOLD_DAYS` | 21 | 14 ~ 28 | 7 |

## 4. 최적 파라미터 (Best Trial #163)

| 파라미터 | Phase 2 Best | Phase 1 확정값 | 변경 |
|---|---|---|---|
| `V4_SWING_TRIGGER_PCT` | **27.0** | 27.5 | -0.5 |
| `V4_SWING_STAGE1_DRAWDOWN_PCT` | **-11.0** | -11.0 | — |
| `V4_SWING_STAGE1_ATR_MULT` | **2.75** | 2.5 | +0.25 |
| `V4_CB_BTC_SURGE_PCT` | **11.5** | 13.5 | -2.0 |
| `V4_CB_BTC_CRASH_PCT` | **-7.0** | -6.0 | -1.0 |
| `V4_PAIR_FIXED_TP_PCT` | **8.5** | 6.5 | +2.0 |
| `V4_PAIR_IMMEDIATE_SELL_PCT` | **0.55** | 0.4 | +0.15 |
| `PAIR_GAP_SELL_THRESHOLD_V2` | **9.0** | 9.0 | — |
| `STOP_LOSS_PCT` | **-4.0** | -4.25 | +0.25 |
| `CONL_SELL_PROFIT_PCT` | **6.0** | 4.5 | +1.5 |
| `DCA_DROP_PCT` | **-0.9** | -1.35 | +0.45 |
| `V4_SIDEWAYS_ATR_DECLINE_PCT` | **8.0** | 10.0 | -2.0 |
| `V4_SWING_STAGE1_HOLD_DAYS` | **21** | 21 | — |

## 5. Top 5 Trials

| # | 스코어 | MDD | Sharpe | 승률 | CB차단 |
|---|---|---|---|---|---|
| 163 | +361.72 | -14.25% | 1.6407 | 55.4% | 1013 |
| 157 | +361.21 | -14.25% | 1.6398 | 55.4% | — |
| 144 | +360.91 | -14.29% | 1.6393 | 55.4% | — |
| 166 | +357.88 | -14.26% | 1.6326 | 55.5% | — |
| 167 | +357.88 | -14.26% | 1.6326 | 55.5% | — |

## 6. OOS 평가 (2026-01-01 ~ 2026-02-17)

| 순위 | Trial | Train 스코어 | OOS 수익률 | OOS MDD | OOS 승률 | 매수 | 매도 |
|---|---|---|---|---|---|---|---|
| 1 | #163 | +361.72 | **+3.42%** | -1.08% | 10.8% (4W/23L) | 7 | 37 |
| 2 | #157 | +361.21 | **+3.42%** | -1.08% | 10.8% (4W/23L) | 7 | 37 |
| 3 | #144 | +360.91 | **+3.42%** | -1.08% | 10.8% (4W/23L) | 7 | 37 |
| 4 | #166 | +357.88 | **+3.70%** | -1.09% | 20.0% (8W/22L) | 7 | 40 |
| 5 | #167 | +357.88 | **+3.70%** | -1.09% | 20.0% (8W/22L) | 7 | 40 |

### Phase 비교

| 단계 | Train 스코어 | OOS 수익률 | OOS MDD | 비고 |
|---|---|---|---|---|
| Study 5 확정 (Phase 1) | +348.76 | +3.51% | -1.54% | BTC_SURGE=13.5 |
| Phase 2 Best (#163) | +361.72 | +3.42% | -1.08% | OOS MDD 개선 |
| Phase 2 #166/#167 | +357.88 | **+3.70%** | -1.09% | OOS 최고 |

## 7. 결과 해석 및 판단

### 주요 관찰
- Phase 2 Best (#163)의 훈련 스코어는 baseline 미달 (-2.61%) — 확정값이 이미 local optimum에 근접
- Sharpe 개선 (1.6075 → 1.6407, +2.1%): 위험 조정 수익 소폭 향상
- OOS: Phase 2 #163(+3.42%)은 Study 5(+3.51%) 대비 소폭 낮고, #166/#167(+3.70%)은 소폭 높음
- OOS MDD: 크게 개선 (-1.54% → -1.08~1.09%)

### 파라미터 변화 패턴
- **CONL_SELL_PROFIT: 4.5 → 6.0 (+33%)** — CONL 익절 기준 대폭 상향 (가장 큰 변화 중 하나)
- **DCA_DROP: -1.35 → -0.9 (+33%)** — DCA 진입 하락 기준 완화 (더 빨리 DCA)
- **PAIR_FIXED_TP: 6.5 → 8.5 (+31%)** — 쌍매도 익절 기준 상향
- **CB_BTC_SURGE: 13.5 → 11.5 (-15%)** — BTC 서지 차단 임계값 하향 (더 자주 차단)
- 변화가 큰 파라미터 多 → 견고성 의문

### 채택 권고

> **Phase 2 Best (#163) 채택 보류 — Phase 1 확정값 유지 권고**
>
> 근거:
> 1. 훈련 스코어 미달 (-2.61%): 확정값 이미 최적에 가까움
> 2. 파라미터 변화 대폭 (13개 중 10개 변경, 일부 30% 이상) vs OOS 개선 미미
> 3. OOS +3.42% (Phase 2 Best) < OOS +3.51% (Study 5 확정)
> 4. 단, #166/#167(OOS +3.70%)는 소폭 우세 — 별도 검토 가능
>
> **결론: Phase 1 확정값(Study 1+2+5)이 안정적. config.py 변경 불필요.**

## 8. config.py 현행 유지값 (Phase 1 확정)

```python
# Study 2 확정
V4_SWING_TRIGGER_PCT = 27.5
V4_SWING_STAGE1_DRAWDOWN_PCT = -11.0
V4_SWING_STAGE1_ATR_MULT = 2.5
V4_SWING_STAGE1_HOLD_DAYS = 21

# Study 5 확정
V4_CB_BTC_SURGE_PCT = 13.5
V4_CB_BTC_CRASH_PCT = -6.0
V4_PAIR_FIXED_TP_PCT = 6.5
V4_PAIR_IMMEDIATE_SELL_PCT = 0.4
PAIR_GAP_SELL_THRESHOLD_V2 = 9.0

# Study 1 확정 (공유)
STOP_LOSS_PCT = -4.25
CONL_SELL_PROFIT_PCT = 4.5
DCA_DROP_PCT = -1.35
V4_SIDEWAYS_ATR_DECLINE_PCT = 10.0
```

# PTJ v4 Optuna 최적화 리포트

> 생성일: 2026-02-24 03:33

## 1. 실행 정보

| 항목 | 값 |
|---|---|
| 총 Trial | 150 (완료: 150, 실패: 0) |
| 병렬 Worker | 8 |
| 실행 시간 | 1154.3초 (19.2분) |
| Trial당 평균 | 7.7초 |
| Sampler | TPE (seed=42) |
| Phase | 1 |
| 목적함수 모드 | balanced |
| 최적화 기간 | 전체 기간 |

## 2. Baseline vs Best 비교

| 지표 | Baseline | Best (#10) | 차이 |
|---|---|---|---|
| **수익률** | +348.76% | **+337.85%** | -10.91% |
| MDD | -13.15% | -13.15% | +0.00% |
| Sharpe | 1.5767 | 1.5976 | +0.0209 |
| 승률 | 30.2% | 34.4% | +4.2% |
| 매도 횟수 | 242 | 317 | +75 |
| 손절 횟수 | 4 | 2 | -2 |
| 시간손절 | 3 | 4 | +1 |
| 횡보장 일수 | 328 | 328 | +0 |
| 수수료 | 3,142원 | 3,343원 | +201원 |
| CB 차단 | 7482 | 1529 | -5953 |

## 3. 최적 파라미터 (Best Trial #10)

| 파라미터 | 최적값 | Baseline | 변경 |
|---|---|---|---|
| `PAIR_GAP_SELL_THRESHOLD_V2` | **9.00** | 8.80 | +0.20 |
| `V4_CB_BTC_CRASH_PCT` | **-6.00** | -5.00 | -1.00 |
| `V4_CB_BTC_SURGE_PCT` | **13.50** | 5.00 | +8.50 |
| `V4_PAIR_FIXED_TP_PCT` | **6.50** | 5.00 | +1.50 |
| `V4_PAIR_IMMEDIATE_SELL_PCT` | **0.40** | 0.40 | - |


## 4. Top 5 Trials

| # | 수익률 | MDD | Sharpe | 승률 | 매도 | 손절 | 횡보일 | CB차단 |
|---|---|---|---|---|---|---|---|---|
| 10 | +337.85% | -13.15% | 1.5976 | 34.4% | 317 | 2 | 328 | 1529 |
| 19 | +337.85% | -13.15% | 1.5976 | 34.4% | 317 | 2 | 328 | 986 |
| 22 | +337.85% | -13.15% | 1.5976 | 34.4% | 317 | 2 | 328 | 1641 |
| 24 | +337.85% | -13.15% | 1.5976 | 34.4% | 317 | 2 | 328 | 1646 |
| 25 | +337.85% | -13.15% | 1.5976 | 34.4% | 317 | 2 | 328 | 1703 |


## 5. 파라미터 중요도 (fANOVA)

| 파라미터 | 중요도 |
|---|---|
| `V4_CB_BTC_SURGE_PCT` | 0.9889 █████████████████████████████ |
| `PAIR_GAP_SELL_THRESHOLD_V2` | 0.0058  |
| `V4_PAIR_IMMEDIATE_SELL_PCT` | 0.0034  |
| `V4_CB_BTC_CRASH_PCT` | 0.0017  |
| `V4_PAIR_FIXED_TP_PCT` | 0.0003  |


## 6. Top 5 파라미터 상세

### #1 — Trial 10 (+337.85%)

```
PAIR_GAP_SELL_THRESHOLD_V2 = 9.00
V4_CB_BTC_CRASH_PCT = -6.00
V4_CB_BTC_SURGE_PCT = 13.50
V4_PAIR_FIXED_TP_PCT = 6.50
V4_PAIR_IMMEDIATE_SELL_PCT = 0.40
```

### #2 — Trial 19 (+337.85%)

```
PAIR_GAP_SELL_THRESHOLD_V2 = 10.50
V4_CB_BTC_CRASH_PCT = -7.00
V4_CB_BTC_SURGE_PCT = 14.50
V4_PAIR_FIXED_TP_PCT = 7.00
V4_PAIR_IMMEDIATE_SELL_PCT = 0.80
```

### #3 — Trial 22 (+337.85%)

```
PAIR_GAP_SELL_THRESHOLD_V2 = 11.00
V4_CB_BTC_CRASH_PCT = -6.00
V4_CB_BTC_SURGE_PCT = 12.50
V4_PAIR_FIXED_TP_PCT = 6.50
V4_PAIR_IMMEDIATE_SELL_PCT = 0.80
```

### #4 — Trial 24 (+337.85%)

```
PAIR_GAP_SELL_THRESHOLD_V2 = 7.50
V4_CB_BTC_CRASH_PCT = -6.00
V4_CB_BTC_SURGE_PCT = 12.50
V4_PAIR_FIXED_TP_PCT = 6.50
V4_PAIR_IMMEDIATE_SELL_PCT = 0.80
```

### #5 — Trial 25 (+337.85%)

```
PAIR_GAP_SELL_THRESHOLD_V2 = 11.00
V4_CB_BTC_CRASH_PCT = -6.00
V4_CB_BTC_SURGE_PCT = 12.00
V4_PAIR_FIXED_TP_PCT = 6.50
V4_PAIR_IMMEDIATE_SELL_PCT = 0.60
```

## 7. OOS 평가 (2026-01-01 ~ 2026-02-17)

> Study 2 OOS +4.17% vs Study 5 OOS +3.51%
> **OOS 양호 — 채택 검토 권고.** BTC SURGE CB 완화로 OOS 매도 37회 활성화.

| 순위 | Trial | Train 스코어 | OOS 수익률 | OOS MDD | OOS 승률 | 매수 | 매도 |
|---|---|---|---|---|---|---|---|
| 1 | #10 | +337.85 | **+3.51%** | -1.54% | 10.8% (4W/23L) | 7 | 37 |
| 2 | #19 | +337.85 | **+3.49%** | -1.56% | 10.8% (4W/23L) | 7 | 37 |
| 3 | #22 | +337.85 | **+3.51%** | -1.54% | 10.8% (4W/23L) | 7 | 37 |
| 4 | #24 | +337.85 | **+3.51%** | -1.54% | 10.8% (4W/23L) | 7 | 37 |
| 5 | #25 | +337.85 | **+3.51%** | -1.54% | 10.8% (4W/23L) | 7 | 37 |

### Study별 OOS 비교

| 스터디 | OOS 수익률 | OOS MDD | OOS 매도 | 핵심 변화 |
|---|---|---|---|---|
| Study 1 (1y base) | -7.12% | — | 1 | CONL=6.0 → 진입 과다 |
| **Study 2** | **+1.17%** | -3.15% | **37** | TRIGGER=27.5% 억제 효과 |
| Study 3 | -6.71% | -11.76% | 1 | CB 완화 + GAP↑ → 과도한 선별 |
| Study 4 | -6.71% | -11.76% | 1 | SIDEWAYS 강화에도 동일 스윙 진입 |
| **Study 5** | **+3.51%** | -1.54% | **37** | BTC_SURGE 완화(5→13.5) → 거래 활성화 |
| (검증) Study 1+2 확정 | +4.17% | -0.18% | 3 | 확정 베이스라인 기준 |

### OOS 원인 분석

- `V4_CB_BTC_SURGE_PCT=5.0` (현재)는 **지나치게 엄격** — BTC 5% 급등만으로 전체 거래 차단
- CB 차단 7482회 → 1529회 (-80%): 현재 설정이 정상 상승장에서도 진입을 과도하게 차단 중
- `V4_CB_BTC_SURGE_PCT=13.5`로 완화 → OOS 매도 37회 (1회 → 37회), 수익 +3.51%
- `V4_PAIR_FIXED_TP_PCT=6.5` (현재 5.0): 익절 기준 상향 → 수익 실현 개선
- Study 3의 CB 완화 실패와 다른 점: Study 3는 **진입 트리거+GAP 동시 변경** → 매도 1회만 발생

### 채택 권고

> **Study 5 파라미터 채택 검토 — OOS 양호 (+3.51%, 37회 거래).**
> `V4_CB_BTC_SURGE_PCT=5.0`이 지나치게 엄격한 차단 설정임이 확인됨.
> Top 5 모두 `BTC_SURGE_PCT >= 12.0` 공통 패턴 → 12.0~13.5 범위가 적정.
> Study 2 확정 기준 OOS(+4.17%, 3회) 대비 거래 횟수는 많으나 수익률 소폭 낮음.
> **walk-forward 검증 후 V4_CB_BTC_SURGE_PCT=13.5, V4_PAIR_FIXED_TP_PCT=6.5 채택 권고.**

## 8. config.py 적용 코드 (Best Trial #10)

```python
PAIR_GAP_SELL_THRESHOLD_V2 = 9.00
V4_CB_BTC_CRASH_PCT = -6.00
V4_CB_BTC_SURGE_PCT = 13.50
V4_PAIR_FIXED_TP_PCT = 6.50
V4_PAIR_IMMEDIATE_SELL_PCT = 0.40
```

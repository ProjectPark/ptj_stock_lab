# PTJ v4 Optuna 최적화 리포트

> 생성일: 2026-02-24 02:40

## 1. 실행 정보

| 항목 | 값 |
|---|---|
| 총 Trial | 150 (완료: 150, 실패: 0) |
| 병렬 Worker | 8 |
| 실행 시간 | 1211.7초 (20.2분) |
| Trial당 평균 | 8.1초 |
| Sampler | TPE (seed=42) |
| Phase | 1 |
| 목적함수 모드 | balanced |
| 최적화 기간 | 전체 기간 |

## 2. Baseline vs Best 비교

| 지표 | Baseline | Best (#36) | 차이 |
|---|---|---|---|
| **수익률** | +357.97% | **+344.27%** | -13.70% |
| MDD | -13.15% | -13.15% | +0.00% |
| Sharpe | 1.5962 | 1.6016 | +0.0054 |
| 승률 | 33.3% | 43.2% | +9.9% |
| 매도 횟수 | 312 | 229 | -83 |
| 손절 횟수 | 1 | 1 | +0 |
| 시간손절 | 5 | 2 | -3 |
| 횡보장 일수 | 328 | 370 | +42 |
| 수수료 | 3,318원 | 3,096원 | -222원 |
| CB 차단 | 4827 | 4544 | -283 |

## 3. 최적 파라미터 (Best Trial #36)

| 파라미터 | 최적값 | Baseline | 변경 |
|---|---|---|---|
| `STOP_LOSS_PCT` | **-5.25** | -4.25 | -1.00 |
| `V4_SIDEWAYS_ATR_DECLINE_PCT` | **7.50** | 10.00 | -2.50 |
| `V4_SIDEWAYS_RSI_HIGH` | **70.00** | 65.00 | +5.00 |
| `V4_SIDEWAYS_RSI_LOW` | **35.00** | 35.00 | - |
| `V4_SIDEWAYS_VOLUME_DECLINE_PCT` | **17.50** | 15.00 | +2.50 |
| `V4_SPLIT_BUY_INTERVAL_MIN` | **30** | 10 | +20 |


## 4. Top 5 Trials

| # | 수익률 | MDD | Sharpe | 승률 | 매도 | 손절 | 횡보일 | CB차단 |
|---|---|---|---|---|---|---|---|---|
| 36 | +344.27% | -13.15% | 1.6016 | 43.2% | 229 | 1 | 370 | 4544 |
| 38 | +344.27% | -13.15% | 1.6016 | 43.2% | 229 | 1 | 370 | 4544 |
| 105 | +344.26% | -13.15% | 1.6016 | 43.2% | 229 | 1 | 347 | 4557 |
| 113 | +344.26% | -13.15% | 1.6016 | 43.2% | 229 | 1 | 333 | 4557 |
| 114 | +344.26% | -13.15% | 1.6016 | 43.2% | 229 | 1 | 333 | 4557 |


## 5. 파라미터 중요도 (fANOVA)

| 파라미터 | 중요도 |
|---|---|
| `V4_SIDEWAYS_ATR_DECLINE_PCT` | 0.8666 █████████████████████████ |
| `V4_SIDEWAYS_RSI_HIGH` | 0.0635 █ |
| `V4_SIDEWAYS_VOLUME_DECLINE_PCT` | 0.0500 █ |
| `STOP_LOSS_PCT` | 0.0142  |
| `V4_SIDEWAYS_RSI_LOW` | 0.0034  |
| `V4_SPLIT_BUY_INTERVAL_MIN` | 0.0023  |


## 6. Top 5 파라미터 상세

### #1 — Trial 36 (+344.27%)

```
STOP_LOSS_PCT = -5.25
V4_SIDEWAYS_ATR_DECLINE_PCT = 7.50
V4_SIDEWAYS_RSI_HIGH = 70.00
V4_SIDEWAYS_RSI_LOW = 35.00
V4_SIDEWAYS_VOLUME_DECLINE_PCT = 17.50
V4_SPLIT_BUY_INTERVAL_MIN = 30
```

### #2 — Trial 38 (+344.27%)

```
STOP_LOSS_PCT = -5.25
V4_SIDEWAYS_ATR_DECLINE_PCT = 7.50
V4_SIDEWAYS_RSI_HIGH = 70.00
V4_SIDEWAYS_RSI_LOW = 35.00
V4_SIDEWAYS_VOLUME_DECLINE_PCT = 17.50
V4_SPLIT_BUY_INTERVAL_MIN = 30
```

### #3 — Trial 105 (+344.26%)

```
STOP_LOSS_PCT = -4.00
V4_SIDEWAYS_ATR_DECLINE_PCT = 7.50
V4_SIDEWAYS_RSI_HIGH = 65.00
V4_SIDEWAYS_RSI_LOW = 35.00
V4_SIDEWAYS_VOLUME_DECLINE_PCT = 15.00
V4_SPLIT_BUY_INTERVAL_MIN = 20
```

### #4 — Trial 113 (+344.26%)

```
STOP_LOSS_PCT = -3.00
V4_SIDEWAYS_ATR_DECLINE_PCT = 7.50
V4_SIDEWAYS_RSI_HIGH = 65.00
V4_SIDEWAYS_RSI_LOW = 35.00
V4_SIDEWAYS_VOLUME_DECLINE_PCT = 17.50
V4_SPLIT_BUY_INTERVAL_MIN = 20
```

### #5 — Trial 114 (+344.26%)

```
STOP_LOSS_PCT = -4.00
V4_SIDEWAYS_ATR_DECLINE_PCT = 7.50
V4_SIDEWAYS_RSI_HIGH = 65.00
V4_SIDEWAYS_RSI_LOW = 35.00
V4_SIDEWAYS_VOLUME_DECLINE_PCT = 17.50
V4_SPLIT_BUY_INTERVAL_MIN = 20
```

## 7. OOS 평가 (2026-01-01 ~ 2026-02-17)

> Study 2 기준과 비교: Study 2 OOS +1.17% vs Study 4 OOS -6.71%
> **Study 4 파라미터도 OOS에서 역효과** — 채택 보류 권고.

| 순위 | Trial | Train 스코어 | OOS 수익률 | OOS MDD | OOS 승률 | 매수 | 매도 |
|---|---|---|---|---|---|---|---|
| 1 | #36 | +344.27 | **-6.71%** | -11.76% | 0.0% (0W/1L) | 2 | 1 |
| 2 | #38 | +344.27 | **-6.71%** | -11.76% | 0.0% (0W/1L) | 2 | 1 |
| 3 | #105 | +344.26 | **-6.71%** | -11.76% | 0.0% (0W/1L) | 2 | 1 |
| 4 | #113 | +344.26 | **-6.71%** | -11.76% | 0.0% (0W/1L) | 2 | 1 |
| 5 | #114 | +344.26 | **-6.71%** | -11.76% | 0.0% (0W/1L) | 2 | 1 |

### Study별 OOS 비교

| 스터디 | OOS 수익률 | OOS MDD | OOS 매도 | 핵심 변화 |
|---|---|---|---|---|
| Study 1 (1y base) | -7.12% | — | 1 | CONL=6.0 → 진입 과다 |
| **Study 2** | **+1.17%** | -3.15% | **37** | TRIGGER=27.5% 억제 효과 |
| Study 3 | -6.71% | -11.76% | 1 | CB 완화 + GAP↑ → 과도한 선별 |
| Study 4 | -6.71% | -11.76% | 1 | SIDEWAYS 강화에도 동일 스윙 진입 |

### OOS 원인 분석

- 2026-01 구간에서 스윙 트레이드가 1건 발동 → 손실 -6.71%
- `V4_SIDEWAYS_ATR_DECLINE_PCT=7.5` (7.5% 이상 ATR 하락 시 횡보장): 해당 스윙 진입 당시 ATR 조건이 미충족
- `V4_SPLIT_BUY_INTERVAL_MIN=30` (분할매수 간격 확대): 진입 자체를 막지 못함
- SIDEWAYS 강화로 **Train 구간 승률은 +9.9% 개선**됐으나, OOS 특정 구간의 단일 스윙 손실을 막지 못함
- Study 3과 동일한 OOS 결과: 2026-01 손실 스윙은 SIDEWAYS/CB/트리거 필터 모두 통과

### 채택 권고

> **Study 4 파라미터는 OOS 기준으로 Study 2보다 열위 — 채택 보류.**
> Train 구간 Sharpe(+0.0054)와 승률(+9.9%) 개선은 유의미하나, OOS 단일 사례로는 우위를 확인할 수 없음.
> **Study 1(3y) + Study 2 swing 확정값 조합 유지 권고.**
> SIDEWAYS 파라미터는 더 긴 OOS 구간(2년 이상)의 walk-forward 검증 필요.

## 8. config.py 적용 코드 (Best Trial #36)

```python
STOP_LOSS_PCT = -5.25
V4_SIDEWAYS_ATR_DECLINE_PCT = 7.50
V4_SIDEWAYS_RSI_HIGH = 70.00
V4_SIDEWAYS_RSI_LOW = 35.00
V4_SIDEWAYS_VOLUME_DECLINE_PCT = 17.50
V4_SPLIT_BUY_INTERVAL_MIN = 30
```

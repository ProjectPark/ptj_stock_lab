# PTJ v4 Optuna 최적화 리포트

> 생성일: 2026-02-24 01:25

## 1. 실행 정보

| 항목 | 값 |
|---|---|
| 총 Trial | 150 (완료: 150, 실패: 0) |
| 병렬 Worker | 8 |
| 실행 시간 | 728.7초 (12.1분) |
| Trial당 평균 | 4.9초 |
| Sampler | TPE (seed=42) |
| Phase | 1 |
| 목적함수 모드 | balanced |
| 최적화 기간 | 전체 기간 |

## 2. Baseline vs Best 비교

| 지표 | Baseline | Best (#149) | 차이 |
|---|---|---|---|
| **수익률** | +14.34% | **+338.27%** | +323.93% |
| MDD | -24.27% | -13.15% | -11.12% |
| Sharpe | 0.3054 | 1.6005 | +1.2951 |
| 승률 | 66.7% | 33.3% | -33.3% |
| 매도 횟수 | 51 | 312 | +261 |
| 손절 횟수 | 0 | 1 | +1 |
| 시간손절 | 0 | 5 | +5 |
| 횡보장 일수 | 321 | 328 | +7 |
| 수수료 | 1,219원 | 3,322원 | +2,103원 |
| CB 차단 | 695 | 4827 | +4132 |

## 3. 최적 파라미터 (Best Trial #149)

| 파라미터 | 최적값 | Baseline | 변경 |
|---|---|---|---|
| `V4_SWING_STAGE1_ATR_MULT` | **2.50** | 1.50 | +1.00 |
| `V4_SWING_STAGE1_DRAWDOWN_PCT` | **-11.00** | -15.00 | +4.00 |
| `V4_SWING_STAGE1_HOLD_DAYS` | **21** | 63 | -42 |
| `V4_SWING_TRIGGER_PCT` | **27.50** | 15.00 | +12.50 |


## 4. Top 5 Trials

| # | 수익률 | MDD | Sharpe | 승률 | 매도 | 손절 | 횡보일 | CB차단 |
|---|---|---|---|---|---|---|---|---|
| 149 | +338.27% | -13.15% | 1.6005 | 33.3% | 312 | 1 | 328 | 4827 |
| 25 | +317.26% | -14.91% | 1.5564 | 33.3% | 312 | 1 | 328 | 4827 |
| 26 | +317.26% | -14.91% | 1.5564 | 33.3% | 312 | 1 | 328 | 4827 |
| 27 | +317.26% | -14.91% | 1.5564 | 33.3% | 312 | 1 | 328 | 4827 |
| 73 | +317.26% | -14.91% | 1.5564 | 33.3% | 312 | 1 | 328 | 4827 |


## 5. 파라미터 중요도 (fANOVA)

| 파라미터 | 중요도 |
|---|---|
| `V4_SWING_TRIGGER_PCT` | 0.9483 ████████████████████████████ |
| `V4_SWING_STAGE1_DRAWDOWN_PCT` | 0.0357 █ |
| `V4_SWING_STAGE1_ATR_MULT` | 0.0103  |
| `V4_SWING_STAGE1_HOLD_DAYS` | 0.0058  |


## 6. Top 5 파라미터 상세

### #1 — Trial 149 (+338.27%)

```
V4_SWING_STAGE1_ATR_MULT = 2.50
V4_SWING_STAGE1_DRAWDOWN_PCT = -11.00
V4_SWING_STAGE1_HOLD_DAYS = 21
V4_SWING_TRIGGER_PCT = 27.50
```

### #2 — Trial 25 (+317.26%)

```
V4_SWING_STAGE1_ATR_MULT = 4.00
V4_SWING_STAGE1_DRAWDOWN_PCT = -12.00
V4_SWING_STAGE1_HOLD_DAYS = 21
V4_SWING_TRIGGER_PCT = 27.50
```

### #3 — Trial 26 (+317.26%)

```
V4_SWING_STAGE1_ATR_MULT = 2.00
V4_SWING_STAGE1_DRAWDOWN_PCT = -12.00
V4_SWING_STAGE1_HOLD_DAYS = 28
V4_SWING_TRIGGER_PCT = 27.50
```

### #4 — Trial 27 (+317.26%)

```
V4_SWING_STAGE1_ATR_MULT = 2.00
V4_SWING_STAGE1_DRAWDOWN_PCT = -12.00
V4_SWING_STAGE1_HOLD_DAYS = 28
V4_SWING_TRIGGER_PCT = 27.50
```

### #5 — Trial 73 (+317.26%)

```
V4_SWING_STAGE1_ATR_MULT = 3.00
V4_SWING_STAGE1_DRAWDOWN_PCT = -12.00
V4_SWING_STAGE1_HOLD_DAYS = 21
V4_SWING_TRIGGER_PCT = 27.50
```

## 7. config.py 적용 코드 (Best Trial #149)

```python
V4_SWING_STAGE1_ATR_MULT = 2.50
V4_SWING_STAGE1_DRAWDOWN_PCT = -11.00
V4_SWING_STAGE1_HOLD_DAYS = 21
V4_SWING_TRIGGER_PCT = 27.50
```

# PTJ v4 Optuna 최적화 리포트

> 생성일: 2026-02-24 00:26

## 1. 실행 정보

| 항목 | 값 |
|---|---|
| 총 Trial | 160 (완료: 150, 실패: 10) |
| 병렬 Worker | 10 |
| 실행 시간 | 659.5초 (11.0분) |
| Trial당 평균 | 4.1초 |
| Sampler | TPE (seed=42) |
| Phase | 1 |
| 목적함수 모드 | balanced |
| 최적화 기간 | 전체 기간 |

## 2. Baseline vs Best 비교

| 지표 | Baseline | Best (#0) | 차이 |
|---|---|---|---|
| **수익률** | +25.92% | **+21.96%** | -3.96% |
| MDD | -9.34% | -9.34% | +0.00% |
| Sharpe | 1.3923 | 1.3923 | +0.0000 |
| 승률 | 68.3% | 68.3% | +0.0% |
| 매도 횟수 | 41 | 41 | +0 |
| 손절 횟수 | 0 | 0 | +0 |
| 시간손절 | 0 | 0 | +0 |
| 횡보장 일수 | 122 | 122 | +0 |
| 수수료 | 510원 | 510원 | +0원 |
| CB 차단 | 592 | 592 | +0 |

## 3. 최적 파라미터 (Best Trial #0)

| 파라미터 | 최적값 | Baseline | 변경 |
|---|---|---|---|
| `CONL_SELL_PROFIT_PCT` | **4.50** | 4.50 | - |
| `DCA_DROP_PCT` | **-1.35** | -1.35 | - |
| `PAIR_GAP_SELL_THRESHOLD_V2` | **6.60** | 6.60 | - |
| `STOP_LOSS_BULLISH_PCT` | **-16.00** | -16.00 | - |
| `V4_DCA_MAX_COUNT` | **1** | 1 | - |
| `V4_PAIR_FIXED_TP_PCT` | **7.50** | 7.50 | - |
| `V4_PAIR_IMMEDIATE_SELL_PCT` | **0.20** | 0.20 | - |


## 4. Top 5 Trials

| # | 수익률 | MDD | Sharpe | 승률 | 매도 | 손절 | 횡보일 | CB차단 |
|---|---|---|---|---|---|---|---|---|
| 0 | +21.96% | -9.34% | 1.3923 | 68.3% | 41 | 0 | 122 | 592 |
| 1 | +21.96% | -9.34% | 1.3923 | 68.3% | 41 | 0 | 122 | 592 |
| 40 | -29.35% | -24.27% | 0.1385 | 70.2% | 57 | 0 | 304 | 695 |
| 41 | -29.35% | -24.27% | 0.1385 | 70.2% | 57 | 0 | 304 | 695 |
| 42 | -29.35% | -24.27% | 0.1385 | 70.2% | 57 | 0 | 304 | 695 |


## 5. 파라미터 중요도 (fANOVA)

| 파라미터 | 중요도 |
|---|---|
| `CONL_SELL_PROFIT_PCT` | 0.6495 ███████████████████ |
| `PAIR_GAP_SELL_THRESHOLD_V2` | 0.2592 ███████ |
| `V4_PAIR_FIXED_TP_PCT` | 0.0574 █ |
| `DCA_DROP_PCT` | 0.0213  |
| `V4_PAIR_IMMEDIATE_SELL_PCT` | 0.0096  |
| `STOP_LOSS_BULLISH_PCT` | 0.0031  |
| `V4_DCA_MAX_COUNT` | 0.0000  |


## 6. Top 5 파라미터 상세

### #1 — Trial 0 (+21.96%)

```
CONL_SELL_PROFIT_PCT = 4.50
DCA_DROP_PCT = -1.35
PAIR_GAP_SELL_THRESHOLD_V2 = 6.60
STOP_LOSS_BULLISH_PCT = -16.00
V4_DCA_MAX_COUNT = 1
V4_PAIR_FIXED_TP_PCT = 7.50
V4_PAIR_IMMEDIATE_SELL_PCT = 0.20
```

### #2 — Trial 1 (+21.96%)

```
CONL_SELL_PROFIT_PCT = 4.50
DCA_DROP_PCT = -1.35
PAIR_GAP_SELL_THRESHOLD_V2 = 6.60
STOP_LOSS_BULLISH_PCT = -16.00
V4_DCA_MAX_COUNT = 1
V4_PAIR_FIXED_TP_PCT = 7.50
V4_PAIR_IMMEDIATE_SELL_PCT = 0.20
```

### #3 — Trial 40 (-29.35%)

```
CONL_SELL_PROFIT_PCT = 6.00
DCA_DROP_PCT = -1.40
PAIR_GAP_SELL_THRESHOLD_V2 = 5.80
STOP_LOSS_BULLISH_PCT = -15.00
V4_DCA_MAX_COUNT = 1
V4_PAIR_FIXED_TP_PCT = 8.50
V4_PAIR_IMMEDIATE_SELL_PCT = 0.20
```

### #4 — Trial 41 (-29.35%)

```
CONL_SELL_PROFIT_PCT = 6.00
DCA_DROP_PCT = -1.40
PAIR_GAP_SELL_THRESHOLD_V2 = 5.40
STOP_LOSS_BULLISH_PCT = -15.00
V4_DCA_MAX_COUNT = 1
V4_PAIR_FIXED_TP_PCT = 8.00
V4_PAIR_IMMEDIATE_SELL_PCT = 0.20
```

### #5 — Trial 42 (-29.35%)

```
CONL_SELL_PROFIT_PCT = 6.00
DCA_DROP_PCT = -1.40
PAIR_GAP_SELL_THRESHOLD_V2 = 5.80
STOP_LOSS_BULLISH_PCT = -15.00
V4_DCA_MAX_COUNT = 1
V4_PAIR_FIXED_TP_PCT = 8.50
V4_PAIR_IMMEDIATE_SELL_PCT = 0.20
```

## 7. config.py 적용 코드 (Best Trial #0)

```python
CONL_SELL_PROFIT_PCT = 4.50
DCA_DROP_PCT = -1.35
PAIR_GAP_SELL_THRESHOLD_V2 = 6.60
STOP_LOSS_BULLISH_PCT = -16.00
V4_DCA_MAX_COUNT = 1
V4_PAIR_FIXED_TP_PCT = 7.50
V4_PAIR_IMMEDIATE_SELL_PCT = 0.20
```

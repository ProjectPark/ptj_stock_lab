# PTJ v4 Optuna 최적화 리포트

> 생성일: 2026-02-24 02:07

## 1. 실행 정보

| 항목 | 값 |
|---|---|
| 총 Trial | 150 (완료: 150, 실패: 0) |
| 병렬 Worker | 8 |
| 실행 시간 | 1497.8초 (25.0분) |
| Trial당 평균 | 10.0초 |
| Sampler | TPE (seed=42) |
| Phase | 1 |
| 목적함수 모드 | balanced |
| 최적화 기간 | 전체 기간 |

## 2. Baseline vs Best 비교

| 지표 | Baseline | Best (#136) | 차이 |
|---|---|---|---|
| **수익률** | +357.97% | **+342.78%** | -15.19% |
| MDD | -13.15% | -13.15% | +0.00% |
| Sharpe | 1.5962 | 1.5989 | +0.0027 |
| 승률 | 33.3% | 39.5% | +6.2% |
| 매도 횟수 | 312 | 258 | -54 |
| 손절 횟수 | 1 | 2 | +1 |
| 시간손절 | 5 | 4 | -1 |
| 횡보장 일수 | 328 | 328 | +0 |
| 수수료 | 3,318원 | 3,181원 | -137원 |
| CB 차단 | 4827 | 1646 | -3181 |

## 3. 최적 파라미터 (Best Trial #136)

| 파라미터 | 최적값 | Baseline | 변경 |
|---|---|---|---|
| `V4_CB_BTC_CRASH_PCT` | **-6.00** | -3.50 | -2.50 |
| `V4_CB_BTC_SURGE_PCT` | **12.00** | 8.50 | +3.50 |
| `V4_CB_VIX_COOLDOWN_DAYS` | **19** | 13 | +6 |
| `V4_CB_VIX_SPIKE_PCT` | **5.50** | 3.00 | +2.50 |
| `V4_COIN_TRIGGER_PCT` | **2.00** | 3.00 | -1.00 |
| `V4_CONL_TRIGGER_PCT` | **5.50** | 3.00 | +2.50 |
| `V4_PAIR_GAP_ENTRY_THRESHOLD` | **4.80** | 2.00 | +2.80 |


## 4. Top 5 Trials

| # | 수익률 | MDD | Sharpe | 승률 | 매도 | 손절 | 횡보일 | CB차단 |
|---|---|---|---|---|---|---|---|---|
| 136 | +342.78% | -13.15% | 1.5989 | 39.5% | 258 | 2 | 328 | 1646 |
| 137 | +342.78% | -13.15% | 1.5989 | 39.5% | 258 | 2 | 328 | 1650 |
| 144 | +342.78% | -13.15% | 1.5989 | 39.5% | 258 | 2 | 328 | 1975 |
| 139 | +342.69% | -13.15% | 1.5979 | 44.1% | 222 | 2 | 328 | 4769 |
| 140 | +342.69% | -13.15% | 1.5979 | 44.1% | 222 | 2 | 328 | 4780 |


## 5. 파라미터 중요도 (fANOVA)

| 파라미터 | 중요도 |
|---|---|
| `V4_CONL_TRIGGER_PCT` | 0.8733 ██████████████████████████ |
| `V4_CB_BTC_SURGE_PCT` | 0.0864 ██ |
| `V4_CB_VIX_COOLDOWN_DAYS` | 0.0180  |
| `V4_PAIR_GAP_ENTRY_THRESHOLD` | 0.0106  |
| `V4_COIN_TRIGGER_PCT` | 0.0058  |
| `V4_CB_VIX_SPIKE_PCT` | 0.0045  |
| `V4_CB_BTC_CRASH_PCT` | 0.0012  |


## 6. Top 5 파라미터 상세

### #1 — Trial 136 (+342.78%)

```
V4_CB_BTC_CRASH_PCT = -6.00
V4_CB_BTC_SURGE_PCT = 12.00
V4_CB_VIX_COOLDOWN_DAYS = 19
V4_CB_VIX_SPIKE_PCT = 5.50
V4_COIN_TRIGGER_PCT = 2.00
V4_CONL_TRIGGER_PCT = 5.50
V4_PAIR_GAP_ENTRY_THRESHOLD = 4.80
```

### #2 — Trial 137 (+342.78%)

```
V4_CB_BTC_CRASH_PCT = -6.00
V4_CB_BTC_SURGE_PCT = 12.00
V4_CB_VIX_COOLDOWN_DAYS = 15
V4_CB_VIX_SPIKE_PCT = 5.50
V4_COIN_TRIGGER_PCT = 2.00
V4_CONL_TRIGGER_PCT = 5.50
V4_PAIR_GAP_ENTRY_THRESHOLD = 4.40
```

### #3 — Trial 144 (+342.78%)

```
V4_CB_BTC_CRASH_PCT = -5.50
V4_CB_BTC_SURGE_PCT = 12.00
V4_CB_VIX_COOLDOWN_DAYS = 15
V4_CB_VIX_SPIKE_PCT = 5.50
V4_COIN_TRIGGER_PCT = 2.00
V4_CONL_TRIGGER_PCT = 5.50
V4_PAIR_GAP_ENTRY_THRESHOLD = 4.60
```

### #4 — Trial 139 (+342.69%)

```
V4_CB_BTC_CRASH_PCT = -5.50
V4_CB_BTC_SURGE_PCT = 7.00
V4_CB_VIX_COOLDOWN_DAYS = 15
V4_CB_VIX_SPIKE_PCT = 5.50
V4_COIN_TRIGGER_PCT = 2.00
V4_CONL_TRIGGER_PCT = 5.50
V4_PAIR_GAP_ENTRY_THRESHOLD = 4.60
```

### #5 — Trial 140 (+342.69%)

```
V4_CB_BTC_CRASH_PCT = -5.50
V4_CB_BTC_SURGE_PCT = 7.00
V4_CB_VIX_COOLDOWN_DAYS = 15
V4_CB_VIX_SPIKE_PCT = 5.50
V4_COIN_TRIGGER_PCT = 2.00
V4_CONL_TRIGGER_PCT = 5.50
V4_PAIR_GAP_ENTRY_THRESHOLD = 4.20
```

## 7. OOS 평가 (2026-01-01 ~ 2026-02-17)

> Study 2 기준과 비교: Study 2 OOS +1.17% vs Study 3 OOS -6.71%
> **Study 3 파라미터가 OOS에서 오히려 역효과** — 채택 보류 권고.

| 순위 | Trial | Train 스코어 | OOS 수익률 | OOS MDD | OOS 승률 | 매수 | 매도 |
|---|---|---|---|---|---|---|---|
| 1 | #136 | +342.78 | **-6.71%** | -11.76% | 0.0% (0W/1L) | 2 | 1 |
| 2 | #137 | +342.78 | **-6.71%** | -11.76% | 0.0% (0W/1L) | 2 | 1 |
| 3 | #144 | +342.78 | **-6.71%** | -11.76% | 0.0% (0W/1L) | 2 | 1 |
| 4 | #139 | +342.69 | **-6.71%** | -11.76% | 0.0% (0W/1L) | 2 | 1 |
| 5 | #140 | +342.69 | **-6.71%** | -11.76% | 0.0% (0W/1L) | 2 | 1 |

### Study별 OOS 비교

| 스터디 | OOS 수익률 | OOS MDD | OOS 매도 | 핵심 변화 |
|---|---|---|---|---|
| Study 1 (1y base) | -7.12% | — | 1 | CONL=6.0 → 진입 과다 |
| **Study 2** | **+1.17%** | -3.15% | **37** | TRIGGER=27.5% 억제 효과 |
| Study 3 | -6.71% | -11.76% | 1 | CB 완화 + GAP↑ → 과도한 선별 |

### OOS 원인 분석

- `V4_PAIR_GAP_ENTRY_THRESHOLD=4.8` (현재 2.0) — pair 진입 조건이 너무 까다로워 일반 장중 거래 거의 없음 (매도 1회)
- `V4_CONL_TRIGGER_PCT=5.5` (현재 3.0) — CONL 진입 사실상 차단
- 결과: 2026-01에서 swing 1건만 발동 → 손실 -6.71%
- **CB 파라미터 완화는 Train에서만 효과**, OOS에서는 오히려 나쁜 조건 진입 허용

### 채택 권고

> **Study 3 파라미터는 OOS 기준으로 Study 2보다 열위 — 채택 보류.**
> Study 1 (3y) + Study 2 swing 확정값 조합 유지 권고.
> CONL_TRIGGER와 PAIR_GAP_ENTRY는 별도 walk-forward 검증 필요.

## 8. config.py 적용 코드 (Best Trial #136)

```python
V4_CB_BTC_CRASH_PCT = -6.00
V4_CB_BTC_SURGE_PCT = 12.00
V4_CB_VIX_COOLDOWN_DAYS = 19
V4_CB_VIX_SPIKE_PCT = 5.50
V4_COIN_TRIGGER_PCT = 2.00
V4_CONL_TRIGGER_PCT = 5.50
V4_PAIR_GAP_ENTRY_THRESHOLD = 4.80
```

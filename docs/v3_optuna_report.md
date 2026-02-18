# PTJ v3 Optuna 최적화 리포트

> 생성일: 2026-02-18 05:47

---

## ⚠️ 데이터 기간 및 검증 방법

### 백테스트 데이터 기간
| 항목 | 기간 | 일수 |
|---|---|---|
| **주식 1분봉** | 2025-01-03 ~ 2026-02-17 | 269 거래일 |
| **Polymarket** | 2025-01-03 ~ 2026-02-17 | 411 일 |
| **환율 데이터** | 2025-01-02 ~ 2026-02-17 | - |

### 검증 방식
- **In-Sample 최적화**: 전체 기간(2025-01-03 ~ 2026-02-17)에 대해 파라미터 최적화 수행
- **⚠️ 주의**: 별도의 Out-of-Sample Test 기간 없음 → **과최적화 위험 존재**
- **권장**: 실전 적용 전 2026년 2월 18일 이후 데이터로 검증 필요

### 과최적화 리스크 완화 전략
1. ✅ **Phase 1 (500 trials)**: 넓은 파라미터 공간 탐색
2. ✅ **Phase 2 (310 trials)**: 상위 클러스터 중심 미세 튜닝
3. ✅ **Top 5 수렴성**: 유사한 파라미터 세트가 반복적으로 상위권 달성
4. ❌ **Walk-forward validation**: 미실시 (향후 권장)

---

## 1. 실행 정보

| 항목 | 값 |
|---|---|
| 총 Trial | 310 (완료: 310, 실패: 0) |
| 병렬 Worker | 10 |
| 실행 시간 | 991.5초 (16.5분) |
| Trial당 평균 | 3.2초 |
| Sampler | TPE (seed=42) |

## 2. Baseline vs Best 비교

| 지표 | Baseline | Best (#{best.number}) | 차이 |
|---|---|---|---|
| **수익률** | -7.38% | **+13.42%** | +20.79% |
| MDD | -10.13% | -2.17% | -7.96% |
| Sharpe | -0.9150 | 2.3021 | +3.2171 |
| 승률 | 57.5% | 43.0% | -14.4% |
| 매도 횟수 | 2255 | 804 | -1451 |
| 손절 횟수 | 72 | 8 | -64 |
| 시간손절 | 35 | 31 | -4 |
| 횡보장 일수 | 15 | 130 | +115 |
| 수수료 | 5,774,855원 | 2,459,521원 | -3,315,335원 |

## 3. 최적 파라미터 (Best Trial #301)

| 파라미터 | 최적값 | Baseline | 변경 |
|---|---|---|---|
| `COIN_SELL_PROFIT_PCT` | **5.00** | 3.00 | +2.00 |
| `CONL_SELL_PROFIT_PCT` | **3.50** | 2.80 | +0.70 |
| `DCA_DROP_PCT` | **-0.95** | -0.50 | -0.45 |
| `MAX_HOLD_HOURS` | **4** | 5 | -1 |
| `PAIR_GAP_SELL_THRESHOLD_V2` | **8.80** | 0.90 | +7.90 |
| `PAIR_SELL_FIRST_PCT` | **0.95** | 0.80 | +0.15 |
| `STOP_LOSS_BULLISH_PCT` | **-14.00** | -8.00 | -6.00 |
| `STOP_LOSS_PCT` | **-5.25** | -3.00 | -2.25 |
| `TAKE_PROFIT_PCT` | **4.00** | 2.00 | +2.00 |
| `V3_COIN_TRIGGER_PCT` | **5.50** | 4.50 | +1.00 |
| `V3_CONL_TRIGGER_PCT` | **6.00** | 4.50 | +1.50 |
| `V3_DCA_MAX_COUNT` | **9** | 4 | +5 |
| `V3_MAX_PER_STOCK_KRW` | **6,000,000** | 7,000,000 | -1000000 |
| `V3_PAIR_GAP_ENTRY_THRESHOLD` | **7.60** | 2.20 | +5.40 |
| `V3_SIDEWAYS_GLD_THRESHOLD` | **0.30** | 0.30 | +0.00 |
| `V3_SIDEWAYS_INDEX_THRESHOLD` | **0.90** | 0.50 | +0.40 |
| `V3_SIDEWAYS_MIN_SIGNALS` | **2** | 3 | -1 |
| `V3_SIDEWAYS_POLY_HIGH` | **0.50** | 0.60 | -0.10 |
| `V3_SIDEWAYS_POLY_LOW` | **0.35** | 0.40 | -0.05 |
| `V3_SPLIT_BUY_INTERVAL_MIN` | **25** | 20 | +5 |

## 4. Top 5 Trials

| # | 수익률 | MDD | Sharpe | 승률 | 매도 | 손절 | 횡보일 |
|---|---|---|---|---|---|---|---|
| 301 | +13.42% | -2.17% | 2.3021 | 43.0% | 804 | 8 | 130 |
| 302 | +13.42% | -2.17% | 2.3021 | 43.0% | 804 | 8 | 130 |
| 308 | +13.42% | -2.17% | 2.3021 | 43.0% | 804 | 8 | 130 |
| 270 | +13.39% | -2.12% | 2.2982 | 43.2% | 869 | 8 | 130 |
| 271 | +13.39% | -2.12% | 2.2982 | 43.2% | 869 | 8 | 130 |

## 6. Top 5 파라미터 상세

### #1 — Trial 301 (+13.42%)

```
COIN_SELL_PROFIT_PCT = 5.00
CONL_SELL_PROFIT_PCT = 3.50
DCA_DROP_PCT = -0.95
MAX_HOLD_HOURS = 4
PAIR_GAP_SELL_THRESHOLD_V2 = 8.80
PAIR_SELL_FIRST_PCT = 0.95
STOP_LOSS_BULLISH_PCT = -14.00
STOP_LOSS_PCT = -5.25
TAKE_PROFIT_PCT = 4.00
V3_COIN_TRIGGER_PCT = 5.50
V3_CONL_TRIGGER_PCT = 6.00
V3_DCA_MAX_COUNT = 9
V3_MAX_PER_STOCK_KRW = 6_000_000
V3_PAIR_GAP_ENTRY_THRESHOLD = 7.60
V3_SIDEWAYS_GLD_THRESHOLD = 0.30
V3_SIDEWAYS_INDEX_THRESHOLD = 0.90
V3_SIDEWAYS_MIN_SIGNALS = 2
V3_SIDEWAYS_POLY_HIGH = 0.50
V3_SIDEWAYS_POLY_LOW = 0.35
V3_SPLIT_BUY_INTERVAL_MIN = 25
```

### #2 — Trial 302 (+13.42%)

```
COIN_SELL_PROFIT_PCT = 5.00
CONL_SELL_PROFIT_PCT = 3.50
DCA_DROP_PCT = -0.95
MAX_HOLD_HOURS = 4
PAIR_GAP_SELL_THRESHOLD_V2 = 9.00
PAIR_SELL_FIRST_PCT = 0.95
STOP_LOSS_BULLISH_PCT = -14.00
STOP_LOSS_PCT = -5.25
TAKE_PROFIT_PCT = 4.00
V3_COIN_TRIGGER_PCT = 5.50
V3_CONL_TRIGGER_PCT = 6.00
V3_DCA_MAX_COUNT = 9
V3_MAX_PER_STOCK_KRW = 6_000_000
V3_PAIR_GAP_ENTRY_THRESHOLD = 7.60
V3_SIDEWAYS_GLD_THRESHOLD = 0.30
V3_SIDEWAYS_INDEX_THRESHOLD = 0.90
V3_SIDEWAYS_MIN_SIGNALS = 2
V3_SIDEWAYS_POLY_HIGH = 0.50
V3_SIDEWAYS_POLY_LOW = 0.35
V3_SPLIT_BUY_INTERVAL_MIN = 25
```

### #3 — Trial 308 (+13.42%)

```
COIN_SELL_PROFIT_PCT = 5.00
CONL_SELL_PROFIT_PCT = 3.50
DCA_DROP_PCT = -0.95
MAX_HOLD_HOURS = 4
PAIR_GAP_SELL_THRESHOLD_V2 = 9.00
PAIR_SELL_FIRST_PCT = 0.95
STOP_LOSS_BULLISH_PCT = -13.50
STOP_LOSS_PCT = -5.25
TAKE_PROFIT_PCT = 4.00
V3_COIN_TRIGGER_PCT = 5.50
V3_CONL_TRIGGER_PCT = 6.00
V3_DCA_MAX_COUNT = 9
V3_MAX_PER_STOCK_KRW = 6_000_000
V3_PAIR_GAP_ENTRY_THRESHOLD = 7.60
V3_SIDEWAYS_GLD_THRESHOLD = 0.30
V3_SIDEWAYS_INDEX_THRESHOLD = 0.90
V3_SIDEWAYS_MIN_SIGNALS = 2
V3_SIDEWAYS_POLY_HIGH = 0.50
V3_SIDEWAYS_POLY_LOW = 0.35
V3_SPLIT_BUY_INTERVAL_MIN = 25
```

### #4 — Trial 270 (+13.39%)

```
COIN_SELL_PROFIT_PCT = 5.00
CONL_SELL_PROFIT_PCT = 3.50
DCA_DROP_PCT = -0.95
MAX_HOLD_HOURS = 4
PAIR_GAP_SELL_THRESHOLD_V2 = 8.60
PAIR_SELL_FIRST_PCT = 0.85
STOP_LOSS_BULLISH_PCT = -13.00
STOP_LOSS_PCT = -5.25
TAKE_PROFIT_PCT = 4.00
V3_COIN_TRIGGER_PCT = 5.50
V3_CONL_TRIGGER_PCT = 6.00
V3_DCA_MAX_COUNT = 9
V3_MAX_PER_STOCK_KRW = 6_000_000
V3_PAIR_GAP_ENTRY_THRESHOLD = 7.40
V3_SIDEWAYS_GLD_THRESHOLD = 0.30
V3_SIDEWAYS_INDEX_THRESHOLD = 0.90
V3_SIDEWAYS_MIN_SIGNALS = 2
V3_SIDEWAYS_POLY_HIGH = 0.50
V3_SIDEWAYS_POLY_LOW = 0.35
V3_SPLIT_BUY_INTERVAL_MIN = 25
```

### #5 — Trial 271 (+13.39%)

```
COIN_SELL_PROFIT_PCT = 5.00
CONL_SELL_PROFIT_PCT = 3.50
DCA_DROP_PCT = -0.95
MAX_HOLD_HOURS = 4
PAIR_GAP_SELL_THRESHOLD_V2 = 8.60
PAIR_SELL_FIRST_PCT = 0.85
STOP_LOSS_BULLISH_PCT = -13.00
STOP_LOSS_PCT = -5.25
TAKE_PROFIT_PCT = 4.00
V3_COIN_TRIGGER_PCT = 5.50
V3_CONL_TRIGGER_PCT = 6.00
V3_DCA_MAX_COUNT = 9
V3_MAX_PER_STOCK_KRW = 6_000_000
V3_PAIR_GAP_ENTRY_THRESHOLD = 7.40
V3_SIDEWAYS_GLD_THRESHOLD = 0.30
V3_SIDEWAYS_INDEX_THRESHOLD = 0.90
V3_SIDEWAYS_MIN_SIGNALS = 2
V3_SIDEWAYS_POLY_HIGH = 0.50
V3_SIDEWAYS_POLY_LOW = 0.35
V3_SPLIT_BUY_INTERVAL_MIN = 25
```

## 7. config.py 적용 코드 (Best Trial)

```python
COIN_SELL_PROFIT_PCT = 5.00
CONL_SELL_PROFIT_PCT = 3.50
DCA_DROP_PCT = -0.95
MAX_HOLD_HOURS = 4
PAIR_GAP_SELL_THRESHOLD_V2 = 8.80
PAIR_SELL_FIRST_PCT = 0.95
STOP_LOSS_BULLISH_PCT = -14.00
STOP_LOSS_PCT = -5.25
TAKE_PROFIT_PCT = 4.00
V3_COIN_TRIGGER_PCT = 5.50
V3_CONL_TRIGGER_PCT = 6.00
V3_DCA_MAX_COUNT = 9
V3_MAX_PER_STOCK_KRW = 6_000_000
V3_PAIR_GAP_ENTRY_THRESHOLD = 7.60
V3_SIDEWAYS_GLD_THRESHOLD = 0.30
V3_SIDEWAYS_INDEX_THRESHOLD = 0.90
V3_SIDEWAYS_MIN_SIGNALS = 2
V3_SIDEWAYS_POLY_HIGH = 0.50
V3_SIDEWAYS_POLY_LOW = 0.35
V3_SPLIT_BUY_INTERVAL_MIN = 25
```

---

## 8. 다음 단계 (Out-of-Sample 검증)

### 권장 검증 절차

**1단계: Forward Test (2026-02-18 ~ 현재)**
```bash
# 신규 데이터 수집 후 최적 파라미터로 백테스트
python backtest_v3.py --start-date 2026-02-18 --params trial_301
```

**2단계: Walk-Forward Optimization**
- Train: 2025-01-03 ~ 2025-12-31 (1년)
- Test: 2026-01-01 ~ 2026-02-17 (1.5개월)
- 비교: In-Sample 최적화 vs Out-of-Sample 성과

**3단계: 민감도 분석**
- 파라미터 ±10% 변동 시 수익률 변화 측정
- 강건한 파라미터 범위 식별

**4단계: 실전 Paper Trading**
- 최적 파라미터로 1개월 모의 거래
- 슬리피지, 체결 불가 등 실전 요소 반영

### 위험 관리 체크리스트

- [ ] Out-of-Sample 테스트 수익률 > Baseline
- [ ] MDD < -5% 유지 확인
- [ ] Sharpe > 1.5 유지 확인
- [ ] 파라미터 민감도 분석 완료
- [ ] 1개월 Paper Trading 성공
- [ ] 최대 손실액 한도 설정 (예: -10%)

---

## 9. 요약

### 핵심 성과
- **Baseline**: -7.38% → **Best**: +13.42% (+20.79%p 개선)
- **리스크 대폭 감소**: MDD -10.13% → -2.17%
- **우수한 위험조정수익**: Sharpe -0.92 → 2.30

### 전략 핵심
1. **보수적 진입**: GAP 7.6% 이상에서만 매수
2. **공격적 DCA**: 최대 9회 분할 매수
3. **넉넉한 손절선**: -5.25% (일반), -14% (강세장)
4. **높은 익절 목표**: COIN 5%, CONL 3.5%
5. **적극적 횡보장 회피**: 지수 임계값 0.9

### ⚠️ 주의사항
본 최적화는 **In-Sample** 데이터로만 수행되었습니다.
**실전 적용 전 반드시 Out-of-Sample 검증이 필요합니다.**

# Optuna 하이퍼파라미터 최적화 — 베스트 프랙티스 가이드

> PTJ 프로젝트 실전 경험 기반

---

## 목차

1. [범위 설정 원칙](#1-범위-설정-원칙)
2. [파라미터 타입별 권장 범위](#2-파라미터-타입별-권장-범위)
3. [Trials 수 결정](#3-trials-수-결정)
4. [과적합 방지 전략](#4-과적합-방지-전략)
5. [반복 최적화 전략](#5-반복-최적화-전략)
6. [실전 체크리스트](#6-실전-체크리스트)

---

## 1. 범위 설정 원칙

### 1-1. 황금률 (Golden Rules)

#### Rule #1: 현재 기본값 기준 ±50~200% 확장
```python
# ✓ GOOD: 기본값 5.0%
suggest_float("param", 2.5, 10.0, step=0.5)  # 50%~200%

# ✗ BAD: 너무 좁음
suggest_float("param", 4.5, 5.5, step=0.1)   # 90%~110%
```

**이유**: 너무 좁으면 글로벌 최적해를 놓칠 수 있음

---

#### Rule #2: 최적값이 범위 끝이면 확장
```python
# Round 1 결과: 최적값 8.0% (상한 10%)
# → Round 2에서 확장 필요

# Round 2
suggest_float("param", 0.5, 15.0, step=0.5)  # ✓ 확장
```

**시각화**:
```
Round 1:  [────────────────────8.0◄]  ← 상한에 붙음
                                10.0
Round 2:  [────────6.0──────────────]  ← 중간으로 이동
          0.5              15.0
```

---

#### Rule #3: Step은 의미 있는 최소 단위로

```python
# Percentage 파라미터
suggest_float("TRIGGER_PCT", 1.0, 12.0, step=0.5)  # ✓ 0.5%p 단위

# Count 파라미터
suggest_int("DCA_COUNT", 1, 15)  # ✓ 정수만 의미 있음

# ✗ BAD: 너무 세밀
suggest_float("TRIGGER_PCT", 1.0, 12.0, step=0.01)  # 노이즈
```

**기준**:
- % 파라미터: 0.5 step (0.1은 너무 세밀, 1.0은 너무 성김)
- Count 파라미터: 정수 (no step)
- Price 파라미터: 유효숫자 2자리

---

#### Rule #4: 도메인 지식 반영 (Domain Constraints)

```python
# ✓ GOOD: 손절은 반드시 음수
suggest_float("STOP_LOSS_PCT", -15.0, -1.0, step=0.5)

# ✗ BAD: 손절이 양수 가능 (말이 안 됨)
suggest_float("STOP_LOSS_PCT", -10.0, 10.0, step=1.0)

# ✓ GOOD: 보유 시간은 장중 시간 고려
suggest_int("HOLD_HOURS", 2, 10)  # 6.5시간/일 × 1.5일

# ✗ BAD: 100시간 (2주)는 너무 김
suggest_int("HOLD_HOURS", 1, 100)
```

---

### 1-2. 범위 확장 의사결정 트리

```
최적값이 어디에 위치하는가?
│
├─ 범위 끝 (상한/하한 10% 이내)
│   └─> 확장 필요 ✓
│
├─ 중간 (40~60%)
│   └─> 현재 범위 유지 ✓
│
└─ 한쪽으로 치우침 (10~40% or 60~90%)
    └─> 범위 재조정 검토
```

**예시**:
```python
# 시나리오 1: 최적값 9.5% (상한 10%)
# → 15.0%까지 확장

# 시나리오 2: 최적값 5.0% (중간)
# → 현재 범위(1.0~10.0%) 유지

# 시나리오 3: 최적값 2.0% (하한 근처)
# → 하한을 0.5%로 낮춤
```

---

## 2. 파라미터 타입별 권장 범위

### 2-1. 매수/매도 트리거 (Trigger %)

**목적**: 진입/청산 조건

```python
# 조건부 매매 트리거
"COIN_TRIGGER_PCT": suggest_float(
    "COIN_TRIGGER_PCT", 1.0, 12.0, step=0.5
)
"CONL_TRIGGER_PCT": suggest_float(
    "CONL_TRIGGER_PCT", 1.0, 12.0, step=0.5
)
```

**범위 근거**:
- **하한 1.0%**: 이하는 노이즈 거래 (빈도 과다)
- **상한 12.0%**: 초과는 기회 상실 (진입 불가)
- **Step 0.5**: 0.5%p 차이면 체감 가능

**실전 결과** (PTJ v2):
- COIN: 3.0% (낮게 — 빠른 진입)
- CONL: 9.0% (높게 — 보수적 진입)

---

### 2-2. 손절 (Stop Loss %)

**목적**: 리스크 관리

```python
# 일반 손절
"STOP_LOSS_PCT": suggest_float(
    "STOP_LOSS_PCT", -12.0, -1.0, step=0.5
)

# 강세장 손절 (더 여유 있게)
"STOP_LOSS_BULLISH_PCT": suggest_float(
    "STOP_LOSS_BULLISH_PCT", -15.0, -5.0, step=0.5
)
```

**범위 근거**:
- **상한 -1.0%**: 이상은 손절 의미 없음 (보호 불가)
- **하한 -12.0% (일반)**: 레버리지 2x 고려 시 적절
- **하한 -15.0% (강세)**: 강세장에서 조정 견디기 위함
- **Step 0.5**: 손절은 민감하므로 세밀하게

**실전 결과** (PTJ v2):
- 일반: -1.5% (완화)
- 강세: -8.0% (중간)

**주의**: 레버리지 배율에 따라 범위 조정
- 1x: -5.0% ~ -0.5%
- 2x: -12.0% ~ -1.0%
- 3x: -20.0% ~ -2.0%

---

### 2-3. DCA (물타기)

**목적**: 평단가 낮추기

```python
"DCA_MAX_COUNT": suggest_int(
    "DCA_MAX_COUNT", 1, 20
)
"DCA_DROP_PCT": suggest_float(
    "DCA_DROP_PCT", -3.0, -0.3, step=0.1
)
```

**범위 근거**:
- **DCA 횟수 상한 20**: 초과 시 자금 부족 위험
- **DCA 트리거 하한 -0.3%**: 이하는 너무 빈번
- **DCA 트리거 상한 -3.0%**: 초과는 DCA 기회 상실

**실전 결과** (PTJ v2):
- 횟수: 13회 (적극적)
- 트리거: -0.5% (기존 유지)

**Trade-off**:
- 많은 DCA → 평단가 ↓, 자금 소진 ↑
- 적은 DCA → 자금 여유 ↑, 회복력 ↓

---

### 2-4. 쌍둥이 매도 갭 (Pair Gap %)

**목적**: 수렴 시 매도

```python
"PAIR_GAP_SELL_THRESHOLD_V2": suggest_float(
    "PAIR_GAP_SELL_THRESHOLD_V2", 0.5, 15.0, step=0.5
)
```

**범위 근거**:
- **하한 0.5%**: 이하는 노이즈 (거래 과다)
- **상한 15.0%**: 초과는 매도 기회 상실
- **Step 0.5**: 갭은 빠르게 변하므로 적당한 간격

**실전 결과** (PTJ v2):
- 최적: 6.0% (중간)
- 기본값 0.9% 대비 큰 폭 상승 → 거래 빈도 감소

**전략별 차이**:
- 고빈도 전략: 0.5~3.0%
- 일중 전략: 3.0~8.0%
- 스윙 전략: 8.0~15.0%

---

### 2-5. 시간 손절 (Time Stop)

**목적**: 장기 보유 방지

```python
"MAX_HOLD_HOURS": suggest_int(
    "MAX_HOLD_HOURS", 2, 10
)
```

**범위 근거**:
- **하한 2시간**: 이하는 너무 짧음 (추세 불가)
- **상한 10시간**: 약 1.5일 (스윙 허용)
- **정수 단위**: 1시간 단위로 충분

**실전 결과** (PTJ v2):
- 최적: 10시간 (최대값)
- 의미: 강세장에서 충분히 보유

**시장별 차이**:
- 일중 전략: 2~6시간
- 스윙 전략: 6~24시간 (3일)
- 포지션 트레이딩: 24~72시간 (1주)

---

### 2-6. 비율/가중치 (Weight %)

**목적**: 자금 배분

```python
"INITIAL_BUY_PCT": suggest_float(
    "INITIAL_BUY_PCT", 5.0, 30.0, step=5.0
)
"BRKU_WEIGHT_PCT": suggest_int(
    "BRKU_WEIGHT_PCT", 5, 20
)
```

**범위 근거**:
- **하한**: 총 자금의 5% (너무 적으면 효과 없음)
- **상한**: 총 자금의 30% (분산 고려)
- **Step 5**: 5%p 단위면 충분

---

## 3. Trials 수 결정

### 3-1. 경험 법칙

```
Trials >= 20 × 파라미터 수
```

| 파라미터 수 | 최소 Trials | 권장 Trials | 안전 Trials |
|---|---|---|---|
| 3~5개 | 50 | 100 | 200 |
| 6~10개 | 100 | 200 | 500 |
| 11~15개 | 200 | 500 | 1,000 |
| 16~20개 | 500 | 1,000 | 2,000+ |

### 3-2. 시간 vs 정확도 Trade-off

**PTJ v2 경험치** (1 trial ≈ 3초):

| Trials | Workers | 시간 (분) | 용도 |
|---|---|---|---|
| 50 | 10 | ~2.5 | 빠른 탐색 |
| 100 | 10 | ~5 | 표준 |
| 200 | 10 | ~10 | 세밀 탐색 |
| 500 | 10 | ~25 | 최종 검증 |

**권장**:
1. 초기 탐색: 50~100 trials
2. 범위 좁혀서: 100~200 trials
3. 최종 검증: 200~500 trials

---

### 3-3. 수렴 판단

**Optuna 진행 중 모니터링**:
```
[100s] Best: +1.6% / MDD -7.9%
[200s] Best: +1.6% / MDD -7.9%  ← 변화 없음
[300s] Best: +1.6% / MDD -7.9%  ← 수렴
```

**조기 종료 기준**:
- 연속 50 trials 동안 개선 없음 → 수렴
- Pareto front가 안정화 → 충분

---

## 4. 과적합 방지 전략

### 4-1. Walk-forward Validation (필수)

```python
# ✓ GOOD: Train/Test 분리
TRAIN_START = date(2025, 2, 18)
TRAIN_END   = date(2025, 11, 14)  # 75%
TEST_START  = date(2025, 11, 15)
TEST_END    = date(2026, 1, 30)   # 25%

# Train에서만 최적화
engine = BacktestEngineV2(
    start_date=TRAIN_START,
    end_date=TRAIN_END,
)

# Test에서 검증
validate_on_test(best_params)
```

**분할 비율**:
- Train: 70~80%
- Test: 20~30%

---

### 4-2. Train/Test 갭 체크

```python
train_ret = 1.57%
test_ret = -2.53%
gap = abs(train_ret - test_ret)  # 4.10pp

# 판단 기준
if gap < 5:
    print("✓ 견고한 파라미터")
elif gap < 10:
    print("△ 수용 가능")
elif gap < 15:
    print("⚠ 과적합 위험")
else:
    print("✗ 과적합")
```

**PTJ v2 결과**:
- Round 1: 3.61pp → 견고
- Round 2: 4.10pp → 견고

---

### 4-3. Multi-objective 사용

```python
# ✓ GOOD: 수익률 + MDD 동시 최적화
def objective(trial):
    # ...
    return (train_ret, train_mdd)  # 2개 목표

study = optuna.create_study(
    directions=["maximize", "minimize"]  # 수익률↑, MDD↓
)
```

**효과**:
- 단일 목표 → 극단적 파라미터 (과적합 위험)
- 다중 목표 → 균형 잡힌 해 (Pareto front)

---

### 4-4. 정규화 (Regularization)

```python
# 파라미터 복잡도 패널티
complexity = sum([
    abs(param - default) / default
    for param, default in zip(params, defaults)
])

# 목표 함수에 반영
return train_ret - 0.1 * complexity
```

**적용 시나리오**:
- 파라미터가 기본값에서 크게 벗어남 → 패널티
- 단순한 전략 선호

---

## 5. 반복 최적화 전략

### 5-1. 3단계 프로세스

#### Phase 1: 넓은 탐색 (Exploration)

```python
# 목적: 대략적인 최적 영역 파악
# Trials: 50~100
# 범위: 기본값 ±100~200%

define_search_space_v1 = {
    "TRIGGER": suggest_float(1.0, 15.0, step=1.0),  # 넓게
    "STOP_LOSS": suggest_float(-15.0, -1.0, step=1.0),
    # ...
}
```

**결과 분석**:
- 최적값 위치 확인
- 범위 끝에 몰리면 확장 필요

---

#### Phase 2: 세밀한 탐색 (Exploitation)

```python
# 목적: Phase 1 최적값 주변 정밀 탐색
# Trials: 100~200
# 범위: Phase 1 최적값 ±30~50%

# Phase 1 최적값이 12.0%였다면
define_search_space_v2 = {
    "TRIGGER": suggest_float(8.0, 16.0, step=0.5),  # 좁게, 세밀하게
    # ...
}
```

---

#### Phase 3: 검증 (Validation)

```python
# 목적: Out-of-sample 검증
# Test 기간에서 성능 확인
# Train/Test 갭 < 10pp 확인
```

---

### 5-2. 실전 예시 (PTJ v2)

```
Round 1 (넓은 탐색):
├─ 파라미터: 5개
├─ PAIR_GAP: 3.0~10.0%
├─ TRIGGER: 2.0~8.0%
└─ 결과: PAIR_GAP 8.0%, TRIGGER 7.0% → 상한 근처

Round 2 (확장 + 파라미터 추가):
├─ 파라미터: 7개 (BULLISH_STOP, HOLD_HOURS 추가)
├─ PAIR_GAP: 0.5~15.0% (확장)
├─ TRIGGER: 1.0~12.0% (확장)
└─ 결과: PAIR_GAP 6.0%, COIN 3.0%, CONL 9.0% → 수렴
```

---

## 6. 실전 체크리스트

### 6-1. 최적화 시작 전

- [ ] 데이터 품질 확인 (누락/이상치 없음)
- [ ] Train/Test 분리 (70~80% / 20~30%)
- [ ] 기본값 성능 측정 (baseline)
- [ ] 파라미터 수 < 10개 (처음에는)
- [ ] 도메인 제약 정의 (손절 < 0 등)

---

### 6-2. 범위 설정 시

- [ ] 기본값 기준 ±50~200% 확장
- [ ] Step은 의미 있는 최소 단위
- [ ] 도메인 제약 반영 (물리적 한계)
- [ ] 이전 최적값이 범위 끝이면 확장
- [ ] 무의미한 범위 제외 (손절 양수 등)

---

### 6-3. Trials 수 결정 시

- [ ] `Trials >= 20 × 파라미터 수` 만족
- [ ] 시간 여유 고려 (1 trial ≈ 3초 기준)
- [ ] 초기는 50~100, 정밀은 200+
- [ ] Workers = CPU 코어 수 고려

---

### 6-4. 결과 검증 시

- [ ] Train/Test 갭 < 10pp
- [ ] Test 성능 > baseline
- [ ] 최적값이 범위 중간에 위치
- [ ] Pareto front에 2개+ 해 존재
- [ ] Sharpe, MDD 등 다중 지표 확인

---

### 6-5. 과적합 방지

- [ ] Walk-forward validation 사용
- [ ] Multi-objective 최적화
- [ ] Train/Test 갭 모니터링
- [ ] 파라미터 복잡도 고려
- [ ] Out-of-sample 검증

---

## 7. 실패 사례 & 해결책

### 7-1. 과적합 (Overfitting)

**증상**:
```
Train: +15.0%
Test:  -5.0%
Gap:   20pp  ← ✗
```

**원인**:
- Train 기간만 최적화
- 파라미터 너무 많음
- 범위 너무 좁음

**해결**:
- Walk-forward validation 적용
- 파라미터 수 줄이기 (< 10개)
- 정규화 추가

---

### 7-2. 수렴 실패 (No Convergence)

**증상**:
```
Trial  1: -10.0%
Trial 50: -9.5%
Trial 100: -9.3%  ← 개선 미미
```

**원인**:
- Trials 수 부족
- 범위 너무 넓음
- 파라미터 너무 많음

**해결**:
- Trials 2배 증가
- Phase 1/2로 나눠서 실행
- 파라미터 줄이기

---

### 7-3. 극단값 선택

**증상**:
```
STOP_LOSS: -1.0%  ← 상한
DCA_COUNT: 20     ← 상한
```

**원인**:
- 범위 설정 오류
- 도메인 제약 미반영

**해결**:
- 범위 재검토
- 도메인 지식 반영
- 제약 조건 추가

---

## 8. 추천 워크플로우

```
1. 기본값 백테스트
   └─> Baseline 성능 측정

2. Phase 1: 넓은 탐색 (50~100 trials)
   ├─ 파라미터: 5~7개
   ├─ 범위: 기본값 ±100%
   └─> 대략적 최적 영역 파악

3. 결과 분석
   ├─ 최적값이 범위 끝? → 확장
   ├─ 추가 파라미터 필요? → 추가
   └─> Phase 2 설계

4. Phase 2: 세밀한 탐색 (100~200 trials)
   ├─ 범위: Phase 1 최적값 ±50%
   ├─ Step 축소
   └─> 정밀 수렴

5. 검증
   ├─ Train/Test 갭 < 10pp?
   ├─ 다중 지표 확인
   └─> 채택 or 재탐색

6. 적용
   └─> config.py 업데이트
```

---

## 9. 참고 자료

### Optuna 공식 문서
- https://optuna.readthedocs.io/
- Multi-objective: https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/005_multi_objective.html

### PTJ 프로젝트 실전 결과
- `docs/optuna_optimization_report.md`
- Round 1: 5개 파라미터, -7.90%
- Round 2: 7개 파라미터, **+1.57%** (Train)

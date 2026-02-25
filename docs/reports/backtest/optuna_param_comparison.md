# Optuna 파라미터 비교 리포트

**생성일**: 2026-02-23
**분석 대상**: `data/optuna/` 내 전체 DB (8개)
**분석 스크립트**: `experiments/optuna_analysis.py`

---

## 1. 요약 테이블

| DB 파일 | Study | 버전 | 전체 Trial | 완료 | Best Score |
|---------|-------|------|-----------|------|-----------|
| optuna_v2.db | ptj_v2_expanded | v2 | 100 | 100 | +1.57% ret (multi-obj) |
| optuna_v3.db | ptj_v3_full | v3 | 339 | 329 | +12.80 |
| optuna_v3_phase2.db | ptj_v3_phase2 | v3 | 310 | 310 | +13.42 |
| optuna_v3_train_test.db | ptj_v3_train_test | v3 | 190 | 130 | +14.03 |
| optuna_v3_train_test_v2.db | ptj_v3_train_test_v2 | v3 | 104 | 94 | +11.94 |
| optuna_v3_train_test_wide.db | ptj_v3_train_test_wide | v3 | 509 | 500 | +19.08 |
| optuna_v4_phase1.db | ptj_v4_phase1 | v4 | 400 | 400 | **+30.43** |
| crash_model_params.db | crash_model_v1 | misc | 300 | 300 | 0.7677 (AUC) |

> **참고**: v2는 Multi-objective (NSGA-II) — Best Score는 Return 기준. v3/v4는 단일 목적함수 (수익률 기반 score). v5 DB 없음.

---

## 2. v2 최적화 결과 (ptj_v2_expanded)

**최적화 방식**: Multi-objective (수익률 maximize + MDD minimize), NSGA-II
**Best Trial**: #11 | Return = +1.57% | MDD = -7.89%
**Pareto front**: 1개 해

### 파라미터 비교

| 파라미터 | Optuna Best | Current config | Δ | 비고 |
|---------|------------|---------------|---|------|
| COIN_TRIGGER_PCT | 3.0 | 3.0 | 0 | 일치 |
| CONL_TRIGGER_PCT | 9.0 | 3.0 | +6.0 | **큰 차이** |
| STOP_LOSS_PCT | -1.5 | -3.0 | +1.5 | 완화 |
| STOP_LOSS_BULLISH_PCT | -8.0 | -8.0 | 0 | 일치 |
| DCA_MAX_COUNT | 13 | 7 | +6 | **큰 차이** |
| PAIR_GAP_SELL_THRESHOLD_V2 | 6.0 | 0.9 | +5.1 | **큰 차이** |
| MAX_HOLD_HOURS | 10 | 5 | +5 | **큰 차이** |

> v2 최적화는 전체 기간 수익률이 낮음(+1.57%) — 이후 v3/v4로 구조 개선됨.

---

## 3. v3 최적화 결과

### 3-1. v3 Phase 2 (ptj_v3_phase2) — 권장 v3 최적값

**Best Trial**: #301 | Score = +13.42
**의미**: v3의 가장 안정적인 최적화 결과 (협소 탐색, 310 trial)

| 파라미터 | Optuna Best | Current config | Δ | 비고 |
|---------|------------|---------------|---|------|
| V3_PAIR_GAP_ENTRY_THRESHOLD | 7.6 | 2.2 | +5.4 | **매우 큰 차이** |
| V3_DCA_MAX_COUNT | 9 | 4 | +5 | **큰 차이** |
| V3_COIN_TRIGGER_PCT | 5.5 | 4.5 | +1.0 | |
| V3_CONL_TRIGGER_PCT | 6.0 | 4.5 | +1.5 | |
| V3_SPLIT_BUY_INTERVAL_MIN | 25 | 20 | +5 | |
| STOP_LOSS_PCT | -5.25 | -3.0 | -2.25 | **더 엄격** |
| STOP_LOSS_BULLISH_PCT | -14.0 | -8.0 | -6.0 | **더 엄격** |
| COIN_SELL_PROFIT_PCT | 5.0 | 3.0 | +2.0 | **큰 차이** |
| CONL_SELL_PROFIT_PCT | 3.5 | 2.8 | +0.7 | |
| DCA_DROP_PCT | -0.95 | -0.5 | -0.45 | |
| TAKE_PROFIT_PCT | 4.0 | 2.0 | +2.0 | **큰 차이** |
| PAIR_GAP_SELL_THRESHOLD_V2 | 8.8 | 0.9 | +7.9 | **매우 큰 차이** |
| PAIR_SELL_FIRST_PCT | 0.95 | 0.8 | +0.15 | |
| V3_SIDEWAYS_MIN_SIGNALS | 2 | 3 | -1 | 완화 |
| V3_SIDEWAYS_INDEX_THRESHOLD | 0.9 | 0.5 | +0.4 | |

### 3-2. v3 Wide (ptj_v3_train_test_wide) — 최고 스코어 v3

**Best Trial**: #392 | Score = +19.08 (v3 최고)
**특징**: 넓은 탐색 범위 (wide search), CONL_TRIGGER 10% 등 극단값 포함

| 파라미터 | v3 Wide Best | v3 Phase2 Best | 비교 |
|---------|------------|---------------|------|
| V3_PAIR_GAP_ENTRY_THRESHOLD | 5.0 | 7.6 | v3p2가 더 보수적 |
| V3_CONL_TRIGGER_PCT | 10.0 | 6.0 | v3w가 극단적 |
| STOP_LOSS_PCT | -10.0 | -5.25 | v3w가 훨씬 완화 |
| V3_SPLIT_BUY_INTERVAL_MIN | 5 | 25 | v3w가 공격적 |

> v3 Wide는 과적합 가능성 있음 (STOP_LOSS -10% 등 극단값). v3 Phase2 결과가 더 안정적.

### 3-3. v3 파라미터 중요도 (fANOVA 기준)

**ptj_v3_full** Top-5:
| 파라미터 | 중요도 |
|---------|-------|
| COIN_SELL_PROFIT_PCT | 0.2032 |
| V3_CONL_TRIGGER_PCT | 0.1317 |
| V3_PAIR_GAP_ENTRY_THRESHOLD | 0.0982 |
| V3_SIDEWAYS_INDEX_THRESHOLD | 0.0969 |
| V3_COIN_TRIGGER_PCT | 0.0909 |

**ptj_v3_train_test_wide** Top-5:
| 파라미터 | 중요도 |
|---------|-------|
| V3_CONL_TRIGGER_PCT | **0.4751** |
| V3_SIDEWAYS_MIN_SIGNALS | 0.1848 |
| MAX_HOLD_HOURS | 0.0548 |
| STOP_LOSS_BULLISH_PCT | 0.0328 |
| V3_COIN_TRIGGER_PCT | 0.0275 |

> **핵심**: CONL_TRIGGER_PCT와 COIN_SELL_PROFIT_PCT가 v3 성과를 가장 많이 결정함.

---

## 4. v4 최적화 결과 (ptj_v4_phase1)

**Best Trial**: #388 | Score = +30.43 (전체 최고)
**최적화 방식**: Single-objective (balanced mode), 400 trials, TPE

### 핵심 파라미터 비교 (주요 차이 항목)

| 파라미터 | Optuna Best | Current config | Δ | 비고 |
|---------|------------|---------------|---|------|
| STOP_LOSS_BULLISH_PCT | -16.0 | -8.0 | -8.0 | **매우 큰 차이** |
| PAIR_GAP_SELL_THRESHOLD_V2 | 6.6 | 0.9 | +5.7 | **매우 큰 차이** |
| V4_PAIR_GAP_ENTRY_THRESHOLD | 2.0 | 2.2 | -0.2 | 거의 일치 |
| V4_CB_VIX_SPIKE_PCT | 3.0 | 6.0 | -3.0 | **더 민감** |
| V4_CB_VIX_COOLDOWN_DAYS | 13 | 7 | +6 | **더 긴 쿨다운** |
| V4_CB_BTC_CRASH_PCT | -3.5 | -5.0 | +1.5 | 더 민감 |
| V4_CB_BTC_SURGE_PCT | 8.5 | 5.0 | +3.5 | 덜 민감 |
| V4_CB_GLD_COOLDOWN_DAYS | 1 | 3 | -2 | **더 짧은 쿨다운** |
| V4_DCA_MAX_COUNT | 1 | 4 | -3 | 물타기 거의 안 함 |
| V4_COIN_TRIGGER_PCT | 3.0 | 4.5 | -1.5 | 더 공격적 진입 |
| V4_CONL_TRIGGER_PCT | 3.0 | 4.5 | -1.5 | 더 공격적 진입 |
| V4_CONL_ADX_MIN | 10.0 | 18.0 | -8.0 | **큰 차이** — 조건 완화 |
| V4_PAIR_FIXED_TP_PCT | 7.5 | 5.0 | +2.5 | 더 높은 익절 목표 |
| V4_PAIR_IMMEDIATE_SELL_PCT | 0.2 | 0.4 | -0.2 | 즉시 매도 비율 축소 |
| V4_SIDEWAYS_ATR_DECLINE_PCT | 10.0 | 20.0 | -10.0 | **큰 차이** |
| V4_SIDEWAYS_VOLUME_DECLINE_PCT | 15.0 | 30.0 | -15.0 | **큰 차이** |
| V4_SIDEWAYS_RSI_HIGH | 65.0 | 55.0 | +10.0 | **큰 차이** |
| V4_SIDEWAYS_RSI_LOW | 35.0 | 45.0 | -10.0 | **큰 차이** |
| V4_SPLIT_BUY_INTERVAL_MIN | 10 | 20 | -10 | 더 빠른 분할 진입 |
| DCA_DROP_PCT | -1.35 | -0.5 | -0.85 | **큰 차이** |
| CONL_SELL_PROFIT_PCT | 4.5 | 2.8 | +1.7 | |
| STOP_LOSS_PCT | -4.25 | -3.0 | -1.25 | 더 엄격 |

### v4 파라미터 중요도 (fANOVA)

| 파라미터 | 중요도 |
|---------|-------|
| V4_PAIR_GAP_ENTRY_THRESHOLD | **0.2911** |
| V4_SIDEWAYS_POLY_HIGH | 0.1447 |
| V4_CB_BTC_SURGE_PCT | 0.0971 |
| V4_MAX_PER_STOCK | 0.0433 |
| DCA_DROP_PCT | 0.0378 |

> **핵심**: v4에서는 `V4_PAIR_GAP_ENTRY_THRESHOLD`가 압도적으로 중요 (29%). 횡보장 판단 기준(`V4_SIDEWAYS_POLY_HIGH`)도 14%로 2위.

---

## 5. Crash Model 분석 (crash_model_v1)

**Best Trial**: #231 | AUC Score = 0.7677
**목적**: 급락 감지 모델 가중치 최적화

| 파라미터 | Best 가중치 | 중요도 | 의미 |
|---------|-----------|-------|------|
| soxl_cutoff | 0.322 | **0.5699** | SOXL 임계값 — 가장 중요 |
| w_btc | 0.320 | 0.1588 | BTC 방향 가중치 |
| w_btc_ret | 0.260 | — | BTC 수익률 가중치 |
| w_mom | 0.145 | — | 모멘텀 가중치 |
| w_vix | 0.050 | 0.1821 | VIX 가중치 |
| w_weekly | 0.058 | — | 주봉 가중치 |
| w_spike | 0.059 | 0.0370 | 스파이크 가중치 |
| mstz_start | 0.590 | 0.0212 | MSTZ 시작 임계값 |

---

## 6. 버전별 최적화 진화 요약

| 항목 | v2 (100T) | v3 Phase2 (310T) | v3 Wide (500T) | v4 Phase1 (400T) |
|-----|---------|----------------|--------------|----------------|
| Best Score | +1.57% ret | +13.42 | +19.08 | **+30.43** |
| PAIR_GAP_SELL | 6.0 | 8.8 | 1.5 | 6.6 |
| STOP_LOSS | -1.5 | -5.25 | -10.0 | -4.25 |
| STOP_LOSS_BULLISH | -8.0 | -14.0 | -15.0 | **-16.0** |
| COIN/CONL Trigger | 3/9% | 5.5/6% | 6/10% | 3/3% |
| DCA_MAX_COUNT | 13 | 9 | 8 | **1** |

> v4에서 DCA 거의 사용하지 않음(1회) — 단번 진입 + 높은 익절 전략으로 전환.

---

## 7. 현재 config vs Optuna Best 주요 divergence

### 공통 패턴 (모든 버전에서 일관):

| 파라미터 | 현재 config | Optuna 방향 | 해석 |
|---------|------------|-----------|------|
| PAIR_GAP_SELL_THRESHOLD_V2 | 0.9% | 6~9% | **현재 너무 낮음** — 갭 수렴 매도 기준 대폭 상향 필요 |
| COIN_SELL_PROFIT_PCT | 3.0% | 4~5% | 수익 실현 목표 상향 필요 |
| TAKE_PROFIT_PCT | 2.0% | 3.5~5.5% | 강세장 즉시 매도 기준 상향 필요 |
| STOP_LOSS_BULLISH_PCT | -8.0% | -12~-16% | 강세장 손절 너무 tight |
| V3_PAIR_GAP_ENTRY_THRESHOLD | 2.2% | 5~8% | 페어 진입 기준 너무 낮음 |

### v4 전용 주요 divergence:

| 파라미터 | 현재 | Best | 해석 |
|---------|------|------|------|
| V4_CB_VIX_SPIKE_PCT | 6.0% | 3.0% | VIX CB 임계값 낮춰야 (더 민감) |
| V4_CB_VIX_COOLDOWN_DAYS | 7일 | 13일 | 쿨다운 연장 필요 |
| V4_CONL_ADX_MIN | 18.0 | 10.0 | CONL 조건 완화 필요 |
| V4_SIDEWAYS_RSI_HIGH/LOW | 55/45 | 65/35 | 횡보장 RSI 범위 확대 필요 |
| V4_SIDEWAYS_ATR_DECLINE_PCT | 20% | 10% | ATR 감소 기준 낮춰야 |
| DCA_DROP_PCT | -0.5% | -1.35% | 물타기 트리거 더 낮게 |

---

## 8. v5 최적화 상태 및 권장사항

**현재 상태**: v5 Optuna DB 없음 — **최적화 미실행**

v5는 v4 구조를 그대로 상속하며 IAU/GDXU Unix 방어모드가 추가됨.
현재 v5 파라미터는 v4 초기값(= v3 최적값 구조)을 그대로 사용 중.

**권장 우선순위**:

1. **즉시 반영 가능** (v3 Phase2 검증 완료):
   - `PAIR_GAP_SELL_THRESHOLD_V2`: 0.9 → 8.8
   - `COIN_SELL_PROFIT_PCT`: 3.0 → 5.0
   - `TAKE_PROFIT_PCT`: 2.0 → 4.0

2. **v4 결과 검토 후 반영** (v5에 포팅):
   - `V5_CB_VIX_SPIKE_PCT`: 6.0 → 3.0
   - `V5_CB_VIX_COOLDOWN_DAYS`: 7 → 13
   - `V5_CONL_ADX_MIN`: 18 → 10
   - `STOP_LOSS_BULLISH_PCT`: -8.0 → -16.0

3. **v5 전용 최적화 실행**:
   ```bash
   pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_v5_optuna.py \
     --stage 2 --n-trials 400 --n-jobs 10 \
     --study-name ptj_v5_phase1 \
     --db sqlite:///data/optuna/optuna_v5_phase1.db
   ```

---

## 9. 주요 관찰 사항

1. **PAIR_GAP_SELL_THRESHOLD_V2 (현재 0.9%) 이상**: 모든 버전에서 최적값이 5~9%. 현재 config가 지나치게 낮아 조기 매도가 빈번할 가능성 있음.

2. **v4 DCA_MAX_COUNT=1**: 물타기를 거의 하지 않는 전략이 v4에서 최고 성과. 분할매수 전략 재검토 필요.

3. **강세장 손절 (STOP_LOSS_BULLISH_PCT) 일관된 완화**: v2(-8%) → v3(-14%) → v4(-16%)로 점점 완화됨. 강세장에서의 일시 하락을 견디는 방향.

4. **V4_PAIR_GAP_ENTRY_THRESHOLD ≈ 2.0**: v4에서 최적값(2.0)이 현재 config(2.2)와 거의 일치 — 페어 진입 기준은 적절.

5. **V3_CONL_TRIGGER_PCT 가장 중요한 파라미터**: fANOVA에서 v3 Wide의 47%, v3 full에서 13% — CONL 진입 타이밍이 핵심.

6. **crash_model soxl_cutoff 중요도 57%**: 급락 모델에서 SOXL 임계값이 압도적으로 중요. 현재 적용 여부 확인 필요.

---

*생성: experiments/optuna_analysis.py*

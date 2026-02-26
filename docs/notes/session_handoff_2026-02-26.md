# 세션 핸드오프 노트 — 2026-02-26

> 작성: 2026-02-26 (세션 종료 시점)
> 목적: 다음 세션에서 현황 파악 없이 바로 이어받기 위한 요약

---

## 현재 실행 중인 Job

**없음.** Jobs 100(v4_study6), 101(optimizer_v5) 모두 TIMEOUT → 결과 로컬 수집 완료.

---

## 1. D2S v3 Walk-Forward 결과 (line_c)

파일: `data/results/optimization/d2s_v3_wf_summary.json`

| 창 | IS 기간 | IS 수익 | IS Sharpe | OOS 수익 | OOS MDD |
|----|---------|--------|-----------|---------|---------|
| W1 | 2025-03~06 | +25.82% | 4.885 | +0.25% | -7.87% |
| W2 | 2025-03~09 | +65.71% | 4.110 | -3.00% | -10.36% |
| W3 | 2025-03~12 | +76.26% | 2.304 | **-20.28%** | **-32.05%** |
| recent_is_r2 | 2025-10~2026-01 | +53.05% | 2.651 | **+2.40%** | -5.35% |
| recent_is_r3 | 2025-10~2026-01 | +87.17% | 1.840 | -0.04% | **-3.80%** |

**핵심 관찰:**
- W3 OOS 참패(-32% MDD) 원인: 2026 급락장
- recent_is_r2가 OOS 최우수 (+2.40%, Sharpe 1.427, MDD -5.35%)
- recent_is_r3은 더 보수적 (거래수 ↓, MDD ↓ → score페널티)

### D2S v3 r5 (R21 제거, No-ROBN)

파일: `data/results/optimization/d2s_v3_regime_r5_norobn_best_result.json`

| 지표 | 값 |
|------|-----|
| Full 수익률 | **+190.38%** |
| MDD | -20.98% |
| Sharpe | 1.844 |
| Best trial | #268 |
| hold_days_max | 10 (bull=bear 단일) |
| bull_tp | 4.5% |
| bear_tp | 7.0% |
| IS 수익 | +53.1% |
| OOS 수익 | +85.75% |

**의미:** R21(hold_days 조건부) 제거 → +1.82%p 개선 확인. E variant(TP=6%, HD=10d) warm-start가 최적으로 수렴.

---

## 2. v4 Study 6 결과

파일: `data/optuna/optuna_v4_study6.log`

| 지표 | 값 |
|------|-----|
| 완료 trials | 487 / 500 |
| Best trial | #48, value=573.74 |
| 기간 | Job 100 20h TIMEOUT (422→487) |

**다음 할 일:** OOS 검증 스크립트 실행 필요 (best #48 파라미터 추출 후 OOS backtest)

v4 baseline (현재 config): +365.70%, MDD -14.30%, Sharpe 1.611

---

## 3. v5 Optimizer 결과

파일: `data/optuna/v5_opt.db`

| Study | 완료 trials | Best value | 의미 |
|-------|------------|-----------|------|
| ptj_v5_s2 | 261 | **-2.4417** ★ | 기본 확장 탐색 |
| ptj_v5_s3 | 300 | 0.1535 | 진입시간 집중 탐색 |
| ptj_v5_s4 | 300 | -0.8044 | 추가 탐색 |
| ptj_v5_s5 | 380 | -0.0115 | 횡보장 확장 |
| ptj_v5_s6 | 200 | -0.1870 | DCA 최소화 |

v5 baseline (현재 config): **-9.76%**, MDD -10.14%, Sharpe -2.26 → 현재 config 매우 나쁨

**s2 best #239 핵심 파라미터:**
- MAX_HOLD_HOURS: 3, TAKE_PROFIT_PCT: 4.5%, STOP_LOSS_PCT: -3.0%
- V5_INITIAL_BUY: (별도 확인 필요)

**다음 할 일:** s2 best 파라미터 OOS 검증 필요

---

## 4. Study 10-12 실험 결과 (look-ahead bias 수정 후)

### Study 10 — Look-ahead Bias 수정 전후

| 구간 | biased (종가체결) | corrected (시가체결+슬리피지) |
|------|-----------------|--------------------------|
| IS | +6.28%, Sharpe 1.305 | +3.29%, Sharpe 0.983 |
| OOS | +0.72%, Sharpe 0.148 | **-8.41%**, Sharpe -0.838 |

→ 수정 후 성능 하락은 look-ahead bias 제거로 인한 정상 현상

### Study 11 — 레짐 감지 방법 Ablation

| 방법 | IS | OOS Sharpe |
|-----|-----|-----------|
| no_regime | +4.07% | -0.858 |
| streak_only | +4.07% | -0.956 |
| ma_cross | +4.07% | -1.074 |
| **full_3signal (v3_current)** | +3.29% | **-0.838 ★** |

→ **v3_current weights 유지 결정** (OOS 최우수)

### Study 12 — market_score weights

| 스킴 | OOS |
|-----|-----|
| **v3_current** | -8.41% ★ OOS 최우수 |
| equal_weight | -11.91% |
| v3_no_gld | -12.16% |

→ **params_d2s.py weights 확정** (변경 불필요)

### Study D2S Entry Cap

| cap | 거래수 | 승률 | 수익률 |
|-----|-------|------|--------|
| 30% (현재) | 243 | 70.4% | +228% |
| **50%** | 280 | 72.1% | **+277%** |
| 80% / off | 280 | 72.1% | +277% (50%와 동일) |

→ **`daily_new_entry_cap` 0.30 → 0.50 으로 완화 권장** (+49%p)

---

## 5. 최근 코드 변경 사항

### 핵심 버그 수정 (커밋 8ca5944, 6df929b)

1. **Look-ahead bias 수정** (`backtest_d2s*.py`):
   - `generate_daily_signals(snap, positions, daily_buy_counts)` → `(snap, positions, {})`
   - T일 종가 신호 → T+1일 시가 체결 (정확한 시뮬레이션)

2. **RSI 버그 수정** (`optimize_d2s_v3_optuna.py`): 이미 수정됨

3. **`--study-name` 인자 미적용 버그** (`optimize_d2s_recent_is.py`): 이미 수정됨

### 새 파일

- `simulation/optimizers/optimize_d2s_v3_r5.py` — R21 제거 Optuna
- `simulation/optimizers/optimize_v5_optuna.py` — s3/s5/s6 변형 추가
- `experiments/study_10_bias_corrected_v3.py`
- `experiments/study_11_corrected_regime_ablation.py`
- `experiments/study_12_corrected_mscore_weights.py`
- `experiments/study_d2s_entry_cap.py`

---

## 6. 다음 우선순위 할 일

### 🔴 높음

1. **v4 Study6 OOS 검증**
   - journal: `data/optuna/optuna_v4_study6.log` (487 trials, best #48, value=573.74)
   - OOS 기간: 2026-01-01 ~
   - 방법: best params 추출 → backtest_v5.py 로 OOS 단독 실행

2. **v5 s2 OOS 검증**
   - DB: `data/optuna/v5_opt.db`, study: `ptj_v5_s2`, best #239, value=-2.4417
   - baseline -9.76% → best trial이 실제로 개선됐는지 확인

3. **D2S recent_is 앙상블 파라미터 결정**
   - recent_is_r2 (OOS +2.40%, Sharpe 1.427) vs r3 (MDD -3.80%) 중 선택
   - 실서버 파라미터 업데이트 검토

### 🟡 중간

4. **D2S v3 r5 파라미터 → 실서버 반영 검토**
   - Full +190.38%, Sharpe 1.844 (best 성능)
   - `hold_days_max=10`, `bull_tp=4.5%`, `bear_tp=7.0%`

5. **`daily_new_entry_cap` 0.30 → 0.50 변경 테스트**
   - Study D2S Entry Cap 결과: +277% vs +228%

6. **Line B (태준수기) 코드 작성 준비**
   - `docs/rules/line_b/` 확정 rules 작성 후 코드 작성 가능
   - 현재 `line_b_taejun/` FROZEN 상태

### 🟢 낮음

7. **v5 s3~s6 OOS 검증** (s2보다 낮은 성능이지만 확인)
8. **Study 11 full_3signal 방식 재고** (no_regime과 차이 미미, 현재 유지)

---

## 7. 주요 파일 경로

| 항목 | 경로 |
|------|------|
| D2S WF 요약 | `data/results/optimization/d2s_v3_wf_summary.json` |
| D2S r5 결과 | `data/results/optimization/d2s_v3_regime_r5_norobn_best_result.json` |
| v4 study6 journal | `data/optuna/optuna_v4_study6.log` |
| v5 SQLite DB | `data/optuna/v5_opt.db` |
| study 11 결과 | `data/results/backtests/study_11_corrected_regime_20260226.json` |
| study 12 결과 | `data/results/backtests/study_12_corrected_weights_20260226.json` |
| entry cap 결과 | `data/results/analysis/d2s_entry_cap_summary.json` |
| D2S 엔진 | `simulation/strategies/line_c_d2s/d2s_engine.py` |
| D2S v3 파라미터 | `simulation/strategies/line_c_d2s/params_d2s.py` |
| D2S v3 백테스트 | `simulation/backtests/backtest_d2s_v3.py` |
| v5 백테스트 | `simulation/backtests/backtest_v5.py` |
| v4 study6 optimizer | `simulation/optimizers/optimize_v4_study6.py` |
| v5 optimizer | `simulation/optimizers/optimize_v5_optuna.py` |

---

## 8. 코드 환경

```bash
pyenv shell ptj_stock_lab   # Python 3.11
ssh gigaflops-proxy          # 클러스터 접속
make slurm-push PROFILE=...  # 코드 전송
make slurm-submit PROFILE=... # Job 제출
make slurm-collect PROFILE=... # 결과 수집
```

**sqsh 이미지**: `/mnt/giga/project/ptj_stock_lab/slurm/images/ptj_stock_lab.sqsh` (489MB, Python 3.10-slim + 패키지)

---

## 9. 최근 커밋 히스토리

```
6df929b feat: look-ahead bias 후속 수정 + v5 optimizer 변형 + study 10-12 실험 추가
3f6ff4a docs: taejun 전략 리뷰 2026-02-26 — rule-verifier P-NEW-07~13 질문 등록
cda76dd feat: study_d2s_1min_dca — 실거래 DCA 패턴 재현 분봉 Study
8ca5944 fix: D2SBacktest look-ahead bias 수정 — T일 종가 신호 → T+1일 시가 체결
acf6f7c fix: optimize_d2s_recent_is.py --study-name 인자 미적용 버그 수정
00e5c6b feat: R20 vs R21 Ablation + Phase 3 Optuna (R21 제거) 제출
```

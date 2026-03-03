# 세션 핸드오프 노트 — 2026-03-01

> 작성: 2026-03-01 (세션 종료 시점)
> 목적: 다음 세션에서 현황 파악 없이 바로 이어받기 위한 요약

---

## 현재 실행 중인 Job

**없음.** Job 158 (study_v5_s2_oos) COMPLETED — 결과 수집 완료.

---

## 1. 이번 세션에서 완료된 작업

### Study 13 — v5 s2 OOS 검증 ✅

**SLURM Job 158** (giganode-51, 1분 53초 완료)

| 구간 | 지표 | baseline | s2_best (#309) | Δ |
|------|------|----------|----------------|---|
| IS | 수익률 | -9.82% | -1.49% | +8.33%p |
| IS | Sharpe | -2.495 | -1.276 | +1.22 |
| IS | 승률 | 35.6% | 48.4% | +12.8%p |
| **OOS** | **수익률** | +0.04% | **+0.36%** | +0.32%p |
| **OOS** | **Sharpe** | -1.423 | **+2.469** | **+3.89** |
| **OOS** | **승률** | 50% | **79.4%** | +29.4%p |
| OOS | MDD | -0.73% | -0.19% | -0.54%p |

판정: **✅ OOS 양전** — v5 파라미터 확정

- 결과 파일: `data/results/backtests/study_13_v5_s2_oos_20260227.json`
- Optuna DB: `data/optuna/v5_opt.db`, study `ptj_v5_s2`, best trial #309 (647 trials 완료)

### config.py 파라미터 반영 ✅

커밋: `f3b329e` — 25개 파라미터 반영

**v5 전용 (명시값으로 변경):**

| 파라미터 | 이전 | 이후 |
|---------|------|------|
| V5_INITIAL_BUY | 2,250 | **1,000** |
| V5_DCA_BUY | 750 | **250** |
| V5_PAIR_GAP_ENTRY_THRESHOLD | 2.2% | **4.0%** |
| V5_DCA_MAX_COUNT | 4 | **3** |
| V5_MAX_PER_STOCK | 5,250 | **9,750** |
| V5_COIN_TRIGGER_PCT | 4.5% | **6.5%** |
| V5_CONL_TRIGGER_PCT | 4.5% | **6.5%** |
| V5_SPLIT_BUY_INTERVAL_MIN | 20분 | **15분** |
| V5_ENTRY_CUTOFF_HOUR | 10 | **12** |
| V5_SIDEWAYS_MIN_SIGNALS | 3/6 | **4/6** |
| V5_SIDEWAYS_POLY_LOW | 0.40 | **0.30** |
| V5_SIDEWAYS_GLD_THRESHOLD | 0.3 | **0.1** |
| V5_SIDEWAYS_INDEX_THRESHOLD | 0.5 | **0.7** |
| V5_CB_GLD_SPIKE_PCT | 3.0% | **3.5%** |
| V5_CB_BTC_CRASH_PCT | -6.0% | **-4.5%** |
| V5_CB_BTC_SURGE_PCT | 13.5% | **4.0%** |

**공유 파라미터 (v2~v5 공통):**

| 파라미터 | 이전 | 이후 |
|---------|------|------|
| MAX_HOLD_HOURS | 5h | **2h** |
| TAKE_PROFIT_PCT | 4.0% | **4.5%** |
| STOP_LOSS_PCT | -3.0% | **-3.5%** |
| STOP_LOSS_BULLISH_PCT | -16.0% | **-5.5%** |
| DCA_DROP_PCT | -1.35% | **-1.8%** |
| COIN_SELL_PROFIT_PCT | 5.0% | **3.5%** |
| CONL_SELL_PROFIT_PCT | 2.8% | **1.5%** |
| PAIR_GAP_SELL_THRESHOLD_V2 | 9.0 | **9.4** |
| PAIR_SELL_FIRST_PCT | 0.80 | **0.50** |

### SSH 설정 수정 ✅

`~/.ssh/config`에 `gigaflops-node54` IdentityFile 추가:
```
Host gigaflops-node54
    HostName giga-flops.com
    User user
    Port 3333
    IdentityFile ~/.ssh/id_rsa   ← 추가됨
```

---

## 2. 현재 Line A 상태

| 항목 | 상태 |
|------|------|
| v5 rules | `docs/rules/line_a/trading_rules_v5.md` — 확정 |
| v5 config | `config.py` — Study 13 best 반영 완료 (커밋 f3b329e) |
| v5 backtest | `simulation/backtests/backtest_v5.py` |
| v5 optimizer | `simulation/optimizers/optimize_v5_optuna.py` |
| v6 rules | `docs/rules/line_a/trading_rules_v6.md` — 설계됨, 코드 미구현 |

---

## 3. 다음 우선순위 할 일

### 🔴 높음

1. **v5 확정 파라미터로 재백테스트 (전 기간)**
   - 방법: `make slurm-run PROFILE=backtest_v5 PARTITION=all ACCOUNT=default`
   - 목적: 25개 파라미터 반영 후 최종 수익률 확인
   - 예상: IS -1.49% / OOS +0.36% / FULL -1.14% (Study 13 결과 재현)

2. **v6 코드 구현 준비**
   - rules: `docs/rules/line_a/trading_rules_v6.md` 이미 작성됨
   - 구현 내용: Polymarket 5-레이어 + BearRegime + 포지션 연속화
   - 파일: `simulation/strategies/line_a/signals_v6.py` (신설 필요)
   - optimizer: `simulation/optimizers/optimize_v6_regime.py` (신설 필요)

### 🟡 중간

3. **v4 Study6 OOS 검증**
   - DB: `data/optuna/optuna_v4_study6.log` (487 trials, best #48, value=573.74)
   - 방법: best params 추출 → backtest_v5.py OOS 단독 실행

4. **D2S v3 recent_is 앙상블 파라미터 실서버 반영 검토**
   - recent_is_r2: OOS +2.40%, Sharpe 1.427, MDD -5.35% ← 최우수
   - 파일: `simulation/strategies/line_c_d2s/params_d2s.py`

---

## 4. 주요 파일 경로

| 항목 | 경로 |
|------|------|
| v5 config | `config.py` |
| v5 backtest | `simulation/backtests/backtest_v5.py` |
| v5 optimizer | `simulation/optimizers/optimize_v5_optuna.py` |
| v5 OOS 검증 스크립트 | `experiments/study_v5_s2_oos.py` |
| Study 13 결과 | `data/results/backtests/study_13_v5_s2_oos_20260227.json` |
| v5 Optuna DB | `data/optuna/v5_opt.db` (study: ptj_v5_s2, best: #309) |
| v6 rules | `docs/rules/line_a/trading_rules_v6.md` |
| SLURM profile | `slurm/profiles/study_v5_s2_oos.conf` |

---

## 5. 코드 환경

```bash
pyenv shell ptj_stock_lab   # Python 3.11
make slurm-run PROFILE=<profile> PARTITION=all ACCOUNT=default
make slurm-log PROFILE=<profile>
make slurm-collect PROFILE=<profile>
```

**클러스터**: `gigaflops-node54` (SSH via `id_rsa`)
**컴퓨트 노드**: `giganode-[51,54,83]`
**컨테이너**: `/mnt/giga/project/ptj_stock_lab/slurm/images/ptj_stock_lab.sqsh`

---

## 6. 최근 커밋 히스토리

```
f3b329e feat: v5 파라미터 Optuna s2 best 반영 (Study 13 OOS 검증 완료)
29449dc docs: LineB VNQ rules v2 승격 — notes → rules 이관
e3c4f78 docs: rule-verifier CRITICAL 3건 수정
```

---

## 7. Line C v3 Study (별도 세션 추가)

> 이번 세션에서 진행한 Line C D2S v3 study 결과

### Line C Study 13 — Bias-corrected Optuna 전체 탐색 (no-ROBN 1.5년) ✅

- **Journal**: `data/optuna/d2s_v3_norobn_s13.log`
- **설정**: IS 2024-09-18~2025-05-31 | OOS 2025-06-01~2026-02-17, 500 trials
- Best Trial: **#466**, IS +98.9%, OOS +162.6%, FULL **+428.72%**, Sharpe 1.212, MDD -22.5%
- ⚠️ OOS > IS 수익 → 과적합 가능성으로 Study 16 WF 검증 실시

### Line C Study 14 — r5 (R21 제거) Bias-corrected 재실행 ❌ 미완

- SCRIPT_ARGS 따옴표 버그로 정상 실행 실패 (biased params만 corrected engine 검증)
- corrected engine으로 FULL +12.44% (vs biased +190.38% → **15배 과대평가 확인**)
- **재실행 필요**: `make slurm-run PROFILE=study_14_corrected_r5 PARTITION=titanx ACCOUNT=default`
- conf 파일은 따옴표 수정 완료 (`slurm/profiles/study_14_corrected_r5.conf`)

### Line C Study 15 — recent_is r2 vs r3 OOS 비교 ✅

- r2 승: OOS +0.35% vs r3 OOS -0.89% (단, OOS 기간 11일로 통계적 유의성 낮음)
- 결과: `data/results/backtests/study_15_recent_is_compare_*.json`

### Line C Study 16 — Study 13 #466 Walk-Forward 검증 ✅

- 결과: `data/results/optimization/study_16_wf_validation_20260301.json`
- **판정: 3/4 창 OOS 양수 — 부분 견고성**

| 창 | OOS 기간 | OOS% | OOS_Sharpe | OOS_MDD |
|---|---|---|---|---|
| W1 | 2025-03-01~05-31 | **+38.6%** | 3.400 | -4.9% |
| W2 | 2025-06-01~08-31 | **+10.4%** | 1.486 | -11.5% |
| W3 | 2025-09-01~11-30 | **-3.6%** ❌ | -0.306 | -12.9% |
| W4 | 2025-12-01~2026-02-28 | **+143.6%** | 2.107 | -22.5% |

- FULL: +428.72%, Sharpe 1.212, MDD -22.5%, WR 67.2%, trades 137

### 생성된 파일

| 파일 | 내용 |
|---|---|
| `slurm/profiles/study_13_corrected_optuna.conf` | Study 13 SLURM 프로파일 |
| `slurm/profiles/study_14_corrected_r5.conf` | Study 14 SLURM 프로파일 (따옴표 수정됨) |
| `slurm/profiles/study_15_recent_is_compare.conf` | Study 15 SLURM 프로파일 |
| `slurm/profiles/study_16_wf_validation.conf` | Study 16 SLURM 프로파일 |
| `experiments/study_15_recent_is_compare.py` | r2/r3 OOS 비교 스크립트 |
| `experiments/study_16_wf_validation.py` | WF 검증 스크립트 |

### Line C 다음 우선순위

1. **Study 14 재실행** (미결): corrected engine r5 500 trials (~16h)
   ```bash
   make slurm-run PROFILE=study_14_corrected_r5 PARTITION=titanx ACCOUNT=default
   ```
2. **Study 13 #466 파라미터 배포 검토**: WF 부분 견고성 → W3 약점 분석 후 실서버 반영 결정
3. **`daily_new_entry_cap` 0.30 → 0.50 변경 테스트**: Entry Cap Study 결과 (+277% vs +228%)

### ⚠️ SCRIPT_ARGS 따옴표 규칙 (필수)

```bash
# ❌ 잘못됨 — 공백 이후 단어가 별도 명령으로 실행됨
SCRIPT_ARGS=--no-robn --n-trials 500

# ✅ 올바름
SCRIPT_ARGS="--no-robn --n-trials 500 --n-jobs 20 --study-name xxx --journal xxx"
```

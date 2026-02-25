# Trading Rules 문서 계보 (Lineage)

> 작성일: 2026-02-21 | 최종 갱신: 2026-02-25
> 목적: 모든 매매 규칙/전략 문서의 전체 목록과 관계를 한곳에 정리하여 꼬이지 않게 관리

---

## 전체 계보도

```
[A. 설계 기반 라인] ─────────────────────────────────────────────
  docs/rules/line_a/    v1 → v2 → v3 → v4 → v5 → v6
  simulation/strategies/line_a/    signals.py, signals_v2.py, signals_v5.py
  규칙서 6개 + 리포트 13개

[B. 카톡 CSV 라인 (태준 수기 전략)] ─────────────────────────────
  docs/notes/line_b/source/    MT_XLY, MT_VNQ, MT_VNQ2, MT_VNQ3, MT_SK, kakaotalk CSV
  docs/notes/line_b/review/    노트 9개 (2/18 ~ 2/23)
  docs/rules/line_b/           예정 (빈 라인)
  simulation/strategies/line_b_taejun/   _FROZEN — 20개 전략 + 8개 인프라 + 7개 필터
  → 정식 rules 문서: ❌ 없음 (rules 작성 시까지 코드 수정 금지)

[C. 행동 추출 라인 (D2S)] ───────────────────────────────────────
  docs/rules/line_c/     attach v1, attach v2, attach v3
  simulation/strategies/line_c_d2s/   d2s_engine.py, params_d2s.py

[D. 장기 거래내역 라인 (2023~2026)] ─────────────────────────────
  docs/rules/line_d/     jun_trade_2023_v1.md ✅
  simulation/strategies/line_d_history/   (빈 디렉토리)
  거래내역 3개 CSV → 실험 3개
```

---

## A. 설계 기반 라인

### A-1. 규칙서 (`docs/rules/line_a/`)

| 파일 | 핵심 내용 | 상태 |
|---|---|---|
| `trading_rules_v1.md` | 기본 5개 규칙 | 완료 |
| `trading_rules_v2.md` | 파라미터화, 임계값 | 완료 |
| `trading_rules_v3.md` | Optuna 최적화 반영 | 완료 |
| `trading_rules_v4.md` | 스윙, CB 감지 | 완료 |
| `trading_rules_v5.md` | 횡보장, 시간 제한, QA 정정, 이머전시/자산모드 | 완료 |
| `trading_rules_v6.md` | Polymarket 레이어드, BearRegime, BITI/SOXS | 완료 |
| `v4_rule_manifest.yaml` | v4 규칙 매니페스트 | 완료 |

### A-2. 관련 리포트

| 파일 | 위치 |
|---|---|
| `backtest_v2_report.md` | `docs/reports/backtest/` |
| `backtest_comparison_report.md` | `docs/reports/backtest/` |
| `backtest_optuna_comparison_report.md` | `docs/reports/backtest/` |
| `backtest_oos_analysis_report.md` | `docs/reports/backtest/` |
| `5min_backtest_report.md` | `docs/reports/backtest/` |
| `5min_backtest_fees_report.md` | `docs/reports/backtest/` |
| `trial_79_analysis.md` | `docs/reports/backtest/` |
| `v4_best_trade_analysis_2026-02-19.md` | `docs/reports/backtest/` |
| `logic_validation_report_2026-02-22.md` | `docs/reports/backtest/` |
| `v3_optuna_report.md` | `docs/reports/optuna/` |
| `v4_optuna_report.md` / `v4_optuna_plan.md` | `docs/reports/optuna/` |
| `optimization_report.md` / `optuna_optimization_report.md` / `optuna_best_practices.md` | `docs/reports/optuna/` |
| `v3_baseline_report.md` / `v4_baseline_report.md` / `v5_baseline_report.md` | `docs/reports/baseline/` |
| `baseline_performance_v4_v5.md` | `docs/reports/baseline/` |
| `v4_baseline_analysis.md` | `docs/reports/baseline/` |
| `stoploss_comparison_1.5_vs_3.0.md` / `stoploss_optimization_report.md` / `stoploss_with_fees_report.md` | `docs/reports/stoploss/` |

### A-3. 구현 코드 (`simulation/strategies/line_a/`)

| 코드 | 대응 |
|---|---|
| `signals.py` | v1 기본 시그널 |
| `signals_v2.py` | v2 파라미터화 |
| `signals_v5.py` | v5 횡보장/시간 제한 |

---

## B. 카톡 CSV 라인 (태준 수기 전략)

### B-1. 원본 소스 (`docs/notes/line_b/source/`)

| 파일 | 내용 |
|---|---|
| `kakaotalk_trading_notes_2026-02-19.csv` | 카카오톡 대화 원본 |
| `MT_XLY.md` | 태준 지시서 — SK리츠→XLY 대입 |
| `MT_VNQ.md` | 태준 지시서 — SK리츠→VNQ 대입 |
| `MT_VNQ2.md` | 태준 지시서 — CI-0 v0.2, Fill Window, 추격매수 금지 |
| `MT_VNQ3.md` | 태준 지시서 — M201 0.55, M28 게이트, SCHD, M200, M300 |
| `MT_SK.md` | 태준 지시서 — 전체 확정판 |

### B-2. 노트 (`docs/notes/line_b/review/`, 시간순)

| 파일 | 날짜 | 핵심 변경 |
|---|---|---|
| `trading_strategy_notes_2026-02-18.md` | 2/18 | 카톡 대화 초기 정리 |
| `taejun_strategy_review_2026-02-19.md` | 2/19 | 11개 전략 구현, Critical 이슈 정리 |
| `taejun_strategy_review_2026-02-20.md` | 2/20 | SETH/BAC 정정, M1~M5, 수익금 분배 |
| `taejun_strategy_review_2026-02-21_XLY.md` | 2/21 | XLY 파생 — M0/M4~M7/M20/M40, 숏 전략 |
| `taejun_strategy_review_2026-02-21_VNQ.md` | 2/21 | VNQ 파생 — M0/M4/M6/M7/M20/M40 |
| `taejun_strategy_review_2026-02-21_SK.md` | 2/21 | SK 확정 — Q1~Q11 전량 확정, 숏 4종목 |
| `taejun_0222_VNQ_reviewed.md` | 2/22 | VNQ — MT_VNQ2 반영, CI-0 v0.2 |
| `taejun_strategy_review_2026-02-22_VNQ.md` | 2/22 | VNQ — MT_VNQ2 상세 구현 |
| `taejun_strategy_review_2026-02-23_VNQ.md` | 2/23 | VNQ — MT_VNQ3 반영, M201/M28/SCHD/M200/M300 |

### B-3. 인프라 모듈 (`simulation/strategies/line_b_taejun/infra/`)

| 모듈 | 코드 | 역할 |
|---|---|---|
| LimitOrder + OrderQueue | `limit_order.py` | 지정가 주문 관리 (CI-0-12/17/18) |
| M200 KillSwitch | `m200_stop.py` | 7개 조건 즉시매도 |
| M201 ImmediateMode | `m201_mode.py` | BTC 확률 급변 즉시전환 |
| M28 PolyGate | `m28_poly_gate.py` | Polymarket BTC/NDX 포지션 게이트 |
| M5 WeightManager | `m5_weight_manager.py` | T1~T4 비중 배분 + 동적 조정 |
| Orchestrator | `orchestrator.py` | 10단계 우선순위 파이프라인 |
| SCHD Master | `schd_master.py` | SCHD 적립 + 매도 차단 + M300 |
| ProfitDistributor | `profit_distributor.py` | 수익금 분배 (SOXL→ROBN→GLD→CONL) |

### B-4. 정식 rules 문서 (`docs/rules/line_b/`)

❌ **없음** — 노트 9개 + 지시서 5개가 산재. 어느 것이 최종본인지 불명확.

승격 예정:
- `taejun_vnq_trading_rules_v1.md` → VNQ 관련 코드 수정 허용
- `taejun_xly_trading_rules_v1.md` → XLY 관련 코드 수정 허용
- `taejun_sk_trading_rules_v1.md` → SK 관련 코드 수정 허용

### B-5. 구현 코드 — ⚠️ _FROZEN (rules 없이 생성됨)

> **경위**: 2/19 `fef5d8c` 리팩터링 커밋에서 카톡 노트 → 곧바로 코드 구현.
> 정식 rules 문서를 거치지 않고 노트에서 직접 코드로 갔기 때문에 rules가 없는 상태.
> `_FROZEN.md`가 루트에 배치되어 수정/추가 금지 상태.
> 향후 `docs/rules/line_b/`에 rules 문서 작성 시 이 코드들의 로직을 역으로 문서화해야 함.

**전략 (`line_b_taejun/strategies/`)**

| 전략 | 코드 |
|---|---|
| 잽모드 SOXL/BITU/TSLL | `jab_soxl.py`, `jab_bitu.py`, `jab_tsll.py` |
| VIX → GDXU | `vix_gold.py` |
| S&P500 편입 | `sp500_entry.py` |
| 저가매수 | `bargain_buy.py` |
| 숏포지션 전환 | `short_macro.py` |
| 리츠 리스크 | `reit_risk.py` |
| 섹터 로테이션 | `sector_rotate.py` |
| 조건부 은행주 | `bank_conditional.py` |
| 숏 잽모드 ETQ | `jab_etq.py` |
| 이머전시 모드 | `emergency_mode.py` |
| Bear Regime | `bear_regime.py` |
| 조건부 코인 | `conditional_coin.py` |
| 조건부 CONL | `conditional_conl.py` |
| 크래시 바이 | `crash_buy.py` |
| SOXL 독립매수 | `soxl_independent.py` |
| 트윈 페어 | `twin_pair.py` |
| Bearish Defense | `bearish_defense.py` |

**필터 (`line_b_taejun/filters/`)**

| 모듈 | 코드 |
|---|---|
| 필터 엔진 | `filters.py` |
| 서킷 브레이커 | `circuit_breaker.py` |
| 손절 | `stop_loss.py` |
| Poly 퀄리티 | `poly_quality.py` |
| 스윙 모드 | `swing_mode.py` |
| 자산 모드 | `asset_mode.py` |

**공통 (`line_b_taejun/common/`)**

| 모듈 | 코드 |
|---|---|
| 전략 베이스 | `base.py` |
| 전략 등록 | `registry.py` |
| 파라미터 | `params.py` |
| 수수료 | `fees.py` |

**포트폴리오 (`line_b_taejun/portfolio/`)**

| 모듈 | 코드 |
|---|---|
| 포트폴리오 관리 | `portfolio.py` |

**기타**

| 모듈 | 코드 |
|---|---|
| 시그널 프리셋 | `signal_presets.py` |
| 복합 시그널 엔진 | `composite_signal_engine.py` |

---

## C. 행동 추출 라인 (D2S)

### C-1. 데이터 소스

| 파일 | 위치 |
|---|---|
| `거래내역서_20250218_20260217_1.csv` | `history/2025/` |

### C-2. 파이프라인 (`experiments/`)

| 단계 | 스크립트 | 산출물 |
|---|---|---|
| Step 1 | `build_decision_log.py` | `decision_log.csv`, `decision_log_trades.csv` |
| Step 2 | `extract_rules.py` | `rule_extraction_report.csv` |
| Step 3 | `extract_rules_v2.py` | `rule_candidates.csv` |
| Step 4 | `extract_rules_v2_patterns.py` | `round_trips.csv` |
| Step 5 | `analysis_technical_indicators.py` | `technical_indicators_report.csv` |
| Step 6 | `analysis_intraday_patterns.py` | `intraday_patterns_report.csv` |
| Step 7 | `analysis_calendar_correlation.py` | `calendar_effects.csv`, `correlation_matrix.csv` |

산출 CSV 위치: `data/results/analysis/` (15개 파일)

### C-3. 규칙서 (`docs/rules/line_c/`)

| 파일 | 핵심 내용 | 상태 |
|---|---|---|
| `trading_rules_attach_v1.md` | D2S R1~R16, 953건 통계 분석 | 완료 |
| `trading_rules_attach_v2.md` | R17 V-바운스, R18 조기 손절, DCA 강화 | 완료 |
| `trading_rules_attach_v3.md` | 레짐 감지(R20/R21) + BB 하드 필터(R19) + Optuna #449 전체 | 완료 (Study 6~9B 검증) |

### C-3b. 관련 리포트

| 파일 | 위치 |
|---|---|
| `d2s_v3_study_report.md` | `docs/reports/backtest/` |

### C-4. 구현 코드 (`simulation/strategies/line_c_d2s/`)

| 코드 | 대응 |
|---|---|
| `d2s_engine.py` | attach v1 R1~R16 |
| `params_d2s.py` | attach v1/v2 파라미터 |

---

## D. 장기 거래내역 라인 (2023~2026)

### D-1. 데이터 소스

| 파일 | 위치 | 건수 |
|---|---|---|
| `거래내역_20231006_20260212.csv` | `history/` | 1,317건 |
| `수익_거래내역.csv` | `history/` | 244건 |
| `손해_거래내역.csv` | `history/` | 79건 |

### D-2. 분석 스크립트

| 스크립트 | 역할 | 산출물 |
|---|---|---|
| `profit_curve_strategy.py` | 수익곡선 + 모멘텀 전략 백테스트 | (stdout) |
| `exp_J_real_trade_analysis.py` | 수익/손해 조건 비교 → 규칙 추출 | `J_extracted_rules.csv`, `J_real_trade_conditions.csv` |
| `trade_history_vs_signal.py` | Polymarket crash_score vs 실매매 | `docs/charts/trade_history_vs_signal.png` |

### D-3. 규칙서 (`docs/rules/line_d/`)

| 파일 | 핵심 내용 | 상태 |
|---|---|---|
| `jun_trade_2023_v1.md` | 2023~2026 실거래 기반, 모멘텀 추세추종 | ✅ 완료 |

---

## 거버넌스 — rules 문서 변경 시 필수 절차

**이 lineage 문서가 Single Source of Truth.**

### notes → rules → code 파이프라인

코드 생성은 반드시 아래 파이프라인을 거쳐야 한다:

```
docs/notes/ (아이디어) → docs/rules/ (확정) → simulation/strategies/ (코드)
                  ❌ 직접 코드 금지        ✅ 코드 허용
```

- `docs/notes/` 단계의 문서만으로 코드를 생성하지 않는다
- `docs/rules/`에 확정 규칙 문서가 존재해야 코드 구현이 허용된다
- rules 문서 없이 생성된 코드는 `_FROZEN` 상태로 수정/추가 금지

### _FROZEN 상태

`_FROZEN.md` 파일이 디렉토리 루트에 존재하면 해당 코드는 동결 상태:
- 새 전략 모듈 추가 금지
- 기존 코드 로직 수정 금지
- 파라미터 변경 금지
- 해제 조건: 대응하는 `docs/rules/` 아래에 rules 문서가 작성되면 해당 부분만 해제

현재 _FROZEN 적용: `simulation/strategies/line_b_taejun/`

### 체크리스트 (rules 문서 생성/수정 시)

```
[ ] 1. 이 문서(trading_rules_lineage.md) 읽기
[ ] 2. 어느 라인에 속하는지 확인 (A/B/C/D)
[ ] 3. 해당 라인의 이전 문서 읽기
[ ] 4. 새 문서 작성
[ ] 5. 이 문서의 해당 라인 테이블에 행 추가/갱신
```

### 변경 이력

| 날짜 | 변경 |
|---|---|
| 2026-02-21 | 초기 작성 — A/B/C/D 4개 라인 + 전체 파일 목록 |
| 2026-02-23 | MT_VNQ2/VNQ3 반영 — 지시서 2개, 노트 2개 추가 + 인프라 모듈 7개(B-3) + 코드 미대응 7개 추가 |
| 2026-02-23 | 4-line 리팩터링 — docs/ + code/ 전면 재편, jab_seth 삭제, notes→rules→code 거버넌스 |
| 2026-02-25 | LineC attach v3 추가 — 레짐 감지(R19~R21) + Optuna #449 + Study 6~9B 검증 |

# 전체 평가 리포트: Rules ↔ Engine ↔ Optuna

> 생성일: 2026-02-23
> 범위: Line A (v2~v5) + Line B (19개 전략) + Line C (D2S v1/v2) + Line D + Optuna 8개 DB
> 팀: coverage-auditor, backtest-runner, line-b-evaluator, optuna-analyzer, rules-drafter

---

## Executive Summary

| 항목 | 결과 |
|------|------|
| Rules ↔ Engine 커버리지 | **88.9%** (72/81) — Line A/B/C 100%, Line D 0% |
| 최고 성과 엔진 | **Line A v4** (+25.49%, Sharpe 1.37, MDD 9.34%) |
| Line C D2S | v2 +10.92% (v1 대비 +2.14%p 개선) |
| Line B 활성 전략 | 3/19 (데이터 부족으로 16개 비활성) |
| Optuna 최고 스코어 | v4 Phase1: **+30.43** (400 trials) |
| v5 Optuna | **미실행** — 최적화 필요 |
| Line B Rules 초안 | 3개 생성 (VNQ/XLY/SK) |

---

## Phase 1: Rules ↔ Engine 커버리지 감사

### 전체 커버리지

| Line | 규칙 수 | 구현 | 커버리지 | 상태 |
|------|:---:|:---:|:---:|------|
| A (설계기반) | 25 | 25 | **100%** | v1~v5 전 규칙 구현 |
| B (태준수기) | 31 | 31 | **100%** | FROZEN — rules 문서 미존재 |
| C (D2S행동추출) | 16 | 16 | **100%** | R1~R16 + R17~R18 완전 구현 |
| D (장기거래) | 9 | 0 | **0%** | 엔진 코드 완전 부재 |
| **전체** | **81** | **72** | **88.9%** | |

### 주요 갭

- **Line D**: `jun_trade_2023_v1.md`에 정의된 E-1~3 (진입), X-1~4 (청산), VIX/BTC 레짐 규칙 9개 → `line_d_history/`에 `__init__.py`만 존재
- **Line B**: 코드는 완전하나 `docs/rules/line_b/` 공식 규칙서 미존재 → Phase 4에서 초안 생성

> 상세: `docs/reports/backtest/rules_coverage_summary.md`

---

## Phase 2-A: Line A 백테스트 성능 비교

### 기간: 2025-02-18 ~ 2026-02-17 (240 거래일)

| 지표 | v2 | v3 | v4 | v5 |
|------|----|----|----|----|
| 수익률 | -37.65% | -17.97% | **+25.49%** | -13.68% |
| MDD | 37.65% | 17.97% | **9.34%** | 14.17% |
| Sharpe | -3.11 | -2.33 | **1.37** | -2.63 |
| 승률 | 40.8% | **52.6%** | 38.8% | 40.3% |
| 매도 횟수 | 3,096 | 1,857 | **85** | 904 |
| 손절 횟수 | 124 | 81 | **1** | 50 |
| 수수료 | $7,891 | ₩5.8M | **$676** | $1,904 |

### 핵심 인사이트

1. **v4가 압도적 우위**: 수익률·MDD·Sharpe 모두 최고. CB + 서킷브레이커가 시장 하락 방어에 효과적
2. **v5 CB 과도 차단**: CB 매수 차단 22,766회 vs v4의 1,630회 → 방어 로직이 수익 기회도 차단
3. **v3 선별 매매 효과**: 갭 기준 강화 + 쿨타임 증가로 거래 -40%, 승률 +11.7%p
4. **거래 극소화 = 수익**: v4의 매도 85회(v2 대비 -97%) → 수수료 절감 + 노이즈 회피

> 상세: `docs/reports/backtest/version_comparison_table.md`

---

## Phase 2-B: Line B 19개 전략 개별 평가

### 기간: 2025-01-03 ~ 2026-01-30 (269 거래일)

| 전략 | BUY | SELL | 진입률 | 비고 |
|------|----:|-----:|-------:|------|
| **twin_pair** | 158 | 101 | 58.7% | 가장 활발 |
| **conditional_coin** | 46 | 0 | 17.1% | ETHU+XXRP+SOLT 조건 |
| **conditional_conl** | 41 | 0 | 15.2% | CONL 트리거 조건 |
| 나머지 16개 | 0 | 0 | 0% | 데이터 부족 |

### 비활성 원인 분류

| 원인 | 전략 수 | 대표 전략 |
|------|:---:|------|
| 종목 미포함 (SOXX, TSLA, VIX 등) | 7 | jab_soxl, jab_bitu, vix_gold, soxl_independent |
| history 메타데이터 부재 | 5 | bargain_buy, sector_rotate, short_macro, reit_risk |
| 외부 모드/데이터 의존 | 4 | emergency_mode, bear_regime_long, bearish_defense, sp500_entry |

### 시사점

- backtest_1min_v2.parquet에 **16종목만** 포함 → 전략 대부분이 추가 종목 필요
- 완전한 평가를 위해 **종목 커버리지 확대** + **history 메타데이터 보강** 필수
- 활성 3개 전략(twin_pair, conditional_coin/conl)이 Line B의 핵심 시그널 소스

> 상세: `docs/reports/backtest/line_b_strategy_stats.md`

---

## Phase 2-C: Line C D2S 백테스트

### 기간: 2025-03-03 ~ 2026-02-17 (242 거래일, 일봉)

| 지표 | D2S v1 | D2S v2 | Δ |
|------|--------|--------|---|
| 수익률 | +8.78% | **+10.92%** | +2.14%p |
| MDD | -35.11% | **-33.16%** | +1.95%p |
| Sharpe | 0.419 | **0.454** | +0.035 |
| 승률 | 67.3% | 66.3% | -1.0%p |
| 총 거래 | 281 | 251 | -30 |
| 수수료 | $1,802 | **$1,669** | -$133 |

### 핵심 인사이트

- v2의 R17(V-바운스 2x) + R18(조기 손절) 추가로 수익률 개선 + MDD 개선
- 승률 소폭 하락(-1%p)은 조기 손절(R18) 효과 — 작은 손실을 빨리 끊음
- MDD -33~35%는 여전히 높음 → 추가 리스크 관리 검토 필요

> 상세: `docs/reports/backtest/d2s_evaluation.md`

---

## Phase 3: Optuna 최적화 결과 분석

### DB 현황 (8개)

| Study | 버전 | Trials | Best Score | 비고 |
|-------|------|:---:|-----------|------|
| ptj_v2_expanded | v2 | 100 | +1.57% | Multi-objective |
| ptj_v3_full | v3 | 329 | +12.80 | |
| ptj_v3_phase2 | v3 | 310 | +13.42 | 권장 v3 값 |
| ptj_v3_train_test | v3 | 130 | +14.03 | |
| ptj_v3_train_test_v2 | v3 | 94 | +11.94 | |
| ptj_v3_train_test_wide | v3 | 500 | +19.08 | 과적합 주의 |
| ptj_v4_phase1 | v4 | 400 | **+30.43** | 전체 최고 |
| crash_model_v1 | misc | 300 | 0.7677 AUC | 급락 모델 |

### config vs Optuna Best — 주요 괴리

| 파라미터 | 현재 config | Optuna 방향 | 일관성 |
|---------|------------|-----------|:---:|
| PAIR_GAP_SELL_THRESHOLD_V2 | 0.9% | **6~9%** | 전 버전 일치 |
| STOP_LOSS_BULLISH_PCT | -8.0% | **-12~-16%** | 점진 완화 |
| DCA_MAX_COUNT | 4~7 | **1** (v4) | v4에서 극적 |
| COIN_SELL_PROFIT_PCT | 3.0% | **4~5%** | v3/v4 일치 |
| TAKE_PROFIT_PCT | 2.0% | **3.5~5.5%** | 상향 필요 |

### v4 파라미터 중요도 Top-3

1. **V4_PAIR_GAP_ENTRY_THRESHOLD** — 29% (현재 2.2 ≈ 최적 2.0, 거의 일치)
2. **V4_SIDEWAYS_POLY_HIGH** — 14%
3. **V4_CB_BTC_SURGE_PCT** — 10%

### v5 최적화 상태

**미실행**. v5 파라미터는 v4 초기값을 그대로 사용 중. 즉시 반영 가능 항목:
- `PAIR_GAP_SELL_THRESHOLD_V2`: 0.9 → 8.8
- `COIN_SELL_PROFIT_PCT`: 3.0 → 5.0
- `STOP_LOSS_BULLISH_PCT`: -8.0 → -16.0

> 상세: `docs/reports/backtest/optuna_param_comparison.md`

---

## Phase 4: Line B Rules 문서 초안

### 생성된 문서

| 문서 | 소스 | 전략 수 | 특징 |
|------|------|:---:|------|
| `taejun_vnq_trading_rules_v1.md` | MT_VNQ3 리뷰 | 13 (A~M) | 가장 포괄적, M0~M300 마스터 플랜 포함 |
| `taejun_xly_trading_rules_v1.md` | MT_XLY 리뷰 | 12 (A~L) | SK리츠→XLY 전환, Polymarket 매핑 |
| `taejun_sk_trading_rules_v1.md` | MT_SK 리뷰 | 12 (A~L) | 원본 SK리츠 기반, KR REIT trio |

### 발견된 마커

| 마커 | 의미 | 개수 |
|------|------|:---:|
| **[미확인]** | 코드에만 존재, 리뷰 노트에 없음 | 17 |
| **[리뷰 미반영]** | 리뷰 노트에 있으나 코드에 미반영 | 주요 1건 |

**[리뷰 미반영] 핵심**: params.py의 BARGAIN_BUY 파라미터가 구버전 값(CONL: 188%, SOXL: 320%) → 리뷰 노트에서는 100%로 변경 지시됨

> 상세: `docs/rules/line_b/taejun_vnq_trading_rules_v1.md` 등

---

## 종합 권고사항

### 즉시 실행 (High Priority)

| # | 액션 | 근거 |
|---|------|------|
| 1 | **v5 Optuna 최적화 실행** | v5 DB 없음, v4 초기값 그대로 사용 중 |
| 2 | **PAIR_GAP_SELL_THRESHOLD_V2 상향** (0.9→6~9%) | 전 버전 Optuna에서 일관된 결과 |
| 3 | **v5 CB 파라미터 튜닝** | CB 22,766회 과도 차단 → v4 수준(1,630회)으로 조정 |
| 4 | **STOP_LOSS_BULLISH_PCT 완화** (-8→-16%) | v2~v4 Optuna 전체에서 완화 방향 일관 |

### 중기 (Medium Priority)

| # | 액션 | 근거 |
|---|------|------|
| 5 | **Line B 데이터 커버리지 확대** | 16/19 전략이 종목 부족으로 비활성 |
| 6 | **Line B Rules 초안 검토 + 확정** | FROZEN 해제를 위한 전제 조건 |
| 7 | **D2S MDD 개선** | -33% MDD → 추가 리스크 관리 필요 |
| 8 | **BARGAIN_BUY 파라미터 업데이트** | 리뷰 노트 지시(100%) vs 코드(188/320%) 불일치 |

### 장기 (Low Priority)

| # | 액션 | 근거 |
|---|------|------|
| 9 | **Line D 엔진 구현** | 9개 규칙 정의됨, 코드 0% |
| 10 | **Line B history 메타데이터 구축** | bargain_buy, sector_rotate 등 활성화 |
| 11 | **v6 설계 시 v4 구조 기반** | v4가 유일 수익, 극소 거래 전략 |

---

## 산출물 목록

### 스크립트 (experiments/)
| 파일 | Phase | 설명 |
|------|-------|------|
| `rules_coverage_audit.py` | 1 | Rules↔Engine 텍스트 패턴 분석 |
| `backtest_evaluation.py` | 2-A | Line A v2~v5 백테스트 실행 |
| `d2s_backtest_evaluation.py` | 2-C | D2S v1/v2 백테스트 실행 |
| `line_b_strategy_evaluation.py` | 2-B | Line B 19개 전략 시그널 평가 |
| `optuna_analysis.py` | 3 | Optuna 8개 DB 분석 |

### 리포트 (docs/reports/backtest/)
| 파일 | Phase | 설명 |
|------|-------|------|
| `rules_coverage_summary.md` | 1 | 커버리지 감사 결과 |
| `version_comparison_table.md` | 2-A | Line A v2~v5 비교 |
| `d2s_evaluation.md` | 2-C | D2S v1/v2 평가 |
| `line_b_strategy_stats.md` | 2-B | Line B 전략별 통계 |
| `optuna_param_comparison.md` | 3 | Optuna 파라미터 비교 |
| `full_evaluation_report.md` | 최종 | **이 문서** |

### Rules 초안 (docs/rules/line_b/)
| 파일 | Phase | 설명 |
|------|-------|------|
| `taejun_vnq_trading_rules_v1.md` | 4 | VNQ 기반 rules 초안 |
| `taejun_xly_trading_rules_v1.md` | 4 | XLY 기반 rules 초안 |
| `taejun_sk_trading_rules_v1.md` | 4 | SK리츠 기반 rules 초안 |

---

*Generated by full-evaluation team (2026-02-23)*

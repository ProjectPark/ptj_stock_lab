# attach v1 Study 진행 노트 (2026-02-26)

## 목표

`docs/rules/line_c/trading_rules_attach_v1.md` 기반 실거래 재현.
코드: `simulation/backtests/backtest_d2s.py` + `simulation/strategies/line_c_d2s/`

**실거래 기준 (2025-02-19 ~ 2026-02-12, 248거래일)**

| 지표 | 목표값 |
|------|--------|
| D2S 4종목 매수건 | 406건 (102일 × 3.98건/일) |
| 라운드트립 | 722건 |
| 승률 | 65.5% |
| 평균 PnL | +7.39% |

> 거래수 매칭은 목표가 아님. **승률 · 평균PnL** 최대화가 핵심.

---

## 완료된 Phase

### Phase 1 — 기준선 백테스트 (Job 119)
`experiments/backtest_d2s_v1_baseline.py`

| 기간 | 승률 | 평균PnL | 거래수 |
|------|------|---------|-------|
| FULL (2025-02-19~) | 67.6% | +3.71% | 105건 |
| WARM (2025-03-03~) | 70.3% | +3.98% | 111건 |

### Phase 2 — Ablation Study (Job 118)
`experiments/study_ablation_rules.py`

주요 발견:
- R14 riskoff OFF → 승률 -2.2%p (가장 중요한 규칙)
- Study B (R13 SPY연속상승 + R5 MSTU riskoff + R8 CONL contrarian) OFF → 승률 +2.8%p 개선
- R13 (spy_streak_max=3) 억제 효과 → 999(사실상 OFF)로 수정

### Phase 4A — 파라미터 최적화 (Job 123)
`experiments/backtest_d2s_v1_optimized.py`

수정된 파라미터 (`simulation/strategies/line_c_d2s/params_d2s.py` D2S_ENGINE):

```python
"optimal_hold_days_max": 5,           # 7 → 5
"mstu_riskoff_contrarian_only": False, # True → False  (Study B)
"robn_riskoff_momentum_boost": False,  # True → False  (Study B)
"conl_contrarian_require_riskoff": False, # True → False  (Study B)
"spy_streak_max": 999,                # 3 → 999 (Study B / R13 OFF)
```

결과 (WARM): 승률 63.2%, 평균PnL +4.19%, 155건

### Phase 4B — 분봉 진입 Study (Job 124)
`experiments/study_d2s_1min.py`

entry_opt 방식 (09:30~10:30 최저점 봉 진입):
- 결과: 65.9% 승률, +6.40% 평균PnL, 243건

### Study D2S 1min DCA (Job 131)
`experiments/study_d2s_1min_dca.py`

history 분석 기반: D2S 4종목 일평균 3.98건/일, DCA 중앙값 범위 1.13%

| 시나리오 | 거래수 | 승률 | 평균PnL |
|---------|-------|------|--------|
| entry_opt | 243 | 66.7% | +6.61% |
| **dca_1pct** | **243** | **70.4%** | **+7.46%** ← 최선 |
| dca_2pct | 243 | 69.5% | +7.52% |
| dca_3pct | 243 | 69.5% | +7.52% |

### Study D2S Entry Cap (Job 155)
`experiments/study_d2s_entry_cap.py`

daily_new_entry_cap 완화 실험 + dca_1pct 고정:

| cap | 거래수 | 승률 | 평균PnL |
|-----|-------|------|--------|
| 0.30 (현재) | 243 | 70.4% | +7.46% |
| 0.50 | 280 | 72.1% | +6.98% |
| 0.80 | 280 | 72.1% | +6.98% |
| 1.00 | 280 | 72.1% | +6.98% |

**결론**: cap 완화 시 거래수↑ 승률↑ 이지만 평균PnL 하락. cap=0.30 유지가 최선.

---

## 확정 파라미터 (attach v1 최종)

**일봉 엔진** (`D2S_ENGINE`, params_d2s.py 이미 반영):
```python
optimal_hold_days_max = 5
mstu_riskoff_contrarian_only = False
robn_riskoff_momentum_boost = False
conl_contrarian_require_riskoff = False
spy_streak_max = 999
daily_new_entry_cap = 0.30  # 유지
```

**분봉 진입 레이어** (study 결과, 아직 엔진 미반영):
```python
entry_mode    = "entry_opt"   # 09:30~10:30 최저점 봉 진입
dca_threshold = 1.0%          # -1% 하락 시 DCA 추가매수
max_dca       = 3             # 레이어 최대 3회
```

**최종 성과 (WARM 기간 기준)**:

| 지표 | 실거래 목표 | 달성값 | 격차 |
|------|-----------|--------|------|
| 승률 | 65.5% | **70.4%** | +4.9%p |
| 평균PnL | +7.39% | **+7.46%** | +0.07%p |

---

## 파일 구조

| 역할 | 파일 |
|------|------|
| 일봉 백테스트 엔진 | `simulation/backtests/backtest_d2s.py` |
| D2S 시그널 엔진 | `simulation/strategies/line_c_d2s/d2s_engine.py` |
| 파라미터 | `simulation/strategies/line_c_d2s/params_d2s.py` (D2S_ENGINE) |
| 분봉 DCA Study | `experiments/study_d2s_1min_dca.py` |
| Entry Cap Study | `experiments/study_d2s_entry_cap.py` |
| 분봉 데이터 | `data/market/ohlcv/backtest_1min.parquet` (2025-01-03~2026-01-30) |
| 결과 CSV | `data/results/analysis/d2s_1min_dca_*.csv` |
| 결과 CSV | `data/results/analysis/d2s_entry_cap_*.csv` |

## 주의사항

- `backtest_d2s.py` look-ahead bias 수정 완료 (T일 종가 신호 → T+1일 시가 체결, 슬리피지 0.05%)
- 분봉 데이터 범위: 2026-01-30 까지 (이후 날짜 없음)
- Polymarket 연동 없음 → BITU 폴백 사용 중

## 다음 가능한 방향

1. **분봉 진입 레이어 엔진 통합**: study 결과(entry_opt + dca_1pct)를 `d2s_engine.py`에 반영
2. **v2 설계 시작**: attach v2 규칙 문서 작성 후 새 엔진 개발
3. **OOS 검증**: 2026-02-01 이후 실거래와 비교 (데이터 수집 필요)

"""
D2S 엔진 전용 파라미터 — line_b_taejun/common/params.py에서 분리
"""

# ============================================================
# D2S 엔진 (D2S-Engine) — 실거래 행동 추출 기반 일봉 전략
# 출처: trading_rules_attach_v1.md (953건 실거래 D2S 분석)
# 2026-02-24: Study A/B/C 반영 (R14 그라데이션, 종목별 역발상, market_score)
# ============================================================
D2S_ENGINE = {
    # ── 종목 유니버스 ─────────────────────────────────────────
    "tickers": ["ROBN", "CONL", "MSTU", "AMDL"],  # 매수 대상만
    "lead_tickers": ["BITU", "NVDL"],               # 선행 지표 (매수 안함)
    "ticker_weights": {  # 실적 기반 종목 비중 (§12-1)
        "ROBN": 0.30, "CONL": 0.20, "MSTU": 0.25, "AMDL": 0.25,
    },

    # ── 쌍둥이 페어 ──────────────────────────────────────────
    "twin_pairs": {
        "coin_MSTU": {"lead": "BITU", "follow": "MSTU"},
        "coin_CONL": {"lead": "BITU", "follow": "CONL"},
        "bank_CONL": {"lead": "ROBN", "follow": "CONL"},
        "semi_AMDL": {"lead": "NVDL", "follow": "AMDL"},
    },

    # ── 0차 게이트: market_score 통합 점수 (Study C, §1-8) ──
    "market_score_suppress": 0.40,        # score < 0.40 → 당일 진입 전면 억제
    "market_score_entry_b": 0.55,         # B등급: 기본 비중 진입
    "market_score_entry_a": 0.60,         # A등급: 최대 비중 진입
    "market_score_weights": {             # 신호별 가중치 (합계 1.0)
        "gld_score":      0.20,           # R1 기반 (p=0.036)
        "spy_score":      0.15,           # R6 기반
        "riskoff_score":  0.25,           # R14 기반 (최강)
        "streak_score":   0.15,           # R13 기반
        "vol_score":      0.15,           # R16 기반 (★ 유일 유의, p=0.041)
        "btc_score":      0.10,           # R3 대용 (BITU 기반)
    },

    # ── 1. 시황 필터 (R1, R3, R13, R14) ─────────────────────
    "gld_suppress_threshold": 1.0,        # GLD ≥ +1.0% → 매수 억제 (p=0.036)
    "btc_up_max": 0.75,                   # BTC up > 0.75 → 매수 억제 (OOS)
    "btc_up_min": 0.40,                   # BTC up < 0.40 → 매수 억제 (OOS 하한)
    "confidence_suppress": True,          # confidence_signal → 매수 억제 (p=0.0003)
    "spy_streak_max": 999,                # SPY streak 필터 비활성화 (Phase 4A)
    "spy_bearish_threshold": -1.0,        # SPY < -1% → 역발상 매수 (50%)

    # ── R14 리스크오프 그라데이션 (Study A, §1-5) ──────────
    "riskoff_gld_up_spy_down": True,      # GLD↑+SPY↓ 활성화 여부
    "riskoff_spy_min_threshold": -1.5,    # SPY 하락 하한: 이 미만 → R14 미발동 + 비중 50% 축소
    "riskoff_gld_optimal_min": 0.5,       # GLD 최적 구간 하한 (Study A Level 2, p=0.042)
    "riskoff_spy_optimal_max": -0.5,      # SPY 최적 구간 상한 (Study A Level 2, p=0.042)
    "riskoff_consecutive_boost": 3,       # 연속 리스크오프 N일 이상 → 특급 신호 (100% 승률)
    "riskoff_panic_size_factor": 0.5,     # SPY < -1.5% 시 비중 축소 계수

    # ── 2. 쌍둥이 갭 (R2, OOS Decision Tree) ─────────────────
    "gap_bank_conl_max": 6.3,             # gap_bank_CONL > 6.3% → 관망 (OOS)
    "robn_pct_max": 2.1,                  # ROBN > +2.1% → 관망 (OOS)

    # ── 3. 기술적 지표 (R7~R9, R16) ─────────────────────────
    "rsi_period": 14,
    "rsi_entry_min": 40,                  # RSI 진입 하한
    "rsi_entry_max": 60,                  # RSI 진입 상한 (85% 승률)
    "rsi_danger_zone": 80,                # RSI > 80 → 진입 금지 (9%)
    "bb_period": 20,
    "bb_std": 2,
    "bb_entry_max": 0.6,                  # %B ≤ 0.6 우대 (87.5%)
    "bb_danger_zone": 1.0,                # %B > 1.0 → 진입 금지 (25.4%)
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "atr_period": 14,
    "atr_high_quantile": 0.75,            # ATR Q4 → 진입 우대 (85.3%)
    "vol_avg_period": 20,
    "vol_entry_min": 1.2,                 # 거래량 하한 (91.2% 승률 구간)
    "vol_entry_max": 2.0,                 # 거래량 상한

    # ── 4. 캘린더 (R15) ─────────────────────────────────────
    "friday_boost": True,                 # 금요일 진입 우대 (88.3%)

    # ── 5. 역발상 진입 — 종목별 차별화 (Study B, §6-1) ──────
    "contrarian_entry_threshold": 0.0,    # 기본: 종목 하락 시 역발상

    # MSTU: R14 발동 시 역발상만 허용. R14 발동 + 순발상 = 절대 금지 (25%)
    "mstu_riskoff_contrarian_only": False,
    # ROBN: R14 발동 시 순발상(모멘텀) 진입 우대 (68.8%)
    "robn_riskoff_momentum_boost": False,
    # CONL: R14 없이 역발상 효과 없음 (46.7%). R14 병행 시만 진입 허용
    "conl_contrarian_require_riskoff": False,
    # AMDL: 금요일 + 역발상 -1.5% 이하 특화 (66.7%)
    "amdl_friday_contrarian_threshold": -1.5,

    # ── 6. 청산 규칙 (R4, R5) ───────────────────────────────
    "take_profit_pct": 5.9,               # 이익실현 기준 gross (중앙값 +5.86%)
    "optimal_hold_days_min": 4,           # 최적 보유 하한
    "optimal_hold_days_max": 5,           # 최적 보유 상한 (초과 시 강제청산)
    "dca_max_daily": 5,                   # 일일 동일종목 매수 상한 (R5)

    # ── 7. 자금 ──────────────────────────────────────────────
    "total_capital": 15_000,              # USD
    "buy_size_large": 0.15,               # 대형 매수 (역발상 진입)
    "buy_size_small": 0.05,               # 소형 매수 (탐색적)
    "daily_new_entry_cap": 0.30,          # 동일 날 신규 진입 총합 ≤ 자본 30% (§12-3)
}

# ============================================================
# D2S 엔진 파라미터 v2 — backtest_d2s_v2.py 전용
# ============================================================
# v1 대비 추가/변경:
#   R17: 충격 V-바운스 포지션 2배 확대
#   R18: BB 하단 돌파 후 조기 손절 (3일 비회복 → 손절)
#   dca_max_layers: 2 (3레이어+ 강력 억제)
D2S_ENGINE_V2 = {
    **D2S_ENGINE,  # v1 파라미터 전체 계승

    # ── v2 신규: R17 충격 V-바운스 확대 ─────────────────────
    "vbounce_bb_threshold": 0.15,         # %B < 0.15 → V-바운스 발동
    "vbounce_crash_threshold": -10.0,     # 종목 -10% 이상 → V-바운스 발동
    "vbounce_score_threshold": 0.87,      # 진입 점수 ≥ 0.87 → V-바운스 발동
    "vbounce_size_multiplier": 2.0,       # buy_size_large × 2.0 (30% 상한)
    "vbounce_size_max": 0.30,             # V-바운스 최대 비중 상한

    # ── v2 신규: R18 BB 하단 돌파 조기 손절 ─────────────────
    "early_stoploss_days": 3,             # BB<0 진입 후 N일 내 미회복 시 손절
    "early_stoploss_recovery": 2.0,       # V-바운스 성공 판단 회복률 +2%

    # ── v2 강화: DCA 레이어 제한 ──────────────────────────────
    "dca_max_layers": 2,                  # 최대 DCA 레이어 (3레이어: 승률 27%)
}

# ============================================================
# D2S 엔진 파라미터 v3 — backtest_d2s_v3.py 전용
# ============================================================
# v2 대비 추가/변경:
#   R19: BB 진입 하드 필터 (%B > 0.30 → 진입 금지, Study G F2: +12.3%p OOS +13.2%p)
#   R20: 레짐 조건부 take_profit (bull=5.0%, bear=6.5%)  ← Optuna #449 역전
#   R21: 레짐 조건부 hold_days   (bull=12일, bear=8일)   ← Optuna #449
#   레짐 감지: SPY streak(5/1) + SMA12(±1.1%/1.5%) + Polymarket BTC(0.55/0.35)
#   근거: Study 5 + D2S v3 Regime Optuna #449 (IS +33.68% / OOS +62.42%, Sharpe 2.22/1.46)
#
# Optuna #449 최적화 기간:
#   IS: 2024-09-18 ~ 2025-05-31  |  OOS: 2025-06-01 ~ 2026-02-17 (no-ROBN 1.5년)
# 추가 검증 완료 (2026-02-25): Study 6~9B
#   Study 6 ROBN 1년: v3 +57.34% vs v2 -1.83%
#   Study 7 IS 구간: v3_IS Sharpe 3.463 (승률 100%)
#   Study 9B weights: v3_current OOS 1위 → weights 확정
D2S_ENGINE_V3 = {
    **D2S_ENGINE_V2,  # v2 파라미터 전체 계승

    # ── v3 신규: R19 BB 진입 하드 필터 (Study G F2) ─────────
    "bb_entry_hard_max": 0.30,            # %B > 0.30 → 진입 금지 (Study G: +12.3%p, OOS +13.2%p)
    "bb_entry_hard_filter": True,         # R19 필터 활성화 여부

    # ── v3 신규: 레짐 감지 활성화 ───────────────────────────
    "regime_enabled": True,               # 레짐 조건부 청산 활성화

    # 레짐 감지 — SPY streak 기반 (Optuna #449)
    "regime_bull_spy_streak": 5,          # 3 → 5 (Optuna: bull 판정 보수화)
    "regime_bear_spy_streak": 1,          # 2 → 1 (Optuna: bear 판정 민감화)

    # 레짐 감지 — SPY SMA 기반 (Optuna #449)
    "regime_spy_sma_period": 12,          # 20 → 12 (단기 SMA 반응성 향상)
    "regime_spy_sma_bull_pct": 1.1,       # 1.0 → 1.1 (Optuna)
    "regime_spy_sma_bear_pct": -1.5,      # -1.0 → -1.5 (Optuna: bear 판정 강화)

    # 레짐 감지 — Polymarket BTC (Optuna #449)
    "regime_btc_bull_threshold": 0.55,    # 0.60 → 0.55 (Risk-on 민감화)
    "regime_btc_bear_threshold": 0.35,    # 0.40 → 0.35 (Risk-off 보수화)

    # ── v3 신규: R20 레짐 조건부 take_profit (Optuna #449 역전) ─
    # ※ Optuna 결과: bull에서도 5.0%가 최적 (빠른 익절이 전반적으로 유리)
    "bull_take_profit_pct": 5.0,          # 5.9 → 5.0 (Optuna 역전 결과)
    "bear_take_profit_pct": 6.5,          # 5.0 → 6.5 (Optuna: bear는 더 기다림)

    # ── v3 신규: R21 레짐 조건부 hold_days (Optuna #449) ────
    "bull_hold_days_max": 12,             # 7 → 12 (Optuna: bull은 더 길게)
    "bear_hold_days_max": 8,              # 4 → 8  (Optuna: bear도 일부 연장)
    "optimal_hold_days_max": 12,          # fallback: bull_hold_days_max 기준

    # ── Optuna #449 최적값 — 시황 필터 오버라이드 ───────────
    "gld_suppress_threshold": 0.5,        # 1.0 → 0.5 (GLD 억제 민감화)
    "btc_up_max": 0.7,                    # 0.75 → 0.7
    "robn_pct_max": 2.5,                  # 2.1 → 2.5
    "gap_bank_conl_max": 5.0,             # 6.3 → 5.0 (쌍둥이 갭 상한 강화)
    "spy_bearish_threshold": -1.25,       # -1.0 → -1.25

    # R14 그라데이션 (Optuna #449)
    "riskoff_spy_min_threshold": -2.8,    # -1.5 → -2.8 (패닉 구간 확장)
    "riskoff_gld_optimal_min": 0.7,       # 0.5 → 0.7
    "riskoff_spy_optimal_max": -0.6,      # -0.5 → -0.6
    "riskoff_consecutive_boost": 2,       # 3 → 2 (연속 리스크오프 기준 완화)
    "riskoff_panic_size_factor": 0.6,     # 0.5 → 0.6

    # market_score 게이트 (Optuna #449)
    "market_score_suppress": 0.45,        # 0.40 → 0.45 (진입 기준 강화)
    "market_score_entry_b": 0.7,          # 0.55 → 0.7
    "market_score_entry_a": 0.8,          # 0.60 → 0.8
    "market_score_weights": {             # w_gld=0.25,w_spy=0.15,w_riskoff=0.2,
        "gld_score":     0.2273,          # w_streak=0.2,w_vol=0.2,w_btc=0.1 → normalize
        "spy_score":     0.1364,          # total=1.1 기준 정규화
        "riskoff_score": 0.1818,
        "streak_score":  0.1818,
        "vol_score":     0.1818,
        "btc_score":     0.0909,
    },

    # ── Optuna #449 최적값 — 기술적 지표 오버라이드 ─────────
    "rsi_entry_min": 37,                  # 40 → 37 (RSI 진입 하한 완화)
    "rsi_entry_max": 69,                  # 60 → 69 (RSI 진입 상한 확대)
    "rsi_danger_zone": 76,                # 80 → 76 (RSI 위험 기준 강화)
    "bb_entry_max": 0.4,                  # 0.6 → 0.4 (BB 진입 상한 강화)
    "bb_danger_zone": 1.1,                # 1.0 → 1.1
    "atr_high_quantile": 0.65,            # 0.75 → 0.65 (ATR Q4 기준 완화)
    "vol_entry_min": 1.1,                 # 1.2 → 1.1
    "vol_entry_max": 3.5,                 # 2.0 → 3.5 (상대 거래량 허용 범위 확대)

    # ── Optuna #449 최적값 — 진입 조건 오버라이드 ───────────
    "contrarian_entry_threshold": -0.5,   # 0.0 → -0.5 (역발상 기준 강화)
    "amdl_friday_contrarian_threshold": -3.0,  # -1.5 → -3.0 (AMDL 금요일 기준 강화)

    # ── Optuna #449 최적값 — V-바운스(R17) 파라미터 ─────────
    "vbounce_bb_threshold": 0.2,          # 0.15 → 0.2 (V-바운스 발동 구간 확대)
    "vbounce_crash_threshold": -12.0,     # -10.0 → -12.0 (더 큰 충격만 발동)
    "vbounce_score_threshold": 0.9,       # 0.87 → 0.9
    "vbounce_size_multiplier": 2.5,       # 2.0 → 2.5

    # ── Optuna #449 최적값 — 조기 손절(R18) 파라미터 ────────
    "early_stoploss_days": 2,             # 3 → 2 (더 빠른 손절 판단)
    "early_stoploss_recovery": 3.0,       # 2.0 → 3.0 (V-바운스 성공 기준 상향)

    # ── Optuna #449 최적값 — 자금/DCA 오버라이드 ────────────
    "buy_size_large": 0.1,                # 0.15 → 0.1 (기본 매수 비중 축소)
    "daily_new_entry_cap": 0.15,          # 0.30 → 0.15 (일일 신규 진입 한도 강화)
    "dca_max_daily": 2,                   # 5 → 2 (일일 동일종목 매수 횟수 제한)
}

# ============================================================
# D2S 엔진 파라미터 v3 No-ROBN — ROBN 제외 1.5년 테스트 전용
# ============================================================
# MSTU 상장일 기준 (2024-09-18 ~): ROBN은 2025-01-31 상장으로 3종목만 사용
# Study 8B 검증 완료 (2026-02-25): 레짐 방법 6종 비교 → v3_3signal 유지 결론
# ticker_weights: ROBN 30% → CONL/MSTU/AMDL 균등 재배분
# twin_pairs: bank_CONL(lead=ROBN) 제거
D2S_ENGINE_V3_NO_ROBN = {
    **D2S_ENGINE_V3,

    # ── ROBN 제외 종목 유니버스 ─────────────────────────────
    "tickers": ["CONL", "MSTU", "AMDL"],
    "ticker_weights": {"CONL": 0.33, "MSTU": 0.34, "AMDL": 0.33},

    # ── twin_pairs: bank_CONL(lead=ROBN) 제거 ─────────────
    "twin_pairs": {
        "coin_MSTU": {"lead": "BITU", "follow": "MSTU"},
        "coin_CONL": {"lead": "BITU", "follow": "CONL"},
        "semi_AMDL": {"lead": "NVDL", "follow": "AMDL"},
    },

    # ── ROBN 전용 규칙 비활성화 ────────────────────────────
    "robn_riskoff_momentum_boost": False,
}

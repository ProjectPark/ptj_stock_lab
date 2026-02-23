"""
D2S 엔진 전용 파라미터 — line_b_taejun/common/params.py에서 분리
"""

# ============================================================
# D2S 엔진 (D2S-Engine) — 실거래 행동 추출 기반 일봉 전략
# 출처: trading_rules_attach_v1.md (953건 실거래 D2S 분석)
# ============================================================
D2S_ENGINE = {
    # ── 종목 유니버스 ─────────────────────────────────────────
    "tickers": ["ROBN", "CONL", "MSTU", "AMDL"],  # 매수 대상만
    "lead_tickers": ["BITU", "NVDL"],               # 선행 지표 (매수 안함)
    "ticker_weights": {  # 실적 기반 종목 비중
        "ROBN": 0.30, "CONL": 0.20, "MSTU": 0.25, "AMDL": 0.25,
    },

    # ── 쌍둥이 페어 ──────────────────────────────────────────
    "twin_pairs": {
        "coin_MSTU": {"lead": "BITU", "follow": "MSTU"},
        "coin_CONL": {"lead": "BITU", "follow": "CONL"},
        "bank_CONL": {"lead": "ROBN", "follow": "CONL"},
        "semi_AMDL": {"lead": "NVDL", "follow": "AMDL"},
    },

    # ── 1. 시황 필터 (R1, R3, R13, R14) ─────────────────────
    "gld_suppress_threshold": 1.0,      # GLD ≥ +1.0% → 매수 억제 (p=0.036)
    "btc_up_max": 0.75,                 # BTC up > 0.75 → 매수 억제 (OOS)
    "confidence_suppress": True,         # confidence_signal → 매수 억제 (p=0.0003)
    "spy_streak_max": 3,                 # SPY 3일+ 연속 상승 → 매수 금지 (27.3%)
    "riskoff_gld_up_spy_down": True,     # GLD↑+SPY↓ → 적극 매수 (86.4%)
    "spy_bearish_threshold": -1.0,       # SPY < -1% → 역발상 매수 (50%)

    # ── 2. 쌍둥이 갭 (R2, OOS Decision Tree) ────────────────
    "gap_bank_conl_max": 6.3,            # gap_bank_CONL > 6.3% → 관망 (OOS)
    "robn_pct_max": 2.1,                # ROBN > +2.1% → 관망 (OOS)

    # ── 3. 기술적 지표 (R7~R9, R16) ─────────────────────────
    "rsi_period": 14,
    "rsi_entry_min": 40,                 # RSI 진입 하한
    "rsi_entry_max": 60,                 # RSI 진입 상한 (85% 승률)
    "rsi_danger_zone": 80,               # RSI > 80 → 진입 금지 (9%)
    "bb_period": 20,
    "bb_std": 2,
    "bb_entry_max": 0.6,                # %B ≤ 0.6 우대 (87.5%)
    "bb_danger_zone": 1.0,              # %B > 1.0 → 진입 금지 (25.4%)
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "atr_period": 14,
    "atr_high_quantile": 0.75,          # ATR Q4 → 진입 우대 (85.3%)
    "vol_avg_period": 20,
    "vol_entry_min": 1.2,               # 거래량 하한 (91.2% 승률 구간, §4-6)
    "vol_entry_max": 2.0,               # 거래량 상한 (91.2% 승률 구간, §4-6)

    # ── 4. 캘린더 (R15) ─────────────────────────────────────
    "friday_boost": True,                # 금요일 진입 우대 (88.3%)

    # ── 5. 진입 규칙 ────────────────────────────────────────
    "contrarian_entry_threshold": 0.0,   # 종목 하락 시 매수 (역발상)

    # ── 6. 청산 규칙 (R4, R5) ───────────────────────────────
    "take_profit_pct": 5.9,              # 이익실현 기준 (중앙값)
    "optimal_hold_days_min": 4,          # 최적 보유 하한
    "optimal_hold_days_max": 7,          # 최적 보유 상한
    "dca_max_daily": 5,                  # 일일 동일종목 매수 상한

    # ── 7. 자금 ──────────────────────────────────────────────
    "total_capital": 15_000,             # USD
    "buy_size_large": 0.15,              # 대형 매수 (역발상 진입)
    "buy_size_small": 0.05,              # 소형 매수 (탐색적)
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
    "vbounce_bb_threshold": 0.15,        # %B < 0.15 → V-바운스 발동 (대수익 평균 0.14)
    "vbounce_crash_threshold": -10.0,    # 종목 -10% 이상 → V-바운스 발동 (대수익 평균 -12.6%)
    "vbounce_score_threshold": 0.87,     # 진입 점수 ≥ 0.87 → V-바운스 발동
    "vbounce_size_multiplier": 2.0,      # buy_size_large × 2.0 (30% 상한)
    "vbounce_size_max": 0.30,            # V-바운스 최대 비중 상한

    # ── v2 신규: R18 BB 하단 돌파 조기 손절 ─────────────────
    "early_stoploss_days": 3,            # BB<0 진입 후 N일 내 미회복 시 손절
    "early_stoploss_recovery": 2.0,      # V-바운스 성공 판단 회복률 +2%

    # ── v2 강화: DCA 레이어 제한 ──────────────────────────────
    "dca_max_layers": 2,                 # 최대 DCA 레이어 (3레이어: 승률 27%)
}

"""
taejun_attach_pattern - 전략 파라미터 정의
==========================================
모든 전략의 조건/임계값을 코드와 분리하여 관리.
출처: kakaotalk_trading_notes_2026-02-19.csv (박태준)
"""

# ============================================================
# 잽모드 SOXL (Jab-SOXL) — 반도체 레버리지 프리마켓 단타
# ============================================================
JAB_SOXL = {
    "poly_ndx_min": 0.51,       # Polymarket NASDAQ 상승 기대 최소
    "gld_min": 0.1,             # GLD 최소 변동률 (%)
    "qqq_min": 0.3,             # QQQ 최소 변동률 (%)
    "soxx_max": -0.2,           # SOXX 최대 변동률 (%) — 이하
    "soxl_max": -0.6,           # SOXL 최대 변동률 (%) — 이하
    "individual": {             # 개별 반도체 종목 최소 변동률 (%)
        "NVDA": 0.9,
        "AMD": 0.9,
        "SMCI": 1.0,
        "KLA": 0.8,
        "AMAT": 0.8,
        "AVGO": 0.55,
        "MPWR": 0.55,
        "TXN": 0.66,
        "ASML": 1.0,
        "LRCX": 0.8,
        "MU": 0.55,
    },
    "target_pct": 0.9,          # 목표 수익률 (%)
    "size": 1.0,                # 매수 비율 (전액)
    "entry_start_kst": (17, 30),  # 진입 시작 (KST)
    "ticker": "SOXL",
}

# ============================================================
# 잽모드 BITU (Jab-BITU) — 비트코인 레버리지 프리마켓 단타
# ============================================================
JAB_BITU = {
    "poly_btc_min": 0.63,       # Polymarket BTC 상승 기대 최소
    "gld_min": 0.1,             # GLD 최소 변동률 (%)
    "bitu_max": -0.4,           # BITU 최대 변동률 (%) — 이하
    "crypto_conditions": {       # 크립토 스팟 최소 변동률 (%)
        "BTC": 0.9,
        "ETH": 0.9,
        "SOL": 2.0,
        "XRP": 5.0,
    },
    "target_pct": 0.9,          # 목표 수익률 (%)
    "size": 1.0,                # 매수 비율 (전액)
    "entry_start_kst": (17, 30),
    "ticker": "BITU",
}

# ============================================================
# 잽모드 TSLL (Jab-TSLL) — 테슬라 레버리지 소액 단타
# ============================================================
JAB_TSLL = {
    "poly_ndx_min": 0.63,       # Polymarket NASDAQ 상승 기대 최소
    "gld_max": 0.3,             # GLD 최대 변동률 (%) — 이하
    "tsll_max": -0.8,           # TSLL 최대 변동률 (%) — 이하
    "tsla_min": 0.5,            # TSLA 최소 변동률 (%)
    "qqq_min": 0.7,             # QQQ 최소 변동률 (%)
    "target_pct": 1.0,          # 목표 수익률 (%)
    "max_amount_krw": 2_000_000,  # 최대 매수 금액 (원)
    "entry_start_kst": (17, 30),
    "ticker": "TSLL",
}

# ============================================================
# 숏 잽모드 SETH (Jab-SETH) — 이더리움 숏 단타
# ============================================================
JAB_SETH = {
    "poly_eth_down_min": 0.12,   # Polymarket ETH 하락 기대 최소 (12%)
    "gld_min": 0.01,             # GLD 최소 변동률 (%)
    "seth_min": 0.0,             # SETH 최소 변동률 (%) — 양전
    "target_pct": 0.5,           # 목표 수익률 (%)
    "size": 1.0,
    "ticker": "SETH",
}

# ============================================================
# VIX 급등 → GDXU (VIX-Gold) — 공포 지수 기반 금광 매수
# ============================================================
VIX_GOLD = {
    "vix_spike_min": 10.0,       # VIX 일간 변동률 최소 (%)
    "poly_down_min": 0.30,       # Polymarket 전반적 하락 기대 (30%)
    "target_pct": 10.0,          # GDXU 목표 수익률 (%)
    "reinvest_ticker": "IAU",    # 매도 후 수익금 재투자 대상
    "ticker": "GDXU",
}

# ============================================================
# S&P500 편입 기업 (SP500-Entry) — 편입 다음 날 매수
# ============================================================
SP500_ENTRY = {
    "poly_ndx_min": 0.51,        # Polymarket NASDAQ 상승 기대 최소
    "gld_block_positive": True,   # GLD 상승시 매수 금지
    "net_income_min": 0.01,       # 재무 순이익 최소 (%)
    "target_pct": 1.5,            # 목표 수익률 (%, 수수료 제외)
    "size": 1.0,
}

# ============================================================
# 저가매수 (Bargain-Buy) — 3년 최고가 대비 폭락시 진입
# ============================================================
BARGAIN_BUY = {
    # 종목별 파라미터 테이블
    # drop_pct: 3년 최고가 대비 하락률 진입 기준 (%)
    # add_drop: 추가매수 트리거 추가 하락률 (%)
    # add_size: 추가매수 비율 (0.5=50%, 1.0=전액)
    # target_pct: 목표 수익률 (%)
    # sell_splits: 분할매도 횟수 (0=전액 매도)
    # reinvest: 매도 수익금 재투자 대상 ("cash"=현금화)
    # split_days: 재투자 분할매수 일수
    # hold_days: 최대 보유 기한 (0=무제한)
    "tickers": {
        "CONL": {
            "drop_pct": -80,
            "add_drop": -3,
            "add_size": 0.5,
            "target_pct": 188,
            "sell_splits": 0,
            "reinvest": "CONL",
            "split_days": 30,
            "hold_days": 0,
        },
        "SOXL": {
            "drop_pct": -90.5,
            "add_drop": -5,
            "add_size": 1.0,
            "target_pct": 320,
            "sell_splits": 6,
            "reinvest": "SOXL",
            "split_days": 30,
            "hold_days": 0,
        },
        "AMDL": {
            "drop_pct": -89,
            "add_drop": -5,
            "add_size": 1.0,
            "target_pct": 40,
            "sell_splits": 6,
            "reinvest": "SOXL",
            "split_days": 30,
            "hold_days": 0,
        },
        "NVDL": {
            "drop_pct": -73,
            "add_drop": -5,
            "add_size": 1.0,
            "target_pct": 200,
            "sell_splits": 6,
            "reinvest": "SOXL",
            "split_days": 30,
            "hold_days": 0,
        },
        "ROBN": {
            "drop_pct": -83,
            "add_drop": -3,
            "add_size": 1.0,
            "target_pct": 200,
            "sell_splits": 6,
            "reinvest": "CONL",
            "split_days": 30,
            "hold_days": 0,
        },
        "ETHU": {
            "drop_pct": -95,
            "add_drop": -10,
            "add_size": 0.1,
            "target_pct": 20,
            "sell_splits": 6,
            "reinvest": "ROBN",
            "split_days": 100,
            "hold_days": 0,
        },
        "BRKU": {
            "drop_pct": -32,
            "add_drop": -3,
            "add_size": 1.0,
            "target_pct": 0.5,
            "sell_splits": 0,
            "reinvest": "cash",
            "split_days": 0,
            "hold_days": 0,
        },
        "NFXL": {
            "drop_pct": -26,
            "add_drop": -20,
            "add_size": 1.0,
            "target_pct": 0.9,
            "sell_splits": 0,
            "reinvest": "cash",
            "split_days": 0,
            "hold_days": 0,
        },
        "SNXX": {
            "drop_pct": -46,
            "add_drop": -10,
            "add_size": 1.0,
            "target_pct": 3,
            "sell_splits": 0,
            "reinvest": "cash",
            "split_days": 0,
            "hold_days": 0,
        },
        "OKLL": {
            "drop_pct": -55,
            "add_drop": -12,
            "add_size": 1.0,
            "target_pct": 3,
            "sell_splits": 0,
            "reinvest": "cash",
            "split_days": 0,
            "hold_days": 0,
        },
        "PLTU": {
            "drop_pct": -44,
            "add_drop": -10,
            "add_size": 1.0,
            "target_pct": 10,
            "sell_splits": 0,
            "reinvest": "cash",
            "split_days": 0,
            "hold_days": 20,
        },
    },
    # 저가매수 금지 조건
    "block_rules": {
        "gld_decline": True,          # 금 하락시 금지
        "poly_ndx_min": 0.49,         # Polymarket 상승 기대 49% 이하시 금지
        "volume_decline_days": 3,     # N일 전후 거래량 감소시 금지
    },
    # 매매 시간
    "trading_hours": {
        "start_kst": (9, 0),          # 오전 9시
        "end_kst": "market_close",    # 장마감까지
    },
}

# ============================================================
# 숏포지션 전환 (Short-Macro) — 부동산 과열시 전면 숏
# ============================================================
SHORT_MACRO = {
    "conditions": {
        "index_ath": True,            # 나스닥/S&P500 역대 최고가
    },
    "action": {
        "sell_all_except": ["GDXU", "IAU", "GLD", "cash"],
        "build_gdxu_pct": 1.0,       # GDXU 100% 구축
        "gdxu_target_pct": 90,       # GDXU +90% 목표
        "reinvest_ticker": "IAU",    # GDXU 매도 후 IAU 매수
    },
    "exit": {
        "exit_type": "full_sell",    # 전액 매도 (분할 아님, 정정 반영)
    },
}

# ============================================================
# 부동산 유동성 감소 이벤트 (REIT-Risk) — 레버리지 매매 중단
# ============================================================
REIT_RISK = {
    "conditions": {
        "reits": ["SK리츠", "TIGER 리츠부동산인프라", "롯데리츠"],
        "reits_up_days": 7,           # 연속 상승 일수
        "reits_avg_min": 0.1,         # 평균 변동률 최소 (%)
        "fear_greed_min": 75,         # 탐욕지수 최소
    },
    "action": {
        "ban_except": ["GDXU"],       # GDXU 제외 레버리지 매매 금지
        "ban_days": 90,               # 금지 기간 (일)
    },
}

# ============================================================
# 섹터 로테이션 (Sector-Rotate) — 4대 섹터 순환 매수
# ============================================================
SECTOR_ROTATE = {
    # 1위 섹터 → 매수 대상 매핑
    "rotation": {
        "bitcoin": {"buy": "SOXL", "interval_days": 3, "qty": 1},
        "semiconductor": {"buy": "ROBN", "interval_days": 7, "qty": 1},
        "bank": {"buy": "GDXU", "interval_days": 14, "qty": 1},
    },
    # 섹터 대표 종목 (수익률 비교용)
    "sector_proxies": {
        "bitcoin": "BITU",
        "semiconductor": "SOXX",
        "bank": "ROBN",
        "gold": "GLD",
    },
    # 리츠 기반 주의사항
    "caution_rules": {
        "gdxu_sk_reit_drop": -1.0,     # SK리츠 -1% → GDXU 매수 조심
        "conl_sk_reit_up_days": 7,      # SK리츠 7일 상승 → CONL 매수 조심
        "long_sk_reit_drop": -1.5,      # SK리츠 -1.5% → 롱포지션 집중
    },
}

# ============================================================
# 조건부 은행주 (Bank-Conditional) — BAC 역전 매매
# ============================================================
BANK_CONDITIONAL = {
    "watch_tickers": ["JPM", "HSBC", "WFC", "RBC", "C"],
    "target_ticker": "BAC",
    "condition": "watch_all_positive_target_negative",
    "amount_krw": 3_000_000,        # 투자 금액 (원)
    "target_pct": 0.5,              # 목표 수익률 (%, 수수료 포함)
    "reinvest": "cash",
}

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
    "vol_entry_min": 0.5,               # 거래량 하한 (콤보용)
    "vol_entry_max": 1.5,               # 거래량 상한 (콤보용)

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

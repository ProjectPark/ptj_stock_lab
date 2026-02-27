"""
taejun_attach_pattern - 전략 파라미터 정의
==========================================
모든 전략의 조건/임계값을 코드와 분리하여 관리.
출처: kakaotalk_trading_notes_2026-02-19.csv (박태준)

Re-export: BaseParams/V5Params 등은 simulation.strategies.params에서
정의되며, line_b_taejun 내부 모듈은 이 파일을 통해 접근한다.
"""

# Re-export dataclass params for line_b_taejun internal use.
# 원본: simulation/strategies/params.py
from simulation.strategies.params import BaseParams, V5Params  # noqa: F401

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
        "KLAC": 0.8,
        "AMAT": 0.8,
        "AVGO": 0.55,
        "MPWR": 0.55,
        "TXN": 0.66,
        "ASML": 1.0,
        "LRCX": 0.8,
        "MU": 0.55,
    },
    "target_pct": 1.15,         # 목표 수익률 (%) — MT_VNQ3
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
        "XRP": 2.5,  # 하루 변동 기준 (장마감 X)
    },
    "target_pct": 1.15,         # 목표 수익률 (%) — MT_VNQ3
    "size": 1.0,                # 매수 비율 (전액)
    "entry_start_kst": (17, 30),
    "ticker": "BITU",
}

# ============================================================
# 잽모드 TSLL (Jab-TSLL) — 테슬라 레버리지 소액 단타
# ============================================================
JAB_TSLL = {
    "poly_ndx_min": 0.63,       # Polymarket NASDAQ 상승 기대 최소
    "gld_max": 0.1,             # GLD 최대 변동률 (%) — 이하 (MT_VNQ3)
    "tsll_max": -0.8,           # TSLL 최대 변동률 (%) — 이하
    "tsla_min": 0.5,            # TSLA 최소 변동률 (%)
    "qqq_min": 0.7,             # QQQ 최소 변동률 (%)
    "target_pct": 1.25,         # 목표 수익률 (%) — MT_VNQ3
    "max_amount_krw": 2_000_000,  # 최대 매수 금액 (원)
    "entry_start_kst": (17, 30),
    "ticker": "TSLL",
}

# ============================================================
# 숏 잽모드 ETQ (Jab-ETQ) — ETH 2x 인버스 단타 (v6 교체)
# ============================================================
# SETH → ETQ 교체: target_pct 0.5 → 0.8 (2x 레버리지 반영)
JAB_ETQ = {
    "poly_down_spread_min": 12.0,  # 최고 하락기대 - 평균 >= 12pp
    "gld_min": 0.01,               # GLD 최소 변동률 (%)
    "etq_min": 0.0,                # ETQ 최소 변동률 (%) — 양전
    "target_pct": 1.05,            # 목표 수익률 (%) — 2x 레버리지 (MT_VNQ3)
    "size": 1.0,
    "ticker": "ETQ",
}

# ============================================================
# VIX 급등 방어모드 (VIX-Gold / Unix(VIX) 방어) — 13-7절
# ============================================================
VIX_GOLD = {
    # 발동 조건
    "vix_spike_min": 10.0,           # Unix(VIX) 일간 변동률 최소 (%)
    # 자금 배분
    "iau_pct": 0.40,                 # IAU 총 투자금의 40%
    "gdxu_pct": 0.30,                # GDXU 총 투자금의 30%
    # GDXU 전술 운용
    "gdxu_min_days": 2,              # 최소 보유 거래일
    "gdxu_max_days": 3,              # 최대 보유 거래일 (강제 청산)
    # IAU 손절
    "iau_stop_pct": -5.0,            # IAU 매수가 대비 -5% 손절
    # IAU 금지 쿨다운 (GDXU -12% 발생 시)
    "gdxu_cooldown_trigger": -12.0,  # GDXU -12% → IAU 금지 시작
    "iau_cooldown_days": 40,         # IAU 금지 거래일 수
    # legacy fields (하위 호환)
    "poly_down_min": 0.30,
    "target_pct": 10.25,          # MT_VNQ3
    "reinvest_ticker": "IAU",
    "ticker": "GDXU",
}

# ============================================================
# S&P500 편입 기업 (SP500-Entry) — 편입 다음 날 매수
# ============================================================
SP500_ENTRY = {
    "poly_ndx_min": 0.51,        # Polymarket NASDAQ 상승 기대 최소
    "gld_block_positive": True,   # GLD 상승시 매수 금지
    "net_income_min": 0.01,       # 재무 순이익 최소 (%)
    "target_pct": 1.75,           # 목표 수익률 (%, 수수료 제외) — MT_VNQ3
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
            "deadline_days": 365,       # 초기 기한 (거래일) — TBD: 사용자 확인 필요
            "deadline_extension": 30,   # 1회 연장 (거래일)
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
            "deadline_days": 365,       # 초기 기한 (거래일) — TBD: 사용자 확인 필요
            "deadline_extension": 30,   # 1회 연장 (거래일)
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
            "add_size": 0,             # MT_VNQ3: 추가매수 없음
            "target_pct": 20,
            "sell_splits": 6,
            "reinvest": "ROBN",
            "split_days": 100,
            "hold_days": 0,
        },
        "BRKU": {
            "drop_pct": -31,           # MT_VNQ3
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
        # SNXX, OKLL 제거 (Q-10)
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
        "reits": ["VNQ"],                                           # MT_VNQ3: Primary REIT
        "reits_kr_aux": ["SK리츠", "TIGER 리츠부동산인프라", "롯데리츠"],  # KR 보조 (결측 시 VNQ만 사용)
        "reits_7d_return_min": 1.0,   # 7일 상승률 1% 이상 (각각)
    },
    "action": {
        "ban_except": ["GDXU"],       # GDXU 제외 레버리지 매매 금지
        "ban_days": 90,               # 금지 기간 (일)
    },
    "cautious_mode": {
        "attack_leverage_pct": 50,    # 공격모드 레버리지 50% 제한
    },
}

# ============================================================
# 섹터 로테이션 (Sector-Rotate) — 4대 섹터 순환 매수
# ============================================================
SECTOR_ROTATE = {
    # 순차 로테이션 (1Y 저가 대비 상승률 기반)
    "rotation_sequence": [
        {
            "name": "bitcoin",
            "proxy": "BITU",
            "buy": "SOXL",
            "activate_pct": 14,       # 1Y 저가 대비 14% 상승시 시작
            "deactivate_pct": 60,     # 60% 상승시 전액매도 → 다음
            "interval_days": 3,
            "qty": 1,
        },
        {
            "name": "semiconductor",
            "proxy": "SOXX",
            "buy": "ROBN",
            "activate_pct": 13,
            "deactivate_pct": 80,
            "interval_days": 7,
            "qty": 1,
        },
        {
            "name": "bank",
            "proxy": "ROBN",
            "buy": "GDXU",
            "activate_pct": 10,
            "deactivate_pct": 50,
            "interval_days": 14,
            "qty": 1,
        },
        {
            "name": "gold",
            "proxy": "GLD",
            "action": "cash",         # 현금 보유
            "activate_pct": 10,
            "fx_hedge": {
                "krw_up_threshold": 7.1,    # 환율 7.1% 상승 → 50% 원화 환전
                "krw_down_threshold": 7.1,  # 환율 7.1% 하락 → 달러 환전
                "hedge_pct": 50,
            },
        },
    ],
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
    "target_pct": 1.05,             # 목표 수익률 (%, 수수료 포함) — MT_VNQ3
    "reinvest": "cash",
}

# ============================================================
# 이머전시 모드 (Emergency Mode) — Polymarket 30pp+ 급변 대응
# ============================================================
EMERGENCY_MODE = {
    "poly_swing_min": 30.0,       # Polymarket 30pp 이상 변동
    "target_net_pct": 0.9,        # 수수료 제외 목표 (0.25% 매수 + 0.25% 매도)
    # 기본: 수익중 포지션 즉시 매도
    "base": {
        "action": "sell_profitable",
    },
    # 모드 1: BTC 급등 → BITU 매수
    "btc_surge": {
        "ticker": "BITU",
        "poly_key": "btc_up",
        "direction": "up",
        "combined_swing_min": 30.0,  # Poly + BTC 합산 30%+
    },
    # 모드 2: NASDAQ 급등 → SOXL 매수
    "ndx_bull": {
        "ticker": "SOXL",
        "poly_key": "ndx_up",
        "direction": "up",
        "combined_swing_min": 30.0,
    },
    # 모드 3: NASDAQ 급락 → SOXS 매수
    "ndx_bear": {
        "ticker": "SOXS",
        "poly_key": "ndx_up",
        "direction": "down",
        "combined_swing_min": 30.0,
    },
}

# ============================================================
# Polymarket 데이터 품질 필터 (Poly-Quality)
# ============================================================
POLY_QUALITY = {
    "min_prob": 0.02,           # 1% 이하 제외
    "max_prob": 0.99,           # 99% 이상 제외
    "min_volatility_hours": 5,  # 5시간 미만 변동 제외
    "stale_pause": True,        # 미갱신시 해당 조건 정지
}

# ============================================================
# 자산 모드 시스템 (Asset Mode) — 공격/방어/조심/이머전시
# ============================================================
ASSET_MODE = {
    "attack_strategies": [
        "jab_soxl", "jab_bitu", "jab_tsll", "jab_etq",
        "bargain_buy", "vix_gold", "sp500_entry", "bank_conditional",
        "short_macro", "emergency_mode",
        "crash_buy", "soxl_independent",          # v5 신규
    ],
    "defense_strategies": ["sector_rotate"],
    "cautious_leverage_pct": 50,
}

# ============================================================
# 서킷 브레이커 파라미터 (Circuit Breaker) — CB-1~6 (1절)
# ============================================================
CIRCUIT_BREAKER = {
    # CB-1: VIX 급등 → 7거래일 신규 매수 금지
    "cb1_vix_min": 6.0,          # VIX 일간 +6% 이상
    "cb1_days": 7,               # 7거래일 신규 매수 금지
    # CB-2: GLD 급등 → 3거래일 신규 매수 금지
    "cb2_gld_min": 3.0,          # GLD +3% 이상
    "cb2_days": 3,               # 3거래일 신규 매수 금지
    # CB-3: BTC 급락 → 조건 해제 시까지
    "cb3_btc_drop": -5.0,        # BTC -5% 이상 하락
    # CB-4: BTC 급등 → 추격매수 금지
    "cb4_btc_surge": 5.0,        # BTC +5% 이상 상승
    # CB-5: 금리 상승 우려 → 모든 신규 매수 금지 + 레버리지 3일 추가 대기
    "cb5_rate_hike_prob": 0.50,  # Polymarket 금리상승 확률 50%
    "cb5_lev_cooldown_days": 3,  # 해제 후 레버리지 ETF 추가 대기 (거래일)
    "cb5_leverage_tickers": [    # 레버리지 ETF 목록 (단계적 해제 대상)
        "BITU", "MSTU", "IRE", "ROBN", "CONL", "ETHU",
        "XXRP", "SOLT", "BRKU", "SOXL", "GDXU", "TSLL",
    ],
    # CB-6: 과열 종목 전환 (+20% → 비레버리지 대체)
    "cb6_surge_min": 20.0,       # +20% 이상 → 과열
    "cb6_recovery_pct": -10.0,   # 고점 대비 -10% → 자동 복귀
    "cb6_mapping": {             # 레버리지 → 비레버리지 매핑
        "SOXL": "SOXX",
        "CONL": "COIN",
        "IRE": "IREN",
        "MSTU": None,            # 대체 없음 → 매수 전면 금지
    },
}

# ============================================================
# ATR 손절 파라미터 (Stop Loss) — 5-1~5-4절
# ============================================================
STOP_LOSS = {
    # ATR 기반 손절
    "atr_period": 14,
    "atr_multiplier": 1.5,           # 일반: 진입가 - 1.5 × ATR
    "atr_multiplier_bullish": 2.5,   # 강세장(Poly NDX>=70%): - 2.5 × ATR
    # 고변동성 손절 — 레버리지 배수별 차등 (5-4절)
    "high_vol_lookback": 5,          # 최근 5거래일 관찰
    "high_vol_min_count": 2,         # 2회 이상 발생 시 발동
    "high_vol_threshold": {          # 레버리지별 일간 등락률 절대값 기준
        1: 10.0,   # 1x: 10% 이상
        2: 15.0,   # 2x: 15% 이상
        3: 20.0,   # 3x: 20% 이상
    },
    "high_vol_stop_pct": {           # 레버리지별 고정 손절 (진입가 대비)
        1: -4.0,   # 1x: -4%
        2: -6.0,   # 2x: -6%
        3: -8.0,   # 3x: -8%
    },
    # 종목별 레버리지 배수 매핑
    "leverage_map": {
        # 1x
        "COIN": 1, "SOXX": 1, "IREN": 1, "GLD": 1, "IAU": 1,
        "SPY": 1, "QQQ": 1, "BAC": 1, "JPM": 1,
        # 2x
        "MSTU": 2, "IRE": 2, "CONL": 2, "BITU": 2, "ROBN": 2,
        "ETHU": 2, "XXRP": 2, "SOLT": 2, "BRKU": 2, "TSLL": 2,
        # 3x
        "SOXL": 3, "GDXU": 3,
    },
}

# ============================================================
# 급락 역매수 파라미터 (Crash Buy) — 5-5절
# ============================================================
CRASH_BUY = {
    "tickers": ["SOXL", "CONL", "IRE"],  # 대상 종목
    "drop_trigger": -30.0,               # 당일 -30% 이상 하락
    "luld_count_min": 3,                 # LULD 거래중단 3회 이상
    "buy_pct": 0.95,                     # 총 투자금의 95% 매수
    "entry_et_hour": 15,                 # 진입 시각 (ET 15:55)
    "entry_et_min": 55,
}

# ============================================================
# SOXL 독립 매매 파라미터 (SOXL Independent) — 4-7절
# ============================================================
SOXL_INDEPENDENT = {
    "soxx_min": 2.0,          # SOXX 당일 +2% 이상
    "adx_min": 20,            # ADX(14) >= 20
    "initial_usd": 2250,      # 초기 진입 금액
    "dca_usd": 750,           # 물타기 1회 금액
    "dca_max": 4,             # 물타기 최대 4회
    "dca_drop_pct": -0.5,     # 물타기 트리거 (1차 진입가 대비 -0.5% 간격)
    "sell_tp_pct": 5.0,       # 고정 익절 +5% (60% 물량)
    "sell_tp_ratio": 0.60,    # 고정 익절 물량 비율
    "sell_momentum_ratio": 0.40,   # 즉시 매도 물량 비율
    "soxx_weak_pct": 0.5,     # SOXX 모멘텀 약화 기준 (+0.5% 미만 or EMA음수)
    "ticker": "SOXL",
}

# ============================================================
# 횡보장 감지 파라미터 (v5 기술지표 6개) — 2-2절
# ============================================================
SIDEWAYS_DETECTOR = {
    # 지표 설정
    "atr_period": 14,
    "atr_ma_period": 20,
    "atr_drop_pct": 0.20,          # 20일 평균 대비 20% 이상 감소
    "volume_ma_period": 20,
    "volume_drop_pct": 0.30,       # 20일 평균 대비 30% 이상 감소
    "ema_period": 20,
    "ema_slope_days": 5,           # 5일 기준 기울기
    "ema_slope_pct": 0.001,        # |기울기| <= 0.1%
    "rsi_period": 14,
    "rsi_lo": 45,                  # RSI 박스권 하한
    "rsi_hi": 55,                  # RSI 박스권 상한
    "bb_period": 20,
    "bb_std": 2,
    "bb_quantile_period": 60,
    "bb_quantile_pct": 0.20,       # 60일 하위 20%
    "hl_max_pct": 2.0,             # 고저차이 2% 이하
    # 판정
    "min_signals": 3,              # 6개 중 3개 이상 충족 → 횡보장
    "tickers": ["SPY", "QQQ"],     # 평가 기준 종목
}

# ============================================================
# Bear Regime 감지기 파라미터 — todd_fuck_v1
# ============================================================
# 급락 체제 선언 조건 (AND):
#   - btc_up < 0.43  (시장 컨센서스 비관, v2 OOS: 0.38→0.43)
#   - btc_monthly_dip > 0.30 (이달 저점 도달 확률 높음)
# 체제 해제: btc_up > 0.57 회복 (히스테리시스, v2 OOS: 0.50→0.57)
# 체제 ON 시: 롱 레버리지 50% 축소 + 인버스 ETF 진입 허용
# ============================================================
BEAR_REGIME = {
    # 진입 조건
    # v2 OOS 최적화(7d-stk4): btc_up_min 0.38→0.43, recovery 0.50→0.57
    "btc_up_min": 0.43,            # btc_up 7d rolling < 43% → 급락 체제 진입
    "monthly_dip_min": 0.30,       # btc_monthly_dip > 30% → 하락 압력 확인
    # 해제 조건 (히스테리시스)
    "recovery_threshold": 0.57,    # btc_up 7d rolling > 57% 회복 시 체제 해제
    # 소프트 경보 (스코어 기반)
    "regime_score_warn": 0.40,     # score > 0.40 → 경보 (WARN)
    "regime_score_bear": 0.55,     # score > 0.55 → 강한 약세 (STRONG_BEAR)
    # 4단계 롱 레버리지 (v6 — NORMAL/WARN/BEAR/STRONG_BEAR)
    "warn_leverage": 0.8,          # WARN: unit_mul × 0.8
    "cautious_leverage": 0.5,      # BEAR: unit_mul × 0.5
    "strong_bear_leverage": 0.3,   # STRONG_BEAR: unit_mul × 0.3
    # 인버스 ETF 매핑
    "btc_bear_ticker": "BITI",     # BTC 약세 시 인버스
    "ndx_bear_ticker": "SOXS",     # NDX 약세 시 인버스
    # 인버스 진입 조건
    "inverse_btc_up_max": 0.35,    # btc_up < 35% → BITI 진입
    "inverse_ndx_up_max": 0.38,    # ndx_up < 38% → SOXS 진입
    # 인버스 투입 비율 (체제별)
    "inv_size_bear": 0.3,          # BEAR: 잔여 현금의 30%
    "inv_size_strong_bear": 0.8,   # STRONG_BEAR: 잔여 현금의 80%
    # 🔍 인버스 청산 조건 (Optuna 탐색 대상)
    "inv_target_pct": 8.0,         # 목표수익률 (%) — 탐색 [3.0, 15.0]
    "inv_stop_loss_pct": 5.0,      # 손절폭 (%, 양수로 저장) — 탐색 [3.0, 10.0]
    "inv_max_hold_days": 30,       # 보유 기한 (일) — 탐색 [10, 60]
    # 인버스 ↔ 롱 섹터 매핑 (STRONG_BEAR 50% 축소 대상)
    # TODO: Orchestrator 구현 후 활성화 (현재 params만 정의)
    "inverse_sector_map": {
        "BITI": ["IREN", "MSTU", "BITU", "CONL", "ROBN"],
        "SOXS": ["SOXL"],
    },
}

# ============================================================
# Polymarket 확률 기반 포지션 크기 연속화 — todd_fuck_v1
# ============================================================
# 이진 게이트(ON/OFF) → btc_up 연속 스케일로 unit_mul 조정.
#
# btc_up 구간:
#   < 0.45  → 롱 진입 차단 (BearRegime 연동)
#   0.45~0.55 → unit_mul × 0.7 (축소)
#   0.55~0.70 → unit_mul × 1.0 (기본)
#   > 0.70  → unit_mul × 1.5 (확대)
# ============================================================
POLY_POSITION_SCALE = {
    "enabled": True,
    # btc_up 구간 경계 [하한1, 하한2, 상한1]
    "btc_up_thresholds": [0.45, 0.55, 0.70],
    # 각 구간별 unit_mul 배율 [차단, 축소, 기본, 확대]
    "unit_mul_factors": [0.0, 0.7, 1.0, 1.5],
}

# ============================================================
# 급등 스윙 모드 파라미터 (Swing Mode) — 13절
# ============================================================
SWING_MODE = {
    # 진입 조건
    "trigger_pct": 15.0,          # +15% 이상 급등
    "tickers": [                   # 대상 종목
        "SOXL", "SOXX", "CONL", "COIN", "IRE", "IREN",
    ],
    # 1단계: 급등 종목 보유
    "phase1_pct": 0.90,            # 90% 투자
    "phase1_months": 3,            # 3개월
    "phase1_stop_pct": -15.0,      # 진입가 대비 -15% 손절
    "phase1_atr_multiplier": 1.5,  # ATR 1.5x 손절
    # 2단계: IAU 안전자산
    "phase2_ticker": "IAU",
    "phase2_pct": 0.70,            # 70% 투자
    "phase2_months": 5,            # 5개월
    "phase2_stop_pct": -5.0,       # -5% 손절
}

# ──────────────────────────────────────────────────────────
# ENGINE_CONFIG — CI-0 v0.2 오케스트레이터 설정값
# 출처: MT_VNQ2.md L2842~2960 (CI-0 v0.2)
# ──────────────────────────────────────────────────────────
ENGINE_CONFIG = {
    # CI-0-3: 주문 상태머신
    # order_expiry_bars: deprecated — order_ttl_sec으로 대체 (MT_VNQ3 §3-3)
    "order_ttl_sec": 120,          # MT_VNQ3 §3-3: 주문 TTL 2분 확정
    "order_retry_max": 3,          # 재시도 최대 횟수
    "order_slip_pct": 0.001,       # 재시도 시 가격 보정 (0.1%)

    # MT_VNQ3 신규 파라미터
    "fill_window_sec": 10,         # MT_VNQ3 §5: Fill Window 10초 룰
    "bid_slip_max_pct": 0.002,     # MT_VNQ3 §7: 매도 즉시성 bid 기반 0.2% 이내
    "shrink_hard_limit": 1.001,    # MT_VNQ3 §1: 100.1% 이상이면 BUY_STOP
    "position_mode": "effective_confirmed",  # MT_VNQ3: 포지션 분리 모드 (effective/confirmed)

    # CI-0-4: "즉시 행동" 정의
    "signal_bar_close": True,      # True = 신호 Bar 종가 기준 지정가
    "api_delay_tolerance_s": 60,   # API 지연 허용 범위 (1분)

    # CI-0-7: session_day 기준
    "session_roll_hour": 17,       # KST 17:30에 세션 날짜 전환
    "session_roll_minute": 30,

    # CI-0-8: M5 차감/초기화
    "m5_reset_on_session_start": True,  # 세션 시작(17:30 KST)에 카운트 초기화
    "m5_deduct_on_fill": True,          # FILLED 완료 시 차감

    # CI-0-9: ProfitDistributor
    "profit_dist_hour_kst": 6,      # 장 마감 후 KST 06:05 실행
    "profit_dist_minute_kst": 5,

    # CI-0-10: 미정 조건 기본값
    "p5_44poly_btc_enabled": False,  # P-5: 44POLYMARKET BTC 비활성화 (정의 전)
    "p8_m80_min_score": 35,          # P-8: M80 최소 매수 점수 (보수적 35점)
    "max_quote_age_s": 120,          # 데이터 freshness 기준 (2분)
    "no_chase_buy": True,              # MT_VNQ3 §6: 추격매수 금지
    "m5_t5_reserve_timeout_sec": 10,   # MT_VNQ3 §10: T5+ 예약 타임아웃
}

"""
product/engine_v4/params.py
============================
v4 엔진 확정 파라미터 (Study 2+5 최적화 반영)
config.py의 V4_* 전체를 DEFAULT_* 상수로 재정의.

Study 결과 반영:
- CB 파라미터: Optuna v4_phase1 + v4_study5 확정값
- 스윙 파라미터: Optuna v4_study2 확정값
- 매도 파라미터: Optuna v4_study5 확정값

사용법:
    from product.engine_v4.params import DEFAULT_PARAMS
    # 또는 개별 상수
    from product.engine_v4.params import DEFAULT_CB_BTC_SURGE_PCT
"""
from __future__ import annotations

# ============================================================
# 종목 / 페어 정의
# ============================================================

DEFAULT_TICKERS: dict[str, dict] = {
    "GLD":  {"name": "금 ETF",          "category": "gold",               "exchange": "AMS"},
    "BITU": {"name": "비트코인 2x",     "category": "twin_coin",          "exchange": "AMS"},
    "MSTU": {"name": "스트래티지 2x",   "category": "twin_coin",          "exchange": "AMS"},
    "IRE":  {"name": "아이렌 2x",       "category": "twin_coin",          "exchange": "NAS"},
    "ROBN": {"name": "로빈후드 2x",     "category": "twin_bank",          "exchange": "AMS"},
    "CONL": {"name": "코인베이스 2x",   "category": "twin_bank",          "exchange": "NAS"},
    "ETHU": {"name": "이더리움 2x",     "category": "conditional",        "exchange": "AMS"},
    "XXRP": {"name": "리플 2x",         "category": "conditional",        "exchange": "AMS"},
    "SOLT": {"name": "솔라나 2x",       "category": "conditional",        "exchange": "NAS"},
    "COIN": {"name": "코인베이스",       "category": "conditional_target", "exchange": "NAS"},
    "BRKU": {"name": "버크셔 2x",       "category": "bearish",            "exchange": "NAS"},
    "SPY":  {"name": "S&P 500 ETF",     "category": "market",             "exchange": "AMS"},
    "QQQ":  {"name": "나스닥 100 ETF",  "category": "market",             "exchange": "NAS"},
}

DEFAULT_TWIN_PAIRS: dict[str, dict] = {
    "coin":  {"lead": "BITU", "follow": ["MSTU", "IRE"], "label": "코인 (BTC ↔ MSTR/IREN)"},
    "bank":  {"lead": "ROBN", "follow": ["CONL"],        "label": "은행 (HOOD ↔ COIN)"},
}

DEFAULT_CONDITIONAL_TRIGGERS: list[str] = ["ETHU", "XXRP", "SOLT"]
DEFAULT_CONDITIONAL_TARGET: str = "COIN"
DEFAULT_BEARISH_TICKERS: list[str] = ["BRKU"]

# CB 과열 대상 / 레버리지 종목
DEFAULT_CB_OVERHEAT_TICKERS: list[str] = ["SOXL", "CONL", "IRE", "MSTU"]
DEFAULT_LEVERAGED_TICKERS: list[str] = [
    "BITU", "MSTU", "IRE", "ROBN", "CONL", "ETHU", "XXRP", "SOLT", "BRKU", "SOXL", "GDXU"
]

# ============================================================
# 자금 파라미터
# ============================================================

DEFAULT_TOTAL_CAPITAL: float = 15_000.0          # 총 투자금 ($)
DEFAULT_INITIAL_BUY: float = 2_250.0             # 초기 진입 금액 ($)
DEFAULT_DCA_BUY: float = 750.0                   # 물타기 1회 금액 ($)
DEFAULT_DCA_MAX_COUNT: int = 4                   # 물타기 최대 횟수
DEFAULT_MAX_PER_STOCK: float = 5_250.0           # 종목당 최대 투입 ($)

# ============================================================
# 매매 파라미터 (기본)
# ============================================================

DEFAULT_PAIR_GAP_ENTRY_THRESHOLD: float = 2.2   # 쌍둥이 ENTRY 갭 기준 (%)
DEFAULT_SPLIT_BUY_INTERVAL_MIN: int = 20         # 중복 주문 쿨타임 (분)
DEFAULT_MAX_DAILY_TRADES_PER_STOCK: int = 1      # 종목당 일일 최대 트레이드

# ── 매도 공통 ──────────────────────────────────────────────
DEFAULT_PAIR_GAP_SELL_THRESHOLD: float = 9.0     # 쌍둥이 매도 갭 기준 (%) # Optuna v4_study5 best  # was: 8.8
DEFAULT_PAIR_SELL_FIRST_PCT: float = 0.80        # 1차 매도 비율 (80%)
DEFAULT_PAIR_SELL_REMAINING_PCT: float = 0.30    # 잔여분 분할매도 비율 (30%)
DEFAULT_PAIR_SELL_INTERVAL_MIN: int = 5          # 분할매도 간격 (분)

# ── 조건부 매매 매도 ───────────────────────────────────────
DEFAULT_COIN_TRIGGER_PCT: float = 4.5            # COIN 매수 트리거 (%)
DEFAULT_COIN_SELL_PROFIT_PCT: float = 5.0        # COIN 일반 매도 순수익 기준 (%)
DEFAULT_COIN_SELL_BEARISH_PCT: float = 0.3       # COIN 하락장 즉시 매도 순수익 기준 (%)
DEFAULT_CONL_TRIGGER_PCT: float = 4.5            # CONL 매수 트리거 (%)
DEFAULT_CONL_SELL_PROFIT_PCT: float = 2.8        # CONL 수익 실현 매도 순수익 기준 (%)
DEFAULT_CONL_SELL_AVG_PCT: float = 1.0           # CONL 매도 — 트리거 평균 하한 (%)

# ── 시간 손절 ──────────────────────────────────────────────
DEFAULT_MAX_HOLD_HOURS: float = 5.0              # 기본 시간 손절 (시간)
DEFAULT_TAKE_PROFIT_PCT: float = 4.0             # 강세장 연장 시 즉시 매도 기준 (%)

# ============================================================
# 진입 시간 파라미터
# ============================================================

DEFAULT_ENTRY_CUTOFF_HOUR: int = 10              # 매수 마감 시간 (ET hour)
DEFAULT_ENTRY_CUTOFF_MINUTE: int = 0             # 매수 마감 시간 (ET minute)
DEFAULT_ENTRY_DEFAULT_START_HOUR: int = 8        # 기본 매수 시작 시간 (ET)
DEFAULT_ENTRY_DEFAULT_START_MINUTE: int = 0
DEFAULT_ENTRY_EARLY_START_HOUR: int = 3          # 조기 진입 시작 시간 (ET 03:30)
DEFAULT_ENTRY_EARLY_START_MINUTE: int = 30
DEFAULT_ENTRY_TREND_ADX_MIN: float = 25.0        # 조기 진입 ADX 기준
DEFAULT_ENTRY_VOLUME_RATIO_MIN: float = 1.5      # 조기 진입 거래량 배수 기준
DEFAULT_ENTRY_EMA_PERIOD: int = 20               # 조기 진입 EMA 기간
DEFAULT_ENTRY_EMA_SLOPE_LOOKBACK: int = 5        # 조기 진입 EMA 기울기 계산 lookback
DEFAULT_CONDITIONAL_EXEMPT_CUTOFF: bool = True   # 조건부 진입 마감 면제 여부

# ============================================================
# 횡보장 감지 파라미터
# ============================================================

DEFAULT_SIDEWAYS_ENABLED: bool = True
DEFAULT_SIDEWAYS_MIN_SIGNALS: int = 3            # 횡보장 판정 최소 충족 지표 수 (5개 중)
DEFAULT_SIDEWAYS_EVAL_INTERVAL_MIN: int = 30     # 횡보장 재평가 간격 (분)
DEFAULT_SIDEWAYS_POLY_LOW: float = 0.40          # 횡보장 Polymarket 확률 하한
DEFAULT_SIDEWAYS_POLY_HIGH: float = 0.60         # 횡보장 Polymarket 확률 상한
DEFAULT_SIDEWAYS_GLD_THRESHOLD: float = 0.3      # 횡보장 GLD |등락률| 기준 (%)
DEFAULT_SIDEWAYS_GAP_FAIL_COUNT: int = 2         # 횡보장 갭 수렴 실패 횟수 기준
DEFAULT_SIDEWAYS_TRIGGER_FAIL_COUNT: int = 2     # 횡보장 트리거 불발 횟수 기준
DEFAULT_SIDEWAYS_INDEX_THRESHOLD: float = 0.5    # 횡보장 SPY/QQQ |등락률| 기준 (%)
# v4 신규 지표
DEFAULT_SIDEWAYS_ATR_DECLINE_PCT: float = 20.0   # ATR 하락 기준 (%)
DEFAULT_SIDEWAYS_VOLUME_DECLINE_PCT: float = 30.0 # 거래량 하락 기준 (%)
DEFAULT_SIDEWAYS_EMA_SLOPE_MAX: float = 0.1      # EMA 기울기 상한 (%)
DEFAULT_SIDEWAYS_RSI_LOW: float = 45.0           # RSI 횡보 하한
DEFAULT_SIDEWAYS_RSI_HIGH: float = 55.0          # RSI 횡보 상한
DEFAULT_SIDEWAYS_BB_WIDTH_PERCENTILE: float = 20.0 # BB폭 하위 백분위 기준
DEFAULT_SIDEWAYS_RANGE_MAX_PCT: float = 2.0      # 장중 고저폭 상한 (%)
DEFAULT_SIDEWAYS_EMA_SLOPE_LOOKBACK_DAYS: int = 5
DEFAULT_SIDEWAYS_BB_PERCENTILE_WINDOW_DAYS: int = 60

# ============================================================
# 서킷브레이커 파라미터
# ============================================================

# ─ CB-1: VIX 급등 ─────────────────────────────────────────
DEFAULT_CB_VIX_SPIKE_PCT: float = 3.0            # VIX 급등 기준 # Optuna v4_phase1 best  # was: 6.0
DEFAULT_CB_VIX_COOLDOWN_DAYS: int = 13           # VIX 급등 시 차단 기간 (거래일) # Optuna v4_phase1 best  # was: 7

# ─ CB-2: GLD 급등 ─────────────────────────────────────────
DEFAULT_CB_GLD_SPIKE_PCT: float = 3.0            # GLD 급등 기준 (+3%)
DEFAULT_CB_GLD_COOLDOWN_DAYS: int = 3            # GLD 급등 시 차단 기간 (거래일)

# ─ CB-3: BTC 급락 ─────────────────────────────────────────
DEFAULT_CB_BTC_CRASH_PCT: float = -6.0           # BTC 급락 기준 # Optuna v4_study5 best  # was: -5.0

# ─ CB-4: BTC 급등 ─────────────────────────────────────────
DEFAULT_CB_BTC_SURGE_PCT: float = 13.5           # BTC 급등 기준 # Optuna v4_study5 best  # was: 5.0

# ─ CB-5: 금리 상승 ────────────────────────────────────────
DEFAULT_CB_RATE_HIKE_PROB_PCT: float = 50.0      # 금리 상승 우려 기준 (%)

# ─ CB-6: 과열 ─────────────────────────────────────────────
DEFAULT_CB_OVERHEAT_PCT: float = 20.0            # 과열 전환 기준 (+20%)
DEFAULT_CB_OVERHEAT_RECOVERY_PCT: float = -10.0  # 과열 해제 기준 (고점 대비 조정 proxy)
DEFAULT_CB_OVERHEAT_SWITCH_MAP: dict[str, str | None] = {
    "SOXL": "SOXX",
    "CONL": "COIN",
    "IRE": "IREN",
    "MSTU": None,
}

# ─ 고변동성 파라미터 ─────────────────────────────────────
DEFAULT_HIGH_VOL_MOVE_PCT: float = 10.0          # 고변동성 판정 변동률 기준 (|%|)
DEFAULT_HIGH_VOL_HIT_COUNT: int = 2              # 고변동성 판정 누적 횟수
DEFAULT_HIGH_VOL_STOP_LOSS_PCT: float = -4.0     # 고변동성 고정 손절 (%)

# ============================================================
# CONL 조건부 진입 필터
# ============================================================

DEFAULT_CONL_ADX_MIN: float = 10.0               # CONL 조건부 진입 ADX 하한 # Optuna v4_phase1 best  # was: 18.0
DEFAULT_CONL_EMA_PERIOD: int = 20                # CONL EMA 기간
DEFAULT_CONL_EMA_SLOPE_LOOKBACK: int = 5         # EMA 기울기 계산용 lookback bars
DEFAULT_CONL_EMA_SLOPE_MIN_PCT: float = 0.0      # EMA 기울기 하한 (%)
DEFAULT_CONL_FIXED_BUY: float = 6_750.0          # CONL 조건부 고정 진입 금액 ($)
DEFAULT_CONL_DCA_ENABLED: bool = False           # CONL 물타기 허용 여부 (v4: 금지)

# ============================================================
# v4 고정 익절 / 갭 수렴 매도
# ============================================================

DEFAULT_PAIR_FIXED_TP_STOCKS: list[str] = ["SOXL", "CONL", "IRE"]  # 40/60 고정 익절 대상
DEFAULT_PAIR_IMMEDIATE_SELL_PCT: float = 0.40    # 갭 수렴 즉시 매도 비율 # Optuna v4_study5 best  # was: 0.80
DEFAULT_PAIR_FIXED_TP_PCT: float = 6.5           # 고정 익절 기준 (%) # Optuna v4_study5 best  # was: 5.0

# ============================================================
# 급락 역매수 (Crash Buy)
# ============================================================

DEFAULT_CRASH_BUY_THRESHOLD_PCT: float = -40.0   # 급락 역매수 트리거 (%)
DEFAULT_CRASH_BUY_TIME_HOUR: int = 15            # 급락 역매수 시각 (ET)
DEFAULT_CRASH_BUY_TIME_MINUTE: int = 55
DEFAULT_CRASH_BUY_WEIGHT_PCT: float = 95.0       # 급락 역매수 비중 (%)
DEFAULT_CRASH_BUY_STOCKS: list[str] = ["SOXL", "CONL", "IRE"]
DEFAULT_CRASH_BUY_FLAT_BAND_PCT: float = 0.1     # 다음날 시가 보합 판정 밴드 (%)
DEFAULT_CRASH_BUY_FLAT_OBSERVE_MIN: int = 30     # 보합 시 관찰 시간 (분)

# ============================================================
# 스윙 매매 파라미터
# ============================================================

DEFAULT_SWING_TRIGGER_PCT: float = 27.5          # 스윙 진입 트리거 (%) # Optuna v4_study2 best  # was: 15.0
DEFAULT_SWING_ELIGIBLE_TICKERS: list[str] = ["SOXL", "SOXX", "CONL", "COIN", "IRE", "IREN"]

# ─ Stage 1 (모멘텀) ────────────────────────────────────────
DEFAULT_SWING_STAGE1_WEIGHT_PCT: float = 90.0    # 스윙 Stage1 진입 비중 (%)
DEFAULT_SWING_STAGE1_HOLD_DAYS: int = 21         # Stage1 최대 보유 기간 (거래일) # Optuna v4_study2 best  # was: 63
DEFAULT_SWING_STAGE1_ATR_MULT: float = 2.5       # Stage1 ATR 손절 배수 # Optuna v4_study2 best  # was: 1.5
DEFAULT_SWING_STAGE1_DRAWDOWN_PCT: float = -11.0 # Stage1 drawdown 손절 (%) # Optuna v4_study2 best  # was: -15.0

# ─ Stage 2 (모멘텀 → GLD 전환) ───────────────────────────
DEFAULT_SWING_STAGE2_GLD_TICKER: str = "GLD"
DEFAULT_SWING_STAGE2_WEIGHT_PCT: float = 70.0    # Stage2 진입 비중 (%)
DEFAULT_SWING_STAGE2_HOLD_DAYS: int = 105        # Stage2 최대 보유 기간 (거래일)
DEFAULT_SWING_STAGE2_STOP_PCT: float = -5.0      # Stage2 손절 기준 (%)

# ─ VIX 트리거 스윙 ───────────────────────────────────────
DEFAULT_SWING_VIX_STAGE1_WEIGHT_PCT: float = 80.0
DEFAULT_SWING_VIX_STAGE1_HOLD_DAYS: int = 105
DEFAULT_SWING_VIX_STAGE2_COOLDOWN_DAYS: int = 63

# ============================================================
# Polymarket / 시장 모드
# ============================================================

DEFAULT_POLYMARKET_BULLISH_THRESHOLD: float = 70.0  # 강세장 모드 진입 기준 (%)
DEFAULT_BEARISH_POLYMARKET_THRESHOLD: float = 20.0  # 하락장 진입 기준 (%)
DEFAULT_BEARISH_DROP_THRESHOLD: float = -6.0        # 방어주 분할매수 진입 (%)
DEFAULT_STOP_LOSS_BULLISH_PCT: float = -16.0        # 강세장 모드 손절 라인 (%)
DEFAULT_DCA_DROP_PCT: float = -1.35                 # 물타기 트리거 하락률 (%)
DEFAULT_BRKU_WEIGHT_PCT: float = 10                 # BRKU 포트폴리오 고정 비중 (%)
DEFAULT_BEARISH_BUY_DAYS: int = 5                   # 방어주 분할매수 기간 (일)

# ============================================================
# DEFAULT_PARAMS 통합 dict
# ============================================================
# 위 상수 전체를 "DEFAULT_" prefix 제거 + snake_case 키로 통합.
# params: dict | None = None 인자에서 None이면 DEFAULT_PARAMS 사용.

DEFAULT_PARAMS: dict = {
    # 종목/페어
    "tickers": DEFAULT_TICKERS,
    "twin_pairs": DEFAULT_TWIN_PAIRS,
    "conditional_triggers": DEFAULT_CONDITIONAL_TRIGGERS,
    "conditional_target": DEFAULT_CONDITIONAL_TARGET,
    "bearish_tickers": DEFAULT_BEARISH_TICKERS,
    "cb_overheat_tickers": DEFAULT_CB_OVERHEAT_TICKERS,
    "leveraged_tickers": DEFAULT_LEVERAGED_TICKERS,
    # 자금
    "total_capital": DEFAULT_TOTAL_CAPITAL,
    "initial_buy": DEFAULT_INITIAL_BUY,
    "dca_buy": DEFAULT_DCA_BUY,
    "dca_max_count": DEFAULT_DCA_MAX_COUNT,
    "max_per_stock": DEFAULT_MAX_PER_STOCK,
    # 기본 매매
    "pair_gap_entry_threshold": DEFAULT_PAIR_GAP_ENTRY_THRESHOLD,
    "split_buy_interval_min": DEFAULT_SPLIT_BUY_INTERVAL_MIN,
    "max_daily_trades_per_stock": DEFAULT_MAX_DAILY_TRADES_PER_STOCK,
    "pair_gap_sell_threshold": DEFAULT_PAIR_GAP_SELL_THRESHOLD,
    "pair_sell_first_pct": DEFAULT_PAIR_SELL_FIRST_PCT,
    "pair_sell_remaining_pct": DEFAULT_PAIR_SELL_REMAINING_PCT,
    "pair_sell_interval_min": DEFAULT_PAIR_SELL_INTERVAL_MIN,
    "coin_trigger_pct": DEFAULT_COIN_TRIGGER_PCT,
    "coin_sell_profit_pct": DEFAULT_COIN_SELL_PROFIT_PCT,
    "coin_sell_bearish_pct": DEFAULT_COIN_SELL_BEARISH_PCT,
    "conl_trigger_pct": DEFAULT_CONL_TRIGGER_PCT,
    "conl_sell_profit_pct": DEFAULT_CONL_SELL_PROFIT_PCT,
    "conl_sell_avg_pct": DEFAULT_CONL_SELL_AVG_PCT,
    "max_hold_hours": DEFAULT_MAX_HOLD_HOURS,
    "take_profit_pct": DEFAULT_TAKE_PROFIT_PCT,
    # 진입 시간
    "entry_cutoff_hour": DEFAULT_ENTRY_CUTOFF_HOUR,
    "entry_cutoff_minute": DEFAULT_ENTRY_CUTOFF_MINUTE,
    "entry_default_start_hour": DEFAULT_ENTRY_DEFAULT_START_HOUR,
    "entry_default_start_minute": DEFAULT_ENTRY_DEFAULT_START_MINUTE,
    "entry_early_start_hour": DEFAULT_ENTRY_EARLY_START_HOUR,
    "entry_early_start_minute": DEFAULT_ENTRY_EARLY_START_MINUTE,
    "entry_trend_adx_min": DEFAULT_ENTRY_TREND_ADX_MIN,
    "entry_volume_ratio_min": DEFAULT_ENTRY_VOLUME_RATIO_MIN,
    "entry_ema_period": DEFAULT_ENTRY_EMA_PERIOD,
    "entry_ema_slope_lookback": DEFAULT_ENTRY_EMA_SLOPE_LOOKBACK,
    "conditional_exempt_cutoff": DEFAULT_CONDITIONAL_EXEMPT_CUTOFF,
    # 횡보장
    "sideways_enabled": DEFAULT_SIDEWAYS_ENABLED,
    "sideways_min_signals": DEFAULT_SIDEWAYS_MIN_SIGNALS,
    "sideways_eval_interval_min": DEFAULT_SIDEWAYS_EVAL_INTERVAL_MIN,
    "sideways_poly_low": DEFAULT_SIDEWAYS_POLY_LOW,
    "sideways_poly_high": DEFAULT_SIDEWAYS_POLY_HIGH,
    "sideways_gld_threshold": DEFAULT_SIDEWAYS_GLD_THRESHOLD,
    "sideways_gap_fail_count": DEFAULT_SIDEWAYS_GAP_FAIL_COUNT,
    "sideways_trigger_fail_count": DEFAULT_SIDEWAYS_TRIGGER_FAIL_COUNT,
    "sideways_index_threshold": DEFAULT_SIDEWAYS_INDEX_THRESHOLD,
    "sideways_atr_decline_pct": DEFAULT_SIDEWAYS_ATR_DECLINE_PCT,
    "sideways_volume_decline_pct": DEFAULT_SIDEWAYS_VOLUME_DECLINE_PCT,
    "sideways_ema_slope_max": DEFAULT_SIDEWAYS_EMA_SLOPE_MAX,
    "sideways_rsi_low": DEFAULT_SIDEWAYS_RSI_LOW,
    "sideways_rsi_high": DEFAULT_SIDEWAYS_RSI_HIGH,
    "sideways_bb_width_percentile": DEFAULT_SIDEWAYS_BB_WIDTH_PERCENTILE,
    "sideways_range_max_pct": DEFAULT_SIDEWAYS_RANGE_MAX_PCT,
    "sideways_ema_slope_lookback_days": DEFAULT_SIDEWAYS_EMA_SLOPE_LOOKBACK_DAYS,
    "sideways_bb_percentile_window_days": DEFAULT_SIDEWAYS_BB_PERCENTILE_WINDOW_DAYS,
    # CB
    "cb_vix_spike_pct": DEFAULT_CB_VIX_SPIKE_PCT,
    "cb_vix_cooldown_days": DEFAULT_CB_VIX_COOLDOWN_DAYS,
    "cb_gld_spike_pct": DEFAULT_CB_GLD_SPIKE_PCT,
    "cb_gld_cooldown_days": DEFAULT_CB_GLD_COOLDOWN_DAYS,
    "cb_btc_crash_pct": DEFAULT_CB_BTC_CRASH_PCT,
    "cb_btc_surge_pct": DEFAULT_CB_BTC_SURGE_PCT,
    "cb_rate_hike_prob_pct": DEFAULT_CB_RATE_HIKE_PROB_PCT,
    "cb_overheat_pct": DEFAULT_CB_OVERHEAT_PCT,
    "cb_overheat_recovery_pct": DEFAULT_CB_OVERHEAT_RECOVERY_PCT,
    "cb_overheat_switch_map": DEFAULT_CB_OVERHEAT_SWITCH_MAP,
    "high_vol_move_pct": DEFAULT_HIGH_VOL_MOVE_PCT,
    "high_vol_hit_count": DEFAULT_HIGH_VOL_HIT_COUNT,
    "high_vol_stop_loss_pct": DEFAULT_HIGH_VOL_STOP_LOSS_PCT,
    # CONL 필터
    "conl_adx_min": DEFAULT_CONL_ADX_MIN,
    "conl_ema_period": DEFAULT_CONL_EMA_PERIOD,
    "conl_ema_slope_lookback": DEFAULT_CONL_EMA_SLOPE_LOOKBACK,
    "conl_ema_slope_min_pct": DEFAULT_CONL_EMA_SLOPE_MIN_PCT,
    "conl_fixed_buy": DEFAULT_CONL_FIXED_BUY,
    "conl_dca_enabled": DEFAULT_CONL_DCA_ENABLED,
    # 고정 익절
    "pair_fixed_tp_stocks": DEFAULT_PAIR_FIXED_TP_STOCKS,
    "pair_immediate_sell_pct": DEFAULT_PAIR_IMMEDIATE_SELL_PCT,
    "pair_fixed_tp_pct": DEFAULT_PAIR_FIXED_TP_PCT,
    # 급락 역매수
    "crash_buy_threshold_pct": DEFAULT_CRASH_BUY_THRESHOLD_PCT,
    "crash_buy_time_hour": DEFAULT_CRASH_BUY_TIME_HOUR,
    "crash_buy_time_minute": DEFAULT_CRASH_BUY_TIME_MINUTE,
    "crash_buy_weight_pct": DEFAULT_CRASH_BUY_WEIGHT_PCT,
    "crash_buy_stocks": DEFAULT_CRASH_BUY_STOCKS,
    "crash_buy_flat_band_pct": DEFAULT_CRASH_BUY_FLAT_BAND_PCT,
    "crash_buy_flat_observe_min": DEFAULT_CRASH_BUY_FLAT_OBSERVE_MIN,
    # 스윙
    "swing_trigger_pct": DEFAULT_SWING_TRIGGER_PCT,
    "swing_eligible_tickers": DEFAULT_SWING_ELIGIBLE_TICKERS,
    "swing_stage1_weight_pct": DEFAULT_SWING_STAGE1_WEIGHT_PCT,
    "swing_stage1_hold_days": DEFAULT_SWING_STAGE1_HOLD_DAYS,
    "swing_stage1_atr_mult": DEFAULT_SWING_STAGE1_ATR_MULT,
    "swing_stage1_drawdown_pct": DEFAULT_SWING_STAGE1_DRAWDOWN_PCT,
    "swing_stage2_gld_ticker": DEFAULT_SWING_STAGE2_GLD_TICKER,
    "swing_stage2_weight_pct": DEFAULT_SWING_STAGE2_WEIGHT_PCT,
    "swing_stage2_hold_days": DEFAULT_SWING_STAGE2_HOLD_DAYS,
    "swing_stage2_stop_pct": DEFAULT_SWING_STAGE2_STOP_PCT,
    "swing_vix_stage1_weight_pct": DEFAULT_SWING_VIX_STAGE1_WEIGHT_PCT,
    "swing_vix_stage1_hold_days": DEFAULT_SWING_VIX_STAGE1_HOLD_DAYS,
    "swing_vix_stage2_cooldown_days": DEFAULT_SWING_VIX_STAGE2_COOLDOWN_DAYS,
    # 시장 모드
    "polymarket_bullish_threshold": DEFAULT_POLYMARKET_BULLISH_THRESHOLD,
    "bearish_polymarket_threshold": DEFAULT_BEARISH_POLYMARKET_THRESHOLD,
    "bearish_drop_threshold": DEFAULT_BEARISH_DROP_THRESHOLD,
    "stop_loss_bullish_pct": DEFAULT_STOP_LOSS_BULLISH_PCT,
    "dca_drop_pct": DEFAULT_DCA_DROP_PCT,
    "brku_weight_pct": DEFAULT_BRKU_WEIGHT_PCT,
    "bearish_buy_days": DEFAULT_BEARISH_BUY_DAYS,
}

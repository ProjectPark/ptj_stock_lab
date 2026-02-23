"""
PTJ 매매법 — 파라미터 중앙 관리
================================
v2~v5 모든 백테스트 엔진과 Optuna 최적화의 단일 파라미터 소스.

- frozen dataclass: mutation 방지, multiprocessing safe
- 상속 구조: BaseParams → V3Params → V4Params → V5Params
- config.py 브릿지: vN_params_from_config() 함수

Usage:
    from strategies.params import V5Params, v5_params_from_config

    params = v5_params_from_config()           # config.py에서 읽기
    params = V5Params.from_dict(trial_dict)    # Optuna trial에서 생성

    from dataclasses import replace
    modified = replace(params, stop_loss_pct=-4.0)  # 복사 + 수정
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


# ============================================================
# FeeConfig
# ============================================================

@dataclass(frozen=True)
class FeeConfig:
    """KIS 해외주식 수수료 구조 (단일 진실점)"""
    commission_pct: float = 0.25
    sec_fee_pct: float = 0.00278
    fx_spread_pct: float = 0.10

    @property
    def buy_fee_pct(self) -> float:
        """매수 총 수수료율: 0.350%"""
        return self.commission_pct + self.fx_spread_pct

    @property
    def sell_fee_pct(self) -> float:
        """매도 총 수수료율: 0.353%"""
        return self.commission_pct + self.sec_fee_pct + self.fx_spread_pct

    @property
    def round_trip_pct(self) -> float:
        """왕복 수수료율: 0.703%"""
        return self.buy_fee_pct + self.sell_fee_pct

    def calc_buy_fee(self, amount: float) -> tuple[float, float]:
        """(net_amount, fee) — 매수 후 실제 투자 금액과 수수료"""
        fee = amount * self.buy_fee_pct / 100
        return amount - fee, fee

    def calc_sell_fee(self, proceeds: float) -> tuple[float, float]:
        """(net_proceeds, fee) — 매도 후 실수령액과 수수료"""
        fee = proceeds * self.sell_fee_pct / 100
        return proceeds - fee, fee


# ============================================================
# BaseParams (v2 기본값)
# ============================================================

@dataclass(frozen=True)
class BaseParams:
    """v2~v5 공통 파라미터 (v2 기본값)"""

    # ── 자금 ────────────────────────────────────────────────────
    total_capital: float = 15_000
    initial_buy: float = 2_250
    dca_buy: float = 750
    max_per_stock: float = 7_500

    # ── 통화/환율 ──────────────────────────────────────────────
    use_krw: bool = False
    exchange_rate: float = 1.0

    # ── 매매 파라미터 ──────────────────────────────────────────
    pair_gap_entry_threshold: float = 1.5
    pair_gap_sell_threshold: float = 0.9
    stop_loss_pct: float = -3.0
    stop_loss_bullish_pct: float = -8.0
    dca_drop_pct: float = -0.5
    dca_max_count: int = 7
    max_hold_hours: int = 5
    take_profit_pct: float = 2.0
    split_buy_interval_min: int = 5

    # ── COIN/CONL ──────────────────────────────────────────────
    coin_trigger_pct: float = 3.0
    conl_trigger_pct: float = 3.0
    coin_sell_profit_pct: float = 3.0
    coin_sell_bearish_pct: float = 0.3
    conl_sell_profit_pct: float = 2.8
    conl_sell_avg_pct: float = 1.0

    # ── 분할매도 ───────────────────────────────────────────────
    pair_sell_first_pct: float = 0.80
    pair_sell_remaining_pct: float = 0.30
    pair_sell_interval_min: int = 5

    # ── 하락장 ─────────────────────────────────────────────────
    polymarket_bullish_threshold: int = 70
    bearish_drop_threshold: float = -6.0
    bearish_buy_days: int = 5
    brku_weight_pct: int = 10
    bearish_polymarket_threshold: int = 20

    # ── 기타 ───────────────────────────────────────────────────
    coin_follow_volatility_gap: float = 0.5

    # ── 수수료 ─────────────────────────────────────────────────
    fee_config: FeeConfig = FeeConfig()

    # ── 자금 유입 ────────────────────────────────────────────
    injection_pct: float = 0.0          # 투입원금 대비 입금 비율 (%, 0=비활성)
    injection_interval_days: int = 0    # 입금 간격 (거래일, 0=비활성, 20≈월간)
    size_by_invested: bool = False      # True: 포지션 사이즈를 투입원금 기반으로

    # ── 직렬화 ─────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """파라미터를 dict로 변환 (fee_config 제외)."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "fee_config"
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BaseParams":
        """dict에서 파라미터 생성 (알 수 없는 키는 무시)."""
        known = {f.name for f in fields(cls)} - {"fee_config"}
        return cls(**{k: v for k, v in d.items() if k in known})


# ============================================================
# V3Params
# ============================================================

@dataclass(frozen=True)
class V3Params(BaseParams):
    """v3 오버라이드 + 횡보장/진입제한 파라미터"""

    # ── v2 오버라이드 (KRW 기준) ───────────────────────────────
    total_capital: float = 20_000_000
    initial_buy: float = 3_000_000
    dca_buy: float = 1_000_000
    max_per_stock: float = 7_000_000
    use_krw: bool = True
    exchange_rate: float = 1_350

    pair_gap_entry_threshold: float = 2.2
    dca_max_count: int = 4
    coin_trigger_pct: float = 4.5
    conl_trigger_pct: float = 4.5
    split_buy_interval_min: int = 20

    # ── v3 신규: 진입 제한 ─────────────────────────────────────
    max_daily_trades_per_stock: int = 1
    entry_cutoff_hour: int = 10
    entry_cutoff_minute: int = 30
    conditional_exempt_cutoff: bool = True

    # ── v3 신규: 횡보장 감지 ───────────────────────────────────
    sideways_enabled: bool = True
    sideways_min_signals: int = 3
    sideways_eval_interval_min: int = 30
    sideways_poly_low: float = 0.40
    sideways_poly_high: float = 0.60
    sideways_gld_threshold: float = 0.3
    sideways_gap_fail_count: int = 2
    sideways_trigger_fail_count: int = 2
    sideways_index_threshold: float = 0.5


# ============================================================
# V4Params
# ============================================================

@dataclass(frozen=True)
class V4Params(V3Params):
    """v4 추가: 서킷브레이커, 스윙, 크래시바이, CONL 필터"""

    # ── v3 오버라이드 (USD 복귀) ───────────────────────────────
    total_capital: float = 15_000
    initial_buy: float = 2_250
    dca_buy: float = 750
    max_per_stock: float = 5_250
    use_krw: bool = False
    exchange_rate: float = 1.0

    entry_cutoff_minute: int = 0    # v3=30 → v4=0

    # ── v4 신규: 진입 시간 ─────────────────────────────────────
    entry_default_start_hour: int = 8
    entry_default_start_minute: int = 0
    entry_early_start_hour: int = 3
    entry_early_start_minute: int = 30
    entry_trend_adx_min: float = 25.0
    entry_volume_ratio_min: float = 1.5
    entry_ema_period: int = 20
    entry_ema_slope_lookback: int = 5

    # ── v4 신규: 횡보장 추가 지표 ──────────────────────────────
    sideways_atr_decline_pct: float = 20.0
    sideways_volume_decline_pct: float = 30.0
    sideways_ema_slope_max: float = 0.1
    sideways_rsi_low: float = 45.0
    sideways_rsi_high: float = 55.0
    sideways_bb_width_percentile: float = 20.0
    sideways_range_max_pct: float = 2.0
    sideways_ema_slope_lookback_days: int = 5
    sideways_bb_percentile_window_days: int = 60

    # ── v4 신규: 서킷브레이커 ──────────────────────────────────
    cb_vix_spike_pct: float = 6.0
    cb_vix_cooldown_days: int = 7
    cb_gld_spike_pct: float = 3.0
    cb_gld_cooldown_days: int = 3
    cb_btc_crash_pct: float = -5.0
    cb_btc_surge_pct: float = 5.0
    cb_rate_hike_prob_pct: float = 50.0
    cb_overheat_pct: float = 20.0
    cb_overheat_recovery_pct: float = -10.0

    # ── v4 신규: 고변동성 ──────────────────────────────────────
    high_vol_move_pct: float = 10.0
    high_vol_hit_count: int = 2
    high_vol_stop_loss_pct: float = -4.0

    # ── v4 신규: CONL 필터 ─────────────────────────────────────
    conl_adx_min: float = 18.0
    conl_ema_period: int = 20
    conl_ema_slope_lookback: int = 5
    conl_ema_slope_min_pct: float = 0.0
    conl_fixed_buy: float = 6_750
    conl_dca_enabled: bool = False

    # ── v4 신규: 분할매도 확장 ─────────────────────────────────
    pair_immediate_sell_pct: float = 0.40
    pair_fixed_tp_pct: float = 5.0

    # ── v4 신규: 급락 역매수 ───────────────────────────────────
    crash_buy_threshold_pct: float = -40.0
    crash_buy_time_hour: int = 15
    crash_buy_time_minute: int = 55
    crash_buy_weight_pct: float = 95.0
    crash_buy_flat_band_pct: float = 0.1
    crash_buy_flat_observe_min: int = 30

    # ── v4 신규: 스윙 매매 ─────────────────────────────────────
    swing_trigger_pct: float = 15.0
    swing_stage1_weight_pct: float = 90.0
    swing_stage1_hold_days: int = 63
    swing_stage1_atr_mult: float = 1.5
    swing_stage1_drawdown_pct: float = -15.0
    swing_stage2_weight_pct: float = 70.0
    swing_stage2_hold_days: int = 105
    swing_stage2_stop_pct: float = -5.0
    swing_vix_stage1_weight_pct: float = 80.0
    swing_vix_stage1_hold_days: int = 105
    swing_vix_stage2_cooldown_days: int = 63


# ============================================================
# V5Params
# ============================================================

@dataclass(frozen=True)
class V5Params(V4Params):
    """v5 추가: 방어모드 (IAU/GDXU), 레버리지별 손절"""

    # ── v4 오버라이드 ──────────────────────────────────────────
    entry_early_start_hour: int = 4    # v4=3 → v5=4
    entry_early_start_minute: int = 0  # v4=30 → v5=0
    conditional_exempt_cutoff: bool = False  # v4=True → v5=False

    # ── v5 신규: CB 레버리지 쿨다운 ────────────────────────────
    cb_rate_leverage_cooldown_days: int = 3

    # ── v5 신규: 레버리지별 고변동성 손절 ──────────────────────
    high_vol_stop_loss_1x_pct: float = -4.0
    high_vol_stop_loss_2x_pct: float = -6.0
    high_vol_stop_loss_3x_pct: float = -8.0

    # ── v5 신규: Unix(VIX) 방어모드 ────────────────────────────
    unix_defense_trigger_pct: float = 10.0
    unix_defense_iau_weight_pct: float = 40.0
    unix_defense_gdxu_weight_pct: float = 30.0
    iau_stop_loss_pct: float = -5.0
    iau_ban_trigger_gdxu_drop_pct: float = -12.0
    iau_ban_duration_trading_days: int = 40
    gdxu_hold_min_days: int = 2
    gdxu_hold_max_days: int = 3
    unix_rebalance_hour: int = 15
    unix_rebalance_minute: int = 55


# ============================================================
# config.py 브릿지 (마이그레이션용)
# ============================================================

def _shared_params_from_config() -> dict[str, Any]:
    """v2~v5 공유 파라미터를 config.py에서 읽는다."""
    import config
    return {
        "stop_loss_pct": config.STOP_LOSS_PCT,
        "stop_loss_bullish_pct": config.STOP_LOSS_BULLISH_PCT,
        "dca_drop_pct": config.DCA_DROP_PCT,
        "max_hold_hours": config.MAX_HOLD_HOURS,
        "take_profit_pct": config.TAKE_PROFIT_PCT,
        "coin_sell_profit_pct": config.COIN_SELL_PROFIT_PCT,
        "coin_sell_bearish_pct": config.COIN_SELL_BEARISH_PCT,
        "conl_sell_profit_pct": config.CONL_SELL_PROFIT_PCT,
        "conl_sell_avg_pct": config.CONL_SELL_AVG_PCT,
        "pair_sell_first_pct": config.PAIR_SELL_FIRST_PCT,
        "pair_sell_remaining_pct": config.PAIR_SELL_REMAINING_PCT,
        "pair_sell_interval_min": config.PAIR_SELL_INTERVAL_MIN,
        "pair_gap_sell_threshold": config.PAIR_GAP_SELL_THRESHOLD_V2,
        "polymarket_bullish_threshold": config.POLYMARKET_BULLISH_THRESHOLD,
        "bearish_drop_threshold": config.BEARISH_DROP_THRESHOLD,
        "bearish_buy_days": config.BEARISH_BUY_DAYS,
        "brku_weight_pct": config.BRKU_WEIGHT_PCT,
        "bearish_polymarket_threshold": config.BEARISH_POLYMARKET_THRESHOLD,
        "coin_follow_volatility_gap": config.COIN_FOLLOW_VOLATILITY_GAP,
        # 자금 유입
        "injection_pct": config.INJECTION_PCT,
        "injection_interval_days": config.INJECTION_INTERVAL_DAYS,
        "size_by_invested": config.SIZE_BY_INVESTED,
    }


def v2_params_from_config() -> BaseParams:
    """config.py v2 파라미터 → BaseParams"""
    import config
    d = _shared_params_from_config()
    d.update({
        "total_capital": config.TOTAL_CAPITAL_USD,
        "initial_buy": config.INITIAL_BUY_USD,
        "dca_buy": config.DCA_BUY_USD,
        "max_per_stock": config.MAX_PER_STOCK_USD,
        "pair_gap_entry_threshold": config.PAIR_GAP_ENTRY_THRESHOLD,
        "dca_max_count": config.DCA_MAX_COUNT,
        "split_buy_interval_min": config.SPLIT_BUY_INTERVAL_MIN,
        "coin_trigger_pct": config.COIN_TRIGGER_PCT,
        "conl_trigger_pct": config.CONL_TRIGGER_PCT,
    })
    return BaseParams(**d)


def v3_params_from_config() -> V3Params:
    """config.py v3 파라미터 → V3Params"""
    import config
    d = _shared_params_from_config()
    d.update({
        "total_capital": config.TOTAL_CAPITAL_KRW,
        "initial_buy": config.INITIAL_BUY_KRW,
        "dca_buy": config.DCA_BUY_KRW,
        "max_per_stock": config.V3_MAX_PER_STOCK_KRW,
        "use_krw": True,
        "exchange_rate": config.EXCHANGE_RATE_KRW,
        "pair_gap_entry_threshold": config.V3_PAIR_GAP_ENTRY_THRESHOLD,
        "dca_max_count": config.V3_DCA_MAX_COUNT,
        "split_buy_interval_min": config.V3_SPLIT_BUY_INTERVAL_MIN,
        "coin_trigger_pct": config.V3_COIN_TRIGGER_PCT,
        "conl_trigger_pct": config.V3_CONL_TRIGGER_PCT,
        # v3 신규
        "max_daily_trades_per_stock": config.V3_MAX_DAILY_TRADES_PER_STOCK,
        "entry_cutoff_hour": config.V3_ENTRY_CUTOFF_HOUR,
        "entry_cutoff_minute": config.V3_ENTRY_CUTOFF_MINUTE,
        "conditional_exempt_cutoff": config.V3_CONDITIONAL_EXEMPT_CUTOFF,
        "sideways_enabled": config.V3_SIDEWAYS_ENABLED,
        "sideways_min_signals": config.V3_SIDEWAYS_MIN_SIGNALS,
        "sideways_eval_interval_min": config.V3_SIDEWAYS_EVAL_INTERVAL_MIN,
        "sideways_poly_low": config.V3_SIDEWAYS_POLY_LOW,
        "sideways_poly_high": config.V3_SIDEWAYS_POLY_HIGH,
        "sideways_gld_threshold": config.V3_SIDEWAYS_GLD_THRESHOLD,
        "sideways_gap_fail_count": config.V3_SIDEWAYS_GAP_FAIL_COUNT,
        "sideways_trigger_fail_count": config.V3_SIDEWAYS_TRIGGER_FAIL_COUNT,
        "sideways_index_threshold": config.V3_SIDEWAYS_INDEX_THRESHOLD,
    })
    return V3Params(**d)


def v4_params_from_config() -> V4Params:
    """config.py v4 파라미터 → V4Params"""
    import config
    d = _shared_params_from_config()
    d.update({
        # 자금
        "total_capital": config.V4_TOTAL_CAPITAL,
        "initial_buy": config.V4_INITIAL_BUY,
        "dca_buy": config.V4_DCA_BUY,
        "max_per_stock": config.V4_MAX_PER_STOCK,
        # 매매
        "pair_gap_entry_threshold": config.V4_PAIR_GAP_ENTRY_THRESHOLD,
        "dca_max_count": config.V4_DCA_MAX_COUNT,
        "split_buy_interval_min": config.V4_SPLIT_BUY_INTERVAL_MIN,
        "coin_trigger_pct": config.V4_COIN_TRIGGER_PCT,
        "conl_trigger_pct": config.V4_CONL_TRIGGER_PCT,
        # v3 계승
        "max_daily_trades_per_stock": config.V4_MAX_DAILY_TRADES_PER_STOCK,
        "entry_cutoff_hour": config.V4_ENTRY_CUTOFF_HOUR,
        "entry_cutoff_minute": config.V4_ENTRY_CUTOFF_MINUTE,
        "conditional_exempt_cutoff": config.V4_CONDITIONAL_EXEMPT_CUTOFF,
        "sideways_enabled": config.V4_SIDEWAYS_ENABLED,
        "sideways_min_signals": config.V4_SIDEWAYS_MIN_SIGNALS,
        "sideways_eval_interval_min": config.V4_SIDEWAYS_EVAL_INTERVAL_MIN,
        "sideways_poly_low": config.V4_SIDEWAYS_POLY_LOW,
        "sideways_poly_high": config.V4_SIDEWAYS_POLY_HIGH,
        "sideways_gld_threshold": config.V4_SIDEWAYS_GLD_THRESHOLD,
        "sideways_gap_fail_count": config.V4_SIDEWAYS_GAP_FAIL_COUNT,
        "sideways_trigger_fail_count": config.V4_SIDEWAYS_TRIGGER_FAIL_COUNT,
        "sideways_index_threshold": config.V4_SIDEWAYS_INDEX_THRESHOLD,
        # v4 진입 시간
        "entry_default_start_hour": config.V4_ENTRY_DEFAULT_START_HOUR,
        "entry_default_start_minute": config.V4_ENTRY_DEFAULT_START_MINUTE,
        "entry_early_start_hour": config.V4_ENTRY_EARLY_START_HOUR,
        "entry_early_start_minute": config.V4_ENTRY_EARLY_START_MINUTE,
        "entry_trend_adx_min": config.V4_ENTRY_TREND_ADX_MIN,
        "entry_volume_ratio_min": config.V4_ENTRY_VOLUME_RATIO_MIN,
        "entry_ema_period": config.V4_ENTRY_EMA_PERIOD,
        "entry_ema_slope_lookback": config.V4_ENTRY_EMA_SLOPE_LOOKBACK,
        # v4 횡보장 추가
        "sideways_atr_decline_pct": config.V4_SIDEWAYS_ATR_DECLINE_PCT,
        "sideways_volume_decline_pct": config.V4_SIDEWAYS_VOLUME_DECLINE_PCT,
        "sideways_ema_slope_max": config.V4_SIDEWAYS_EMA_SLOPE_MAX,
        "sideways_rsi_low": config.V4_SIDEWAYS_RSI_LOW,
        "sideways_rsi_high": config.V4_SIDEWAYS_RSI_HIGH,
        "sideways_bb_width_percentile": config.V4_SIDEWAYS_BB_WIDTH_PERCENTILE,
        "sideways_range_max_pct": config.V4_SIDEWAYS_RANGE_MAX_PCT,
        "sideways_ema_slope_lookback_days": config.V4_SIDEWAYS_EMA_SLOPE_LOOKBACK_DAYS,
        "sideways_bb_percentile_window_days": config.V4_SIDEWAYS_BB_PERCENTILE_WINDOW_DAYS,
        # v4 서킷브레이커
        "cb_vix_spike_pct": config.V4_CB_VIX_SPIKE_PCT,
        "cb_vix_cooldown_days": config.V4_CB_VIX_COOLDOWN_DAYS,
        "cb_gld_spike_pct": config.V4_CB_GLD_SPIKE_PCT,
        "cb_gld_cooldown_days": config.V4_CB_GLD_COOLDOWN_DAYS,
        "cb_btc_crash_pct": config.V4_CB_BTC_CRASH_PCT,
        "cb_btc_surge_pct": config.V4_CB_BTC_SURGE_PCT,
        "cb_rate_hike_prob_pct": config.V4_CB_RATE_HIKE_PROB_PCT,
        "cb_overheat_pct": config.V4_CB_OVERHEAT_PCT,
        "cb_overheat_recovery_pct": config.V4_CB_OVERHEAT_RECOVERY_PCT,
        # v4 고변동성
        "high_vol_move_pct": config.V4_HIGH_VOL_MOVE_PCT,
        "high_vol_hit_count": config.V4_HIGH_VOL_HIT_COUNT,
        "high_vol_stop_loss_pct": config.V4_HIGH_VOL_STOP_LOSS_PCT,
        # v4 CONL 필터
        "conl_adx_min": config.V4_CONL_ADX_MIN,
        "conl_ema_period": config.V4_CONL_EMA_PERIOD,
        "conl_ema_slope_lookback": config.V4_CONL_EMA_SLOPE_LOOKBACK,
        "conl_ema_slope_min_pct": config.V4_CONL_EMA_SLOPE_MIN_PCT,
        "conl_fixed_buy": config.V4_CONL_FIXED_BUY,
        "conl_dca_enabled": config.V4_CONL_DCA_ENABLED,
        # v4 분할매도 확장
        "pair_immediate_sell_pct": config.V4_PAIR_IMMEDIATE_SELL_PCT,
        "pair_fixed_tp_pct": config.V4_PAIR_FIXED_TP_PCT,
        # v4 급락 역매수
        "crash_buy_threshold_pct": config.V4_CRASH_BUY_THRESHOLD_PCT,
        "crash_buy_time_hour": config.V4_CRASH_BUY_TIME_HOUR,
        "crash_buy_time_minute": config.V4_CRASH_BUY_TIME_MINUTE,
        "crash_buy_weight_pct": config.V4_CRASH_BUY_WEIGHT_PCT,
        "crash_buy_flat_band_pct": config.V4_CRASH_BUY_FLAT_BAND_PCT,
        "crash_buy_flat_observe_min": config.V4_CRASH_BUY_FLAT_OBSERVE_MIN,
        # v4 스윙
        "swing_trigger_pct": config.V4_SWING_TRIGGER_PCT,
        "swing_stage1_weight_pct": config.V4_SWING_STAGE1_WEIGHT_PCT,
        "swing_stage1_hold_days": config.V4_SWING_STAGE1_HOLD_DAYS,
        "swing_stage1_atr_mult": config.V4_SWING_STAGE1_ATR_MULT,
        "swing_stage1_drawdown_pct": config.V4_SWING_STAGE1_DRAWDOWN_PCT,
        "swing_stage2_weight_pct": config.V4_SWING_STAGE2_WEIGHT_PCT,
        "swing_stage2_hold_days": config.V4_SWING_STAGE2_HOLD_DAYS,
        "swing_stage2_stop_pct": config.V4_SWING_STAGE2_STOP_PCT,
        "swing_vix_stage1_weight_pct": config.V4_SWING_VIX_STAGE1_WEIGHT_PCT,
        "swing_vix_stage1_hold_days": config.V4_SWING_VIX_STAGE1_HOLD_DAYS,
        "swing_vix_stage2_cooldown_days": config.V4_SWING_VIX_STAGE2_COOLDOWN_DAYS,
    })
    return V4Params(**d)


def v5_params_from_config() -> V5Params:
    """config.py v5 파라미터 → V5Params"""
    import config
    d = _shared_params_from_config()
    d.update({
        # 자금
        "total_capital": config.V5_TOTAL_CAPITAL,
        "initial_buy": config.V5_INITIAL_BUY,
        "dca_buy": config.V5_DCA_BUY,
        "max_per_stock": config.V5_MAX_PER_STOCK,
        # 매매
        "pair_gap_entry_threshold": config.V5_PAIR_GAP_ENTRY_THRESHOLD,
        "dca_max_count": config.V5_DCA_MAX_COUNT,
        "split_buy_interval_min": config.V5_SPLIT_BUY_INTERVAL_MIN,
        "coin_trigger_pct": config.V5_COIN_TRIGGER_PCT,
        "conl_trigger_pct": config.V5_CONL_TRIGGER_PCT,
        # v3 계승
        "max_daily_trades_per_stock": config.V5_MAX_DAILY_TRADES_PER_STOCK,
        "entry_cutoff_hour": config.V5_ENTRY_CUTOFF_HOUR,
        "entry_cutoff_minute": config.V5_ENTRY_CUTOFF_MINUTE,
        "conditional_exempt_cutoff": config.V5_CONDITIONAL_EXEMPT_CUTOFF,
        "sideways_enabled": config.V5_SIDEWAYS_ENABLED,
        "sideways_min_signals": config.V5_SIDEWAYS_MIN_SIGNALS,
        "sideways_eval_interval_min": config.V5_SIDEWAYS_EVAL_INTERVAL_MIN,
        "sideways_poly_low": config.V5_SIDEWAYS_POLY_LOW,
        "sideways_poly_high": config.V5_SIDEWAYS_POLY_HIGH,
        "sideways_gld_threshold": config.V5_SIDEWAYS_GLD_THRESHOLD,
        "sideways_gap_fail_count": config.V5_SIDEWAYS_GAP_FAIL_COUNT,
        "sideways_trigger_fail_count": config.V5_SIDEWAYS_TRIGGER_FAIL_COUNT,
        "sideways_index_threshold": config.V5_SIDEWAYS_INDEX_THRESHOLD,
        # v4 계승: 진입 시간
        "entry_default_start_hour": config.V5_ENTRY_DEFAULT_START_HOUR,
        "entry_default_start_minute": config.V5_ENTRY_DEFAULT_START_MINUTE,
        "entry_early_start_hour": config.V5_ENTRY_EARLY_START_HOUR,
        "entry_early_start_minute": config.V5_ENTRY_EARLY_START_MINUTE,
        "entry_trend_adx_min": config.V5_ENTRY_TREND_ADX_MIN,
        "entry_volume_ratio_min": config.V5_ENTRY_VOLUME_RATIO_MIN,
        "entry_ema_period": config.V5_ENTRY_EMA_PERIOD,
        "entry_ema_slope_lookback": config.V5_ENTRY_EMA_SLOPE_LOOKBACK,
        # v4 계승: 서킷브레이커
        "cb_vix_spike_pct": config.V5_CB_VIX_SPIKE_PCT,
        "cb_vix_cooldown_days": config.V5_CB_VIX_COOLDOWN_DAYS,
        "cb_gld_spike_pct": config.V5_CB_GLD_SPIKE_PCT,
        "cb_gld_cooldown_days": config.V5_CB_GLD_COOLDOWN_DAYS,
        "cb_btc_crash_pct": config.V5_CB_BTC_CRASH_PCT,
        "cb_btc_surge_pct": config.V5_CB_BTC_SURGE_PCT,
        "cb_rate_hike_prob_pct": config.V5_CB_RATE_HIKE_PROB_PCT,
        "cb_overheat_pct": config.V5_CB_OVERHEAT_PCT,
        "cb_overheat_recovery_pct": config.V5_CB_OVERHEAT_RECOVERY_PCT,
        # v4 계승: 고변동성
        "high_vol_move_pct": config.V5_HIGH_VOL_MOVE_PCT,
        "high_vol_hit_count": config.V5_HIGH_VOL_HIT_COUNT,
        # v5 신규: CB 레버리지 쿨다운
        "cb_rate_leverage_cooldown_days": config.V5_CB_RATE_LEVERAGE_COOLDOWN_DAYS,
        # v5 신규: 레버리지별 손절
        "high_vol_stop_loss_1x_pct": config.V5_HIGH_VOL_STOP_LOSS_1X_PCT,
        "high_vol_stop_loss_2x_pct": config.V5_HIGH_VOL_STOP_LOSS_2X_PCT,
        "high_vol_stop_loss_3x_pct": config.V5_HIGH_VOL_STOP_LOSS_3X_PCT,
        # v5 신규: Unix 방어모드
        "unix_defense_trigger_pct": config.V5_UNIX_DEFENSE_TRIGGER_PCT,
        "unix_defense_iau_weight_pct": config.V5_UNIX_DEFENSE_IAU_WEIGHT_PCT,
        "unix_defense_gdxu_weight_pct": config.V5_UNIX_DEFENSE_GDXU_WEIGHT_PCT,
        "iau_stop_loss_pct": config.V5_IAU_STOP_LOSS_PCT,
        "iau_ban_trigger_gdxu_drop_pct": config.V5_IAU_BAN_TRIGGER_GDXU_DROP_PCT,
        "iau_ban_duration_trading_days": config.V5_IAU_BAN_DURATION_TRADING_DAYS,
        "gdxu_hold_min_days": config.V5_GDXU_HOLD_MIN_DAYS,
        "gdxu_hold_max_days": config.V5_GDXU_HOLD_MAX_DAYS,
        "unix_rebalance_hour": config.V5_UNIX_REBALANCE_HOUR,
        "unix_rebalance_minute": config.V5_UNIX_REBALANCE_MINUTE,
    })
    return V5Params(**d)

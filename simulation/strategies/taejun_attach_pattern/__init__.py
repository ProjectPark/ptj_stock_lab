"""
taejun_attach_pattern
=====================
박태준 매매 전략 패턴 — 개별 전략 모듈 패키지.

Usage:
    from strategies.taejun_attach_pattern import get_strategy, list_strategies

    strat = get_strategy("bargain_buy")
    signal = strat.generate_signal(market_data)
"""
from .asset_mode import AssetMode, AssetModeManager
from .bear_regime import BearRegimeDetector, scale_unit_mul_by_poly
from .base import Action, BaseStrategy, ExitReason, MarketData, Position, Signal
from .circuit_breaker import CircuitBreaker, CBStatus
from .composite_signal_engine import CompositeSignalEngine
from .d2s_engine import D2SEngine, DailySnapshot, D2SPosition, TechnicalPreprocessor
from .fees import FeeCalculator, FeeConfig
from .m201_mode import M201Action, M201ImmediateMode, M201Signal
from .poly_quality import PolyQualityFilter
from .portfolio import PortfolioManager
from .registry import get_strategy, list_strategies
from .signal_presets import composite_from_base_params
from .stop_loss import StopLossCalculator
from .swing_mode import SwingModeManager, SwingPhase

__all__ = [
    "Action",
    "AssetMode",
    "AssetModeManager",
    "BearRegimeDetector",
    "scale_unit_mul_by_poly",
    "BaseStrategy",
    "CBStatus",
    "CircuitBreaker",
    "CompositeSignalEngine",
    "D2SEngine",
    "D2SPosition",
    "DailySnapshot",
    "ExitReason",
    "FeeCalculator",
    "FeeConfig",
    "M201Action",
    "M201ImmediateMode",
    "M201Signal",
    "MarketData",
    "PolyQualityFilter",
    "PortfolioManager",
    "Position",
    "Signal",
    "StopLossCalculator",
    "SwingModeManager",
    "SwingPhase",
    "TechnicalPreprocessor",
    "composite_from_base_params",
    "get_strategy",
    "list_strategies",
]

# 전략 모듈 임포트 — register 데코레이터가 실행되면서 자동 등록
from . import bargain_buy as _bargain_buy  # noqa: F401, E402
from . import jab_bitu as _jab_bitu  # noqa: F401, E402
from . import jab_tsll as _jab_tsll  # noqa: F401, E402
from . import jab_seth as _jab_seth  # noqa: F401, E402
from . import jab_soxl as _jab_soxl  # noqa: F401, E402
from . import vix_gold as _vix_gold  # noqa: F401, E402
from . import sp500_entry as _sp500_entry  # noqa: F401, E402
from . import sector_rotate as _sector_rotate  # noqa: F401, E402
from . import bank_conditional as _bank_conditional  # noqa: F401, E402
from . import short_macro as _short_macro  # noqa: F401, E402
from . import reit_risk as _reit_risk  # noqa: F401, E402
from . import twin_pair as _twin_pair  # noqa: F401, E402
from . import conditional_coin as _conditional_coin  # noqa: F401, E402
from . import conditional_conl as _conditional_conl  # noqa: F401, E402
from . import bearish_defense as _bearish_defense  # noqa: F401, E402
from . import emergency_mode as _emergency_mode  # noqa: F401, E402
from . import bear_regime as _bear_regime  # noqa: F401, E402
from . import crash_buy as _crash_buy  # noqa: F401, E402
from . import soxl_independent as _soxl_independent  # noqa: F401, E402
from . import jab_etq as _jab_etq  # noqa: F401, E402

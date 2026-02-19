"""
taejun_attach_pattern
=====================
박태준 매매 전략 패턴 — 개별 전략 모듈 패키지.

Usage:
    from strategies.taejun_attach_pattern import get_strategy, list_strategies

    strat = get_strategy("bargain_buy")
    signal = strat.generate_signal(market_data)
"""
from .base import Action, BaseStrategy, ExitReason, MarketData, Position, Signal
from .d2s_engine import D2SEngine, DailySnapshot, D2SPosition, TechnicalPreprocessor
from .fees import FeeCalculator, FeeConfig
from .portfolio import PortfolioManager
from .registry import get_strategy, list_strategies

__all__ = [
    "Action",
    "BaseStrategy",
    "D2SEngine",
    "D2SPosition",
    "DailySnapshot",
    "ExitReason",
    "FeeCalculator",
    "FeeConfig",
    "MarketData",
    "PortfolioManager",
    "Position",
    "Signal",
    "TechnicalPreprocessor",
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

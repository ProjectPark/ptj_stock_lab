"""
product/engine_v4/__init__.py
==============================
v4 엔진 공개 API.

사용법:
    from product.engine_v4 import generate_v4_signals
    from product.engine_v4.params import DEFAULT_PARAMS
"""
from __future__ import annotations

try:
    from .signals_v4 import generate_v4_signals
    __all__ = ["generate_v4_signals"]
except ImportError:
    __all__ = []

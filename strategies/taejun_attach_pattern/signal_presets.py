"""
Signal Engine Presets — BaseParams → CompositeSignalEngine 팩토리
================================================================
버전별 파라미터 → CompositeSignalEngine 인스턴스를 생성하는 편의 함수.

Usage:
    from strategies.taejun_attach_pattern.signal_presets import composite_from_base_params
    from strategies.params import v5_params_from_config

    engine = composite_from_base_params(v5_params_from_config())
"""
from __future__ import annotations

from strategies.params import BaseParams

from .composite_signal_engine import CompositeSignalEngine


def composite_from_base_params(params: BaseParams) -> CompositeSignalEngine:
    """BaseParams → CompositeSignalEngine 팩토리."""
    return CompositeSignalEngine.from_base_params(params)

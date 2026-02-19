"""
taejun_attach_pattern - 전략 레지스트리
======================================
데코레이터로 전략을 등록하고, 이름으로 인스턴스를 조회한다.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseStrategy

_REGISTRY: dict[str, type[BaseStrategy]] = {}


def register(cls: type[BaseStrategy]) -> type[BaseStrategy]:
    """클래스 데코레이터: 전략을 레지스트리에 등록한다."""
    _REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str, params: dict | None = None) -> BaseStrategy:
    """이름으로 전략 인스턴스를 생성한다."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown strategy: {name!r}. Available: {available}")
    return _REGISTRY[name](params)


def list_strategies() -> list[str]:
    """등록된 전략 이름 목록을 반환한다."""
    return sorted(_REGISTRY.keys())

"""
Polymarket 지표 시스템
=====================
예측 시장 확률을 시장 심리 선행 지표로 활용.

사용법:
    from polymarket import fetch_all_indicators, compute_composite_signal

    results = fetch_all_indicators()
    signal = compute_composite_signal(results)
"""

from polymarket.poly_fetcher import fetch_all_indicators
from polymarket.poly_signals import compute_composite_signal

__all__ = ["fetch_all_indicators", "compute_composite_signal"]

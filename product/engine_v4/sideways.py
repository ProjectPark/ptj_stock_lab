"""
product/engine_v4/sideways.py
==============================
v4 횡보장 감지 순수 함수.
DataFrame 의존 없음 — 사전 계산된 지표 dict만 입력.
"""
from __future__ import annotations


def check_sideways(
    indicators: dict,
    params: dict,
) -> dict:
    """
    횡보장 여부 판정.

    Args:
        indicators: {
            "atr_decline": bool,      # ATR이 20일 평균보다 20% 이상 낮음
            "volume_decline": bool,   # 거래량이 20일 평균보다 30% 이상 낮음
            "ema_flat": bool,         # EMA20 기울기 절대값 <= 0.1%
            "rsi_box": bool,          # RSI 45~55 박스권
            "bb_narrow": bool,        # BB폭 60일 하위 20% 이내
            "range_narrow": bool,     # 장중 고저폭 <= range_max_pct (선택적)
        }
        params: DEFAULT_PARAMS

    Returns:
        {
            "is_sideways": bool,
            "triggered_count": int,   # 충족된 지표 수
            "signals": dict[str, bool],  # 각 지표 충족 여부
        }

    판정: triggered_count >= sideways_min_signals → is_sideways=True
    """
    min_signals: int = params.get("sideways_min_signals", 3)

    # 5개 필수 지표
    core_keys = ["atr_decline", "volume_decline", "ema_flat", "rsi_box", "bb_narrow"]

    signals: dict[str, bool] = {}
    for key in core_keys:
        signals[key] = bool(indicators.get(key, False))

    # range_narrow 는 indicators에 있을 때만 카운트
    if "range_narrow" in indicators:
        signals["range_narrow"] = bool(indicators["range_narrow"])

    triggered_count = sum(1 for v in signals.values() if v)
    is_sideways = triggered_count >= min_signals

    return {
        "is_sideways": is_sideways,
        "triggered_count": triggered_count,
        "signals": signals,
    }

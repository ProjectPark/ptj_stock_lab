"""
product/engine_v4/conditional.py
==================================
v4 조건부 진입 필터 순수 함수.

backtest_v4.py _passes_conl_entry_filter() 로직을 순수 함수로 이식.
"""
from __future__ import annotations


def check_conl_entry_filter(
    indicators: dict,
    params: dict,
) -> dict:
    """
    CONL 조건부 진입 ADX/EMA 필터.

    Args:
        indicators: {"adx": float, "ema_slope_pct": float}
                    (사전 계산된 지표값, backtest_v4.py _prepare_conl_indicators 참조)
        params: DEFAULT_PARAMS

    Returns:
        {"passes": bool, "reason": str}
        reason: "ok" | "no_indicator" | "adx_fail" | "ema_fail" | "adx_ema_fail"
    """
    conl_adx_min: float = params.get("conl_adx_min", 10.0)
    conl_ema_slope_min_pct: float = params.get("conl_ema_slope_min_pct", 0.0)

    # 지표값 없음 → 차단 (backtest_v4.py 라인 671~674)
    if indicators is None:
        return {"passes": False, "reason": "no_indicator"}

    adx = indicators.get("adx")
    ema_slope_pct = indicators.get("ema_slope_pct")

    if adx is None or ema_slope_pct is None:
        return {"passes": False, "reason": "no_indicator"}

    # backtest_v4.py 라인 675~676
    adx_ok: bool = adx >= conl_adx_min
    ema_ok: bool = ema_slope_pct > conl_ema_slope_min_pct

    if adx_ok and ema_ok:
        return {"passes": True, "reason": "ok"}

    if not adx_ok and not ema_ok:
        return {"passes": False, "reason": "adx_ema_fail"}

    if not adx_ok:
        return {"passes": False, "reason": "adx_fail"}

    # not ema_ok
    return {"passes": False, "reason": "ema_fail"}

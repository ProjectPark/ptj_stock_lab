"""
product/engine_v4/sell.py
==========================
v4 매도 조건 판정 순수 함수 (5가지 조건).
"""
from __future__ import annotations


def check_atr_stop(
    entry_price: float,
    cur_price: float,
    entry_atr: float,
    market_mode: str,
    is_high_vol: bool,
    params: dict,
) -> bool:
    """
    ATR 기반 가격 손절 판정.

    소스: backtest_v4.py _process_sells() 섹션 1 (라인 1824~1845)

    로직:
    - atr_mult = 2.5 if market_mode == "bullish" else 1.5
    - atr_stop = entry_price - (atr_mult * max(entry_atr, 0.01))
    - is_high_vol → high_vol_stop = entry_price * (1 + high_vol_stop_loss_pct/100)
                     atr_stop = max(atr_stop, high_vol_stop)
    - cur_price <= atr_stop → True
    """
    high_vol_stop_loss_pct: float = params.get("high_vol_stop_loss_pct", -4.0)

    atr_mult = 2.5 if market_mode == "bullish" else 1.5
    atr_stop = entry_price - (atr_mult * max(entry_atr, 0.01))

    if is_high_vol:
        high_vol_stop = entry_price * (1.0 + high_vol_stop_loss_pct / 100.0)
        atr_stop = max(atr_stop, high_vol_stop)

    return cur_price <= atr_stop


def check_time_stop(
    elapsed_market_min: float,
    market_mode: str,
    net_pnl_pct: float,
    params: dict,
) -> dict:
    """
    시간 손절 판정.

    소스: backtest_v4.py _process_sells() 섹션 2 (라인 1847~1871)

    Returns:
        {"should_sell": bool, "carry": bool}
        - should_sell=True: 즉시 매도
        - carry=True: 강세장 익절 미달 → 다음날 carry로 보유

    로직:
    - elapsed_market_min < max_hold_hours * 60 → {"should_sell": False, "carry": False}
    - elapsed >= limit:
        - bullish: net_pnl_pct >= take_profit_pct → {"should_sell": True, "carry": False}
                   else → {"should_sell": False, "carry": True}
        - 기타: → {"should_sell": True, "carry": False}
    """
    max_hold_hours: float = params.get("max_hold_hours", 5.0)
    take_profit_pct: float = params.get("take_profit_pct", 4.0)

    time_limit_min = max_hold_hours * 60

    if elapsed_market_min < time_limit_min:
        return {"should_sell": False, "carry": False}

    if market_mode == "bullish":
        if net_pnl_pct >= take_profit_pct:
            return {"should_sell": True, "carry": False}
        else:
            return {"should_sell": False, "carry": True}
    else:
        return {"should_sell": True, "carry": False}


def check_fixed_tp(
    net_pnl_pct: float,
    ticker: str,
    overheat_origin: str,
    signal_type: str,
    params: dict,
) -> bool:
    """
    v4 고정 익절 판정 (pair_fixed_tp_pct 이상).

    소스: backtest_v4.py _process_sells() 섹션 5 (라인 1910~1927)

    조건:
    - signal_type == "twin"
    - (overheat_origin or ticker) in pair_fixed_tp_stocks
    - net_pnl_pct >= pair_fixed_tp_pct
    """
    pair_fixed_tp_stocks: list[str] = params.get("pair_fixed_tp_stocks", ["SOXL", "CONL", "IRE"])
    pair_fixed_tp_pct: float = params.get("pair_fixed_tp_pct", 6.5)

    if signal_type != "twin":
        return False

    origin = overheat_origin or ticker
    if origin not in pair_fixed_tp_stocks:
        return False

    return net_pnl_pct >= pair_fixed_tp_pct


def check_staged_sell_trigger(
    gap_pct: float,
    ticker: str,
    overheat_origin: str,
    signal_type: str,
    params: dict,
) -> dict:
    """
    쌍둥이 갭 수렴 매도 트리거 판정.

    소스: backtest_v4.py _process_sells() 섹션 7 (라인 1941~1989)

    Returns:
        {
            "trigger": bool,
            "mode": "fixed_tp" | "staged",  # fixed_tp 대상이면 즉시 40%+고정익절, 아니면 분할매도
            "immediate_sell_pct": float,     # 즉시 매도 비율
        }

    로직:
    - gap_pct <= pair_gap_sell_threshold → trigger=True
    - origin in pair_fixed_tp_stocks → mode="fixed_tp", immediate_sell_pct=pair_immediate_sell_pct
    - 아닌 경우 → mode="staged", immediate_sell_pct=pair_sell_first_pct
    """
    pair_gap_sell_threshold: float = params.get("pair_gap_sell_threshold", 9.0)
    pair_fixed_tp_stocks: list[str] = params.get("pair_fixed_tp_stocks", ["SOXL", "CONL", "IRE"])
    pair_immediate_sell_pct: float = params.get("pair_immediate_sell_pct", 0.40)
    pair_sell_first_pct: float = params.get("pair_sell_first_pct", 0.80)

    if gap_pct > pair_gap_sell_threshold:
        return {"trigger": False, "mode": "", "immediate_sell_pct": 0.0}

    origin = overheat_origin or ticker
    if signal_type == "twin" and origin in pair_fixed_tp_stocks:
        return {
            "trigger": True,
            "mode": "fixed_tp",
            "immediate_sell_pct": pair_immediate_sell_pct,
        }

    return {
        "trigger": True,
        "mode": "staged",
        "immediate_sell_pct": pair_sell_first_pct,
    }

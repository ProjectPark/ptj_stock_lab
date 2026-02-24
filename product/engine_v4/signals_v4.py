"""
product/engine_v4/signals_v4.py
=================================
v4 엔진 통합 진입점.

generate_v4_signals() 는 ptj_stock의 generate_all_signals() 를 완전 대체한다.
내부적으로 circuit_breaker / sideways / swing / twin_pairs / conditional / sell
모듈을 조합해 단일 결과 dict를 반환한다.

사용법:
    from product.engine_v4 import generate_v4_signals
    from product.engine_v4.params import DEFAULT_PARAMS

    result = generate_v4_signals(
        {"BITU": {"change_pct": 14.0, "close": 50.0}},
        poly_probs={"rate_hike": 0.55},
    )
    cb = result["circuit_breaker"]   # CBState
    assert cb.block_new_buys         # BTC 급등 CB-4 발동
"""
from __future__ import annotations

from .circuit_breaker import CBState, evaluate_circuit_breaker
from .sideways import check_sideways
from .swing import SwingSignal, check_swing_entry
from .twin_pairs import evaluate_twin_pairs
from .conditional import check_conl_entry_filter
from .sell import check_atr_stop, check_time_stop, check_fixed_tp, check_staged_sell_trigger
from .params import DEFAULT_PARAMS


def generate_v4_signals(
    changes: dict[str, dict],
    *,
    indicators: dict | None = None,
    positions: dict | None = None,
    poly_probs: dict | None = None,
    cb_state: CBState | None = None,
    is_swing_active: bool = False,
    params: dict | None = None,
) -> dict:
    """
    v4 통합 신호 생성.

    Args:
        changes: 티커별 변동 정보.
            형식: {
                "BITU": {"change_pct": 3.2, "close": 45.0},
                "VIX":  {"change_pct": 4.0},
                "GLD":  {"change_pct": 0.5},
                ...
            }
            - BITU.change_pct → BTC 프록시 (circuit_breaker CB-3/4)
            - VIX.change_pct  → CB-1 / 스윙 VIX 트리거
            - GLD.change_pct  → CB-2

        indicators: 사전 계산된 기술 지표 (sideways 판정 + CONL 필터).
            형식: {
                "sideways": {
                    "atr_decline": bool,
                    "volume_decline": bool,
                    "ema_flat": bool,
                    "rsi_box": bool,
                    "bb_narrow": bool,
                    "range_narrow": bool,   # 선택
                },
                "conl": {
                    "adx": float,
                    "ema_slope_pct": float,
                },
            }
            None이면 횡보장/CONL 필터 비활성.

        positions: 현재 보유 포지션 dict.
            형식: {ticker: {"entry_price": float, "entry_atr": float,
                            "signal_type": str, "overheat_origin": str,
                            "elapsed_market_min": float, "net_pnl_pct": float}}
            None이면 매도 신호 비활성.

        poly_probs: Polymarket 확률 dict.
            형식: {"rate_hike": 0.55, "btc_up": 0.70, ...}
            None이면 CB-5(금리) 비활성.

        cb_state: 이전 호출에서 반환된 CBState (cooldown 카운터 보존).
            None이면 초기 상태(CBState())로 시작.

        is_swing_active: 현재 스윙 모드 활성 여부.
            True면 swing / twin_pairs / conditional 신호 생성 건너뜀.

        params: 파라미터 오버라이드 dict.
            None이면 DEFAULT_PARAMS 사용.

    Returns:
        {
            "circuit_breaker": CBState,
                # block_all, block_new_buys, active_rules 포함
                # 다음 호출 시 이 CBState를 cb_state 인자로 전달해야 cooldown 유지됨

            "sideways": {
                "is_sideways": bool,
                "triggered_count": int,
                "signals": {지표별 충족 여부},
            },
                # indicators["sideways"] 없으면 {"is_sideways": False, ...}

            "swing": SwingSignal,
                # should_enter, trigger_type, targets, weight_pct

            "twin_pairs": [
                {
                    "pair": str,
                    "lead": str,
                    "follow": str,
                    "signal": "BUY" | "SELL" | "NONE",
                    "gap_pct": float,
                    "lead_pct": float,
                    "follow_pct": float,
                    "reason": str,
                },
                ...
            ],

            "conditional": {
                "conl": {"passes": bool, "reason": str},
            },
                # indicators["conl"] 없으면 {"conl": {"passes": False, "reason": "no_indicator"}}

            "sell_signals": {
                ticker: {
                    "atr_stop": bool,
                    "time_stop": {"should_sell": bool, "carry": bool},
                    "fixed_tp": bool,
                    "staged_sell": {"trigger": bool, "mode": str, "immediate_sell_pct": float},
                },
                ...
            },
                # positions 없으면 {}
        }

    Examples:
        >>> result = generate_v4_signals(
        ...     {"BITU": {"change_pct": 14.0, "close": 50.0}},
        ... )
        >>> result["circuit_breaker"].block_new_buys
        True  # BTC 14% 급등 → CB-4 발동
    """
    p: dict = params if params is not None else DEFAULT_PARAMS
    state: CBState = cb_state if cb_state is not None else CBState()

    # ── 1. 서킷브레이커 ────────────────────────────────────────────
    new_cb: CBState = evaluate_circuit_breaker(changes, poly_probs, state, p)

    # ── 2. 횡보장 ──────────────────────────────────────────────────
    sideways_indicators = (indicators or {}).get("sideways", {})
    if sideways_indicators:
        sideways_result = check_sideways(sideways_indicators, p)
    else:
        sideways_result = {"is_sideways": False, "triggered_count": 0, "signals": {}}

    # ── 3. 스윙 진입 ───────────────────────────────────────────────
    swing_signal: SwingSignal = check_swing_entry(changes, is_swing_active, p)

    # ── 4. 쌍둥이 페어 ────────────────────────────────────────────
    if is_swing_active or sideways_result["is_sideways"] or new_cb.block_all:
        twin_pairs_result: list[dict] = []
    else:
        twin_pairs_result = evaluate_twin_pairs(changes, p)

    # ── 5. 조건부 진입 필터 ────────────────────────────────────────
    conl_indicators = (indicators or {}).get("conl", {})
    if conl_indicators:
        conl_filter = check_conl_entry_filter(conl_indicators, p)
    else:
        conl_filter = {"passes": False, "reason": "no_indicator"}

    conditional_result = {"conl": conl_filter}

    # ── 6. 포지션별 매도 신호 ──────────────────────────────────────
    sell_signals: dict[str, dict] = {}
    if positions:
        for ticker, pos in positions.items():
            entry_price: float = pos.get("entry_price", 0.0)
            entry_atr: float = pos.get("entry_atr", 0.0)
            signal_type: str = pos.get("signal_type", "")
            overheat_origin: str = pos.get("overheat_origin", "")
            elapsed_market_min: float = pos.get("elapsed_market_min", 0.0)
            net_pnl_pct: float = pos.get("net_pnl_pct", 0.0)
            market_mode: str = pos.get("market_mode", "neutral")
            is_high_vol: bool = bool(pos.get("is_high_vol", False))
            cur_price: float = pos.get("cur_price", entry_price)
            gap_pct: float = pos.get("gap_pct", float("inf"))

            sell_signals[ticker] = {
                "atr_stop": check_atr_stop(
                    entry_price, cur_price, entry_atr, market_mode, is_high_vol, p
                ),
                "time_stop": check_time_stop(elapsed_market_min, market_mode, net_pnl_pct, p),
                "fixed_tp": check_fixed_tp(net_pnl_pct, ticker, overheat_origin, signal_type, p),
                "staged_sell": check_staged_sell_trigger(
                    gap_pct, ticker, overheat_origin, signal_type, p
                ),
            }

    return {
        "circuit_breaker": new_cb,
        "sideways": sideways_result,
        "swing": swing_signal,
        "twin_pairs": twin_pairs_result,
        "conditional": conditional_result,
        "sell_signals": sell_signals,
    }

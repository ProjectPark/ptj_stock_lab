"""
PTJ 매매법 v4 - 시그널 엔진 (선별 매매형)
==========================================
v2 시그널을 기반으로 v4 변경사항 적용:
- 쌍둥이 ENTRY 갭: 1.5% → 2.2%
- COIN/CONL 트리거: 3.0% → 4.5%
- 횡보장 감지 (5개 지표 중 3개 이상 → 현금 100%)
- 진입 시간 제한 (V4_ENTRY_CUTOFF 이후 매수 금지)

config 의존 없음 — 모든 파라미터는 인자로 전달.
v2 시그널 함수를 재사용하고, v4 고유 로직만 추가.
"""

from __future__ import annotations

import signals_v2


# ---------------------------------------------------------------------------
# 1. Market mode (v4: sideways 추가)
# ---------------------------------------------------------------------------

def determine_market_mode_v4(
    poly_probs: dict | None,
    sideways_active: bool = False,
) -> str:
    """Polymarket 확률 + 횡보장 상태로 시황 모드를 결정한다.

    우선순위: sideways > bearish > bullish > normal

    Parameters
    ----------
    poly_probs : dict | None
        ``{"btc_up": float, "ndx_up": float, "eth_up": float}``
    sideways_active : bool
        횡보장 모드 활성 여부 (외부에서 evaluate_sideways로 판정).

    Returns
    -------
    str
        ``"sideways"`` | ``"bearish"`` | ``"bullish"`` | ``"normal"``
    """
    if sideways_active:
        return "sideways"

    # v2 로직 그대로 사용
    return signals_v2.determine_market_mode(poly_probs)


# ---------------------------------------------------------------------------
# 2. Sideways detection (v4 신규)
# ---------------------------------------------------------------------------

def evaluate_sideways(
    indicators: dict[str, bool] | None = None,
    poly_probs: dict | None = None,
    changes: dict | None = None,
    gap_fail_count: int = 0,
    trigger_fail_count: int = 0,
    *,
    poly_low: float = 0.40,
    poly_high: float = 0.60,
    gld_threshold: float = 0.3,
    gap_fail_threshold: int = 2,
    trigger_fail_threshold: int = 2,
    index_threshold: float = 0.5,
    min_signals: int = 3,
) -> dict:
    """횡보장 지표를 평가하여 횡보장 여부를 판정한다.

    Parameters
    ----------
    poly_probs : dict | None
        Polymarket 일일 확률.
    changes : dict
        현재 가격 변동 ``{ticker: {"change_pct": float, ...}}``.
    gap_fail_count : int
        당일 갭 ≥ ENTRY 후 수렴 실패 횟수 (엔진에서 추적).
    trigger_fail_count : int
        당일 트리거 발동 후 목표 수익 미달 횟수 (엔진에서 추적).
    poly_low / poly_high : float
        Polymarket 횡보 범위 (기본 0.40~0.60).
    gld_threshold : float
        GLD |등락률| 기준 (기본 0.3%).
    gap_fail_threshold : int
        갭 수렴 실패 횟수 기준 (기본 2회).
    trigger_fail_threshold : int
        트리거 불발 횟수 기준 (기본 2회).
    index_threshold : float
        SPY·QQQ |등락률| 기준 (기본 0.5%).
    min_signals : int
        횡보장 판정 최소 충족 지표 수 (기본 3).

    Returns
    -------
    dict
        ``indicators``: 각 지표 충족 여부,
        ``count``: 충족 지표 수,
        ``is_sideways``: 횡보장 판정 결과,
        ``message``: 판정 설명.
    """
    if indicators is not None:
        total = len(indicators) if indicators else 0
        count = sum(1 for v in indicators.values() if v)
        is_sideways = count >= min_signals
        met = [k for k, v in indicators.items() if v]
        if is_sideways:
            message = f"횡보장 감지 ({count}/{total} 충족: {', '.join(met)}) → 현금 100%"
        else:
            message = f"횡보장 아님 ({count}/{total} 충족, 기준 {min_signals}개)"
        return {
            "indicators": indicators,
            "count": count,
            "is_sideways": is_sideways,
            "message": message,
        }

    indicators = {}

    # 1. Polymarket 확률 40~60% 범위
    if poly_probs is not None:
        btc_up = poly_probs.get("btc_up", 0.5)
        poly_in_range = poly_low <= btc_up <= poly_high
    else:
        poly_in_range = False  # 데이터 없으면 평가 불가 → 미충족
    indicators["poly_range"] = poly_in_range

    # 2. GLD |등락률| ≤ threshold
    changes = changes or {}
    gld_data = changes.get("GLD", {})
    gld_pct = abs(gld_data.get("change_pct", 0.0))
    indicators["gld_flat"] = gld_pct <= gld_threshold

    # 3. 쌍둥이 갭 수렴 실패
    indicators["gap_fail"] = gap_fail_count >= gap_fail_threshold

    # 4. COIN/CONL 트리거 불발
    indicators["trigger_fail"] = trigger_fail_count >= trigger_fail_threshold

    # 5. SPY·QQQ 모두 |등락률| ≤ threshold
    spy_pct = abs(changes.get("SPY", {}).get("change_pct", 0.0))
    qqq_pct = abs(changes.get("QQQ", {}).get("change_pct", 0.0))
    indicators["index_flat"] = spy_pct <= index_threshold and qqq_pct <= index_threshold

    count = sum(1 for v in indicators.values() if v)
    is_sideways = count >= min_signals

    met = [k for k, v in indicators.items() if v]
    total = len(indicators)
    if is_sideways:
        message = f"횡보장 감지 ({count}/{total} 충족: {', '.join(met)}) → 현금 100%"
    else:
        message = f"횡보장 아님 ({count}/{total} 충족, 기준 {min_signals}개)"

    return {
        "indicators": indicators,
        "count": count,
        "is_sideways": is_sideways,
        "message": message,
    }


# ---------------------------------------------------------------------------
# 3. Twin pairs (v4: entry threshold 변경)
# ---------------------------------------------------------------------------

def check_twin_pairs_v4(
    changes: dict,
    pairs: dict,
    entry_threshold: float = 2.2,
) -> list[dict]:
    """쌍둥이 페어 갭 분석 — v4: ENTRY 기준 2.2%.

    Parameters
    ----------
    changes : dict
        가격 변동 딕셔너리.
    pairs : dict
        페어 설정.
    entry_threshold : float
        ENTRY 진입 갭 기준 (기본 2.2%).

    Returns
    -------
    list[dict]
        각 follow 별 시그널.
    """
    results: list[dict] = []

    for _key, pair_cfg in pairs.items():
        lead = pair_cfg["lead"]
        follows = pair_cfg["follow"]
        label = pair_cfg.get("label", _key)

        lead_data = changes.get(lead, {})
        lead_pct = lead_data.get("change_pct", 0.0)

        for follow_ticker in follows:
            follow_data = changes.get(follow_ticker, {})
            follow_pct = follow_data.get("change_pct", 0.0)
            gap = lead_pct - follow_pct

            if gap <= 0.9 and follow_pct > 0:
                signal = "SELL"
                message = (
                    f"{label} | {follow_ticker} 갭 {gap:+.2f}% ≤ 0.9% "
                    f"& 양봉 → 매도"
                )
            elif gap >= entry_threshold:
                signal = "ENTRY"
                message = (
                    f"{label} | {lead} +{lead_pct:.2f}% vs "
                    f"{follow_ticker} +{follow_pct:.2f}% (갭 {gap:+.2f}%) → 진입"
                )
            else:
                signal = "HOLD"
                message = (
                    f"{label} | {lead} {lead_pct:+.2f}% vs "
                    f"{follow_ticker} {follow_pct:+.2f}% (갭 {gap:+.2f}%) → 대기"
                )

            results.append({
                "pair": label,
                "lead": lead,
                "follow": follow_ticker,
                "lead_pct": lead_pct,
                "follow_pct": follow_pct,
                "gap": gap,
                "signal": signal,
                "message": message,
            })

    return results


# ---------------------------------------------------------------------------
# 4. Aggregate (v4)
# ---------------------------------------------------------------------------

def generate_all_signals_v4(
    changes: dict,
    poly_probs: dict | None = None,
    pairs: dict | None = None,
    sideways_active: bool = False,
    *,
    # v4 파라미터
    entry_threshold: float = 2.2,
    coin_trigger_pct: float = 4.5,
    coin_sell_profit_pct: float = 3.0,
    coin_sell_bearish_pct: float = 0.3,
    conl_trigger_pct: float = 4.5,
    conl_sell_profit_pct: float = 2.8,
    conl_sell_avg_pct: float = 1.0,
    stop_loss_normal: float = -3.0,
    stop_loss_bullish: float = -8.0,
    brku_weight: float = 10.0,
) -> dict:
    """v4 시그널 전체 생성.

    v2 함수를 재사용하되 v4 파라미터로 호출한다.

    Returns
    -------
    dict
        ``market_mode``, ``gold``, ``twin_pairs``, ``conditional_coin``,
        ``conditional_conl``, ``stop_loss``, ``bearish``
    """
    if pairs is None:
        pairs = {}

    # 1. 시황 판단 (횡보장 반영)
    market_mode = determine_market_mode_v4(poly_probs, sideways_active)

    # 2. 금 시그널 (v2 동일)
    gld_data = changes.get("GLD", {})
    gld_pct = gld_data.get("change_pct", 0.0)
    gold = signals_v2.check_gold_signal_v2(gld_pct)

    # 3. 쌍둥이 페어 (v4: entry threshold 2.2%)
    twin_pairs = check_twin_pairs_v4(changes, pairs, entry_threshold=entry_threshold)

    # 4. 조건부 COIN (v4: trigger 4.5%)
    conditional_coin = signals_v2.check_conditional_coin_v2(
        changes,
        mode=market_mode if market_mode != "sideways" else "normal",
        trigger_pct=coin_trigger_pct,
        sell_profit_pct=coin_sell_profit_pct,
        sell_bearish_pct=coin_sell_bearish_pct,
    )

    # 5. 조건부 CONL (v4: trigger 4.5%)
    conditional_conl = signals_v2.check_conditional_conl_v2(
        changes,
        trigger_pct=conl_trigger_pct,
        sell_profit_pct=conl_sell_profit_pct,
        sell_avg_pct=conl_sell_avg_pct,
    )

    # 6. 손절 (v2 동일)
    effective_mode = market_mode if market_mode != "sideways" else "normal"
    stop_loss = signals_v2.check_stop_loss_v2(
        changes,
        mode=effective_mode,
        normal_pct=stop_loss_normal,
        bullish_pct=stop_loss_bullish,
    )

    # 7. 하락장 방어 (v2 동일)
    bearish = signals_v2.check_bearish_v2(effective_mode, brku_weight_pct=brku_weight)

    return {
        "market_mode": market_mode,
        "gold": gold,
        "twin_pairs": twin_pairs,
        "conditional_coin": conditional_coin,
        "conditional_conl": conditional_conl,
        "stop_loss": stop_loss,
        "bearish": bearish,
    }

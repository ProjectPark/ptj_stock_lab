"""
PTJ 매매법 v2 - 시그널 엔진
============================
v2 매매 규칙을 순수 함수로 구현. config 의존 없음 — 모든 파라미터는 인자로 전달.

주요 변경 (v1 대비):
- Polymarket 기반 시황 판단 (bullish/bearish/normal)
- 쌍둥이 multi-follow + 0.9% 매도
- CONL 조건부 매수/매도 (ETHU/XXRP/SOLT 각각 +3%)
- 손절 라인 시황 연동 (-3% / -8%)
- 하락장: BRKU 10% 고정
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# 1. Market mode
# ---------------------------------------------------------------------------

def determine_market_mode(poly_probs: dict | None) -> str:
    """Polymarket 일일 확률로부터 시황 모드를 결정한다.

    Parameters
    ----------
    poly_probs : dict | None
        ``{"btc_up": float, "ndx_up": float, "eth_up": float}``
        각 값은 0.0 ~ 1.0 사이의 확률. None이면 ``"normal"`` 반환.

    Returns
    -------
    str
        ``"bullish"`` | ``"bearish"`` | ``"normal"``
    """
    if poly_probs is None:
        return "normal"

    btc_up = poly_probs.get("btc_up", 0.0)
    ndx_up = poly_probs.get("ndx_up", 0.0)
    eth_up = poly_probs.get("eth_up", 0.0)

    # Bullish: BTC 상승확률 >= 70%
    if btc_up >= 0.70:
        return "bullish"

    # Bearish: 세 지표 모두 <= 20%
    if btc_up <= 0.20 and ndx_up <= 0.20 and eth_up <= 0.20:
        return "bearish"

    return "normal"


# ---------------------------------------------------------------------------
# 2. Gold signal
# ---------------------------------------------------------------------------

def check_gold_signal_v2(gld_pct: float) -> dict:
    """GLD 변동률 기반 매매 허용 여부를 판단한다.

    Parameters
    ----------
    gld_pct : float
        GLD 변동률(%) — 양수이면 매매 금지, 음수이면 추가 매수 허용.

    Returns
    -------
    dict
        ``ticker``, ``change_pct``, ``warning``, ``allow_extra_buy``, ``message``
    """
    warning = gld_pct > 0
    allow_extra_buy = gld_pct < 0

    if warning:
        message = f"GLD +{gld_pct:.2f}% — 금 양봉, 전체 매매 금지"
    elif allow_extra_buy:
        message = f"GLD {gld_pct:.2f}% — 금 음봉, 현금 추가 매수 허용"
    else:
        message = "GLD 0.00% — 변동 없음, 기존 포지션 유지"

    return {
        "ticker": "GLD",
        "change_pct": gld_pct,
        "warning": warning,
        "allow_extra_buy": allow_extra_buy,
        "message": message,
    }


# ---------------------------------------------------------------------------
# 3. Twin pairs
# ---------------------------------------------------------------------------

def check_twin_pairs_v2(changes: dict, pairs: dict) -> list[dict]:
    """쌍둥이 페어 갭 분석 — multi-follow + 0.9 % 매도 기준.

    Parameters
    ----------
    changes : dict
        ``{ticker: {"change_pct": float, ...}, ...}``
    pairs : dict
        ``{key: {"lead": str, "follow": [str, ...], "label": str}, ...}``

    Returns
    -------
    list[dict]
        각 follow 별 시그널 딕셔너리 리스트.
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
            elif gap >= 1.5:
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
# 4. Conditional COIN
# ---------------------------------------------------------------------------

def check_conditional_coin_v2(
    changes: dict,
    mode: str = "normal",
    triggers: list[str] | None = None,
    target: str = "COIN",
    trigger_pct: float = 3.0,
    sell_profit_pct: float = 3.0,
    sell_bearish_pct: float = 0.3,
) -> dict:
    """ETHU/XXRP/SOLT 각각 +3% 이상이면 COIN 매수 시그널. (v2 변경)

    Parameters
    ----------
    changes : dict
        가격 변동 딕셔너리.
    mode : str
        시황 모드 (``"normal"`` | ``"bullish"`` | ``"bearish"``).
    triggers : list[str] | None
        트리거 종목 리스트. 기본값 ``["ETHU", "XXRP", "SOLT"]``.
    target : str
        매수 대상 종목. 기본값 ``"COIN"``.
    trigger_pct : float
        각 트리거의 최소 변동률 기준 (기본 3.0%).
    sell_profit_pct : float
        일반 시황 COIN 매도 순수익 기준 (기본 3.0%).
    sell_bearish_pct : float
        하락장 COIN 즉시 매도 순수익 기준 (기본 0.3%).

    Returns
    -------
    dict
        ``triggers``, ``all_above_threshold``, ``buy_signal``, ``target``,
        ``target_pct``, ``trigger_avg_pct``, ``sell_target_pct``, ``message``
    """
    if triggers is None:
        triggers = ["ETHU", "XXRP", "SOLT"]

    trigger_info: dict[str, dict] = {}
    all_above = True
    pct_sum = 0.0

    for ticker in triggers:
        data = changes.get(ticker, {})
        pct = data.get("change_pct", 0.0)
        above = pct >= trigger_pct
        if not above:
            all_above = False
        pct_sum += pct
        trigger_info[ticker] = {"change_pct": pct, "above_threshold": above}

    trigger_avg = pct_sum / len(triggers) if triggers else 0.0

    target_data = changes.get(target, {})
    target_pct = target_data.get("change_pct", 0.0)

    # 시황별 매도 기준
    sell_target = sell_bearish_pct if mode == "bearish" else sell_profit_pct

    if all_above:
        parts = ", ".join(
            f"{t} {trigger_info[t]['change_pct']:+.2f}%" for t in triggers
        )
        message = (
            f"{parts} 각각 ≥ +{trigger_pct:.1f}% → {target} 매수 "
            f"(매도 기준: 순수익 +{sell_target:.1f}%)"
        )
    else:
        below = [t for t in triggers if not trigger_info[t]["above_threshold"]]
        message = (
            f"{', '.join(below)} 미달 (기준 +{trigger_pct:.1f}%) "
            f"→ {target} 매수 보류"
        )

    return {
        "triggers": trigger_info,
        "all_above_threshold": all_above,
        "buy_signal": all_above,
        "target": target,
        "target_pct": target_pct,
        "trigger_avg_pct": round(trigger_avg, 4),
        "sell_target_pct": sell_target,
        "message": message,
    }


# ---------------------------------------------------------------------------
# 5. Conditional CONL
# ---------------------------------------------------------------------------

def check_conditional_conl_v2(
    changes: dict,
    trigger_pct: float = 3.0,
    sell_profit_pct: float = 2.8,
    sell_avg_pct: float = 1.0,
) -> dict:
    """CONL 조건부 매수/매도 — ETHU/XXRP/SOLT 각각 +3 % 이상이면 매수.

    Parameters
    ----------
    changes : dict
        가격 변동 딕셔너리.
    trigger_pct : float
        각 트리거의 최소 변동률 (기본 3.0%).
    sell_profit_pct : float
        CONL 순이익 매도 기준 (기본 2.8%).
    sell_avg_pct : float
        트리거 평균이 이 이하로 하락 시 매도 (기본 1.0%).

    Returns
    -------
    dict
        ``triggers``, ``all_above_threshold``, ``trigger_avg_pct``,
        ``sell_on_avg_drop``, ``conl_pct``, ``buy_signal``, ``message``
    """
    trigger_tickers = ["ETHU", "XXRP", "SOLT"]
    trigger_info: dict[str, dict] = {}
    all_above = True
    pct_sum = 0.0

    for ticker in trigger_tickers:
        data = changes.get(ticker, {})
        pct = data.get("change_pct", 0.0)
        above = pct >= trigger_pct
        if not above:
            all_above = False
        pct_sum += pct
        trigger_info[ticker] = {"change_pct": pct, "above_threshold": above}

    trigger_avg = pct_sum / len(trigger_tickers) if trigger_tickers else 0.0
    sell_on_avg_drop = trigger_avg < sell_avg_pct

    conl_data = changes.get("CONL", {})
    conl_pct = conl_data.get("change_pct", 0.0)

    if all_above:
        message = (
            f"ETHU/XXRP/SOLT 각각 ≥ +{trigger_pct:.1f}% → CONL 매수 "
            f"(평균 {trigger_avg:+.2f}%)"
        )
    else:
        below = [
            t for t in trigger_tickers
            if not trigger_info[t]["above_threshold"]
        ]
        message = (
            f"{', '.join(below)} 미달 (기준 +{trigger_pct:.1f}%) "
            f"→ CONL 매수 보류 (평균 {trigger_avg:+.2f}%)"
        )

    return {
        "triggers": trigger_info,
        "all_above_threshold": all_above,
        "trigger_avg_pct": round(trigger_avg, 4),
        "sell_on_avg_drop": sell_on_avg_drop,
        "conl_pct": conl_pct,
        "buy_signal": all_above,
        "message": message,
    }


# ---------------------------------------------------------------------------
# 6. Stop loss
# ---------------------------------------------------------------------------

def check_stop_loss_v2(
    changes: dict,
    mode: str = "normal",
    normal_pct: float = -3.0,
    bullish_pct: float = -8.0,
) -> list[dict]:
    """시황 연동 손절 체크.

    Parameters
    ----------
    changes : dict
        가격 변동 딕셔너리.
    mode : str
        ``"normal"`` | ``"bullish"`` | ``"bearish"``
    normal_pct : float
        일반/하락장 손절 기준 (기본 -3.0%).
    bullish_pct : float
        상승장 손절 기준 (기본 -8.0%).

    Returns
    -------
    list[dict]
        손절 해당 종목만 반환.
    """
    threshold = bullish_pct if mode == "bullish" else normal_pct
    results: list[dict] = []

    for ticker, data in changes.items():
        pct = data.get("change_pct", 0.0)
        if pct <= threshold:
            results.append({
                "ticker": ticker,
                "change_pct": pct,
                "stop_loss": True,
                "threshold": threshold,
                "message": (
                    f"{ticker} {pct:+.2f}% ≤ {threshold:.1f}% "
                    f"({mode} 모드) → 손절"
                ),
            })

    return results


# ---------------------------------------------------------------------------
# 7. Bearish defense
# ---------------------------------------------------------------------------

def check_bearish_v2(
    mode: str,
    brku_weight_pct: float = 10.0,
) -> dict:
    """하락장 방어 — BRKU 비중 고정.

    Parameters
    ----------
    mode : str
        시황 모드.
    brku_weight_pct : float
        BRKU 포트폴리오 비중 (기본 10%).

    Returns
    -------
    dict
        ``mode``, ``buy_brku``, ``brku_weight_pct``, ``message``
    """
    buy_brku = mode == "bearish"

    if buy_brku:
        message = f"하락장 감지 → BRKU {brku_weight_pct:.1f}% 매수, 나머지 현금 보유"
    else:
        message = f"시황 {mode} → BRKU 매수 불필요, 현금 보유"

    return {
        "mode": mode,
        "buy_brku": buy_brku,
        "brku_weight_pct": brku_weight_pct,
        "message": message,
    }


# ---------------------------------------------------------------------------
# 8. Aggregate
# ---------------------------------------------------------------------------

def generate_all_signals_v2(
    changes: dict,
    poly_probs: dict | None = None,
    pairs: dict | None = None,
    triggers: list[str] | None = None,
    coin_trigger_pct: float = 3.0,
    coin_sell_profit_pct: float = 3.0,
    coin_sell_bearish_pct: float = 0.3,
    conl_trigger_pct: float = 3.0,
    conl_sell_profit_pct: float = 2.8,
    conl_sell_avg_pct: float = 1.0,
    stop_loss_normal: float = -3.0,
    stop_loss_bullish: float = -8.0,
    brku_weight: float = 10.0,
) -> dict:
    """v2 시그널 전체 생성.

    Parameters
    ----------
    changes : dict
        ``{ticker: {"change_pct": float, ...}, ...}``
    poly_probs : dict | None
        Polymarket 확률. None이면 normal 모드.
    pairs : dict | None
        쌍둥이 페어 설정. None이면 빈 dict 사용.
    triggers : list[str] | None
        조건부 매수 트리거 종목. None이면 기본값 사용.
    coin_trigger_pct : float
        COIN 매수 트리거 — 각 트리거 종목 최소 변동률 (기본 3.0%).
    coin_sell_profit_pct : float
        COIN 일반 매도 순수익 기준 (기본 3.0%).
    coin_sell_bearish_pct : float
        COIN 하락장 즉시 매도 순수익 기준 (기본 0.3%).
    conl_trigger_pct : float
        CONL 매수 트리거 기준 (기본 3.0%).
    conl_sell_profit_pct : float
        CONL 순이익 매도 기준 (기본 2.8%).
    conl_sell_avg_pct : float
        CONL 트리거 평균 매도 기준 (기본 1.0%).
    stop_loss_normal : float
        일반/하락장 손절 기준 (기본 -3.0%).
    stop_loss_bullish : float
        상승장 손절 기준 (기본 -8.0%).
    brku_weight : float
        BRKU 포트폴리오 비중 (기본 10.0%).

    Returns
    -------
    dict
        ``market_mode``, ``gold``, ``twin_pairs``, ``conditional_coin``,
        ``conditional_conl``, ``stop_loss``, ``bearish``
    """
    if pairs is None:
        pairs = {}

    # 1. 시황 판단
    market_mode = determine_market_mode(poly_probs)

    # 2. 금 시그널
    gld_data = changes.get("GLD", {})
    gld_pct = gld_data.get("change_pct", 0.0)
    gold = check_gold_signal_v2(gld_pct)

    # 3. 쌍둥이 페어
    twin_pairs = check_twin_pairs_v2(changes, pairs)

    # 4. 조건부 COIN
    conditional_coin = check_conditional_coin_v2(
        changes,
        mode=market_mode,
        triggers=triggers,
        trigger_pct=coin_trigger_pct,
        sell_profit_pct=coin_sell_profit_pct,
        sell_bearish_pct=coin_sell_bearish_pct,
    )

    # 5. 조건부 CONL
    conditional_conl = check_conditional_conl_v2(
        changes,
        trigger_pct=conl_trigger_pct,
        sell_profit_pct=conl_sell_profit_pct,
        sell_avg_pct=conl_sell_avg_pct,
    )

    # 6. 손절
    stop_loss = check_stop_loss_v2(
        changes,
        mode=market_mode,
        normal_pct=stop_loss_normal,
        bullish_pct=stop_loss_bullish,
    )

    # 7. 하락장 방어
    bearish = check_bearish_v2(market_mode, brku_weight_pct=brku_weight)

    return {
        "market_mode": market_mode,
        "gold": gold,
        "twin_pairs": twin_pairs,
        "conditional_coin": conditional_coin,
        "conditional_conl": conditional_conl,
        "stop_loss": stop_loss,
        "bearish": bearish,
    }

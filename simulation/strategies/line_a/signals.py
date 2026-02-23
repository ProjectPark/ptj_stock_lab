# DEPRECATED: line_b_taejun 전략 엔진이 완전 대체. 호환성을 위해 보존.
"""
PTJ 매매법 - 시그널 엔진
========================
5가지 매매 규칙을 독립 함수로 구현:
  1. 금 시황 체크 (매매 금지 판단)
  2. 쌍둥이 페어 갭 시그널
  3. 조건부 매매 시그널 (COIN)
  4. 손절 체크
  5. 하락장 매매 시그널
"""
from __future__ import annotations

import config


def check_gold_signal(changes: dict[str, dict]) -> dict:
    """규칙 1: 금 상승 시 매매 금지.

    Returns:
        {"ticker": str, "change_pct": float, "warning": bool, "message": str}
    """
    gld = changes.get(config.GOLD_TICKER, {})
    change = gld.get("change_pct", 0.0)
    warning = change > 0
    if warning:
        msg = f"금(GLD) {change:+.2f}% — 매매 금지 권고"
    else:
        msg = f"금(GLD) {change:+.2f}% — 매매 가능"
    return {
        "ticker": config.GOLD_TICKER,
        "change_pct": change,
        "warning": warning,
        "message": msg,
    }


def check_twin_pairs(changes: dict[str, dict]) -> list[dict]:
    """규칙 2: 쌍둥이 페어 갭 분석.

    각 페어에 대해:
    - gap = lead_change - follow_change
    - |gap| <= SELL_THRESHOLD → 매도 시그널
    - |gap| >= ENTRY_THRESHOLD → 매수 검토 시그널

    Returns:
        [{"pair": str, "lead": str, "follow": str, "lead_pct": float,
          "follow_pct": float, "gap": float, "signal": str, "message": str}]
    """
    results = []
    for key, pair in config.TWIN_PAIRS.items():
        lead = pair["lead"]
        follow = pair["follow"]
        lead_chg = changes.get(lead, {}).get("change_pct", 0.0)
        follow_chg = changes.get(follow, {}).get("change_pct", 0.0)
        gap = round(lead_chg - follow_chg, 2)
        abs_gap = abs(gap)

        if abs_gap <= config.PAIR_GAP_SELL_THRESHOLD:
            signal = "SELL"
            msg = f"갭 {gap:+.2f}% (≤{config.PAIR_GAP_SELL_THRESHOLD}%) — 매도 시그널"
        elif abs_gap >= config.PAIR_GAP_ENTRY_THRESHOLD:
            signal = "ENTRY"
            msg = f"갭 {gap:+.2f}% (≥{config.PAIR_GAP_ENTRY_THRESHOLD}%) — 매수 검토"
        else:
            signal = "HOLD"
            msg = f"갭 {gap:+.2f}% — 관망"

        results.append({
            "pair": pair["label"],
            "lead": lead,
            "follow": follow,
            "lead_pct": lead_chg,
            "follow_pct": follow_chg,
            "gap": gap,
            "signal": signal,
            "message": msg,
        })
    return results


def check_conditional_buy(changes: dict[str, dict]) -> dict:
    """규칙 3: 조건부 매매 — ETHU/XXRP/SOLT 모두 양전 시 COIN 매수.

    Returns:
        {"triggers": dict, "all_positive": bool, "target": str,
         "target_pct": float, "message": str}
    """
    triggers = {}
    for t in config.CONDITIONAL_TRIGGERS:
        chg = changes.get(t, {}).get("change_pct", 0.0)
        triggers[t] = {"change_pct": chg, "positive": chg > 0}

    all_pos = all(v["positive"] for v in triggers.values())
    target_chg = changes.get(config.CONDITIONAL_TARGET, {}).get("change_pct", 0.0)

    if all_pos:
        msg = (
            f"ETHU/XXRP/SOLT 전체 양전 — "
            f"{config.CONDITIONAL_TARGET} 매수 시그널 (현재 {target_chg:+.2f}%)"
        )
    else:
        neg = [t for t, v in triggers.items() if not v["positive"]]
        msg = f"{', '.join(neg)} 음전 — 조건 미충족"

    return {
        "triggers": triggers,
        "all_positive": all_pos,
        "target": config.CONDITIONAL_TARGET,
        "target_pct": target_chg,
        "message": msg,
    }


def check_stop_loss(changes: dict[str, dict]) -> list[dict]:
    """규칙 4: 손절 라인 체크 (-3%).

    Returns:
        [{"ticker": str, "change_pct": float, "stop_loss": bool, "message": str}]
    """
    alerts = []
    for ticker, info in changes.items():
        chg = info.get("change_pct", 0.0)
        if chg <= config.STOP_LOSS_PCT:
            alerts.append({
                "ticker": ticker,
                "change_pct": chg,
                "stop_loss": True,
                "message": f"{ticker} {chg:+.2f}% — 손절 라인 도달 ({config.STOP_LOSS_PCT}%)",
            })
    return alerts


def check_bearish_signals(changes: dict[str, dict]) -> dict:
    """규칙 5: 하락장 매매 시그널.

    SPY/QQQ 하락 시 금 2x 또는 방어주 매수 검토.

    Returns:
        {"market_down": bool, "gold_up": bool, "spy_pct": float,
         "qqq_pct": float, "bearish_picks": list, "message": str}
    """
    spy_chg = changes.get("SPY", {}).get("change_pct", 0.0)
    qqq_chg = changes.get("QQQ", {}).get("change_pct", 0.0)
    gld_chg = changes.get(config.GOLD_TICKER, {}).get("change_pct", 0.0)

    market_down = spy_chg < 0 and qqq_chg < 0
    gold_up = gld_chg > 0

    picks = []
    for t in config.BEARISH_TICKERS:
        chg = changes.get(t, {}).get("change_pct", 0.0)
        picks.append({
            "ticker": t,
            "name": config.TICKERS[t]["name"],
            "change_pct": chg,
        })

    if market_down and gold_up:
        msg = (
            f"시장 하락 (SPY {spy_chg:+.2f}%, QQQ {qqq_chg:+.2f}%) + "
            f"금 상승 — 금 2x ETF 매수 검토"
        )
    elif market_down:
        msg = f"시장 하락 (SPY {spy_chg:+.2f}%, QQQ {qqq_chg:+.2f}%) — 방어주 검토"
    else:
        msg = f"시장 정상 (SPY {spy_chg:+.2f}%, QQQ {qqq_chg:+.2f}%)"

    return {
        "market_down": market_down,
        "gold_up": gold_up,
        "spy_pct": spy_chg,
        "qqq_pct": qqq_chg,
        "bearish_picks": picks,
        "message": msg,
    }


def generate_all_signals(changes: dict[str, dict]) -> dict:
    """모든 시그널을 한번에 생성."""
    return {
        "gold": check_gold_signal(changes),
        "twin_pairs": check_twin_pairs(changes),
        "conditional": check_conditional_buy(changes),
        "stop_loss": check_stop_loss(changes),
        "bearish": check_bearish_signals(changes),
    }

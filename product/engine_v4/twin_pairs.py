"""
product/engine_v4/twin_pairs.py
================================
v4 쌍둥이 페어 진입/매도 신호 순수 함수.

backtest_v4.py / TwinPairStrategy.evaluate() 로직을 순수 함수로 이식.
"""
from __future__ import annotations


def evaluate_twin_pairs(
    changes: dict[str, dict],
    params: dict,
) -> list[dict]:
    """
    쌍둥이 페어 진입/매도 신호 생성.

    Args:
        changes: {"BITU": {"change_pct": 3.2, "close": 45.0},
                  "MSTU": {"change_pct": 1.5}, ...}
        params: DEFAULT_PARAMS

    Returns:
        [
            {
                "pair": "coin",
                "lead": "BITU",
                "follow": "MSTU",
                "signal": "BUY" | "SELL" | "NONE",
                "gap_pct": 1.7,
                "lead_pct": 3.2,
                "follow_pct": 1.5,
                "reason": "gap_entry" | "gap_converge" | "no_signal",
            },
            ...
        ]

    BUY 조건: gap_pct >= pair_gap_entry_threshold  (lead - follow 갭)
    SELL 조건: gap_pct <= pair_gap_sell_threshold 이고 follow 수익 (수렴)
    """
    twin_pairs: dict = params.get("twin_pairs", {})
    entry_threshold: float = params.get("pair_gap_entry_threshold", 2.2)
    sell_threshold: float = params.get("pair_gap_sell_threshold", 9.0)

    results: list[dict] = []

    for pair_key, pair_cfg in twin_pairs.items():
        lead: str = pair_cfg["lead"]
        follows: list[str] = pair_cfg.get("follow", [])
        label: str = pair_cfg.get("label", pair_key)

        lead_data: dict = changes.get(lead, {})
        lead_pct: float = lead_data.get("change_pct", 0.0)

        for follow_ticker in follows:
            follow_data: dict = changes.get(follow_ticker, {})
            follow_pct: float = follow_data.get("change_pct", 0.0)

            # 갭 = 선행 변동률 - 후행 변동률
            gap_pct: float = lead_pct - follow_pct

            if gap_pct >= entry_threshold:
                signal = "BUY"
                reason = "gap_entry"
            elif gap_pct <= sell_threshold and follow_pct > 0:
                # 갭 수렴 + follow 양봉 → 매도 신호
                signal = "SELL"
                reason = "gap_converge"
            else:
                signal = "NONE"
                reason = "no_signal"

            results.append(
                {
                    "pair": label,
                    "lead": lead,
                    "follow": follow_ticker,
                    "signal": signal,
                    "gap_pct": gap_pct,
                    "lead_pct": lead_pct,
                    "follow_pct": follow_pct,
                    "reason": reason,
                }
            )

    return results

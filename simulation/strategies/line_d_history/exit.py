"""
JUN 매매법 v2 — 청산 신호
===========================
X-1: 목표 수익 +15%
X-2: 추세 붕괴 (MA20 -15% 이하)
X-3: 손절 -20%
X-4: 시간 초과 45거래일

우선순위: X-1 > X-3 > X-2 > X-4
출처: docs/rules/line_d/jun_trade_2023_v2.md  섹션 8
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExitResult:
    should_exit: bool
    reason: Optional[str] = None   # TARGET / STOP / TREND_BREAK / TIME / None
    pnl_pct: Optional[float] = None


def check_exit(
    avg_price: float,
    current_price: float,
    ma20: float,
    holding_days: int,
    params,
) -> ExitResult:
    """
    청산 조건 순서대로 평가 후 첫 번째 hit 반환.

    avg_price    : 평균 매수단가 (USD)
    current_price: 현재가 (USD)
    ma20         : 당일 MA20
    holding_days : 보유 거래일 수
    """
    pnl_pct = (current_price / avg_price - 1) * 100

    # X-1: 목표 수익
    if pnl_pct >= params.target_pct:
        return ExitResult(should_exit=True, reason="TARGET", pnl_pct=round(pnl_pct, 2))

    # X-3: 손절 (-20%)
    if pnl_pct <= params.stop_loss_pct:
        return ExitResult(should_exit=True, reason="STOP", pnl_pct=round(pnl_pct, 2))

    # X-2: 추세 붕괴 (현재가 < MA20 * (1 - 15%))
    if ma20 > 0 and current_price < ma20 * (1 + params.ma20_breakdown_pct / 100):
        return ExitResult(should_exit=True, reason="TREND_BREAK", pnl_pct=round(pnl_pct, 2))

    # X-4: 시간 초과
    if holding_days >= params.max_holding_days:
        return ExitResult(should_exit=True, reason="TIME", pnl_pct=round(pnl_pct, 2))

    return ExitResult(should_exit=False)

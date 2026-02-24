"""
JUN 매매법 v2 — 포지션 상태 관리
===================================
단일 포지션의 매수·피라미딩·청산을 추적한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class Position:
    ticker: str
    open_date: date              # 최초 매수일 (US 거래일)
    shares: float                # 총 보유 수량
    avg_price_usd: float         # 평균 매수단가 (USD)
    total_cost_usd: float        # 총 매수금액 (USD)
    add_count: int = 0           # 피라미딩 횟수 (최대 2회)
    last_buy_price_usd: float = 0.0   # 마지막 매수단가 (피라미딩 기준)
    entry_times: list = field(default_factory=list)  # 각 매수 timestamp

    # ── 메서드 ──────────────────────────────────────────────────────────────
    def add_shares(self, shares: float, price_usd: float, ts=None) -> None:
        """피라미딩 추가 매수 처리."""
        new_cost = shares * price_usd
        total_shares = self.shares + shares
        self.avg_price_usd = (self.total_cost_usd + new_cost) / total_shares
        self.total_cost_usd += new_cost
        self.shares = total_shares
        self.last_buy_price_usd = price_usd
        self.add_count += 1
        if ts is not None:
            self.entry_times.append(ts)

    def can_pyramid(self, current_price: float, rsi: float, params) -> bool:
        """피라미딩 가능 여부 확인."""
        if self.add_count >= params.pyramid_max_adds:
            return False
        if self.last_buy_price_usd <= 0:
            return False
        gain = (current_price / self.last_buy_price_usd - 1) * 100
        if gain < params.pyramid_gain_pct:
            return False
        if rsi < params.rsi_entry_min:
            return False
        return True

    def pnl_pct(self, current_price: float) -> float:
        """현재가 기준 손익률(%)."""
        if self.avg_price_usd <= 0:
            return 0.0
        return (current_price / self.avg_price_usd - 1) * 100

    def holding_days(self, today: date) -> int:
        """보유 거래일 수 (캘린더 일 기준 근사)."""
        return (today - self.open_date).days


@dataclass
class ClosedTrade:
    ticker: str
    open_date: date
    close_date: date
    avg_price_usd: float
    close_price_usd: float
    shares: float
    add_count: int
    pnl_pct: float
    pnl_usd: float
    exit_reason: str              # TARGET / STOP / TREND_BREAK / TIME

    @classmethod
    def from_position(
        cls,
        pos: Position,
        close_date: date,
        close_price: float,
        exit_reason: str,
    ) -> "ClosedTrade":
        pnl_pct = (close_price / pos.avg_price_usd - 1) * 100
        pnl_usd = (close_price - pos.avg_price_usd) * pos.shares
        return cls(
            ticker=pos.ticker,
            open_date=pos.open_date,
            close_date=close_date,
            avg_price_usd=pos.avg_price_usd,
            close_price_usd=close_price,
            shares=pos.shares,
            add_count=pos.add_count,
            pnl_pct=round(pnl_pct, 2),
            pnl_usd=round(pnl_usd, 4),
            exit_reason=exit_reason,
        )

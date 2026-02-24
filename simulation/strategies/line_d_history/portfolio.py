"""
JUN 매매법 v2 — 포트폴리오 관리
===================================
복수 포지션 보유·신규 진입 사이징·상태 추적.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from .indicators import vix_position_size
from .position import ClosedTrade, Position


class Portfolio:
    def __init__(self, params, fx_usd_krw: float = 1350.0):
        self.params = params
        self.fx = fx_usd_krw            # USD → KRW 환율 (고정 또는 일별 갱신)
        self.positions: dict[str, Position] = {}   # ticker → Position
        self.closed_trades: list[ClosedTrade] = []
        self.daily_snapshots: list[dict] = []

    # ── 포지션 사이징 ──────────────────────────────────────────────────────
    def position_size_usd(self, vix: float) -> float:
        """VIX 레짐별 1회 매수 금액(USD) 반환."""
        krw = vix_position_size(vix, self.params)
        return krw / self.fx if self.fx > 0 else 0.0

    # ── 신규 진입 ──────────────────────────────────────────────────────────
    def open_position(
        self,
        ticker: str,
        price_usd: float,
        vix: float,
        trade_date: date,
        ts=None,
    ) -> bool:
        """신규 포지션 개설. 이미 보유 중이면 False."""
        if ticker in self.positions:
            return False
        size_usd = self.position_size_usd(vix)
        if size_usd <= 0:
            return False
        shares = size_usd / price_usd
        pos = Position(
            ticker=ticker,
            open_date=trade_date,
            shares=shares,
            avg_price_usd=price_usd,
            total_cost_usd=size_usd,
            last_buy_price_usd=price_usd,
            entry_times=[ts] if ts else [],
        )
        self.positions[ticker] = pos
        return True

    # ── 피라미딩 ──────────────────────────────────────────────────────────
    def add_to_position(
        self,
        ticker: str,
        price_usd: float,
        vix: float,
        ts=None,
    ) -> bool:
        """피라미딩 추가 매수. 포지션 없거나 조건 미충족이면 False."""
        pos = self.positions.get(ticker)
        if pos is None:
            return False
        size_usd = self.position_size_usd(vix)
        if size_usd <= 0:
            return False
        shares = size_usd / price_usd
        pos.add_shares(shares, price_usd, ts)
        return True

    # ── 청산 ──────────────────────────────────────────────────────────────
    def close_position(
        self,
        ticker: str,
        price_usd: float,
        close_date: date,
        exit_reason: str,
    ) -> Optional[ClosedTrade]:
        """포지션 청산 후 ClosedTrade 반환."""
        pos = self.positions.pop(ticker, None)
        if pos is None:
            return None
        trade = ClosedTrade.from_position(pos, close_date, price_usd, exit_reason)
        self.closed_trades.append(trade)
        return trade

    # ── 스냅샷 ────────────────────────────────────────────────────────────
    def snapshot(self, today: date, price_map: dict[str, float]) -> None:
        """일별 포트폴리오 평가액 스냅샷 저장."""
        total_usd = sum(
            pos.shares * price_map.get(t, pos.avg_price_usd)
            for t, pos in self.positions.items()
        )
        self.daily_snapshots.append({
            "date": today,
            "open_positions": len(self.positions),
            "tickers": list(self.positions.keys()),
            "total_value_usd": round(total_usd, 2),
        })

    # ── 집계 ──────────────────────────────────────────────────────────────
    def summary(self) -> dict:
        trades = self.closed_trades
        if not trades:
            return {"trades": 0}
        wins   = [t for t in trades if t.pnl_pct > 0]
        losses = [t for t in trades if t.pnl_pct <= 0]
        exit_counts = {}
        for t in trades:
            exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1
        return {
            "trades":      len(trades),
            "win_rate":    round(len(wins) / len(trades) * 100, 1),
            "avg_pnl_pct": round(sum(t.pnl_pct for t in trades) / len(trades), 2),
            "total_pnl_usd": round(sum(t.pnl_usd for t in trades), 2),
            "wins":        len(wins),
            "losses":      len(losses),
            "exit_reasons": exit_counts,
        }

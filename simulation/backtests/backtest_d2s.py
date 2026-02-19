#!/usr/bin/env python3
"""
D2S 백테스트 — 일봉 기반 실거래 행동 추출 전략 검증
=====================================================
trading_rules_attach_v1.md의 D2S 규칙(R1~R16)을 market_daily.parquet로 검증.

기존 v2~v5 백테스트(1분봉)와 달리 일봉 단위로 작동한다.
쌍둥이 갭, 기술적 지표, 시황 필터, 캘린더 효과를 종합 평가.

Usage:
    pyenv shell ptj_stock_lab && python backtests/backtest_d2s.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from simulation.backtests import backtest_common

from simulation.strategies.taejun_attach_pattern.d2s_engine import (
    D2SEngine,
    D2SPosition,
    DailySnapshot,
    TechnicalPreprocessor,
)
from simulation.strategies.taejun_attach_pattern.params import D2S_ENGINE

# ============================================================
# 수수료 상수
# ============================================================
BUY_FEE_PCT = 0.35     # 매수 수수료 (수수료 0.25% + 환전 0.10%)
SELL_FEE_PCT = 0.353    # 매도 수수료 (수수료 0.25% + SEC 0.003% + 환전 0.10%)


# Polymarket 로더는 backtest_common.load_polymarket_daily를 사용


# ============================================================
# 백테스트 엔진
# ============================================================

@dataclass
class TradeRecord:
    """거래 기록."""
    date: date
    ticker: str
    side: str
    price: float
    qty: float
    amount: float
    fee: float
    pnl: float
    pnl_pct: float
    reason: str
    score: float = 0.0


class D2SBacktest:
    """D2S 일봉 백테스트 엔진."""

    def __init__(
        self,
        params: dict | None = None,
        start_date: date = date(2025, 3, 3),   # 기술적 지표 워밍업 후
        end_date: date = date(2026, 2, 17),
        use_fees: bool = True,
    ):
        self.params = params or D2S_ENGINE
        self.start_date = start_date
        self.end_date = end_date
        self.use_fees = use_fees

        self.engine = D2SEngine(self.params)
        self.preprocessor = TechnicalPreprocessor(self.params)

        # 상태
        self.cash = self.params["total_capital"]
        self.initial_capital = self.params["total_capital"]
        self.positions: dict[str, D2SPosition] = {}
        self.trades: list[TradeRecord] = []
        self.equity_curve: list[tuple[date, float]] = []

    # ------------------------------------------------------------------
    # 데이터 로드
    # ------------------------------------------------------------------

    def _load_data(self) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict]:
        """market_daily + Polymarket 데이터를 로드한다."""
        data_dir = _ROOT / "data"
        market_path = data_dir / "market" / "daily" / "market_daily.parquet"
        poly_dir = data_dir / "polymarket"

        print("[1/3] 데이터 로드")
        df = pd.read_parquet(market_path)
        print(f"  market_daily: {df.shape[0]} days, {len(set(c[0] for c in df.columns))} tickers")

        # 기술적 지표 계산
        print("[2/3] 기술적 지표 계산")
        tech = self.preprocessor.compute(df)
        print(f"  {len(tech)} tickers processed")

        # Polymarket (backtest_common의 파서 재사용)
        poly = backtest_common.load_polymarket_daily(poly_dir)

        return df, tech, poly

    # ------------------------------------------------------------------
    # 스냅샷 빌드
    # ------------------------------------------------------------------

    def _build_snapshot(
        self,
        trading_date: date,
        tech: dict[str, pd.DataFrame],
        poly: dict,
        spy_streak: int,
    ) -> DailySnapshot | None:
        """특정 날짜의 DailySnapshot을 빌드한다."""
        snap = DailySnapshot(
            trading_date=trading_date,
            weekday=trading_date.weekday(),
        )

        ts = pd.Timestamp(trading_date)
        found_any = False

        for ticker, ind_df in tech.items():
            if ts not in ind_df.index:
                continue
            row = ind_df.loc[ts]
            if pd.isna(row.get("close", np.nan)):
                continue

            found_any = True
            snap.closes[ticker] = float(row["close"])
            snap.opens[ticker] = float(row.get("open", row["close"]))
            snap.highs[ticker] = float(row.get("high", row["close"]))
            snap.lows[ticker] = float(row.get("low", row["close"]))
            snap.volumes[ticker] = float(row.get("volume", 0))
            snap.changes[ticker] = float(row.get("change_pct", 0))

            rsi_val = row.get("rsi")
            if not pd.isna(rsi_val):
                snap.rsi[ticker] = float(rsi_val)

            snap.macd_bullish[ticker] = bool(row.get("macd_bullish", False))

            bb_val = row.get("bb_pct_b")
            if not pd.isna(bb_val):
                snap.bb_pct_b[ticker] = float(bb_val)

            atr_val = row.get("atr")
            if not pd.isna(atr_val):
                snap.atr[ticker] = float(atr_val)

            aq_val = row.get("atr_quantile")
            if not pd.isna(aq_val):
                snap.atr_quantile[ticker] = float(aq_val)

            rv_val = row.get("rel_volume")
            if not pd.isna(rv_val):
                snap.rel_volume[ticker] = float(rv_val)

        if not found_any:
            return None

        # Polymarket
        poly_day = poly.get(trading_date, {})
        if "btc_up" in poly_day:
            snap.poly_btc_up = poly_day["btc_up"]

        snap.spy_streak = spy_streak

        return snap

    # ------------------------------------------------------------------
    # 매수/매도 실행
    # ------------------------------------------------------------------

    def _execute_buy(
        self, ticker: str, size: float, snap: DailySnapshot,
        reason: str, score: float,
    ) -> TradeRecord | None:
        """매수를 실행한다."""
        price = snap.closes.get(ticker, 0)
        if price <= 0 or self.cash <= 0:
            return None

        amount = self.cash * size
        amount = min(amount, self.cash)
        if amount < 1.0:
            return None

        fee = amount * BUY_FEE_PCT / 100 if self.use_fees else 0
        net_amount = amount - fee
        qty = net_amount / price

        # 포지션 업데이트
        existing = self.positions.get(ticker)
        if existing:
            total_cost = existing.cost_basis + net_amount
            total_qty = existing.qty + qty
            existing.entry_price = total_cost / total_qty
            existing.qty = total_qty
            existing.cost_basis = total_cost
            existing.dca_count += 1
        else:
            self.positions[ticker] = D2SPosition(
                ticker=ticker,
                entry_price=price,
                qty=qty,
                entry_date=snap.trading_date,
                cost_basis=net_amount,
            )

        self.cash -= amount

        trade = TradeRecord(
            date=snap.trading_date, ticker=ticker, side="BUY",
            price=price, qty=qty, amount=amount, fee=fee,
            pnl=0, pnl_pct=0, reason=reason, score=score,
        )
        self.trades.append(trade)
        return trade

    def _execute_sell(
        self, ticker: str, size: float, snap: DailySnapshot, reason: str,
    ) -> TradeRecord | None:
        """매도를 실행한다."""
        pos = self.positions.get(ticker)
        if not pos:
            return None

        price = snap.closes.get(ticker, 0)
        if price <= 0:
            return None

        sell_qty = pos.qty * size
        proceeds = sell_qty * price
        fee = proceeds * SELL_FEE_PCT / 100 if self.use_fees else 0
        net_proceeds = proceeds - fee

        cost_basis = pos.entry_price * sell_qty
        pnl = net_proceeds - cost_basis
        pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0

        pos.qty -= sell_qty
        pos.cost_basis -= cost_basis
        if pos.qty < 0.001:
            del self.positions[ticker]

        self.cash += net_proceeds

        trade = TradeRecord(
            date=snap.trading_date, ticker=ticker, side="SELL",
            price=price, qty=sell_qty, amount=proceeds, fee=fee,
            pnl=pnl, pnl_pct=pnl_pct, reason=reason,
        )
        self.trades.append(trade)
        return trade

    # ------------------------------------------------------------------
    # 메인 루프
    # ------------------------------------------------------------------

    def run(self, verbose: bool = True) -> "D2SBacktest":
        """백테스트를 실행한다."""
        df, tech, poly = self._load_data()

        # 거래일 목록
        all_dates = sorted(df.index)
        trading_dates = [
            d.date() if hasattr(d, 'date') else d
            for d in all_dates
            if self.start_date <= (d.date() if hasattr(d, 'date') else d) <= self.end_date
        ]

        if verbose:
            print(f"\n[3/3] D2S 백테스트 실행")
            print(f"  기간: {trading_dates[0]} ~ {trading_dates[-1]} ({len(trading_dates)}일)")
            print(f"  초기 자본: ${self.initial_capital:,.0f}")
            print()

        # SPY 연속 상승 추적
        spy_streak = 0

        for i, td in enumerate(trading_dates):
            snap = self._build_snapshot(td, tech, poly, spy_streak)
            if snap is None:
                continue

            # 당일 매수 횟수 추적
            daily_buy_counts: dict[str, int] = {}

            # 시그널 생성
            signals = self.engine.generate_daily_signals(
                snap, self.positions, daily_buy_counts,
            )

            # 시그널 실행 (매도 먼저, 매수 후)
            for sig in signals:
                if sig["action"] == "SELL":
                    self._execute_sell(
                        sig["ticker"], sig["size"], snap, sig["reason"],
                    )

            for sig in signals:
                if sig["action"] == "BUY":
                    trade = self._execute_buy(
                        sig["ticker"], sig["size"], snap,
                        sig["reason"], sig.get("score", 0),
                    )
                    if trade:
                        daily_buy_counts[sig["ticker"]] = (
                            daily_buy_counts.get(sig["ticker"], 0) + 1
                        )

            # SPY streak 업데이트
            spy_pct = snap.changes.get("SPY", 0)
            if spy_pct > 0:
                spy_streak += 1
            else:
                spy_streak = 0

            # 자산 스냅샷
            equity = self.cash + sum(
                snap.closes.get(t, pos.entry_price) * pos.qty
                for t, pos in self.positions.items()
            )
            self.equity_curve.append((td, equity))

            # 진행 표시
            if verbose and ((i + 1) % 50 == 0 or i == len(trading_dates) - 1):
                ret_pct = (equity / self.initial_capital - 1) * 100
                n_pos = len(self.positions)
                n_trades = len(self.trades)
                print(
                    f"  [{i+1:>3}/{len(trading_dates)}] {td}  "
                    f"자산: ${equity:,.0f} ({ret_pct:+.1f}%)  "
                    f"포지션: {n_pos}개  거래: {n_trades}건"
                )

        if verbose:
            print("\n  백테스트 완료!")

        return self

    # ------------------------------------------------------------------
    # 리포트
    # ------------------------------------------------------------------

    def report(self) -> dict:
        """종합 리포트를 생성한다."""
        sell_trades = [t for t in self.trades if t.side == "SELL"]
        buy_trades = [t for t in self.trades if t.side == "BUY"]

        wins = [t for t in sell_trades if t.pnl > 0]
        losses = [t for t in sell_trades if t.pnl <= 0]

        final_equity = self.equity_curve[-1][1] if self.equity_curve else self.initial_capital
        total_return = (final_equity / self.initial_capital - 1) * 100

        # MDD
        peak = 0
        mdd = 0
        for _, eq in self.equity_curve:
            peak = max(peak, eq)
            dd = (eq - peak) / peak * 100 if peak > 0 else 0
            mdd = min(mdd, dd)

        # Sharpe (일봉 기준)
        if len(self.equity_curve) > 1:
            equities = [eq for _, eq in self.equity_curve]
            returns = pd.Series(equities).pct_change().dropna()
            if returns.std() > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(252)
            else:
                sharpe = 0
        else:
            sharpe = 0

        total_fees = sum(t.fee for t in self.trades)
        total_pnl = sum(t.pnl for t in sell_trades)
        avg_pnl_pct = np.mean([t.pnl_pct for t in sell_trades]) if sell_trades else 0

        report = {
            "period": f"{self.start_date} ~ {self.end_date}",
            "trading_days": len(self.equity_curve),
            "initial_capital": self.initial_capital,
            "final_equity": round(final_equity, 2),
            "total_return_pct": round(total_return, 2),
            "mdd_pct": round(mdd, 2),
            "sharpe_ratio": round(sharpe, 3),
            "total_trades": len(self.trades),
            "buy_trades": len(buy_trades),
            "sell_trades": len(sell_trades),
            "win_trades": len(wins),
            "lose_trades": len(losses),
            "win_rate": round(len(wins) / len(sell_trades) * 100, 1) if sell_trades else 0,
            "avg_pnl_pct": round(avg_pnl_pct, 2),
            "total_pnl": round(total_pnl, 2),
            "total_fees": round(total_fees, 2),
            "remaining_positions": len(self.positions),
        }
        return report

    def print_report(self) -> None:
        """리포트를 출력한다."""
        r = self.report()
        print("\n" + "=" * 60)
        print("  D2S 백테스트 리포트 (attach v1)")
        print("=" * 60)
        print(f"  기간: {r['period']}")
        print(f"  거래일: {r['trading_days']}일")
        print(f"  초기 자본: ${r['initial_capital']:,.0f}")
        print(f"  최종 자산: ${r['final_equity']:,.0f}")
        print(f"  총 수익률: {r['total_return_pct']:+.2f}%")
        print(f"  MDD: {r['mdd_pct']:.2f}%")
        print(f"  Sharpe: {r['sharpe_ratio']:.3f}")
        print(f"  ---")
        print(f"  총 거래: {r['total_trades']}건 (매수 {r['buy_trades']} / 매도 {r['sell_trades']})")
        print(f"  승률: {r['win_rate']}% ({r['win_trades']}승 / {r['lose_trades']}패)")
        print(f"  평균 수익률: {r['avg_pnl_pct']:+.2f}%")
        print(f"  총 PnL: ${r['total_pnl']:+,.2f}")
        print(f"  총 수수료: ${r['total_fees']:,.2f}")
        print(f"  잔여 포지션: {r['remaining_positions']}개")
        print("=" * 60)

        # 종목별 통계
        if self.trades:
            print("\n  종목별 매도 성과:")
            sell_trades = [t for t in self.trades if t.side == "SELL"]
            ticker_stats: dict[str, list] = {}
            for t in sell_trades:
                ticker_stats.setdefault(t.ticker, []).append(t)
            for ticker in sorted(ticker_stats):
                trades = ticker_stats[ticker]
                wins = sum(1 for t in trades if t.pnl > 0)
                wr = wins / len(trades) * 100 if trades else 0
                avg = np.mean([t.pnl_pct for t in trades])
                total = sum(t.pnl for t in trades)
                print(
                    f"    {ticker:6s}: {len(trades):3d}건  "
                    f"승률 {wr:5.1f}%  평균 {avg:+6.2f}%  PnL ${total:+,.0f}"
                )

    def save_trades(self, path: Path | None = None) -> Path:
        """거래 기록을 CSV로 저장한다."""
        if path is None:
            path = _ROOT / "data" / "results" / "backtests" / "d2s_trades.csv"
        path.parent.mkdir(parents=True, exist_ok=True)

        records = []
        for t in self.trades:
            records.append({
                "date": t.date,
                "ticker": t.ticker,
                "side": t.side,
                "price": round(t.price, 4),
                "qty": round(t.qty, 4),
                "amount": round(t.amount, 2),
                "fee": round(t.fee, 2),
                "pnl": round(t.pnl, 2),
                "pnl_pct": round(t.pnl_pct, 2),
                "reason": t.reason,
                "score": round(t.score, 3),
            })
        df = pd.DataFrame(records)
        df.to_csv(path, index=False)
        print(f"\n  거래 기록 저장: {path}")
        return path


# ============================================================
# 메인
# ============================================================

def main():
    bt = D2SBacktest()
    bt.run(verbose=True)
    bt.print_report()
    bt.save_trades()


if __name__ == "__main__":
    main()

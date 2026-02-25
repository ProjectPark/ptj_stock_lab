#!/usr/bin/env python3
"""
D2S 백테스트 v2 — V-바운스 + 조기 손절 + DCA 레이어 제한
============================================================
trading_rules_attach_v2.md의 신규 규칙 검증:
  R17: 충격 V-바운스 포지션 2배 확대 (%B<0.15 + crash>10% + score≥0.87)
  R18: BB 하단 돌파 조기 손절 (3일 비회복 → 손절)
  DCA 레이어 제한: 최대 2레이어 (3레이어+ 승률 27% 이하)

v1 대비 개선 목표:
  - 강제청산(-$383 평균) → 조기 손절(소규모 손실)로 전환
  - 대박 이벤트 시 포지션 자동 2배 확대
  - 레이어 누적 손실 방지

Usage:
    pyenv shell ptj_stock_lab && python backtests/backtest_d2s_v2.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from simulation.strategies.line_c_d2s.params_d2s import D2S_ENGINE_V2
from simulation.strategies.line_c_d2s.d2s_engine import D2SPosition, DailySnapshot

from simulation.backtests.backtest_d2s import D2SBacktest, TradeRecord, BUY_FEE_PCT, SELL_FEE_PCT


class D2SBacktestV2(D2SBacktest):
    """D2S 백테스트 v2 — R17/R18/DCA 레이어 제한 추가.

    변경 사항
    ---------
    * _execute_buy() : DCA 레이어 체크 + V-바운스(R17) 자동 2배 사이징
    * run()          : R18 조기 손절 스캔 추가
    * print_report() : v2 라벨 + R17/R18 발생 통계
    """

    def __init__(
        self,
        params: dict | None = None,
        start_date: date = date(2025, 3, 3),
        end_date: date = date(2026, 2, 17),
        use_fees: bool = True,
        data_path=None,
        slippage_pct: float = 0.05,
    ):
        super().__init__(
            params=params or D2S_ENGINE_V2,
            start_date=start_date,
            end_date=end_date,
            use_fees=use_fees,
            data_path=data_path,
            slippage_pct=slippage_pct,
        )
        # BB 하단 돌파 포지션 메타 — ticker → breach_date
        self.position_meta: dict[str, date | None] = {}
        # 통계용 카운터
        self._r17_count = 0
        self._r18_count = 0

    # ------------------------------------------------------------------
    # R17: V-바운스 조건 체크
    # ------------------------------------------------------------------

    def _is_vbounce(self, ticker: str, snap: DailySnapshot, score: float) -> bool:
        """R17 V-바운스 발동 조건: %B<threshold + crash>threshold + score≥threshold."""
        bb = snap.bb_pct_b.get(ticker, 1.0)
        crash = snap.changes.get(ticker, 0.0)
        return (
            bb < self.params["vbounce_bb_threshold"]
            and crash < self.params["vbounce_crash_threshold"]
            and score >= self.params["vbounce_score_threshold"]
        )

    # ------------------------------------------------------------------
    # 매수 실행 — DCA 레이어 체크 + R17 사이징
    # ------------------------------------------------------------------

    def _execute_buy(
        self, ticker: str, size: float, snap: DailySnapshot,
        reason: str, score: float,
    ) -> TradeRecord | None:
        """v2 매수: DCA 레이어 제한 + V-바운스 2배 확대."""
        pos = self.positions.get(ticker)

        # ── DCA 레이어 제한 (R5 v2) ──
        if pos is not None:
            max_layers = self.params["dca_max_layers"]
            vbounce = self._is_vbounce(ticker, snap, score)
            if pos.dca_count >= max_layers and not vbounce:
                # 레이어 초과 + V-바운스 아님 → 매수 금지
                return None

        # ── V-바운스 사이징 (R17) ──
        vbounce = self._is_vbounce(ticker, snap, score)
        if vbounce:
            size = min(
                size * self.params["vbounce_size_multiplier"],
                self.params["vbounce_size_max"],
            )
            reason = f"[R17:vbounce×{self.params['vbounce_size_multiplier']:.0f}] {reason}"
            self._r17_count += 1

        # ── 실제 매수 (부모 메서드) ──
        trade = super()._execute_buy(ticker, size, snap, reason, score)

        # ── BB 하단 돌파 모니터링 (R18용) ──
        if trade is not None:
            bb = snap.bb_pct_b.get(ticker, 1.0)
            if bb < 0:
                # 처음 돌파한 날만 기록
                if self.position_meta.get(ticker) is None:
                    self.position_meta[ticker] = snap.trading_date
            elif ticker not in self.position_meta:
                self.position_meta[ticker] = None

        return trade

    # ------------------------------------------------------------------
    # R18: BB 하단 돌파 조기 손절 체크
    # ------------------------------------------------------------------

    def _check_r18_exit(self, ticker: str, pos: D2SPosition, snap: DailySnapshot) -> str | None:
        """R18 조기 손절 조건 확인.

        Returns
        -------
        str | None
            손절 사유 문자열 (손절 안 하면 None).
        """
        breach_date = self.position_meta.get(ticker)
        if breach_date is None:
            return None

        current_price = snap.closes.get(ticker, 0)
        if current_price <= 0 or pos.entry_price <= 0:
            return None

        days_since = (snap.trading_date - breach_date).days
        pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100

        early_days = self.params["early_stoploss_days"]
        early_recovery = self.params["early_stoploss_recovery"]

        if days_since > early_days and pnl_pct < early_recovery:
            return (
                f"R18:early_stoploss "
                f"(bb<0 {days_since}일 경과, pnl={pnl_pct:+.1f}% < +{early_recovery}%)"
            )
        return None

    # ------------------------------------------------------------------
    # 메인 루프 — R18 조기 손절 추가
    # ------------------------------------------------------------------

    def run(self, verbose: bool = True) -> "D2SBacktestV2":
        """v2 백테스트 실행 — R18 조기 손절 포함 (look-ahead bias 수정)."""
        df, tech, poly = self._load_data()

        all_dates = sorted(df.index)
        trading_dates = [
            d.date() if hasattr(d, "date") else d
            for d in all_dates
            if self.start_date <= (d.date() if hasattr(d, "date") else d) <= self.end_date
        ]

        if verbose:
            print(f"\n[3/3] D2S v2 백테스트 실행 (look-ahead bias 수정)")
            print(f"  기간: {trading_dates[0]} ~ {trading_dates[-1]} ({len(trading_dates)}일)")
            print(f"  초기 자본: ${self.initial_capital:,.0f}")
            print(f"  R17 V-바운스 임계: %B<{self.params['vbounce_bb_threshold']} "
                  f"+ crash<{self.params['vbounce_crash_threshold']}% "
                  f"+ score≥{self.params['vbounce_score_threshold']}")
            print(f"  R18 조기 손절: {self.params['early_stoploss_days']}일 내 "
                  f"+{self.params['early_stoploss_recovery']}% 미회복")
            print(f"  DCA 최대 레이어: {self.params['dca_max_layers']}")
            print()

        spy_streak = 0

        # T일 결정 → T+1일 시가에 체결
        pending_signals: list[dict] = []

        for i, td in enumerate(trading_dates):
            snap = self._build_snapshot(td, tech, poly, spy_streak)
            if snap is None:
                continue

            daily_buy_counts: dict[str, int] = {}

            # ── [A] 전날 결정 시그널을 오늘 시가에 체결 ────────────
            if pending_signals:
                # 매도 먼저
                for sig in pending_signals:
                    if sig["action"] == "SELL":
                        self._execute_sell(sig["ticker"], sig["size"], snap, sig["reason"])
                        if sig["ticker"] not in self.positions:
                            self.position_meta.pop(sig["ticker"], None)

                # 매수 (daily_entry_cap 체크 포함)
                daily_entry_cap = self.params.get("daily_new_entry_cap", 1.0)
                daily_entry_used = 0.0
                for sig in pending_signals:
                    if sig["action"] == "BUY":
                        entry_fraction = sig["size"]
                        if daily_entry_used + entry_fraction > daily_entry_cap:
                            continue
                        trade = self._execute_buy(
                            sig["ticker"], sig["size"], snap,
                            sig["reason"], sig.get("score", 0),
                        )
                        if trade:
                            daily_entry_used += entry_fraction
                            daily_buy_counts[sig["ticker"]] = (
                                daily_buy_counts.get(sig["ticker"], 0) + 1
                            )

            # ── [B] 오늘 종가 데이터로 시그널 결정 (내일 실행 예약) ──
            # daily_buy_counts는 오늘 체결 추적용 — 내일 신호 생성에는 빈 dict 전달
            new_signals = self.engine.generate_daily_signals(
                snap, self.positions, {},
            )

            # R18: 조기 손절 시그널 추가
            already_selling = {s["ticker"] for s in new_signals if s["action"] == "SELL"}
            for ticker, pos in list(self.positions.items()):
                if ticker in already_selling:
                    continue
                r18_reason = self._check_r18_exit(ticker, pos, snap)
                if r18_reason:
                    new_signals.append({
                        "action": "SELL",
                        "ticker": ticker,
                        "size": 1.0,
                        "reason": r18_reason,
                        "score": 0,
                    })
                    self._r18_count += 1

            pending_signals = new_signals

            # ── [C] SPY streak 업데이트 (오늘 종가 기반) ─────────────
            spy_pct = snap.changes.get("SPY", 0)
            spy_streak = spy_streak + 1 if spy_pct > 0 else 0

            # ── [D] 자산 스냅샷 (오늘 종가 기준) ─────────────────────
            equity = self.cash + sum(
                snap.closes.get(t, pos.entry_price) * pos.qty
                for t, pos in self.positions.items()
            )
            self.equity_curve.append((td, equity))

            if verbose and ((i + 1) % 50 == 0 or i == len(trading_dates) - 1):
                ret_pct = (equity / self.initial_capital - 1) * 100
                print(
                    f"  [{i+1:>3}/{len(trading_dates)}] {td}  "
                    f"자산: ${equity:,.0f} ({ret_pct:+.1f}%)  "
                    f"포지션: {len(self.positions)}개  "
                    f"R17: {self._r17_count}회  R18: {self._r18_count}회"
                )

        if verbose:
            print("\n  v2 백테스트 완료!")

        return self

    # ------------------------------------------------------------------
    # 리포트 — v2 추가 통계
    # ------------------------------------------------------------------

    def print_report(self) -> None:
        """v2 리포트 출력."""
        r = self.report()
        print("\n" + "=" * 60)
        print("  D2S 백테스트 리포트 (attach v2)")
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
        print(f"  ---")
        print(f"  [R17] V-바운스 확대 발동: {self._r17_count}회")
        print(f"  [R18] 조기 손절 발동: {self._r18_count}회")
        print("=" * 60)

        # 청산 유형별 분석
        sell_trades = [t for t in self.trades if t.side == "SELL"]
        tp_trades = [t for t in sell_trades if "take_profit" in t.reason]
        r18_trades = [t for t in sell_trades if "R18" in t.reason]
        hd_trades = [t for t in sell_trades if "hold_days" in t.reason]

        print("\n  청산 유형별 성과:")
        for label, tlist in [
            ("익절(take_profit)", tp_trades),
            ("R18 조기손절     ", r18_trades),
            ("강제청산(hold_days)", hd_trades),
        ]:
            if tlist:
                avg_pnl = np.mean([t.pnl for t in tlist])
                wr = sum(1 for t in tlist if t.pnl > 0) / len(tlist) * 100
                print(f"    {label}: {len(tlist):3d}건  승률 {wr:5.1f}%  평균 PnL ${avg_pnl:+.0f}")

        # 종목별 통계
        if self.trades:
            print("\n  종목별 매도 성과:")
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


# ============================================================
# 메인 — v1과 v2 나란히 비교
# ============================================================

def main():
    from simulation.backtests.backtest_d2s import D2SBacktest

    print("=" * 60)
    print("  [v1] attach v1 기준선")
    print("=" * 60)
    bt_v1 = D2SBacktest()
    bt_v1.run(verbose=False)
    bt_v1.print_report()

    print("\n")
    print("=" * 60)
    print("  [v2] attach v2 (R17+R18+DCA 제한)")
    print("=" * 60)
    bt_v2 = D2SBacktestV2()
    bt_v2.run(verbose=True)
    bt_v2.print_report()
    bt_v2.save_trades(
        Path(_ROOT) / "data" / "results" / "backtests" / "d2s_trades_v2.csv"
    )

    # 핵심 지표 비교 출력
    r1 = bt_v1.report()
    r2 = bt_v2.report()
    print("\n" + "=" * 60)
    print("  v1 vs v2 핵심 비교")
    print("=" * 60)
    for key, label in [
        ("total_return_pct", "총 수익률"),
        ("win_rate",         "승률"),
        ("mdd_pct",          "MDD"),
        ("sharpe_ratio",     "Sharpe"),
        ("sell_trades",      "총 청산"),
        ("avg_pnl_pct",      "평균 수익률"),
    ]:
        v1_val = r1[key]
        v2_val = r2[key]
        diff = v2_val - v1_val if isinstance(v1_val, (int, float)) else "-"
        sign = "+" if isinstance(diff, float) and diff > 0 else ""
        print(f"  {label:12s}:  v1={v1_val:>8}   v2={v2_val:>8}   Δ={sign}{diff}")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
D2S 백테스트 v3 — 레짐 감지 + 조건부 청산 파라미터
====================================================
trading_rules_attach_v3.md (Optuna #449 반영) 근거:
  R19: BB 진입 하드 필터 (%B > 0.30 → 진입 금지, Study G F2: OOS +13.2%p)
  R20: 레짐 조건부 take_profit (Bull=5.0%, Bear=6.5%)  ← Optuna #449 역전 발견
  R21: 레짐 조건부 hold_days   (Bull=12일, Bear=8일)   ← Optuna #449

레짐 감지 방법 (3차원 다수결):
  - SPY streak 기반: ≥5일 연속 상승 → Bull, ≥1일 연속 하락 → Bear
  - SPY SMA12 기반: SPY > SMA+1.1% → Bull, SPY < SMA-1.5% → Bear
  - Polymarket BTC 확률: btc_up > 0.55 → Bull, btc_up < 0.35 → Bear
  - 복합 판정: 3개 신호 중 2개 이상 일치 → 레짐 확정 (다수결)

검증 결과 (Optuna #449, no-ROBN 1.5년):
  IS  2024-09-18 ~ 2025-05-31: +33.68%, MDD -9.2%, Sharpe 2.220
  OOS 2025-06-01 ~ 2026-02-17: +62.42%, MDD -16.0%, Sharpe 1.461
  전체 (1.5년): +118.05%, MDD -16.0%, Sharpe 1.506

Usage:
    pyenv shell ptj_stock_lab && python simulation/backtests/backtest_d2s_v3.py
"""
from __future__ import annotations

import sys
from collections import deque
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from simulation.strategies.line_c_d2s.params_d2s import D2S_ENGINE_V3
from simulation.strategies.line_c_d2s.d2s_engine import D2SPosition, DailySnapshot
from simulation.backtests.backtest_d2s_v2 import D2SBacktestV2, TradeRecord


class D2SBacktestV3(D2SBacktestV2):
    """D2S 백테스트 v3 — 레짐 감지 + R19/R20/R21.

    변경 사항
    ---------
    * __init__()      : SPY 가격 이력 버퍼 초기화
    * _detect_regime(): SPY streak + SMA 기반 bull/bear/neutral 분류
    * check_exit_v3() : 레짐 조건부 take_profit / hold_days 적용
    * _execute_buy()  : R19 BB 진입 하드 필터 적용
    * run()           : 레짐 추적 + 레짐별 통계 집계
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
            params=params or D2S_ENGINE_V3,
            start_date=start_date,
            end_date=end_date,
            use_fees=use_fees,
            data_path=data_path,
            slippage_pct=slippage_pct,
        )
        # SPY 종가 이력 (SMA 계산용)
        sma_period = self.params.get("regime_spy_sma_period", 20)
        self._spy_closes: deque[float] = deque(maxlen=sma_period)

        # 레짐 추적
        self._current_regime: str = "neutral"
        self._spy_down_streak: int = 0  # SPY 연속 하락일 수

        # 통계 카운터 (레짐별)
        self._regime_days: dict[str, int] = {"bull": 0, "bear": 0, "neutral": 0}
        self._regime_exits: dict[str, list] = {"bull": [], "bear": [], "neutral": []}
        self._r19_blocked: int = 0  # BB 하드 필터 차단 건수
        self._r20_r21_applied: dict[str, int] = {"bull": 0, "bear": 0, "neutral": 0}

    # ------------------------------------------------------------------
    # 레짐 감지
    # ------------------------------------------------------------------

    def _detect_regime(
        self, spy_streak: int, spy_close: float | None,
        poly_btc_up: float | None = None,
    ) -> str:
        """SPY streak + SMA + Polymarket BTC 3차원 레짐 분류.

        Parameters
        ----------
        spy_streak : int
            SPY 연속 상승일 수 (외부 루프에서 전달).
        spy_close : float | None
            당일 SPY 종가 (SMA 계산용).
        poly_btc_up : float | None
            Polymarket BTC up 확률 (Risk-on/off 강도 조정).

        Returns
        -------
        str
            "bull" | "bear" | "neutral"
        """
        if not self.params.get("regime_enabled", True):
            return "neutral"

        bull_streak_th = self.params.get("regime_bull_spy_streak", 3)
        bear_streak_th = self.params.get("regime_bear_spy_streak", 2)

        # ── 1차: SPY streak 기반 ─────────────────────────────
        streak_regime = "neutral"
        if spy_streak >= bull_streak_th:
            streak_regime = "bull"
        elif self._spy_down_streak >= bear_streak_th:
            streak_regime = "bear"

        # ── 2차: SPY SMA 기반 ───────────────────────────────
        sma_regime = "neutral"
        if spy_close is not None and len(self._spy_closes) >= 5:
            sma = float(np.mean(self._spy_closes))
            bull_pct = self.params.get("regime_spy_sma_bull_pct", 1.0) / 100.0
            bear_pct = abs(self.params.get("regime_spy_sma_bear_pct", -1.0)) / 100.0
            if spy_close > sma * (1 + bull_pct):
                sma_regime = "bull"
            elif spy_close < sma * (1 - bear_pct):
                sma_regime = "bear"

        # ── 3차: Polymarket BTC 확률 기반 (Risk-on/off) ─────
        # btc_up > regime_btc_bull_th → Risk-on → Bull 가중
        # btc_up < regime_btc_bear_th → Risk-off → Bear 가중
        poly_regime = "neutral"
        if poly_btc_up is not None:
            btc_bull_th = self.params.get("regime_btc_bull_threshold", 0.60)
            btc_bear_th = self.params.get("regime_btc_bear_threshold", 0.40)
            if poly_btc_up > btc_bull_th:
                poly_regime = "bull"
            elif poly_btc_up < btc_bear_th:
                poly_regime = "bear"

        # ── 복합 판정: 2/3 다수결 ─────────────────────────────
        signals = [streak_regime, sma_regime, poly_regime]
        bull_cnt = signals.count("bull")
        bear_cnt = signals.count("bear")

        if bull_cnt >= 2:
            return "bull"
        if bear_cnt >= 2:
            return "bear"
        # streak 우선 (단일 강한 신호)
        if streak_regime != "neutral":
            return streak_regime
        if sma_regime != "neutral":
            return sma_regime
        return "neutral"

    # ------------------------------------------------------------------
    # R19: BB 진입 하드 필터 오버라이드
    # ------------------------------------------------------------------

    def _execute_buy(
        self, ticker: str, size: float, snap: DailySnapshot,
        reason: str, score: float,
    ) -> TradeRecord | None:
        """v3 매수: R19 BB 하드 필터 추가."""
        # R19: %B > hard_max → 진입 금지 (Study G F2: +12.3%p)
        if self.params.get("bb_entry_hard_filter", True):
            bb = snap.bb_pct_b.get(ticker, 0.5)
            hard_max = self.params.get("bb_entry_hard_max", 0.30)
            if bb > hard_max:
                self._r19_blocked += 1
                return None

        return super()._execute_buy(ticker, size, snap, reason, score)

    # ------------------------------------------------------------------
    # R20/R21: 레짐 조건부 청산 체크
    # ------------------------------------------------------------------

    def _check_regime_exit(
        self, ticker: str, pos: D2SPosition, snap: DailySnapshot, regime: str,
    ) -> dict[str, bool | str]:
        """레짐 조건부 take_profit + hold_days 청산 체크.

        Returns
        -------
        dict
            should_exit: bool, reason: str
        """
        current_price = snap.closes.get(ticker, 0)
        if current_price <= 0 or pos.entry_price <= 0:
            return {"should_exit": False, "reason": ""}

        pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
        hold_days = (snap.trading_date - pos.entry_date).days

        # 레짐별 파라미터 선택
        if regime == "bull":
            tp = self.params.get("bull_take_profit_pct",
                                 self.params.get("take_profit_pct", 5.9))
            hd = self.params.get("bull_hold_days_max",
                                 self.params.get("optimal_hold_days_max", 7))
        else:  # bear / neutral
            tp = self.params.get("bear_take_profit_pct",
                                 self.params.get("take_profit_pct", 5.0))
            hd = self.params.get("bear_hold_days_max",
                                 self.params.get("optimal_hold_days_max", 4))

        # R20: 레짐 조건부 이익실현
        if pnl_pct >= tp:
            return {"should_exit": True,
                    "reason": f"R20:regime_tp[{regime}] pnl={pnl_pct:+.1f}% >= {tp}%"}

        # R21: 레짐 조건부 강제청산
        if hold_days > hd:
            return {"should_exit": True,
                    "reason": f"R21:regime_hd[{regime}] hold={hold_days}d > {hd}d"}

        return {"should_exit": False, "reason": ""}

    # ------------------------------------------------------------------
    # 메인 루프 — 레짐 감지 + R20/R21 적용
    # ------------------------------------------------------------------

    def run(self, verbose: bool = True) -> "D2SBacktestV3":
        """v3 백테스트 실행 — 레짐 감지 + 조건부 청산 (look-ahead bias 수정)."""
        df, tech, poly = self._load_data()

        all_dates = sorted(df.index)
        trading_dates = [
            d.date() if hasattr(d, "date") else d
            for d in all_dates
            if self.start_date <= (d.date() if hasattr(d, "date") else d) <= self.end_date
        ]

        if verbose:
            print(f"\n[3/3] D2S v3 백테스트 실행 (레짐 감지 + R19/R20/R21 + look-ahead bias 수정)")
            print(f"  기간: {trading_dates[0]} ~ {trading_dates[-1]} ({len(trading_dates)}일)")
            print(f"  초기 자본: ${self.initial_capital:,.0f}")
            p = self.params
            print(f"  R19 BB 하드 필터: %B ≤ {p.get('bb_entry_hard_max', 0.30)} "
                  f"({'ON' if p.get('bb_entry_hard_filter', True) else 'OFF'})")
            print(f"  R20 Bull TP={p.get('bull_take_profit_pct', 5.9)}%  "
                  f"Bear TP={p.get('bear_take_profit_pct', 5.0)}%")
            print(f"  R21 Bull hd={p.get('bull_hold_days_max', 7)}d  "
                  f"Bear hd={p.get('bear_hold_days_max', 4)}d")
            print(f"  레짐 감지: streak(bull≥{p.get('regime_bull_spy_streak', 3)}, "
                  f"bear≥{p.get('regime_bear_spy_streak', 2)}) + "
                  f"SMA{p.get('regime_spy_sma_period', 20)}")
            print()

        spy_streak = 0

        # T일 결정 → T+1일 시가에 체결
        pending_signals: list[dict] = []

        for i, td in enumerate(trading_dates):
            snap = self._build_snapshot(td, tech, poly, spy_streak)
            if snap is None:
                continue

            # ── [A] 전날 결정 시그널을 오늘 시가에 체결 ────────────
            daily_buy_counts: dict[str, int] = {}

            if pending_signals:
                # 매도 먼저 (signal에 저장된 결정 레짐 사용)
                for sig in pending_signals:
                    if sig["action"] == "SELL":
                        pnl = self._execute_sell(sig["ticker"], sig["size"], snap, sig["reason"])
                        if sig["ticker"] not in self.positions:
                            self.position_meta.pop(sig["ticker"], None)
                            sig_regime = sig.get("signal_regime", "neutral")
                            self._regime_exits[sig_regime].append(
                                pnl.pnl_pct if pnl is not None else 0
                            )

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

            self._current_td = td  # 서브클래스에서 날짜 접근 가능 (VIX/MA 조회용)

            # SPY 종가 업데이트 (SMA 계산용 — 오늘 종가 기반)
            spy_close = snap.closes.get("SPY", None)
            if spy_close:
                self._spy_closes.append(float(spy_close))

            # 레짐 감지 (SPY streak + SMA + Polymarket BTC)
            poly_btc = snap.poly_btc_up
            regime = self._detect_regime(spy_streak, spy_close, poly_btc)
            self._current_regime = regime
            self._regime_days[regime] = self._regime_days.get(regime, 0) + 1

            # 기본 시그널 (엔진 R1~R18)
            # daily_buy_counts는 오늘 체결 추적용 — 내일 신호 생성에는 빈 dict 전달
            new_signals = self.engine.generate_daily_signals(
                snap, self.positions, {},
            )

            # R20/R21: 레짐 조건부 청산 오버라이드
            # 기존 SELL 시그널 제거 → 레짐 버전으로 교체
            non_sell = [s for s in new_signals if s["action"] != "SELL"]
            regime_sell = []

            for ticker, pos in list(self.positions.items()):
                exit_ctx = self._check_regime_exit(ticker, pos, snap, regime)
                if exit_ctx["should_exit"]:
                    regime_sell.append({
                        "action": "SELL",
                        "ticker": ticker,
                        "size": 1.0,
                        "reason": exit_ctx["reason"],
                        "score": 0,
                        "market_score": 0,
                        "signal_regime": regime,  # 결정 레짐 저장 (통계용)
                    })
                    self._r20_r21_applied[regime] = (
                        self._r20_r21_applied.get(regime, 0) + 1
                    )

            new_signals = non_sell + regime_sell

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
                        "signal_regime": regime,
                    })
                    self._r18_count += 1

            pending_signals = new_signals

            # ── [C] SPY streak 업데이트 (오늘 종가 기반) ─────────────
            spy_pct = snap.changes.get("SPY", 0)
            if spy_pct > 0:
                spy_streak += 1
                self._spy_down_streak = 0
            else:
                spy_streak = 0
                self._spy_down_streak += 1

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
                    f"레짐: {regime:7s}  "
                    f"R17:{self._r17_count} R18:{self._r18_count} R19:{self._r19_blocked}"
                )

        if verbose:
            print("\n  v3 백테스트 완료!")

        return self

    # ------------------------------------------------------------------
    # 리포트 — v3 레짐 통계 추가
    # ------------------------------------------------------------------

    def print_report(self) -> None:
        """v3 리포트 출력."""
        super().print_report()

        print("\n  [v3 레짐 감지 통계]")
        total_days = sum(self._regime_days.values()) or 1
        for regime in ["bull", "bear", "neutral"]:
            days = self._regime_days.get(regime, 0)
            exits = self._regime_exits.get(regime, [])
            applied = self._r20_r21_applied.get(regime, 0)
            avg_pnl = float(np.mean(exits)) if exits else 0.0
            print(
                f"    {regime:7s}: {days:3d}일 ({days/total_days*100:.0f}%)  "
                f"청산 {len(exits):3d}건  avg_pnl ${avg_pnl:+.0f}  "
                f"R20/R21 발동 {applied}회"
            )
        print(f"  [R19] BB 하드 필터 차단: {self._r19_blocked}건")


# ============================================================
# 메인 — v2 vs v3 비교
# ============================================================

def main():
    from simulation.backtests.backtest_d2s_v2 import D2SBacktestV2

    print("=" * 60)
    print("  [v2] attach v2 기준선")
    print("=" * 60)
    bt_v2 = D2SBacktestV2()
    bt_v2.run(verbose=False)
    bt_v2.print_report()

    print("\n")
    print("=" * 60)
    print("  [v3] attach v3 (R19+R20+R21 레짐 조건부)")
    print("=" * 60)
    bt_v3 = D2SBacktestV3()
    bt_v3.run(verbose=True)
    bt_v3.print_report()

    # 핵심 비교
    r2 = bt_v2.report()
    r3 = bt_v3.report()
    print("\n" + "=" * 60)
    print("  v2 vs v3 핵심 비교")
    print("=" * 60)
    for key, label in [
        ("total_return_pct", "총 수익률"),
        ("win_rate",         "승률"),
        ("mdd_pct",          "MDD"),
        ("sharpe_ratio",     "Sharpe"),
        ("sell_trades",      "총 청산"),
        ("avg_pnl_pct",      "평균 수익률"),
    ]:
        v2_val = r2.get(key, 0)
        v3_val = r3.get(key, 0)
        diff = v3_val - v2_val if isinstance(v2_val, (int, float)) else "-"
        sign = "+" if isinstance(diff, float) and diff > 0 else ""
        print(f"  {label:12s}:  v2={v2_val:>8}   v3={v3_val:>8}   Δ={sign}{diff:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

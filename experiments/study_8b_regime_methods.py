#!/usr/bin/env python3
"""
Study 8B — 레짐 감지 방법 6종 비교
====================================
목적: Study 8에서 no_regime이 가장 좋았는데, v3 구현 한계일 수 있음.
      VIX 기반, MA 크로스오버 등 다른 방법론과 비교해서
      "레짐 감지 자체가 필요 없는가" vs "더 나은 방법이 있는가" 검증.

6가지 모드:
  v3_3signal  : 기존 D2SBacktestV3 그대로 (streak+SMA+Poly 다수결)
  no_regime   : regime_enabled=False (tp=5.0% / hd=12일 고정)
  vix_based   : VIX>20→Bear, VIX<15→Bull, 그 외 Neutral
  ma_cross    : SPY 50MA>200MA→Bull, 50MA<200MA→Bear
  streak_sma  : streak+SMA만 (Poly neutral 처리)
  streak_only : streak만 (SMA/Poly neutral 처리)
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from simulation.backtests.backtest_d2s_v3 import D2SBacktestV3
from simulation.strategies.line_c_d2s.params_d2s import D2S_ENGINE_V3_NO_ROBN

RESULTS_DIR = _PROJECT_ROOT / "data" / "results" / "backtests"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = date(2024, 9, 18)
END_DATE = date(2026, 2, 17)


# ============================================================
# VIX / SPY MA 데이터 로더
# ============================================================

def load_vix_series() -> pd.Series:
    """VIX 종가를 date-indexed Series로 로드."""
    vix_path = _PROJECT_ROOT / "data" / "market" / "daily" / "vix_daily.parquet"
    if not vix_path.exists():
        print("  WARNING: vix_daily.parquet not found")
        return pd.Series(dtype=float)

    df = pd.read_parquet(vix_path)
    # 컬럼: symbol, timestamp, open, high, low, close, volume, date
    df["date"] = pd.to_datetime(df["date"]).dt.date
    vix_series = df.set_index("date")["close"]
    vix_series = vix_series[~vix_series.index.duplicated(keep="last")]
    print(f"  VIX 로드: {len(vix_series)}일 ({vix_series.index.min()} ~ {vix_series.index.max()})")
    return vix_series


def load_spy_ma_series() -> pd.Series:
    """SPY 50MA > 200MA 여부를 date-indexed bool Series로 생성.

    market_daily.parquet에서 SPY Close를 로드 → 50일/200일 이동평균 계산.
    """
    market_path = _PROJECT_ROOT / "data" / "market" / "daily" / "market_daily.parquet"
    if not market_path.exists():
        print("  WARNING: market_daily.parquet not found")
        return pd.Series(dtype=bool)

    df = pd.read_parquet(market_path)
    try:
        spy_close = df[("SPY", "Close")]
    except KeyError:
        print("  WARNING: SPY not found in market_daily.parquet")
        return pd.Series(dtype=bool)

    ma50 = spy_close.rolling(50, min_periods=50).mean()
    ma200 = spy_close.rolling(200, min_periods=200).mean()
    golden = ma50 > ma200

    # date index 변환
    result = pd.Series(index=[d.date() for d in golden.index], data=golden.values)
    result = result[~result.index.duplicated(keep="last")]
    valid = result.dropna()
    print(f"  SPY MA cross 로드: {len(valid)}일 유효 (50MA>200MA 비율: {valid.sum()}/{len(valid)})")
    return result


# ============================================================
# D2SBacktestRegimeMethods — 레짐 감지 방법 교체
# ============================================================

class D2SBacktestRegimeMethods(D2SBacktestV3):
    """레짐 감지 방법 6종 교체 가능 백테스트."""

    def __init__(
        self,
        regime_method: str = "v3_3signal",
        vix_series: pd.Series | None = None,
        spy_ma_series: pd.Series | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.regime_method = regime_method
        self.vix_series = vix_series          # pd.Series(date → vix_close)
        self.spy_ma_series = spy_ma_series    # pd.Series(date → bool: 50MA>200MA)
        self._current_date: date | None = None

    def _detect_regime(
        self, spy_streak: int, spy_close: float | None,
        poly_btc_up: float | None = None,
    ) -> str:
        if self.regime_method == "no_regime":
            return "neutral"

        elif self.regime_method == "vix_based":
            if self.vix_series is not None and self._current_date in self.vix_series.index:
                vix = self.vix_series[self._current_date]
                if not pd.isna(vix):
                    if vix < 15:
                        return "bull"
                    if vix > 20:
                        return "bear"
            return "neutral"

        elif self.regime_method == "ma_cross":
            if self.spy_ma_series is not None and self._current_date in self.spy_ma_series.index:
                is_golden = self.spy_ma_series[self._current_date]
                if not pd.isna(is_golden):
                    return "bull" if is_golden else "bear"
            return "neutral"

        elif self.regime_method == "streak_sma":
            # streak + SMA만 (Poly neutral 처리)
            return super()._detect_regime(spy_streak, spy_close, poly_btc_up=None)

        elif self.regime_method == "streak_only":
            # streak만 (SMA/Poly 전부 무시)
            bull_th = self.params.get("regime_bull_spy_streak", 5)
            bear_th = self.params.get("regime_bear_spy_streak", 1)
            if spy_streak >= bull_th:
                return "bull"
            if self._spy_down_streak >= bear_th:
                return "bear"
            return "neutral"

        else:  # v3_3signal
            return super()._detect_regime(spy_streak, spy_close, poly_btc_up)

    def run(self, verbose: bool = True) -> "D2SBacktestRegimeMethods":
        """run() 오버라이드 — 매 날짜마다 self._current_date 설정."""
        df, tech, poly = self._load_data()

        all_dates = sorted(df.index)
        trading_dates = [
            d.date() if hasattr(d, "date") else d
            for d in all_dates
            if self.start_date <= (d.date() if hasattr(d, "date") else d) <= self.end_date
        ]

        if not trading_dates:
            if verbose:
                print(f"  WARNING: No trading dates in range {self.start_date}~{self.end_date}")
            return self

        if verbose:
            print(f"\n[3/3] D2S v3 백테스트 (regime_method={self.regime_method})")
            print(f"  기간: {trading_dates[0]} ~ {trading_dates[-1]} ({len(trading_dates)}일)")
            print(f"  초기 자본: ${self.initial_capital:,.0f}")
            print()

        spy_streak = 0

        for i, td in enumerate(trading_dates):
            self._current_date = td  # VIX/MA 조회용

            snap = self._build_snapshot(td, tech, poly, spy_streak)
            if snap is None:
                continue

            # SPY 종가 업데이트 (SMA 계산용)
            spy_close = snap.closes.get("SPY", None)
            if spy_close:
                self._spy_closes.append(float(spy_close))

            # 레짐 감지 (오버라이드된 _detect_regime 사용)
            poly_btc = snap.poly_btc_up if snap is not None else None
            regime = self._detect_regime(spy_streak, spy_close, poly_btc)
            self._current_regime = regime
            self._regime_days[regime] = self._regime_days.get(regime, 0) + 1

            daily_buy_counts: dict[str, int] = {}

            # 기본 시그널 (R1~R18)
            signals = self.engine.generate_daily_signals(
                snap, self.positions, daily_buy_counts,
            )

            # R20/R21: 레짐 조건부 청산 오버라이드
            non_sell = [s for s in signals if s["action"] != "SELL"]
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
                    })
                    self._r20_r21_applied[regime] = (
                        self._r20_r21_applied.get(regime, 0) + 1
                    )

            signals = non_sell + regime_sell

            # R18: 조기 손절 시그널 추가
            already_selling = {s["ticker"] for s in signals if s["action"] == "SELL"}
            for ticker, pos in list(self.positions.items()):
                if ticker in already_selling:
                    continue
                r18_reason = self._check_r18_exit(ticker, pos, snap)
                if r18_reason:
                    signals.append({
                        "action": "SELL",
                        "ticker": ticker,
                        "size": 1.0,
                        "reason": r18_reason,
                        "score": 0,
                    })
                    self._r18_count += 1

            # 매도 먼저
            for sig in signals:
                if sig["action"] == "SELL":
                    pnl = self._execute_sell(sig["ticker"], sig["size"], snap, sig["reason"])
                    if sig["ticker"] not in self.positions:
                        self.position_meta.pop(sig["ticker"], None)
                        self._regime_exits[regime].append(
                            pnl.pnl_pct if pnl is not None else 0
                        )

            # 매수
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
                self._spy_down_streak = 0
            else:
                spy_streak = 0
                self._spy_down_streak += 1

            # 자산 스냅샷
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
            print("\n  백테스트 완료!")

        return self


# ============================================================
# 실행 함수
# ============================================================

def run_method(label, method, params, start, end, vix_series, spy_ma_series):
    """단일 레짐 메서드 백테스트 실행."""
    print(f"\n{'='*60}")
    print(f"  [{label}] regime_method={method}")
    print(f"{'='*60}")
    bt = D2SBacktestRegimeMethods(
        regime_method=method,
        vix_series=vix_series,
        spy_ma_series=spy_ma_series,
        params=params,
        start_date=start,
        end_date=end,
    )
    bt.run(verbose=False)
    bt.print_report()

    # 레짐 분포 추출
    total_days = sum(bt._regime_days.values()) or 1
    regime_dist = {
        r: f"{bt._regime_days.get(r, 0)}d({bt._regime_days.get(r, 0)/total_days*100:.0f}%)"
        for r in ["bull", "bear", "neutral"]
    }

    report = bt.report()
    report["regime_distribution"] = bt._regime_days
    report["regime_method"] = method
    return report, regime_dist


def main():
    print("=" * 70)
    print("  Study 8B: 레짐 감지 방법 6종 비교")
    print(f"  기간: {START_DATE} ~ {END_DATE}")
    print("=" * 70)

    # 외부 데이터 로드
    print("\n[0] 외부 데이터 로드")
    vix_series = load_vix_series()
    spy_ma_series = load_spy_ma_series()

    configs = [
        ("v3_3signal",   "v3_3signal",   "기존 v3 (streak+SMA+Poly 다수결)"),
        ("no_regime",    "no_regime",    "레짐 감지 없음 (neutral 고정)"),
        ("vix_based",    "vix_based",    "VIX 기반 (>20=Bear, <15=Bull)"),
        ("ma_cross",     "ma_cross",     "SPY 골든크로스 (50MA>200MA)"),
        ("streak_sma",   "streak_sma",   "streak+SMA만 (Poly 제거)"),
        ("streak_only",  "streak_only",  "streak만 (SMA/Poly 제거)"),
    ]

    results = {}
    regime_dists = {}
    for key, method, _ in configs:
        report, rdist = run_method(
            key, method, D2S_ENGINE_V3_NO_ROBN,
            START_DATE, END_DATE, vix_series, spy_ma_series,
        )
        results[key] = report
        regime_dists[key] = rdist

    # 비교 요약
    print("\n" + "=" * 90)
    print("  Study 8B — 레짐 감지 방법 6종 비교 요약")
    print("=" * 90)
    print(
        f"  {'모드':16s}  {'수익률':>10}  {'MDD':>8}  {'Sharpe':>8}  "
        f"{'승률':>8}  {'Bull':>8}  {'Bear':>8}  {'Neutral':>8}"
    )
    print(f"  {'-'*85}")
    for key, _, desc in configs:
        r = results[key]
        rd = regime_dists[key]
        print(
            f"  {key:16s}  "
            f"{r.get('total_return_pct', 0):>9.1f}%  "
            f"{r.get('mdd_pct', 0):>7.1f}%  "
            f"{r.get('sharpe_ratio', 0):>8.3f}  "
            f"{r.get('win_rate', 0):>7.1f}%  "
            f"{rd.get('bull', ''):>8s}  "
            f"{rd.get('bear', ''):>8s}  "
            f"{rd.get('neutral', ''):>8s}"
        )
    print("=" * 90)

    # JSON 저장
    result = {
        "study": "study_8b_regime_methods",
        "run_date": datetime.now().isoformat(),
        "period": {"start": str(START_DATE), "end": str(END_DATE)},
        **results,
    }
    out_path = RESULTS_DIR / f"study_8b_regime_methods_{datetime.now().strftime('%Y%m%d')}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()

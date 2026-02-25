#!/usr/bin/env python3
"""
Study 11 — 레짐 감지 방법 ablation (look-ahead bias 수정 후)
=============================================================
목적: Study 10 수정 후에도 no_regime / streak_only가 여전히 우위인지 확인.
     역전 시 → 레짐 감지 재검토, 유지 시 → v4에서 레짐 삭제 확정.

6가지 레짐 방법 (Study 8B와 동일):
  no_regime   : regime_enabled=False (TP/HD 고정)
  streak_only : SPY streak만 (SMA/Poly 무시)
  ma_cross    : SPY 50MA>200MA→Bull, 50MA<200MA→Bear
  full_3signal: streak+SMA+Poly 다수결 (v3 default)
  no_poly     : streak+SMA만 (Poly neutral 처리)
  v3_current  : full_3signal과 동일 (v3_current params 기준)

검증 기간:
  IS  : 2024-09-18 ~ 2025-05-31
  OOS : 2025-06-01 ~ 2026-02-17

Usage:
    pyenv shell ptj_stock_lab && python experiments/study_11_corrected_regime_ablation.py
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

IS_START   = date(2024, 9, 18)
IS_END     = date(2025, 5, 31)
OOS_START  = date(2025, 6, 1)
OOS_END    = date(2026, 2, 17)
FULL_START = date(2024, 9, 18)
FULL_END   = date(2026, 2, 17)

PERIODS = [
    ("IS",   IS_START,   IS_END),
    ("OOS",  OOS_START,  OOS_END),
    ("FULL", FULL_START, FULL_END),
]


# ============================================================
# SPY MA 크로스 시리즈 로더
# ============================================================

def load_spy_ma_series() -> pd.Series:
    """SPY 50MA>200MA 여부를 date-indexed bool Series로 생성."""
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

    result = pd.Series(index=[d.date() for d in golden.index], data=golden.values)
    result = result[~result.index.duplicated(keep="last")]
    valid = result.dropna()
    print(f"  SPY MA cross: {len(valid)}일 유효 (50MA>200MA: {valid.sum()}/{len(valid)})")
    return result


# ============================================================
# 레짐 방법 교체 가능한 백테스트 클래스
# (corrected v3 run()을 그대로 사용 — only _detect_regime 오버라이드)
# ============================================================

class D2SBacktestRegimeCorrected(D2SBacktestV3):
    """레짐 감지 방법 교체 가능 백테스트 (look-ahead bias 수정 버전).

    부모 클래스(D2SBacktestV3)의 수정된 run()을 그대로 사용하고
    _detect_regime()만 오버라이드한다.
    """

    def __init__(
        self,
        regime_method: str = "full_3signal",
        spy_ma_series: pd.Series | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.regime_method = regime_method
        self.spy_ma_series = spy_ma_series  # date → bool (50MA>200MA)

    def _detect_regime(
        self, spy_streak: int, spy_close: float | None,
        poly_btc_up: float | None = None,
    ) -> str:
        """레짐 감지 방법 선택."""
        if self.regime_method == "no_regime":
            return "neutral"

        elif self.regime_method == "streak_only":
            bull_th = self.params.get("regime_bull_spy_streak", 3)
            bear_th = self.params.get("regime_bear_spy_streak", 2)
            if spy_streak >= bull_th:
                return "bull"
            if self._spy_down_streak >= bear_th:
                return "bear"
            return "neutral"

        elif self.regime_method == "ma_cross":
            td = getattr(self, "_current_td", None)
            if self.spy_ma_series is not None and td is not None:
                if td in self.spy_ma_series.index:
                    val = self.spy_ma_series[td]
                    if not pd.isna(val):
                        return "bull" if bool(val) else "bear"
            return "neutral"

        elif self.regime_method == "no_poly":
            # streak + SMA만 (Poly neutral 처리)
            return super()._detect_regime(spy_streak, spy_close, poly_btc_up=None)

        else:
            # full_3signal / v3_current: 기본 3차원 다수결
            return super()._detect_regime(spy_streak, spy_close, poly_btc_up)


# ============================================================
# 실험 정의
# ============================================================

METHODS = [
    "no_regime",
    "streak_only",
    "ma_cross",
    "full_3signal",
    "no_poly",
    "v3_current",
]

METHOD_DESC = {
    "no_regime":    "레짐 없음 (TP/HD 고정)",
    "streak_only":  "SPY streak만",
    "ma_cross":     "SPY 50MA>200MA",
    "full_3signal": "streak+SMA+Poly 다수결",
    "no_poly":      "streak+SMA만 (Poly 제외)",
    "v3_current":   "v3 현재 (full_3signal과 동일)",
}


def run_method(method: str, params: dict, start: date, end: date,
               spy_ma: pd.Series) -> dict:
    bt = D2SBacktestRegimeCorrected(
        regime_method=method,
        spy_ma_series=spy_ma,
        params=params,
        start_date=start,
        end_date=end,
    )
    bt.run(verbose=False)
    r = bt.report()
    r["method"] = method
    r["regime_days"] = bt._regime_days
    r["r20_r21_applied"] = dict(bt._r20_r21_applied)
    return r


def main():
    print("=" * 80)
    print("  Study 11 — 레짐 감지 방법 ablation (look-ahead bias 수정 후)")
    print(f"  실행 시각: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 80)

    # SPY MA 시리즈 로드 (ma_cross용)
    spy_ma = load_spy_ma_series()

    params = dict(D2S_ENGINE_V3_NO_ROBN)

    # 기간별 결과 수집
    all_results: list[dict] = []
    period_results: dict[str, dict[str, dict]] = {}  # period → method → result

    for period_label, start, end in PERIODS:
        print(f"\n  [{period_label}] {start} ~ {end}")
        period_results[period_label] = {}

        for method in METHODS:
            r = run_method(method, params, start, end, spy_ma)
            r["period"] = period_label
            period_results[period_label][method] = r
            all_results.append(r)
            print(f"    {method:<15}: 수익률={r['total_return_pct']:+.2f}%  "
                  f"MDD={r['mdd_pct']:.2f}%  Sharpe={r['sharpe_ratio']:.3f}  "
                  f"승률={r['win_rate']:.1f}%")

    # ── 결과 비교 테이블 ──────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  결과 비교 테이블 (IS / OOS / FULL)")
    print("=" * 80)

    header = f"  {'방법':<16} {'IS 수익률':>10} {'OOS 수익률':>10} {'FULL 수익률':>11} {'IS Sharpe':>10} {'OOS Sharpe':>10}"
    print(header)
    print("  " + "-" * 70)

    for method in METHODS:
        is_r   = period_results["IS"][method]
        oos_r  = period_results["OOS"][method]
        full_r = period_results["FULL"][method]
        mark = " ★" if method == "no_regime" else ""
        print(
            f"  {method:<16}"
            f"{is_r['total_return_pct']:>9.2f}% "
            f"{oos_r['total_return_pct']:>9.2f}% "
            f"{full_r['total_return_pct']:>10.2f}% "
            f"{is_r['sharpe_ratio']:>9.3f} "
            f"{oos_r['sharpe_ratio']:>9.3f}"
            f"{mark}"
        )

    # ── 결론 분석 ─────────────────────────────────────────────────
    oos_by_method = {m: period_results["OOS"][m]["total_return_pct"] for m in METHODS}
    best_method = max(oos_by_method, key=oos_by_method.get)
    no_regime_rank = sorted(oos_by_method, key=oos_by_method.get, reverse=True).index("no_regime") + 1

    print(f"\n  OOS 최우수: {best_method} ({oos_by_method[best_method]:+.2f}%)")
    print(f"  no_regime OOS 순위: {no_regime_rank}/{len(METHODS)}")

    if no_regime_rank <= 2:
        print("  → ✅ no_regime 우위 유지 — v4에서 레짐 감지 삭제 확정")
    else:
        print("  → ⚠️ no_regime 우위 역전 — 레짐 방법 재검토 필요")

    # ── 저장 ──────────────────────────────────────────────────────
    out_path = RESULTS_DIR / f"study_11_corrected_regime_{datetime.now():%Y%m%d}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  결과 저장: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

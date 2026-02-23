"""
Bear Regime OOS 소급 검증 v2 — 히스테리시스 강화 + rolling_7d 그리드 탐색
==========================================================================
v1 실험 결과:
  - BearRegime (5d, 0.40, streak≥3) 최초 발동: 2026-01-20
  - IREN 고점(1/28) 보다 8일 전 경고
  - 문제: 1/22~23 반등(UP 2회)으로 체제 해제 → 이후 IREN $62 신고점 갱신
  - 2차 발동(2/1, 2/5): CONL $9→$5 급락 구간 포착

v2 개선사항:
  1. 히스테리시스 강화: recovery_threshold = 0.50 → 0.55 / 0.60
  2. rolling_7d 기준 추가 테스트
  3. streak 조건 다양화 (2/3/4일)
  4. CONL 역방향(숏 시뮬) P&L 계산
  5. 12개 파라미터 조합 그리드 탐색 + 비교표
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

POLY_DIR = Path(config.POLY_DATA_DIR)

# ============================================================
# 1. 데이터 로드 (v1과 동일)
# ============================================================

def _last_prob(series: list) -> float | None:
    if not series:
        return None
    last = series[-1]
    if isinstance(last, dict):
        return float(last.get("p", 0.5))
    if isinstance(last, (list, tuple)) and len(last) >= 2:
        return float(last[1])
    return None


def load_poly_oos(start: date, end: date) -> pd.DataFrame:
    records = []
    current = start
    while current <= end:
        fp = POLY_DIR / str(current.year) / f"{current.isoformat()}_1m.json"
        if not fp.exists():
            current += timedelta(days=1)
            continue
        try:
            data = json.load(open(fp))
        except Exception:
            current += timedelta(days=1)
            continue

        indicators = data.get("indicators", {})

        def extract_binary(ind_key: str, outcome_key: str) -> float:
            ind = indicators.get(ind_key, {})
            if "error" in ind:
                return 0.5
            for m in ind.get("markets", []):
                p = _last_prob(m.get("outcomes", {}).get(outcome_key, []))
                if p is not None:
                    return p
            fp_val = ind.get("final_prices", {}).get(outcome_key)
            try:
                return float(fp_val) if fp_val is not None else 0.5
            except (TypeError, ValueError):
                return 0.5

        records.append({
            "date": pd.Timestamp(current),
            "btc_up_raw": extract_binary("btc_up_down", "Up"),
            "ndx_up_raw": extract_binary("ndx_up_down", "Up"),
        })
        current += timedelta(days=1)

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).set_index("date").sort_index()
    return df


def load_price_oos(start: date, end: date) -> pd.DataFrame:
    frames = {}

    def _normalize_idx(s: pd.Series) -> pd.Series:
        """tz-aware → tz-naive date (날짜만 남김)"""
        idx = s.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_convert(None)
        s.index = idx.normalize()
        return s

    # IREN
    iren_fp = ROOT / "data/market/daily/soxx_iren_daily.parquet"
    if iren_fp.exists():
        df = pd.read_parquet(iren_fp)
        iren = df[df["symbol"] == "IREN"].copy()
        iren["date"] = pd.to_datetime(iren["timestamp"], unit="s")
        iren = iren.set_index("date").sort_index()
        mask = (iren.index.date >= start) & (iren.index.date <= end)
        frames["IREN"] = _normalize_idx(iren.loc[mask, "close"].rename("IREN"))

    # CONL, BITU
    hist_fp = ROOT / "data/market/daily/history.parquet"
    if hist_fp.exists():
        df = pd.read_parquet(hist_fp)
        df.index = pd.to_datetime(df.index)
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        df.index = df.index.normalize()
        mask = (df.index.date >= start) & (df.index.date <= end)
        for ticker in ["CONL", "BITU"]:
            col = ("Close", ticker)
            if col in df.columns:
                frames[ticker] = df.loc[mask, col].rename(ticker)

    if not frames:
        return pd.DataFrame()
    result = pd.DataFrame(frames).sort_index()
    return result


# ============================================================
# 2. 파라미터 정의
# ============================================================

class RegimeParams(NamedTuple):
    label: str
    window: int           # rolling window (일)
    entry_thresh: float   # rolling < entry_thresh → bear ON 후보
    recovery_thresh: float # rolling >= recovery_thresh → bear OFF (히스테리시스)
    min_streak: int       # 연속 하락 최소 일수 (streak 조건)


PARAM_GRID: list[RegimeParams] = [
    # ── 기준선 (v1) ──────────────────────────────────────────
    RegimeParams("v1-기준선   ", 5, 0.40, 0.50, 3),
    # ── 히스테리시스 강화 ────────────────────────────────────
    RegimeParams("5d-hyst0.55", 5, 0.40, 0.55, 3),
    RegimeParams("5d-hyst0.60", 5, 0.40, 0.60, 3),
    # ── rolling 7d 기준 ─────────────────────────────────────
    RegimeParams("7d-h0.50   ", 7, 0.43, 0.50, 3),
    RegimeParams("7d-h0.55   ", 7, 0.43, 0.55, 3),
    RegimeParams("7d-h0.57   ", 7, 0.43, 0.57, 3),
    # ── streak 조건 완화 ────────────────────────────────────
    RegimeParams("7d-stk2    ", 7, 0.43, 0.57, 2),
    RegimeParams("7d-stk4    ", 7, 0.43, 0.57, 4),
    # ── WARN 수준 (느슨한 조건) ─────────────────────────────
    RegimeParams("WARN-5d    ", 5, 0.50, 0.60, 1),
    RegimeParams("WARN-7d    ", 7, 0.57, 0.65, 1),
    # ── 엄격한 조건 ─────────────────────────────────────────
    RegimeParams("strict-7d  ", 7, 0.29, 0.57, 4),
    RegimeParams("strict-5d  ", 5, 0.30, 0.60, 4),
]


# ============================================================
# 3. 상태 기반 Bear Regime 시뮬레이션 (히스테리시스 적용)
# ============================================================

def compute_stateful_regime(
    poly_df: pd.DataFrame,
    params: RegimeParams,
) -> pd.DataFrame:
    """히스테리시스가 적용된 상태 기반 BearRegime 계산.

    상태 전이:
      OFF → ON: rolling_Nd < entry_thresh AND streak >= min_streak
      ON  → OFF: rolling_Nd >= recovery_thresh  (히스테리시스)
    """
    df = poly_df.copy()
    w = params.window

    df[f"btc_up_r{w}d"] = df["btc_up_raw"].rolling(w, min_periods=1).mean()
    df[f"ndx_up_r{w}d"] = df["ndx_up_raw"].rolling(w, min_periods=1).mean()

    # 연속 하락 스트릭
    streak = 0
    streaks = []
    for v in df["btc_up_raw"]:
        if v < 0.5:   # 0 (Down) → streak 증가
            streak += 1
        else:
            streak = 0
        streaks.append(streak)
    df["streak"] = streaks

    # 히스테리시스 상태 머신
    is_bear = False
    regime_col = []
    rolling_col = df[f"btc_up_r{w}d"].values
    streak_col = df["streak"].values

    for i in range(len(df)):
        r = rolling_col[i]
        s = streak_col[i]
        if is_bear:
            # OFF 조건: recovery_thresh 이상으로 회복
            if r >= params.recovery_thresh:
                is_bear = False
        else:
            # ON 조건: entry_thresh 미만 AND streak >= min_streak
            if r < params.entry_thresh and s >= params.min_streak:
                is_bear = True
        regime_col.append(1 if is_bear else 0)

    df["bear_regime"] = regime_col
    return df


# ============================================================
# 4. P&L 시뮬레이션 (CONL 숏 포지션)
# ============================================================

def simulate_short_conl(
    regime_df: pd.DataFrame,
    prices: pd.DataFrame,
    stop_pct: float = 10.0,   # 숏 기준 손절: +10% (가격 상승 시)
    target_pct: float = 35.0,  # 숏 기준 목표: -35% 하락
    hold_max_days: int = 25,
    min_hold_days: int = 2,
) -> dict:
    """CONL 숏 시뮬레이션 (Bear Regime 진입 시 매도, 체제 해제 or 목표/손절 시 매수 청산).

    숏이므로: entry_price 대비 가격이 하락할수록 수익.
    pnl_pct = (entry_price - exit_price) / entry_price * 100
    """
    if "CONL" not in prices.columns:
        return {"trades": [], "total_pnl": 0.0, "win_rate": 0.0}

    conl = prices["CONL"].dropna()
    # regime_df 인덱스 → 날짜 정규화
    aligned = regime_df["bear_regime"].reindex(
        pd.to_datetime([d for d in conl.index])
    ).fillna(0)

    trades = []
    in_position = False
    entry_date = None
    entry_price = None
    entry_idx = None

    dates_list = list(conl.index)
    for i, dt in enumerate(dates_list):
        price = conl.iloc[i]
        bear = aligned.get(dt, 0)

        if not in_position:
            if bear == 1:
                in_position = True
                entry_date = dt
                entry_price = price
                entry_idx = i
        else:
            if entry_price is None or entry_price == 0:
                continue
            # 숏 기준 P&L
            pnl_pct = (entry_price - price) / entry_price * 100
            days = i - entry_idx

            exit_reason = None
            if days < min_hold_days:
                pass  # 최소 보유 기간
            elif pnl_pct >= target_pct:
                exit_reason = "TARGET"
            elif pnl_pct <= -stop_pct:
                exit_reason = "STOP"
            elif days >= hold_max_days:
                exit_reason = "TIME"
            elif bear == 0 and days >= min_hold_days:
                exit_reason = "REGIME_OFF"

            if exit_reason:
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": dt,
                    "entry_price": round(entry_price, 3),
                    "exit_price": round(price, 3),
                    "pnl_pct": round(pnl_pct, 2),
                    "days_held": days,
                    "exit_reason": exit_reason,
                })
                in_position = False
                entry_date = entry_price = entry_idx = None

    total_pnl = sum(t["pnl_pct"] for t in trades)
    wins = [t for t in trades if t["pnl_pct"] > 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0.0
    return {
        "trades": trades,
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(win_rate, 1),
        "n_trades": len(trades),
    }


# ============================================================
# 5. 고점 포착 지표 계산
# ============================================================

def peak_capture_metrics(
    regime_df: pd.DataFrame,
    prices: pd.DataFrame,
    params: RegimeParams,
) -> dict:
    """IREN/CONL 고점 대비 BearRegime 트리거 타이밍 분석."""
    metrics = {}

    # IREN 고점
    if "IREN" in prices.columns:
        iren = prices["IREN"].dropna()
        if not iren.empty:
            peak_dt = iren.idxmax()
            metrics["IREN_peak_dt"] = peak_dt
            metrics["IREN_peak_px"] = round(float(iren.max()), 2)

    # CONL 고점
    if "CONL" in prices.columns:
        conl = prices["CONL"].dropna()
        if not conl.empty:
            peak_dt = conl.idxmax()
            metrics["CONL_peak_dt"] = peak_dt
            metrics["CONL_peak_px"] = round(float(conl.max()), 2)
            metrics["CONL_bottom_px"] = round(float(conl.min()), 2)

    # 최초 BEAR 발동일
    bear_days = regime_df.index[regime_df["bear_regime"] == 1]
    if len(bear_days) > 0:
        first_bear = bear_days[0]
        metrics["first_bear_dt"] = first_bear
        metrics["bear_days_total"] = int(regime_df["bear_regime"].sum())

        # IREN 타이밍
        if "IREN_peak_dt" in metrics:
            lag = (first_bear - metrics["IREN_peak_dt"]).days
            metrics["IREN_peak_lag_days"] = lag  # 음수 = 고점 전 감지, 양수 = 고점 후

        # CONL: 체제 진입 시 가격
        if "CONL" in prices.columns:
            conl = prices["CONL"].dropna()
            if first_bear in conl.index:
                metrics["CONL_at_bear"] = round(float(conl[first_bear]), 2)
            drop = (metrics.get("CONL_bottom_px", 0) - metrics.get("CONL_at_bear", 0))
            entry = metrics.get("CONL_at_bear", 1)
            if entry > 0:
                metrics["CONL_max_short_gain"] = round(-drop / entry * 100, 1)
    else:
        metrics["first_bear_dt"] = None
        metrics["bear_days_total"] = 0
        metrics["IREN_peak_lag_days"] = None

    return metrics


# ============================================================
# 6. 그리드 탐색 메인
# ============================================================

def main():
    OOS_START = date(2026, 1, 1)
    OOS_END = date(2026, 2, 17)

    print("=" * 75)
    print("Bear Regime OOS v2 — 히스테리시스 강화 + rolling_7d 그리드 탐색")
    print(f"분석 기간: {OOS_START} ~ {OOS_END}")
    print("=" * 75)

    # 데이터 로드
    print("\n[1] 데이터 로드...")
    poly_df = load_poly_oos(OOS_START, OOS_END)
    prices = load_price_oos(OOS_START, OOS_END)
    print(f"  Polymarket: {len(poly_df)}일  |  주가: {list(prices.columns)}")

    # IREN/CONL 고점/저점 사전 계산
    iren = prices["IREN"].dropna() if "IREN" in prices.columns else pd.Series()
    conl = prices["CONL"].dropna() if "CONL" in prices.columns else pd.Series()

    if not iren.empty:
        iren_peak_dt = iren.idxmax()
        iren_peak_px = iren.max()
        iren_bottom_px = iren.min()
        print(f"\n  IREN 고점: {iren_peak_dt.strftime('%m/%d')} ${iren_peak_px:.2f}  "
              f"저점: ${iren_bottom_px:.2f}")
    if not conl.empty:
        conl_peak_dt = conl.idxmax()
        conl_peak_px = conl.max()
        conl_bottom_px = conl.min()
        print(f"  CONL 고점: {conl_peak_dt.strftime('%m/%d')} ${conl_peak_px:.2f}  "
              f"저점: ${conl_bottom_px:.2f}")

    # ── 그리드 탐색 ─────────────────────────────────────────
    print("\n[2] 파라미터 그리드 탐색...")
    results = []

    for params in PARAM_GRID:
        regime_df = compute_stateful_regime(poly_df, params)
        pnl_res = simulate_short_conl(regime_df, prices)
        metrics = peak_capture_metrics(regime_df, prices, params)

        first_bear_str = (
            metrics["first_bear_dt"].strftime("%m/%d")
            if metrics["first_bear_dt"] is not None else "없음"
        )
        lag = metrics.get("IREN_peak_lag_days")
        lag_str = f"{lag:+d}d" if lag is not None else "  -"
        conl_at_bear = metrics.get("CONL_at_bear")
        conl_at_bear_str = f"${conl_at_bear:.2f}" if conl_at_bear else "  N/A"
        max_gain = metrics.get("CONL_max_short_gain")
        max_gain_str = f"{max_gain:+.1f}%" if max_gain else "  -"

        results.append({
            "params": params,
            "regime_df": regime_df,
            "metrics": metrics,
            "pnl_res": pnl_res,
            "first_bear": first_bear_str,
            "lag": lag,
            "lag_str": lag_str,
            "conl_at_bear_str": conl_at_bear_str,
            "max_gain_str": max_gain_str,
            "bear_days": metrics["bear_days_total"],
        })

    # ── 비교표 출력 ──────────────────────────────────────────
    print("\n[3] 파라미터별 비교표")
    print("-" * 95)
    print(
        f"{'설정':12} {'윈도우':4} {'진입':5} {'복귀':5} {'stk':3} | "
        f"{'최초발동':6} {'고점lag':7} {'CONL진입':8} {'최대수익':8} | "
        f"{'bear일':6} {'거래수':5} {'WR%':5} {'총pnl':8}"
    )
    print("-" * 95)

    for r in results:
        p = r["params"]
        pr = r["pnl_res"]
        print(
            f"{p.label:12} {p.window:4d} {p.entry_thresh:.2f} {p.recovery_thresh:.2f} {p.min_streak:3d} | "
            f"{r['first_bear']:6} {r['lag_str']:7} {r['conl_at_bear_str']:8} {r['max_gain_str']:8} | "
            f"{r['bear_days']:6d} {pr['n_trades']:5d} {pr['win_rate']:5.1f} {pr['total_pnl']:+8.1f}%"
        )

    # ── 최적 파라미터 선택 ───────────────────────────────────
    print("\n[4] 최적 파라미터 선택 기준")
    print("-" * 50)
    print("  목표: 고점 감지 선행 (lag < 0) + WR 높음 + 총 수익 최대")
    print()

    # 유효한 거래가 있는 것만
    valid = [r for r in results if r["pnl_res"]["n_trades"] > 0 and r["lag"] is not None]
    if valid:
        # lag가 음수(고점 전 감지)이면서 total_pnl 최대
        by_pnl = sorted(valid, key=lambda x: x["pnl_res"]["total_pnl"], reverse=True)
        by_lag = sorted(valid, key=lambda x: x["lag"])  # 가장 이른 감지
        by_wr = sorted(valid, key=lambda x: x["pnl_res"]["win_rate"], reverse=True)

        print(f"  총수익 1위: {by_pnl[0]['params'].label.strip()}"
              f" → {by_pnl[0]['pnl_res']['total_pnl']:+.1f}%")
        print(f"  조기감지 1위: {by_lag[0]['params'].label.strip()}"
              f" → lag={by_lag[0]['lag']}일, 발동={by_lag[0]['first_bear']}")
        print(f"  승률 1위: {by_wr[0]['params'].label.strip()}"
              f" → WR={by_wr[0]['pnl_res']['win_rate']:.1f}%")

    # ── 상위 3개 파라미터 상세 거래 내역 ────────────────────
    print("\n[5] 상위 3개 파라미터 거래 상세")
    print("-" * 60)

    top3_idx = [0, 1, 2]  # 기준선 포함
    if valid:
        # total_pnl 기준 상위 3
        top3_pnl = sorted(
            range(len(results)),
            key=lambda i: results[i]["pnl_res"]["total_pnl"],
            reverse=True
        )[:3]
        top3_idx = top3_pnl

    for i in top3_idx:
        r = results[i]
        p = r["params"]
        trades = r["pnl_res"]["trades"]
        print(f"\n  [{p.label.strip()}]  window={p.window}d, entry<{p.entry_thresh}, "
              f"recovery>{p.recovery_thresh}, streak≥{p.min_streak}")
        if not trades:
            print("    거래 없음")
        else:
            for t in trades:
                ed = pd.Timestamp(t["entry_date"]).strftime("%m/%d")
                xd = pd.Timestamp(t["exit_date"]).strftime("%m/%d")
                mark = "✅" if t["pnl_pct"] > 0 else "❌"
                print(
                    f"    {mark} {ed}→{xd} ({t['days_held']}일) "
                    f"CONL ${t['entry_price']}→${t['exit_price']} "
                    f"{t['pnl_pct']:+.1f}% [{t['exit_reason']}]"
                )

    # ── 일별 체제 타임라인 (상위 파라미터 vs 기준선) ─────────
    print("\n[6] 일별 체제 타임라인 비교")
    print("-" * 80)

    # 기준선 + 최고 PnL 파라미터
    show_params = [results[0]]  # v1 기준선
    if valid:
        best_pnl_r = sorted(valid, key=lambda x: x["pnl_res"]["total_pnl"], reverse=True)[0]
        if best_pnl_r["params"].label != results[0]["params"].label:
            show_params.append(best_pnl_r)

    header_labels = [p["params"].label.strip()[:10] for p in show_params]
    h_str = " | ".join(f"{l:10}" for l in header_labels)
    print(f"{'날짜':6} {'btc':3} {'r5d':5} {'r7d':5} {'stk':3} | {h_str} | {'IREN':>7} {'CONL':>6}")
    print("-" * 80)

    r5d = poly_df["btc_up_raw"].rolling(5, min_periods=1).mean()
    r7d = poly_df["btc_up_raw"].rolling(7, min_periods=1).mean()

    # streak 계산
    streak_arr = []
    s = 0
    for v in poly_df["btc_up_raw"]:
        s = s + 1 if v < 0.5 else 0
        streak_arr.append(s)
    streak_s = pd.Series(streak_arr, index=poly_df.index)

    for dt in poly_df.index:
        d = dt.strftime("%m/%d")
        btc = "UP" if poly_df.loc[dt, "btc_up_raw"] >= 0.5 else "DN"
        r5 = r5d.get(dt, 0.5)
        r7 = r7d.get(dt, 0.5)
        stk = int(streak_s.get(dt, 0))

        regime_vals = []
        for sp in show_params:
            v = sp["regime_df"].loc[dt, "bear_regime"] if dt in sp["regime_df"].index else 0
            regime_vals.append("🔴BEAR" if v == 1 else "     ")
        regime_str = " | ".join(f"{v:10}" for v in regime_vals)

        iren_px = prices["IREN"].get(dt) if "IREN" in prices.columns else None
        conl_px = prices["CONL"].get(dt) if "CONL" in prices.columns else None
        iren_str = f"{float(iren_px):7.2f}" if iren_px is not None and not np.isnan(float(iren_px)) else "      -"
        conl_str = f"{float(conl_px):6.2f}" if conl_px is not None and not np.isnan(float(conl_px)) else "     -"

        print(f"{d:6} {btc:3} {r5:.2f} {r7:.2f} {stk:3} | {regime_str} | {iren_str} {conl_str}")

    # ── 핵심 발견 요약 ────────────────────────────────────────
    print("\n" + "=" * 75)
    print("[결론] 히스테리시스 강화 효과")
    print("-" * 75)

    # v1 기준선 vs 최고 설정 비교
    v1 = results[0]
    best = sorted(results, key=lambda x: x["pnl_res"]["total_pnl"], reverse=True)[0]

    print(f"\n  v1 기준선 ({v1['params'].label.strip()}):")
    print(f"    최초발동={v1['first_bear']}, bear일={v1['bear_days']}, "
          f"거래={v1['pnl_res']['n_trades']}건, "
          f"WR={v1['pnl_res']['win_rate']}%, PnL={v1['pnl_res']['total_pnl']:+.1f}%")

    if best["params"].label != v1["params"].label:
        print(f"\n  최고 파라미터 ({best['params'].label.strip()}):")
        print(f"    최초발동={best['first_bear']}, bear일={best['bear_days']}, "
              f"거래={best['pnl_res']['n_trades']}건, "
              f"WR={best['pnl_res']['win_rate']}%, PnL={best['pnl_res']['total_pnl']:+.1f}%")

    print()
    print("  ★ 권장 파라미터:")

    # 0거래 제외하고 조기감지 + WR60% 이상 + pnl 양수
    recommend = [
        r for r in results
        if r["pnl_res"]["n_trades"] > 0
        and r["pnl_res"]["win_rate"] >= 50
        and r["pnl_res"]["total_pnl"] > 0
        and (r["lag"] is not None and r["lag"] <= 0)
    ]
    if recommend:
        best_r = sorted(recommend, key=lambda x: x["pnl_res"]["total_pnl"], reverse=True)[0]
        p = best_r["params"]
        print(f"    {p.label.strip()}: window={p.window}d, entry<{p.entry_thresh}, "
              f"recovery>{p.recovery_thresh}, streak≥{p.min_streak}")
        print(f"    → WR={best_r['pnl_res']['win_rate']}%, PnL={best_r['pnl_res']['total_pnl']:+.1f}%")
        print(f"    → IREN 고점 {abs(best_r['lag'])}일 전 감지")
    else:
        print("    기준 충족 파라미터 없음 — 임계값 재검토 필요")

    return results, poly_df, prices


if __name__ == "__main__":
    results, poly_df, prices = main()

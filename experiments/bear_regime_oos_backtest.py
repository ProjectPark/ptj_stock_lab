"""
Bear Regime OOS 소급 검증 실험 — todd_fuck_v1
==============================================
2026-01-01 ~ 2026-02-17 (OOS 구간) 에서:

1. Polymarket btc_up 일별 방향 시계열 추출 (binary → rolling 연속화)
2. BearRegime 트리거 타이밍 vs IREN/CONL 고점 분석
3. BITI 진입 시나리오 수익 계산

데이터 제약사항:
  - btc_up_down은 final_prices (0=Down, 1=Up) 기준 binary
  - btc_monthly_dip은 time series 비어있어 직접 계산 불가
  → 3/5/7일 rolling avg로 연속 신호 근사
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

# ============================================================
# 1. Polymarket 데이터 로드 (binary btc_up)
# ============================================================

POLY_DIR = Path(config.POLY_DATA_DIR)  # data/polymarket/  (연도별 하위폴더)


def _last_prob_from_series(series: list) -> float | None:
    """time series에서 마지막 확률 추출."""
    if not series:
        return None
    last = series[-1]
    if isinstance(last, dict):
        return float(last.get("p", 0.5))
    if isinstance(last, (list, tuple)) and len(last) >= 2:
        return float(last[1])
    return None


def load_poly_oos(start: date, end: date) -> pd.DataFrame:
    """OOS 구간 Polymarket 데이터 로드.

    Returns DataFrame with columns:
      date, btc_up_raw, ndx_up_raw, btc_upside_pressure (if available)
    """
    records = []
    current = start
    while current <= end:
        year = current.year
        fp = POLY_DIR / str(current.year) / f"{current.isoformat()}_1m.json"
        if not fp.exists():
            current = date(current.year, current.month, current.day)
            # 다음날로
            from datetime import timedelta
            current = current + timedelta(days=1)
            continue

        try:
            with open(fp) as f:
                data = json.load(f)
        except Exception:
            from datetime import timedelta
            current = current + timedelta(days=1)
            continue

        indicators = data.get("indicators", {})

        # btc_up_down: final_prices 기반 binary
        btc_raw = 0.5
        btc_ind = indicators.get("btc_up_down", {})
        if "error" not in btc_ind:
            # time series 우선
            markets = btc_ind.get("markets", [])
            if markets:
                up_series = markets[0].get("outcomes", {}).get("Up", [])
                p = _last_prob_from_series(up_series)
                if p is not None:
                    btc_raw = p
            # final_prices fallback
            if btc_raw == 0.5:
                fp_prices = btc_ind.get("final_prices", {})
                if "Up" in fp_prices:
                    try:
                        btc_raw = float(fp_prices["Up"])
                    except (ValueError, TypeError):
                        pass

        # ndx_up_down
        ndx_raw = 0.5
        ndx_ind = indicators.get("ndx_up_down", {})
        if "error" not in ndx_ind:
            markets = ndx_ind.get("markets", [])
            if markets:
                up_series = markets[0].get("outcomes", {}).get("Up", [])
                p = _last_prob_from_series(up_series)
                if p is not None:
                    ndx_raw = p
            if ndx_raw == 0.5:
                fp_prices = ndx_ind.get("final_prices", {})
                if "Up" in fp_prices:
                    try:
                        ndx_raw = float(fp_prices["Up"])
                    except (ValueError, TypeError):
                        pass

        # btc_above_today: 상방 압력 (Yes 확률 평균)
        btc_upside = None
        above_ind = indicators.get("btc_above_today", {})
        if "error" not in above_ind:
            probs = []
            for m in above_ind.get("markets", []):
                yes_series = m.get("outcomes", {}).get("Yes", [])
                p = _last_prob_from_series(yes_series)
                if p is not None:
                    probs.append(p)
            if probs:
                btc_upside = sum(probs) / len(probs)

        # btc_monthly: reach/dip (시계열 없으면 None)
        btc_monthly_dip = None
        btc_monthly_reach = None
        monthly_ind = indicators.get("btc_monthly", {})
        if "error" not in monthly_ind:
            reach_probs, dip_probs = [], []
            for m in monthly_ind.get("markets", []):
                q = m.get("question", "").lower()
                yes_series = m.get("outcomes", {}).get("Yes", [])
                p = _last_prob_from_series(yes_series)
                if p is None:
                    continue
                if "reach" in q:
                    reach_probs.append(p)
                elif "dip" in q:
                    dip_probs.append(p)
            if reach_probs:
                btc_monthly_reach = max(reach_probs)
            if dip_probs:
                btc_monthly_dip = max(dip_probs)

        records.append({
            "date": current,
            "btc_up_raw": btc_raw,
            "ndx_up_raw": ndx_raw,
            "btc_upside_pressure": btc_upside,
            "btc_monthly_reach": btc_monthly_reach,
            "btc_monthly_dip": btc_monthly_dip,
        })

        from datetime import timedelta
        current = current + timedelta(days=1)

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


# ============================================================
# 2. 주가 데이터 로드
# ============================================================

def load_price_oos(start: date, end: date) -> pd.DataFrame:
    """IREN, CONL, BITU 일별 종가 로드."""
    frames = {}

    # IREN: soxx_iren_daily.parquet
    iren_fp = ROOT / "data/market/daily/soxx_iren_daily.parquet"
    if iren_fp.exists():
        df = pd.read_parquet(iren_fp)
        iren = df[df["symbol"] == "IREN"].copy()
        iren["date"] = pd.to_datetime(iren["timestamp"], unit="s").dt.date
        iren = iren.set_index("date").sort_index()
        mask = (iren.index >= start) & (iren.index <= end)
        frames["IREN"] = iren.loc[mask, "close"].rename("IREN")

    # CONL, BITU: history.parquet
    hist_fp = ROOT / "data/market/daily/history.parquet"
    if hist_fp.exists():
        df = pd.read_parquet(hist_fp)
        df.index = pd.to_datetime(df.index).date
        mask = (df.index >= start) & (df.index <= end)
        for ticker in ["CONL", "BITU"]:
            col = ("Close", ticker)
            if col in df.columns:
                frames[ticker] = df.loc[mask, col].rename(ticker)

    if not frames:
        return pd.DataFrame()

    result = pd.DataFrame(frames)
    result.index = pd.to_datetime(result.index)
    return result.sort_index()


# ============================================================
# 3. Bear Regime 신호 계산
# ============================================================

def compute_bear_signals(poly_df: pd.DataFrame) -> pd.DataFrame:
    """Polymarket 데이터 → BearRegime 신호 계산.

    btc_up_raw는 binary(0/1)이므로 rolling window로 연속화:
      rolling_3d_btc_up: 3일 rolling 평균
      rolling_5d_btc_up: 5일 rolling 평균
      rolling_7d_btc_up: 7일 rolling 평균

    BearRegime 조건 (proxy):
      rolling_5d_btc_up < 0.40  (5일 중 2일 이하 상승)
      consecutive_down_streak >= 3일
    """
    df = poly_df.copy()

    # Rolling 평균 (거래일 기준)
    for w in [3, 5, 7]:
        df[f"btc_up_r{w}d"] = df["btc_up_raw"].rolling(w, min_periods=1).mean()
        df[f"ndx_up_r{w}d"] = df["ndx_up_raw"].rolling(w, min_periods=1).mean()

    # 연속 하락 스트릭
    streak = 0
    streaks = []
    for v in df["btc_up_raw"]:
        if v == 0.0:
            streak += 1
        else:
            streak = 0
        streaks.append(streak)
    df["btc_down_streak"] = streaks

    # BearRegime 프록시 (5일 rolling)
    # 조건: rolling_5d < 0.40 AND streak >= 3
    df["bear_regime_proxy"] = (
        (df["btc_up_r5d"] < 0.40) & (df["btc_down_streak"] >= 3)
    ).astype(int)

    # Soft Warning: rolling_5d < 0.50
    df["bear_warn"] = (df["btc_up_r5d"] < 0.50).astype(int)

    # btc_upside_pressure 활용 (데이터 있는 경우만)
    has_upside = df["btc_upside_pressure"].notna()
    if has_upside.any():
        # upside_pressure < 0.40 → 추가 하락 압력
        df["upside_low"] = (df["btc_upside_pressure"] < 0.40).astype(float)
        df["upside_low"] = df["upside_low"].where(has_upside, other=np.nan)

    return df


# ============================================================
# 4. 가상 BITI 매매 시뮬레이션
# ============================================================

BITI_PROXY_DAILY_PCT = None  # BITI 실제 데이터 없으면 BTC 역방향으로 근사


def simulate_biti_trades(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    entry_col: str = "bear_regime_proxy",
    stop_pct: float = -10.0,
    target_pct: float = 30.0,
    hold_max_days: int = 20,
) -> pd.DataFrame:
    """BearRegime 신호 발생 시 BITI 매수 시뮬레이션.

    BITI가 없으면 CONL 가격의 역방향 (CONL 하락 = BITI 상승)으로 근사.
    1달러 기준 P&L.
    """
    if "BITU" in prices.columns:
        # BITU의 역방향으로 BITI 근사 (실제 BITI = -1x BTC)
        # BITU = 2x BTC, BITI = -1x BTC → 근사: BITU 반대방향
        bitu = prices["BITU"].dropna()
        # BITI 근사: BITU 가격 변화의 -0.5배 (BITI=-1x, BITU=+2x)
        bitu_ret = bitu.pct_change()
        biti_idx = bitu.index
        biti_cumret = (1 - bitu_ret * 0.5).cumprod()
        biti_proxy = pd.Series(
            biti_cumret.values / biti_cumret.iloc[0] * 10.0,  # $10 기준 가격
            index=biti_idx,
            name="BITI_proxy"
        )
    else:
        biti_proxy = None

    trades = []
    position_open = False
    entry_date = None
    entry_price = None

    signal_dates = signals.index[signals[entry_col] == 1]

    for i, dt in enumerate(signals.index):
        dt_date = dt.date() if hasattr(dt, 'date') else dt

        # 포지션 없으면 진입 검토
        if not position_open and dt in signal_dates:
            if biti_proxy is not None and dt in biti_proxy.index:
                price = biti_proxy[dt]
                position_open = True
                entry_date = dt
                entry_price = price
                entry_day = i

        # 포지션 보유 중이면 청산 검토
        if position_open and biti_proxy is not None:
            if dt in biti_proxy.index:
                current_price = biti_proxy[dt]
                pnl_pct = (current_price - entry_price) / entry_price * 100
                days_held = i - entry_day

                exit_reason = None
                if pnl_pct >= target_pct:
                    exit_reason = "TARGET"
                elif pnl_pct <= stop_pct:
                    exit_reason = "STOP"
                elif days_held >= hold_max_days:
                    exit_reason = "TIME"
                elif signals.loc[dt, entry_col] == 0 and days_held >= 3:
                    exit_reason = "REGIME_OFF"

                if exit_reason:
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": dt,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "pnl_pct": round(pnl_pct, 2),
                        "days_held": days_held,
                        "exit_reason": exit_reason,
                    })
                    position_open = False
                    entry_date = None
                    entry_price = None

    return pd.DataFrame(trades)


# ============================================================
# 5. 메인 분석
# ============================================================

def main():
    OOS_START = date(2026, 1, 1)
    OOS_END = date(2026, 2, 17)

    print("=" * 65)
    print("Bear Regime OOS 소급 검증 — todd_fuck_v1")
    print(f"분석 기간: {OOS_START} ~ {OOS_END}")
    print("=" * 65)

    # 데이터 로드
    print("\n[1] 데이터 로드 중...")
    poly_df = load_poly_oos(OOS_START, OOS_END)
    prices = load_price_oos(OOS_START, OOS_END)
    print(f"  Polymarket: {len(poly_df)}일")
    print(f"  주가 종목: {list(prices.columns)}")

    # BearRegime 신호 계산
    print("\n[2] Bear Regime 신호 계산...")
    signals = compute_bear_signals(poly_df)
    signals.index = pd.to_datetime(signals.index)

    # ── 일별 시계열 출력 ──────────────────────────────────────
    print("\n[3] 일별 Polymarket + 주가 타임라인")
    print("-" * 75)

    # 가격 데이터 align
    timeline = signals.copy()
    for ticker in prices.columns:
        timeline[ticker] = prices[ticker]

    # 가격 변화율
    for ticker in prices.columns:
        timeline[f"{ticker}_chg"] = prices[ticker].pct_change() * 100

    header = (
        f"{'날짜':10} {'btc_up':6} {'r5d':5} {'streak':6} "
        f"{'warn':4} {'bear':4} | "
        f"{'IREN':>7} {'IREN%':>6} | "
        f"{'CONL':>6} {'CONL%':>6}"
    )
    print(header)
    print("-" * 75)

    for dt, row in timeline.iterrows():
        d = dt.strftime("%m/%d") if hasattr(dt, 'strftime') else str(dt)
        btc_raw = row.get("btc_up_raw", 0.5)
        r5d = row.get("btc_up_r5d", 0.5)
        streak = int(row.get("btc_down_streak", 0))
        warn = "⚠" if row.get("bear_warn", 0) else " "
        bear = "🔴" if row.get("bear_regime_proxy", 0) else " "

        iren = row.get("IREN", None)
        iren_chg = row.get("IREN_chg", None)
        conl = row.get("CONL", None)
        conl_chg = row.get("CONL_chg", None)

        iren_str = f"{iren:7.2f}" if iren is not None and not np.isnan(iren) else "      -"
        iren_chg_str = f"{iren_chg:+6.1f}%" if iren_chg is not None and not np.isnan(iren_chg) else "      -"
        conl_str = f"{conl:6.2f}" if conl is not None and not np.isnan(conl) else "     -"
        conl_chg_str = f"{conl_chg:+6.1f}%" if conl_chg is not None and not np.isnan(conl_chg) else "      -"

        line = (
            f"{d:10} {'UP' if btc_raw >= 0.5 else 'DN':6} {r5d:.2f} {streak:6d} "
            f"{warn:4} {bear:4} | "
            f"{iren_str} {iren_chg_str} | "
            f"{conl_str} {conl_chg_str}"
        )
        print(line)

    # ── BearRegime 트리거 분석 ────────────────────────────────
    print("\n[4] Bear Regime 트리거 분석")
    print("-" * 50)

    regime_days = signals[signals["bear_regime_proxy"] == 1]
    warn_days = signals[(signals["bear_warn"] == 1) & (signals["bear_regime_proxy"] == 0)]

    print(f"  WARN 발생일:        {len(warn_days)}일")
    print(f"  BEAR REGIME 발생일: {len(regime_days)}일")

    if len(regime_days) > 0:
        first_bear = regime_days.index[0]
        print(f"  최초 BEAR 진입:     {first_bear.strftime('%Y-%m-%d')}")

        # 최초 BEAR 시점의 IREN/CONL 가격
        if first_bear in timeline.index:
            r = timeline.loc[first_bear]
            iren_at_bear = r.get("IREN")
            conl_at_bear = r.get("CONL")
            iren_at_bear_str = f"${iren_at_bear:.2f}" if iren_at_bear and not np.isnan(iren_at_bear) else "N/A"
            conl_at_bear_str = f"${conl_at_bear:.2f}" if conl_at_bear and not np.isnan(conl_at_bear) else "N/A"
            print(f"  IREN @ 최초 BEAR:   {iren_at_bear_str}")
            print(f"  CONL @ 최초 BEAR:   {conl_at_bear_str}")

        # IREN 고점 vs BEAR 트리거 타이밍
        if "IREN" in timeline.columns:
            iren_series = timeline["IREN"].dropna()
            if not iren_series.empty:
                iren_peak_dt = iren_series.idxmax()
                iren_peak_price = iren_series.max()
                print(f"\n  IREN 실제 고점:     {iren_peak_dt.strftime('%Y-%m-%d')} (${iren_peak_price:.2f})")

                # 고점 → 최초 BEAR 사이 일수
                lag = (first_bear - iren_peak_dt).days
                print(f"  고점 → BEAR 트리거: +{lag}일 후" if lag >= 0 else f"  BEAR 트리거 → 고점: {-lag}일 전")

        if "CONL" in timeline.columns:
            conl_series = timeline["CONL"].dropna()
            if not conl_series.empty:
                conl_peak_dt = conl_series.idxmax()
                conl_peak_price = conl_series.max()
                print(f"  CONL 실제 고점:     {conl_peak_dt.strftime('%Y-%m-%d')} (${conl_peak_price:.2f})")

    # ── 가상 BITI 매매 결과 ───────────────────────────────────
    print("\n[5] 가상 BITI 진입 시뮬레이션")
    print("-" * 50)
    trades = simulate_biti_trades(signals, prices)
    if trades.empty:
        print("  BITU 데이터 없거나 BITI 신호 미발생")
    else:
        print(f"  총 거래 수: {len(trades)}")
        for _, t in trades.iterrows():
            print(
                f"  {t['entry_date'].strftime('%m/%d')} → {t['exit_date'].strftime('%m/%d')} "
                f"({t['days_held']}일) {t['pnl_pct']:+.1f}% [{t['exit_reason']}]"
            )

    # ── WARN 기준 조기 감지 분석 ──────────────────────────────
    print("\n[6] Soft Warning 기반 조기 감지 분석")
    print("-" * 50)
    if "CONL" in timeline.columns:
        conl_series = timeline["CONL"].dropna()
        for dt, row in timeline.iterrows():
            if row.get("bear_warn", 0) == 1:
                conl_now = conl_series.get(dt)
                if conl_now and not np.isnan(conl_now):
                    print(
                        f"  WARN {dt.strftime('%m/%d')}: "
                        f"r5d={row['btc_up_r5d']:.2f}, streak={int(row['btc_down_streak'])}, "
                        f"CONL=${conl_now:.2f}"
                    )

    # ── Polymarket 수준 한계 분석 ─────────────────────────────
    print("\n[7] 데이터 한계 및 개선 방향")
    print("-" * 50)
    has_upside = signals["btc_upside_pressure"].notna().sum()
    has_monthly_dip = signals["btc_monthly_dip"].notna().sum()
    print(f"  btc_upside_pressure 유효 일수: {has_upside}/{len(signals)}")
    print(f"  btc_monthly_dip 유효 일수:     {has_monthly_dip}/{len(signals)}")
    print()
    print("  ⚠️  현재 Polymarket 저장 방식의 한계:")
    print("    - btc_up_down: final_prices (binary 0/1) — 연속 확률 없음")
    print("    - btc_monthly: CLOB time series 비어있음 → 확률 추출 불가")
    print("    - btc_above_today: 동일 문제")
    print()
    print("  개선 방향:")
    print("    - collect_poly_history_async.py로 장 중 5분봉 수집 강화")
    print("    - 장 중 btc_up 스냅샷을 별도 로그로 저장")
    print("    - 또는 rolling_5d_btc_up을 공식 Bear Regime 지표로 채택")

    # ── 최종 결론 ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("[결론]")
    print("-" * 65)

    if len(regime_days) > 0:
        first_bear = regime_days.index[0]
        if "IREN" in timeline.columns:
            iren_series = timeline["IREN"].dropna()
            iren_peak_dt = iren_series.idxmax()
            iren_at_bear = timeline.loc[first_bear, "IREN"] if first_bear in timeline.index else None
            iren_at_bottom = iren_series.min()

            print(f"  BearRegime (proxy) 최초 발동: {first_bear.strftime('%Y-%m-%d')}")
            if iren_at_bear and not np.isnan(iren_at_bear):
                drop_from_bear_to_bottom = (iren_at_bottom - iren_at_bear) / iren_at_bear * 100
                print(f"  IREN @ 체제 진입: ${iren_at_bear:.2f}")
                print(f"  IREN @ 최저점:    ${iren_at_bottom:.2f} ({drop_from_bear_to_bottom:+.1f}%)")
                if iren_at_bear < iren_at_bottom:
                    print("  → IREN이 체제 진입 후 반등. 이미 저점에서 감지됨.")
                else:
                    print(f"  → 체제 진입 이후 추가 하락: {drop_from_bear_to_bottom:.1f}%")
                    print("    BITI 진입 시 이 구간이 수익 구간")
    else:
        print("  ⚠️  Rolling 5d < 40% + streak 3일 조건으로는 Bear Regime 미발동")
        print("  → 조건 완화 필요: rolling_5d < 0.50 (WARN 수준)")

    print()
    print("  Rolling btc_up_r5d 전 기간 평균:",
          f"{signals['btc_up_r5d'].mean():.3f}")
    print("  Rolling btc_up_r5d 최솟값:",
          f"{signals['btc_up_r5d'].min():.3f} "
          f"({signals['btc_up_r5d'].idxmin().strftime('%Y-%m-%d')})")
    print()

    return signals, timeline, trades


if __name__ == "__main__":
    signals, timeline, trades = main()

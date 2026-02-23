"""
profit_strategy_optimize.py
============================
수익 극대화를 위한 다중 전략 그리드 서치 + BTC 국면 필터

Flow:
  1. BTC/MSTR 데이터 추가 fetch
  2. 단일 종목 파라미터 그리드 서치 (MSTU, CONL 중심)
  3. BTC 국면 필터 적용 버전 vs 미적용 비교
  4. 최적 파라미터 조합 TOP 20 출력
  5. 종목 조합 포트폴리오 백테스트 (집중투자 vs 분산)

실행:
    pyenv shell ptj_stock_lab && python experiments/profit_strategy_optimize.py
"""
from __future__ import annotations

import itertools
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

OHLCV_PATH  = ROOT / "data" / "market" / "daily" / "profit_curve_ohlcv.parquet"
BTC_PATH    = ROOT / "data" / "market" / "daily" / "btc_mstr_daily.parquet"
FETCH_START = "2023-09-01"

FOCUS_TICKERS = ["MSTU", "CONL", "PTIR", "IREN", "ROBN", "NVDL", "AMDL"]


# ── 지표 ─────────────────────────────────────────────────────────────────────
def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("Date").reset_index(drop=True)
    df["ma10"]  = df["Close"].rolling(10).mean()
    df["ma20"]  = df["Close"].rolling(20).mean()
    df["ma60"]  = df["Close"].rolling(60).mean()
    df["rsi14"] = rsi(df["Close"], 14)
    df["atr14"] = (df["High"] - df["Low"]).rolling(14).mean()
    df["pct_ma20"] = (df["Close"] - df["ma20"]) / df["ma20"] * 100
    df["pct_ma60"] = (df["Close"] - df["ma60"]) / df["ma60"] * 100
    # 20일 내 최고가 대비 낙폭
    df["roll_hi20"] = df["High"].rolling(20).max()
    df["dd20"]      = (df["Close"] - df["roll_hi20"]) / df["roll_hi20"] * 100
    return df


# ── BTC 국면 판단 ─────────────────────────────────────────────────────────────
def fetch_btc_mstr():
    """BTC-USD + MSTR 일봉 수집 → BTC_PATH 저장"""
    print("[BTC/MSTR 수집]")
    raw = yf.download("BTC-USD MSTR", start=FETCH_START, auto_adjust=True, progress=False)
    frames = []
    for sym in ["BTC-USD", "MSTR"]:
        if sym not in raw["Close"].columns:
            continue
        df = raw.xs(sym, axis=1, level=1)[["Open","High","Low","Close","Volume"]].copy()
        df.index.name = "Date"
        df = df.dropna(subset=["Close"]).reset_index()
        df["ticker"] = sym
        frames.append(df)
        print(f"  {sym}: {len(df)}일")
    merged = pd.concat(frames, ignore_index=True)
    merged.to_parquet(BTC_PATH, index=False)
    print(f"  저장 → {BTC_PATH}")
    return merged


def load_btc_regime() -> pd.DataFrame:
    """BTC-USD에 MA20/MA60 추가해 국면 컬럼 생성"""
    if not BTC_PATH.exists():
        fetch_btc_mstr()
    df = pd.read_parquet(BTC_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    btc = df[df["ticker"] == "BTC-USD"].copy().sort_values("Date")
    btc["btc_ma20"]  = btc["Close"].rolling(20).mean()
    btc["btc_ma60"]  = btc["Close"].rolling(60).mean()
    # 국면: 1=강세(MA20>MA60), 0=약세
    btc["bull"]      = (btc["Close"] > btc["btc_ma60"]).astype(int)
    btc["bull_str"]  = (btc["Close"] > btc["btc_ma20"]).astype(int)  # 단기 강세
    return btc[["Date", "Close", "btc_ma20", "btc_ma60", "bull", "bull_str"]].rename(
        columns={"Close": "btc_close"}
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 핵심 백테스트 엔진
# ═══════════════════════════════════════════════════════════════════════════════
def run_backtest(
    df: pd.DataFrame,
    ticker: str,
    # 진입 조건
    entry_pct_ma20: float,    # MA20 대비 최소 % (양수=추세추종, 음수=역발상)
    rsi_min: float,           # 최소 RSI
    rsi_max: float,           # 최대 RSI (100이면 무제한)
    # 피라미딩
    pyramid_pct: float,       # 추가 매수 트리거 (상승 % 또는 하락 %)
    pyramid_mode: str,        # "rise"=추세, "drop"=역발상
    max_pyramid: int,
    # 청산
    target_pct: float,
    stop_pct: float,
    hold_days: int,
    # 국면 필터
    btc_regime: pd.DataFrame | None,
    require_bull: bool,
    # 포지션 크기
    unit_usd: float = 740.0,
    fx: float       = 1350.0,
) -> dict:
    """단일 ticker 백테스트 → 성과 dict 반환"""
    df = df.copy()

    # BTC 국면 조인
    if btc_regime is not None:
        df = df.merge(btc_regime[["Date", "bull", "bull_str"]], on="Date", how="left")
        df["bull"]     = df["bull"].fillna(method="ffill").fillna(0)
        df["bull_str"] = df["bull_str"].fillna(method="ffill").fillna(0)
    else:
        df["bull"] = 1
        df["bull_str"] = 1

    trades   = []
    position = []      # [(date, qty, price)]
    p_count  = 0
    last_buy = None

    for _, row in df.iterrows():
        if pd.isna(row.get("ma20")) or pd.isna(row.get("rsi14")):
            continue

        price   = row["Close"]
        d       = row["Date"]
        pma20   = row["pct_ma20"]
        r14     = row["rsi14"]
        is_bull = int(row["bull"])
        has_pos = bool(position)

        # ── 청산 ──
        if has_pos:
            tqty  = sum(p[1] for p in position)
            tcost = sum(p[1] * p[2] for p in position)
            avg   = tcost / tqty
            pp    = (price - avg) / avg * 100
            held  = (d - position[0][0]).days
            trend_break = pma20 < -20.0

            if pp >= target_pct or pp <= stop_pct or held >= hold_days or trend_break:
                pnl_usd = tqty * price - tcost
                trades.append({
                    "Date": d, "ticker": ticker,
                    "PnL_USD": pnl_usd,
                    "PnL_KRW": pnl_usd * fx,
                    "PnL_pct": pp,
                    "HeldDays": held,
                    "Reason": ("TARGET" if pp >= target_pct
                               else "TREND" if trend_break
                               else "STOP"  if pp <= stop_pct
                               else "TIME"),
                    "bull_at_entry": position[0][3] if len(position[0]) > 3 else 1,
                })
                position, p_count, last_buy = [], 0, None
                continue

            # 국면 바뀌면 강제 청산
            if require_bull and not is_bull and pp > -5:
                pnl_usd = tqty * price - tcost
                trades.append({
                    "Date": d, "ticker": ticker,
                    "PnL_USD": pnl_usd,
                    "PnL_KRW": pnl_usd * fx,
                    "PnL_pct": pp,
                    "HeldDays": held,
                    "Reason": "REGIME_EXIT",
                    "bull_at_entry": position[0][3] if len(position[0]) > 3 else 1,
                })
                position, p_count, last_buy = [], 0, None
                continue

        # ── 국면 필터 ──
        if require_bull and not is_bull:
            continue

        # ── 신규 진입 ──
        if not has_pos:
            cond = (pma20 >= entry_pct_ma20) and (rsi_min <= r14 <= rsi_max)
            if cond:
                qty = unit_usd / price
                position.append((d, qty, price, is_bull))
                last_buy = price
                p_count  = 0

        # ── 피라미딩 ──
        elif p_count < max_pyramid and last_buy is not None:
            if pyramid_mode == "rise":
                trigger = (price - last_buy) / last_buy * 100 >= pyramid_pct
            else:
                trigger = (price - last_buy) / last_buy * 100 <= -pyramid_pct
            if trigger and (rsi_min <= r14 <= rsi_max):
                qty = unit_usd / price
                position.append((d, qty, price, is_bull))
                last_buy = price
                p_count  += 1

    if not trades:
        return {"ticker": ticker, "n_trades": 0, "pnl_krw": 0,
                "win_rate": 0, "avg_pct": 0, "sharpe": 0}

    tdf = pd.DataFrame(trades)
    wins = (tdf["PnL_KRW"] > 0).sum()
    return {
        "ticker":    ticker,
        "n_trades":  len(tdf),
        "n_wins":    int(wins),
        "win_rate":  round(wins / len(tdf) * 100, 1),
        "pnl_krw":   round(tdf["PnL_KRW"].sum()),
        "avg_pct":   round(tdf["PnL_pct"].mean(), 2),
        "max_dd_pct": round(tdf["PnL_pct"].min(), 2),
        "trades_df": tdf,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 그리드 서치
# ═══════════════════════════════════════════════════════════════════════════════
GRID = {
    "entry_pct_ma20":  [-5, 0, 5, 10, 15],
    "rsi_min":         [45, 50, 55, 60, 65],
    "rsi_max":         [80, 90, 100],
    "target_pct":      [8, 12, 15, 20, 25, 30],
    "stop_pct":        [-10, -15, -20, -25],
    "hold_days":       [20, 30, 45, 60],
    "pyramid_pct":     [5, 8, 10],
    "pyramid_mode":    ["rise"],   # 모멘텀 전략 고정
    "max_pyramid":     [1, 2, 3],
    "require_bull":    [False, True],
}

# 조합 수 줄이기: 핵심 파라미터만 선택 (나머지 고정)
GRID_REDUCED = {
    "entry_pct_ma20":  [-5, 0, 5, 10, 15],
    "rsi_min":         [45, 55, 65],
    "target_pct":      [8, 12, 15, 20, 25],
    "stop_pct":        [-10, -15, -20],
    "hold_days":       [20, 30, 45],
    "require_bull":    [False, True],
}
# 고정값
FIXED = dict(rsi_max=100, pyramid_pct=7, pyramid_mode="rise", max_pyramid=2)


def run_grid_search(ticker: str, ohlcv_enr: pd.DataFrame,
                    btc: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    keys = list(GRID_REDUCED.keys())
    combos = list(itertools.product(*[GRID_REDUCED[k] for k in keys]))
    results = []

    if verbose:
        print(f"\n[그리드 서치] {ticker}  조합 수: {len(combos)}")

    for combo in combos:
        params = dict(zip(keys, combo))
        params.update(FIXED)
        r = run_backtest(
            df            = ohlcv_enr,
            ticker        = ticker,
            btc_regime    = btc,
            unit_usd      = 740.0,
            fx            = 1350.0,
            **{k: params[k] for k in [
                "entry_pct_ma20","rsi_min","rsi_max",
                "pyramid_pct","pyramid_mode","max_pyramid",
                "target_pct","stop_pct","hold_days","require_bull"
            ]}
        )
        r.update(params)
        results.append(r)

    df_res = pd.DataFrame([{k: v for k, v in r.items() if k != "trades_df"}
                            for r in results])
    # 수익 있는 것만
    df_res = df_res[df_res["n_trades"] >= 3].copy()
    df_res = df_res.sort_values("pnl_krw", ascending=False).reset_index(drop=True)

    if verbose:
        print(f"  총 {len(df_res)}개 유효 조합 중 TOP 10:")
        show_cols = ["pnl_krw","win_rate","n_trades","avg_pct","max_dd_pct",
                     "entry_pct_ma20","rsi_min","target_pct","stop_pct","hold_days","require_bull"]
        print(df_res[show_cols].head(10).to_string(index=False))

    return df_res, results


# ═══════════════════════════════════════════════════════════════════════════════
# 포트폴리오 백테스트 (최적 파라미터 조합)
# ═══════════════════════════════════════════════════════════════════════════════
def run_portfolio(
    best_params: dict,
    ohlcv_all: pd.DataFrame,
    tickers: list[str],
    btc: pd.DataFrame,
    unit_usd_per_ticker: float = 740.0,
    fx: float = 1350.0,
) -> pd.DataFrame:
    all_trades = []
    for t in tickers:
        sub = ohlcv_all[ohlcv_all["ticker"] == t].copy()
        if len(sub) < 30:
            continue
        sub = add_indicators(sub)
        r = run_backtest(
            df=sub, ticker=t, btc_regime=btc,
            unit_usd=unit_usd_per_ticker, fx=fx, **best_params
        )
        if r["n_trades"] > 0 and "trades_df" in r:
            all_trades.append(r["trades_df"])

    if not all_trades:
        return pd.DataFrame()

    combined = pd.concat(all_trades, ignore_index=True).sort_values("Date")
    combined["cum_pnl_krw"] = combined["PnL_KRW"].cumsum()
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    # ── 데이터 로드 ──
    print("=== 데이터 로드 ===")
    ohlcv = pd.read_parquet(OHLCV_PATH)
    ohlcv["Date"] = pd.to_datetime(ohlcv["Date"])

    if not BTC_PATH.exists():
        fetch_btc_mstr()
    btc = load_btc_regime()
    print(f"BTC 국면 데이터: {btc['Date'].min().date()} ~ {btc['Date'].max().date()}")
    bull_pct = btc["bull"].mean() * 100
    print(f"BTC 강세 국면 비율: {bull_pct:.1f}%\n")

    # ── 종목별 지표 추가 ──
    enriched = {}
    for t in FOCUS_TICKERS:
        sub = ohlcv[ohlcv["ticker"] == t].copy()
        if len(sub) >= 30:
            enriched[t] = add_indicators(sub)

    # ── 1. MSTU + CONL 그리드 서치 (핵심 2종목) ──
    print("=" * 60)
    print("1. MSTU / CONL 파라미터 그리드 서치")
    print("=" * 60)

    best_by_ticker = {}
    all_grid_results = []

    for t in ["MSTU", "CONL"]:
        if t not in enriched:
            continue
        df_res, raw_results = run_grid_search(t, enriched[t], btc)
        best_by_ticker[t] = df_res.iloc[0].to_dict()
        all_grid_results.append(df_res.assign(ticker_src=t))

    # ── 2. 전 종목 공통 최적 파라미터 탐색 ──
    print("\n" + "=" * 60)
    print("2. 전 종목 통합 그리드 서치 (포트폴리오 최적화)")
    print("=" * 60)

    keys = list(GRID_REDUCED.keys())
    combos = list(itertools.product(*[GRID_REDUCED[k] for k in keys]))
    print(f"  조합 수: {len(combos)}  ×  {len(enriched)}종목")

    portfolio_results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        params.update(FIXED)

        bt_params = {k: params[k] for k in [
            "entry_pct_ma20","rsi_min","rsi_max",
            "pyramid_pct","pyramid_mode","max_pyramid",
            "target_pct","stop_pct","hold_days","require_bull"
        ]}

        total_pnl   = 0
        total_trades = 0
        total_wins  = 0
        min_pct     = 0

        for t, df_enr in enriched.items():
            r = run_backtest(df=df_enr, ticker=t, btc_regime=btc,
                             unit_usd=740.0, fx=1350.0, **bt_params)
            total_pnl    += r["pnl_krw"]
            total_trades += r["n_trades"]
            total_wins   += r.get("n_wins", 0)
            min_pct       = min(min_pct, r.get("max_dd_pct", 0))

        portfolio_results.append({
            **params,
            "total_pnl_krw": total_pnl,
            "total_trades":  total_trades,
            "win_rate": round(total_wins / total_trades * 100, 1) if total_trades > 0 else 0,
            "worst_trade_pct": min_pct,
        })

    port_df = pd.DataFrame(portfolio_results)
    port_df = port_df[port_df["total_trades"] >= 10].sort_values(
        "total_pnl_krw", ascending=False
    ).reset_index(drop=True)

    print(f"\n[포트폴리오 최적 TOP 20]")
    show = ["total_pnl_krw","win_rate","total_trades","worst_trade_pct",
            "entry_pct_ma20","rsi_min","target_pct","stop_pct","hold_days",
            "require_bull","max_pyramid"]
    print(port_df[show].head(20).to_string(index=False))

    # ── 3. 상위 3개 파라미터 조합으로 전략 상세 비교 ──
    print("\n" + "=" * 60)
    print("3. 상위 3개 조합 상세 백테스트")
    print("=" * 60)

    for rank in range(min(3, len(port_df))):
        row = port_df.iloc[rank]
        bp = {k: row[k] for k in [
            "entry_pct_ma20","rsi_min","rsi_max",
            "pyramid_pct","pyramid_mode","max_pyramid",
            "target_pct","stop_pct","hold_days","require_bull"
        ]}
        print(f"\n[#{rank+1}] PnL {row['total_pnl_krw']:,.0f}원 | "
              f"entry_ma20={bp['entry_pct_ma20']} rsi≥{bp['rsi_min']} "
              f"tgt={bp['target_pct']}% stop={bp['stop_pct']}% "
              f"hold={bp['hold_days']}d bull={bp['require_bull']}")

        for t, df_enr in enriched.items():
            r = run_backtest(df=df_enr, ticker=t, btc_regime=btc,
                             unit_usd=740.0, fx=1350.0, **bp)
            if r["n_trades"] > 0:
                print(f"  {t:6s}: {r['pnl_krw']:>12,.0f}원  "
                      f"승률={r['win_rate']:5.1f}%  "
                      f"건수={r['n_trades']:3d}  "
                      f"평균={r['avg_pct']:+.1f}%  "
                      f"최대손실={r.get('max_dd_pct',0):+.1f}%")

    # ── 4. 집중투자 전략: MSTU + CONL 2종목 집중 ──
    print("\n" + "=" * 60)
    print("4. MSTU + CONL 집중투자 (단위 3배 = 2,220 USD/매수)")
    print("=" * 60)

    best_params = port_df.iloc[0]
    bp_best = {k: best_params[k] for k in [
        "entry_pct_ma20","rsi_min","rsi_max",
        "pyramid_pct","pyramid_mode","max_pyramid",
        "target_pct","stop_pct","hold_days","require_bull"
    ]}

    for t in ["MSTU", "CONL"]:
        if t not in enriched:
            continue
        r = run_backtest(df=enriched[t], ticker=t, btc_regime=btc,
                         unit_usd=2220.0, fx=1350.0, **bp_best)
        if r["n_trades"] > 0:
            tdf = r["trades_df"]
            tdf["cum"] = tdf["PnL_KRW"].cumsum()
            print(f"\n{t} 집중투자 (단위 2,220 USD):")
            print(f"  총 PnL    : {r['pnl_krw']:>15,.0f} 원")
            print(f"  승률      : {r['win_rate']}%")
            print(f"  건수      : {r['n_trades']}")
            print(f"  평균수익률 : {r['avg_pct']:+.2f}%")
            print(f"  최대손실   : {r.get('max_dd_pct',0):+.2f}%")
            print(f"  최종 누적  : {tdf['cum'].iloc[-1]:>15,.0f} 원")
            # 월별 PnL
            tdf["월"] = tdf["Date"].dt.to_period("M")
            monthly = tdf.groupby("월")["PnL_KRW"].sum()
            print(f"  월별 PnL:")
            for m, v in monthly.items():
                bar = "█" * min(30, max(0, int(abs(v) / 500000)))
                sign = "+" if v >= 0 else "-"
                print(f"    {m}  {sign}{bar}  {v:>+12,.0f}")

    print("\n✅ 분석 완료")


if __name__ == "__main__":
    main()

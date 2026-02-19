"""
exp_B_new_indicators.py — 새로운 기술 지표 조합 실험
=====================================================
Bollinger Band %B + squeeze
MACD histogram 방향
ADX 추세 강도
거래량 폭발 (volume surge)
단기/중기 모멘텀 듀얼
"""
import sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import itertools
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

OUT = ROOT / "experiments" / "results" / "B_new_indicators.csv"
OHLCV_PATH = ROOT / "data" / "market" / "daily" / "profit_curve_ohlcv.parquet"
BTC_PATH   = ROOT / "data" / "market" / "daily" / "btc_mstr_daily.parquet"

TICKERS = ["MSTU", "CONL", "PTIR", "IREN", "NVDL", "AMDL", "ROBN"]


def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def add_indicators(df):
    df = df.copy().sort_values("Date").reset_index(drop=True)
    c = df["Close"]
    # 기본
    df["ma10"]  = c.rolling(10).mean()
    df["ma20"]  = c.rolling(20).mean()
    df["ma60"]  = c.rolling(60).mean()
    df["rsi14"] = rsi(c, 14)
    df["pct_ma20"] = (c - df["ma20"]) / df["ma20"] * 100

    # Bollinger Band (20, 2σ)
    std20 = c.rolling(20).std()
    df["bb_upper"] = df["ma20"] + 2 * std20
    df["bb_lower"] = df["ma20"] - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["ma20"] * 100
    df["bb_pct"]   = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]) * 100  # 0=하단, 100=상단
    # BB squeeze: 밴드 폭이 20일 최솟값 근처
    df["bb_squeeze"] = (df["bb_width"] < df["bb_width"].rolling(20).quantile(0.2)).astype(int)

    # MACD (12, 26, 9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    df["macd_cross_up"] = ((df["macd"] > df["macd_signal"]) &
                           (df["macd"].shift(1) <= df["macd_signal"].shift(1))).astype(int)

    # ADX (14)
    high, low = df["High"], df["Low"]
    tr = pd.concat([high - low,
                    (high - c.shift(1)).abs(),
                    (low  - c.shift(1)).abs()], axis=1).max(axis=1)
    dm_plus  = ((high - high.shift(1)).clip(lower=0)
                .where(high - high.shift(1) > low.shift(1) - low, 0))
    dm_minus = ((low.shift(1) - low).clip(lower=0)
                .where(low.shift(1) - low > high - high.shift(1), 0))
    atr14 = tr.ewm(span=14, adjust=False).mean()
    df["adx"] = (
        (dm_plus.ewm(span=14, adjust=False).mean() / atr14 -
         dm_minus.ewm(span=14, adjust=False).mean() / atr14).abs()
        .ewm(span=14, adjust=False).mean() * 100
    )
    df["di_plus"]  = dm_plus.ewm(span=14, adjust=False).mean() / atr14 * 100
    df["di_minus"] = dm_minus.ewm(span=14, adjust=False).mean() / atr14 * 100
    df["adx_trend"] = (df["adx"] > 25).astype(int)  # ADX>25 = 추세 있음

    # 거래량 지표
    df["vol_ma20"]  = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_ma20"]  # 1 이상 = 평균 초과
    df["vol_surge"] = (df["vol_ratio"] > 1.5).astype(int)

    # 단기/중기 모멘텀 듀얼
    df["mom5"]  = c.pct_change(5) * 100
    df["mom10"] = c.pct_change(10) * 100
    df["mom20"] = c.pct_change(20) * 100
    df["dual_mom_up"] = ((df["mom5"] > 0) & (df["mom20"] > 0)).astype(int)

    # 가격이 최근 N일 신고가 돌파 (브레이크아웃)
    df["high10"] = df["High"].rolling(10).max().shift(1)
    df["high20"] = df["High"].rolling(20).max().shift(1)
    df["breakout10"] = (c > df["high10"]).astype(int)
    df["breakout20"] = (c > df["high20"]).astype(int)

    # MA 정렬 (triple MA: 10>20>60 모두 상승 정렬)
    df["ma_aligned"] = ((df["ma10"] > df["ma20"]) & (df["ma20"] > df["ma60"])).astype(int)

    return df


def backtest(df, entry_fn, target_pct, stop_pct, hold_days,
             unit_usd=740.0, fx=1350.0):
    """entry_fn(row) -> True/False"""
    trades, pos, last_buy = [], [], None
    for _, row in df.iterrows():
        if pd.isna(row.get("ma20")): continue
        price, d = row["Close"], row["Date"]
        pma20 = row["pct_ma20"]
        has_pos = bool(pos)

        if has_pos:
            tq = sum(p[1] for p in pos); tc = sum(p[1]*p[2] for p in pos)
            avg = tc/tq; pp = (price-avg)/avg*100
            held = (d - pos[0][0]).days
            if pp >= target_pct or pp <= stop_pct or held >= hold_days or pma20 < -25:
                pnl = tq*price - tc
                trades.append({"Date":d,"PnL_KRW":pnl*fx,"PnL_pct":pp,"HeldDays":held,
                                "Reason":"T" if pp>=target_pct else "S" if pp<=stop_pct else "X"})
                pos, last_buy = [], None
                continue

        if not has_pos and entry_fn(row):
            qty = unit_usd / price
            pos.append((d, qty, price)); last_buy = price

    if not trades: return {"n":0,"pnl":0,"wr":0,"avg":0}
    tdf = pd.DataFrame(trades)
    return {"n":len(tdf),"pnl":round(tdf["PnL_KRW"].sum()),
            "wr":round((tdf["PnL_KRW"]>0).mean()*100,1),
            "avg":round(tdf["PnL_pct"].mean(),2)}


# ── 진입 전략 정의 ─────────────────────────────────────────────────────────────
# BTC 국면 필터 포함 (항상 적용)
def load_btc_filter():
    btc_df = pd.read_parquet(BTC_PATH)
    btc_df["Date"] = pd.to_datetime(btc_df["Date"])
    btc = btc_df[btc_df["ticker"]=="BTC-USD"].sort_values("Date").copy()
    btc["btc_ma60"] = btc["Close"].rolling(60).mean()
    btc["bull"] = (btc["Close"] > btc["btc_ma60"]).astype(int)
    return btc[["Date","bull"]].set_index("Date")["bull"]

BTC_BULL = None  # 전역 BTC 필터 (main에서 로드)

def is_bull(date):
    global BTC_BULL
    return bool(BTC_BULL.get(date, 0)) if BTC_BULL is not None else True


STRATEGIES = {
    # ── BB 기반 ──
    "bb_high_pct": lambda r: r["bb_pct"] > 60 and is_bull(r["Date"]),
    "bb_high_pct_vol": lambda r: r["bb_pct"] > 60 and r["vol_ratio"] > 1.2 and is_bull(r["Date"]),
    "bb_squeeze_break": lambda r: r["bb_squeeze"] == 0 and r["bb_pct"] > 55 and is_bull(r["Date"]),
    "bb_mid_up":   lambda r: 40 < r["bb_pct"] < 80 and r["rsi14"] > 55 and is_bull(r["Date"]),

    # ── MACD 기반 ──
    "macd_cross_bull": lambda r: r["macd_cross_up"] == 1 and r["macd"] > 0 and is_bull(r["Date"]),
    "macd_positive_rsi": lambda r: r["macd_hist"] > 0 and r["rsi14"] > 55 and is_bull(r["Date"]),
    "macd_hist_grow": lambda r: r["macd_hist"] > 0 and r["pct_ma20"] > 0 and is_bull(r["Date"]),

    # ── ADX 추세 강도 ──
    "adx_trend_up": lambda r: r["adx"] > 25 and r["di_plus"] > r["di_minus"] and is_bull(r["Date"]),
    "adx_strong": lambda r: r["adx"] > 30 and r["pct_ma20"] > 0 and is_bull(r["Date"]),
    "adx_di_cross": lambda r: r["di_plus"] > r["di_minus"] and r["rsi14"] > 50 and is_bull(r["Date"]),

    # ── 거래량 기반 ──
    "vol_surge_up": lambda r: r["vol_surge"] == 1 and r["pct_ma20"] > 0 and is_bull(r["Date"]),
    "vol_surge_rsi": lambda r: r["vol_ratio"] > 1.5 and r["rsi14"] > 55 and is_bull(r["Date"]),
    "vol_high_break": lambda r: r["vol_ratio"] > 1.3 and r["breakout10"] == 1 and is_bull(r["Date"]),

    # ── 브레이크아웃 ──
    "breakout10_bull": lambda r: r["breakout10"] == 1 and is_bull(r["Date"]),
    "breakout20_bull": lambda r: r["breakout20"] == 1 and is_bull(r["Date"]),
    "breakout10_rsi": lambda r: r["breakout10"] == 1 and r["rsi14"] > 55 and is_bull(r["Date"]),

    # ── 듀얼 모멘텀 ──
    "dual_mom_rsi": lambda r: r["dual_mom_up"] == 1 and r["rsi14"] > 55 and is_bull(r["Date"]),
    "triple_ma_align": lambda r: r["ma_aligned"] == 1 and r["rsi14"] > 55 and is_bull(r["Date"]),
    "mom20_rsi": lambda r: r["mom20"] > 10 and r["rsi14"] > 50 and is_bull(r["Date"]),

    # ── 복합 다중 조건 ──
    "all_green": lambda r: (r["pct_ma20"] > 0 and r["rsi14"] > 55 and
                            r["macd_hist"] > 0 and r["adx"] > 20 and is_bull(r["Date"])),
    "bb_macd_vol": lambda r: (r["bb_pct"] > 55 and r["macd_hist"] > 0 and
                               r["vol_ratio"] > 1.1 and is_bull(r["Date"])),
    "breakout_macd_adx": lambda r: (r["breakout10"] == 1 and r["macd_hist"] > 0 and
                                     r["adx"] > 20 and is_bull(r["Date"])),
    "triple_align_vol": lambda r: (r["ma_aligned"] == 1 and r["vol_ratio"] > 1.2 and
                                    r["rsi14"] > 55 and is_bull(r["Date"])),
    # 역발상 (BTC 상관없이)
    "rsi_oversold_recover": lambda r: r["rsi14"] < 35 and r["macd_hist"] > r.get("macd_hist", 0),
    "bb_lower_bounce": lambda r: r["bb_pct"] < 20 and r["rsi14"] < 40 and r["vol_ratio"] > 1.2,
}


def main():
    global BTC_BULL
    print("[B] 새 기술 지표 조합 실험 시작")
    ohlcv = pd.read_parquet(OHLCV_PATH); ohlcv["Date"] = pd.to_datetime(ohlcv["Date"])

    BTC_BULL = load_btc_filter()

    results = []
    for t in TICKERS:
        sub = ohlcv[ohlcv["ticker"]==t].copy()
        if len(sub) < 60: continue
        sub = add_indicators(sub)

        for strat_name, entry_fn in STRATEGIES.items():
            for (tgt, stp, hld) in [(15,-20,30),(20,-20,30),(25,-20,30),
                                     (20,-15,20),(25,-15,20),(30,-20,45),
                                     (15,-10,20),(12,-10,15)]:
                try:
                    r = backtest(sub, entry_fn, tgt, stp, hld)
                    if r["n"] >= 3:
                        results.append({"ticker":t,"strategy":strat_name,
                                        "target":tgt,"stop":stp,"hold":hld,**r})
                except Exception:
                    pass

    df_res = pd.DataFrame(results).sort_values("pnl", ascending=False)
    df_res.to_csv(OUT, index=False)
    print(f"\n[B] 결과 저장: {OUT}  ({len(df_res)}개 조합)")
    print("\nTOP 15:")
    print(df_res[["ticker","strategy","target","stop","hold",
                   "pnl","wr","n","avg"]].head(15).to_string(index=False))

if __name__ == "__main__": main()

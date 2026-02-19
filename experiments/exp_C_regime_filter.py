"""
exp_C_regime_filter.py — 멀티-레짐 필터 & 변동성 조건 실험
=============================================================
VIX 대용 (ATR 기반 자체 변동성 레짐)
BTC + QQQ 듀얼 필터
MA 중첩 정렬 (10>20>60)
BTC 상승 강도 단계별 포지션 사이즈
주별 / 월별 캘린더 효과
"""
import sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

OUT = ROOT / "experiments" / "results" / "C_regime_filter.csv"
OHLCV_PATH = ROOT / "data" / "market" / "daily" / "profit_curve_ohlcv.parquet"
BTC_PATH   = ROOT / "data" / "market" / "daily" / "btc_mstr_daily.parquet"
MKT_PATH   = ROOT / "data" / "market" / "daily" / "market_daily.parquet"

TICKERS = ["MSTU", "CONL", "PTIR", "IREN", "NVDL", "AMDL", "ROBN"]


def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def prep_ticker(df):
    df = df.copy().sort_values("Date").reset_index(drop=True)
    c = df["Close"]
    df["ma10"]     = c.rolling(10).mean()
    df["ma20"]     = c.rolling(20).mean()
    df["ma60"]     = c.rolling(60).mean()
    df["rsi14"]    = rsi(c, 14)
    df["pct_ma20"] = (c - df["ma20"]) / df["ma20"] * 100
    df["atr14"]    = (df["High"]-df["Low"]).rolling(14).mean()
    df["vol_atr"]  = df["atr14"] / c * 100  # 변동성 %
    df["ma_aligned"] = ((df["ma10"] > df["ma20"]) & (df["ma20"] > df["ma60"])).astype(int)
    df["dow"] = df["Date"].dt.dayofweek   # 0=월, 4=금
    df["month"] = df["Date"].dt.month
    return df


def build_btc_regime(btc_df):
    btc = btc_df[btc_df["ticker"]=="BTC-USD"].copy().sort_values("Date")
    btc["btc_ma20"]  = btc["Close"].rolling(20).mean()
    btc["btc_ma60"]  = btc["Close"].rolling(60).mean()
    btc["btc_ma120"] = btc["Close"].rolling(120).mean()
    btc["btc_rsi"]   = rsi(btc["Close"], 14)
    btc["btc_ret5"]  = btc["Close"].pct_change(5) * 100
    btc["btc_ret20"] = btc["Close"].pct_change(20) * 100
    # 레짐 단계: 0=약세, 1=보통, 2=강세
    btc["btc_regime"] = 0
    btc.loc[btc["Close"] > btc["btc_ma60"],  "btc_regime"] = 1
    btc.loc[(btc["Close"] > btc["btc_ma20"]) & (btc["btc_rsi"] > 55), "btc_regime"] = 2
    return btc.set_index("Date")[["btc_ma20","btc_ma60","btc_ma120",
                                    "btc_rsi","btc_ret5","btc_ret20","btc_regime"]]


def build_qqq_regime(mkt_df):
    if ("QQQ","Close") not in mkt_df.columns: return None
    q = mkt_df[("QQQ","Close")].copy().to_frame("Close")
    q["qqq_ma20"] = q["Close"].rolling(20).mean()
    q["qqq_ma60"] = q["Close"].rolling(60).mean()
    q["qqq_bull"] = (q["Close"] > q["qqq_ma60"]).astype(int)
    q.index.name = "Date"
    q = q.reset_index()
    return q.set_index("Date")[["qqq_bull"]]


def run_regime_bt(df, entry_fn, target_pct, stop_pct, hold_days,
                   unit_usd_fn=None, fx=1350.0):
    """unit_usd_fn(row) → 포지션 크기 (레짐별 가변)"""
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
                trades.append({"Date":d,"PnL_KRW":pnl*fx,"PnL_pct":pp,"HeldDays":held})
                pos, last_buy = [], None
                continue

        if not has_pos and entry_fn(row):
            usd = unit_usd_fn(row) if unit_usd_fn else 740.0
            qty = usd / price
            pos.append((d, qty, price)); last_buy = price

    if not trades: return {"n":0,"pnl":0,"wr":0,"avg":0}
    tdf = pd.DataFrame(trades)
    return {"n":len(tdf),"pnl":round(tdf["PnL_KRW"].sum()),
            "wr":round((tdf["PnL_KRW"]>0).mean()*100,1),
            "avg":round(tdf["PnL_pct"].mean(),2)}


def main():
    print("[C] 멀티-레짐 필터 실험 시작")
    ohlcv = pd.read_parquet(OHLCV_PATH); ohlcv["Date"] = pd.to_datetime(ohlcv["Date"])
    btc_df = pd.read_parquet(BTC_PATH);  btc_df["Date"] = pd.to_datetime(btc_df["Date"])
    mkt_df = pd.read_parquet(MKT_PATH)

    btc_reg = build_btc_regime(btc_df)
    qqq_reg = build_qqq_regime(mkt_df)

    results = []
    for t in TICKERS:
        sub = ohlcv[ohlcv["ticker"]==t].copy()
        if len(sub) < 60: continue
        sub = prep_ticker(sub)
        sub = sub.merge(btc_reg.reset_index(), on="Date", how="left")
        for col in ["btc_ma20","btc_ma60","btc_rsi","btc_ret5","btc_ret20","btc_regime"]:
            if col in sub.columns:
                sub[col] = sub[col].fillna(method="ffill")

        if qqq_reg is not None:
            sub = sub.merge(qqq_reg.reset_index(), on="Date", how="left")
            sub["qqq_bull"] = sub["qqq_bull"].fillna(method="ffill").fillna(0)
        else:
            sub["qqq_bull"] = 1

        # 레짐별 포지션 크기 함수
        def unit_by_regime(row):
            reg = row.get("btc_regime", 1)
            if reg == 2: return 1480.0   # 강세: 2배
            if reg == 1: return 740.0    # 보통: 1배
            return 0.0                   # 약세: 진입 안 함

        # ── 레짐 전략 목록 ──
        REGIME_STRATEGIES = {
            # BTC 단계별
            "btc_regime2_only":       lambda r: r.get("btc_regime",0) == 2 and r["rsi14"] > 50,
            "btc_regime1plus":        lambda r: r.get("btc_regime",0) >= 1 and r["pct_ma20"] > 0,
            "btc_regime2_rsi":        lambda r: r.get("btc_regime",0) == 2 and r["rsi14"] > 55,
            "btc_ret5_pos_regime1":   lambda r: r.get("btc_ret5",0) > 5 and r.get("btc_regime",0) >= 1,
            "btc_ret20_strong":       lambda r: r.get("btc_ret20",0) > 15 and r["pct_ma20"] > 0,

            # BTC + QQQ 듀얼 필터
            "dual_btc_qqq":           lambda r: r.get("btc_regime",0) >= 1 and r.get("qqq_bull",0) == 1 and r["rsi14"] > 50,
            "dual_btc_qqq_strong":    lambda r: r.get("btc_regime",0) == 2 and r.get("qqq_bull",0) == 1 and r["rsi14"] > 55,
            "dual_btc_qqq_pct":       lambda r: r.get("btc_regime",0) >= 1 and r.get("qqq_bull",0) == 1 and r["pct_ma20"] > 0,

            # MA 정렬
            "triple_ma_bull":         lambda r: r["ma_aligned"] == 1 and r.get("btc_regime",0) >= 1,
            "triple_ma_strong_btc":   lambda r: r["ma_aligned"] == 1 and r.get("btc_regime",0) == 2,
            "triple_ma_qqq":          lambda r: r["ma_aligned"] == 1 and r.get("qqq_bull",0) == 1,

            # 변동성 레짐 (낮은 변동성 = 추세 강함)
            "low_vol_trend":          lambda r: r.get("vol_atr",999) < r.get("vol_atr",999) and r["pct_ma20"] > 0,
            "low_vol_bull":           lambda r: r.get("vol_atr",99) < 8 and r.get("btc_regime",0) >= 1 and r["rsi14"] > 50,
            "low_vol_breakout":       lambda r: r.get("vol_atr",99) < 8 and r["pct_ma20"] > 5 and r["rsi14"] > 55,

            # 캘린더 효과 (월, 화가 강세 경향)
            "mon_tue_bull":           lambda r: r["dow"] in [0,1] and r.get("btc_regime",0) >= 1 and r["rsi14"] > 55,
            "not_friday_bull":        lambda r: r["dow"] != 4 and r.get("btc_regime",0) >= 1 and r["pct_ma20"] > 0,
            "month_q4_bull":          lambda r: r["month"] in [10,11,12] and r.get("btc_regime",0) >= 1,
            "month_q2q3_bull":        lambda r: r["month"] in [4,5,6,7] and r.get("btc_regime",0) >= 1,

            # BTC 단기 모멘텀 + 레짐
            "btc_momentum_entry":     lambda r: r.get("btc_ret5",0) > 3 and r.get("btc_regime",0) >= 1 and r["rsi14"] > 50,
            "btc_rsi_high_regime":    lambda r: r.get("btc_rsi",0) > 60 and r["pct_ma20"] > 0,

            # 가변 포지션 (btc_regime에 따라 크기 조절)
            "variable_size_regime":   lambda r: r.get("btc_regime",0) >= 1 and r["pct_ma20"] > 0 and r["rsi14"] > 50,
        }

        for strat_name, entry_fn in REGIME_STRATEGIES.items():
            for (tgt, stp, hld) in [(15,-20,30),(20,-20,30),(25,-20,30),
                                     (20,-15,20),(25,-15,20),(30,-20,45),(15,-15,20)]:
                try:
                    # 고정 포지션
                    r = run_regime_bt(sub, entry_fn, tgt, stp, hld)
                    if r["n"] >= 3:
                        results.append({"ticker":t,"strategy":strat_name,"pos_mode":"fixed",
                                        "target":tgt,"stop":stp,"hold":hld,**r})

                    # 레짐별 가변 포지션 (variable_size_regime에서만)
                    if strat_name == "variable_size_regime":
                        r2 = run_regime_bt(sub, entry_fn, tgt, stp, hld,
                                           unit_usd_fn=unit_by_regime)
                        if r2["n"] >= 3:
                            results.append({"ticker":t,"strategy":strat_name+"_var","pos_mode":"variable",
                                            "target":tgt,"stop":stp,"hold":hld,**r2})
                except Exception:
                    pass

    df_res = pd.DataFrame(results).sort_values("pnl", ascending=False)
    df_res.to_csv(OUT, index=False)
    print(f"\n[C] 결과 저장: {OUT}  ({len(df_res)}개 조합)")
    print("\nTOP 15:")
    print(df_res[["ticker","strategy","pos_mode","target","stop","hold",
                   "pnl","wr","n","avg"]].head(15).to_string(index=False))

if __name__ == "__main__": main()

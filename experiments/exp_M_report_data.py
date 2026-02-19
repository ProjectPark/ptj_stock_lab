"""
exp_M_report_data.py — 상위 5개 전략 × 원본/보수적 파라미터 비교 데이터 수집
==============================================================================
출력: experiments/results/M_report_data.json
"""
import sys, warnings, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OHLCV_PATH = ROOT / "data" / "market" / "daily" / "profit_curve_ohlcv.parquet"
BTC_PATH   = ROOT / "data" / "market" / "daily" / "btc_mstr_daily.parquet"
EXTRA_PATH = ROOT / "data" / "market" / "daily" / "extra_signals_daily.parquet"
OUT_JSON   = ROOT / "experiments" / "results" / "M_report_data.json"

TICKERS    = ["IREN", "CONL", "PTIR"]
TRAIN_START = pd.Timestamp("2023-09-01")
TRAIN_END   = pd.Timestamp("2025-12-31")
OOS_START   = pd.Timestamp("2026-01-01")
OOS_END     = pd.Timestamp("2026-02-19")


# ─────────────────────────────────────────────────────────────────────
def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def add_features(df):
    df = df.copy().sort_values("Date").reset_index(drop=True)
    c  = df["Close"]
    df["ma5"]  = c.rolling(5).mean()
    df["ma20"] = c.rolling(20).mean()
    df["ma60"] = c.rolling(60).mean()
    df["pct_ma20"] = (c - df["ma20"]) / df["ma20"] * 100
    df["rsi7"]  = rsi(c, 7)
    df["rsi14"] = rsi(c, 14)
    df["rsi21"] = rsi(c, 21)
    df["ret1"]  = c.pct_change(1) * 100
    df["ret3"]  = c.pct_change(3) * 100
    df["ret5"]  = c.pct_change(5) * 100
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd_hist"] = (ema12-ema26) - (ema12-ema26).ewm(span=9,adjust=False).mean()
    high, low = df["High"], df["Low"]
    tr  = pd.concat([high-low,(high-c.shift(1)).abs(),(low-c.shift(1)).abs()],axis=1).max(axis=1)
    dmp = (high-high.shift(1)).clip(lower=0).where((high-high.shift(1))>(low.shift(1)-low),0)
    dmm = (low.shift(1)-low).clip(lower=0).where((low.shift(1)-low)>(high-high.shift(1)),0)
    atr14 = tr.ewm(span=14,adjust=False).mean()
    df["di_plus"]  = dmp.ewm(span=14,adjust=False).mean()/atr14*100
    df["di_minus"] = dmm.ewm(span=14,adjust=False).mean()/atr14*100
    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    up = (c > c.shift(1)).astype(int)
    df["consec_up"] = up.groupby(((up==0)|(up!=up.shift(1))).cumsum()).cumsum()*up
    return df

def build_macro(btc_df, extra_df):
    btc = btc_df[btc_df["ticker"]=="BTC-USD"].copy().sort_values("Date")
    c   = btc["Close"]
    btc["btc_ma20"]  = c.rolling(20).mean()
    btc["btc_ma60"]  = c.rolling(60).mean()
    btc["btc_rsi14"] = rsi(c, 14)
    btc["btc_regime"] = 0
    btc.loc[c > btc["btc_ma60"], "btc_regime"] = 1
    btc.loc[(c > btc["btc_ma20"]) & (btc["btc_rsi14"] > 55), "btc_regime"] = 2
    mstr = btc_df[btc_df["ticker"]=="MSTR"].copy().sort_values("Date")
    mc   = mstr["Close"]
    mstr["mstr_pct_ma20"] = (mc-mc.rolling(20).mean())/mc.rolling(20).mean()*100
    vix = extra_df[extra_df["ticker"]=="VIX"].copy().sort_values("Date")
    qqq = extra_df[extra_df["ticker"]=="QQQ"].copy().sort_values("Date")
    qc  = qqq["Close"]
    qqq["qqq_ma60"]  = qc.rolling(60).mean()
    qqq["qqq_rsi14"] = rsi(qc, 14)
    qqq["qqq_bull"]  = (qc > qqq["qqq_ma60"]).astype(int)
    macro = (
        btc.set_index("Date")[["btc_ma20","btc_ma60","btc_rsi14","btc_regime"]]
        .join(mstr.set_index("Date")[["mstr_pct_ma20"]], how="outer")
        .join(vix.set_index("Date")[["Close"]].rename(columns={"Close":"vix"}), how="outer")
        .join(qqq.set_index("Date")[["qqq_rsi14","qqq_bull"]], how="outer")
    )
    return macro.ffill()


# ─────────────────────────────────────────────────────────────────────
# 진입 함수 정의
# ─────────────────────────────────────────────────────────────────────
def make_entry(strategy, p):
    if strategy == "di_surge":
        if p.get("original"):
            def fn(r): return (r["di_plus"]>r["di_minus"] and r["ret1"]>3.0 and r.get("btc_regime",0)==2)
        else:
            def fn(r): return (
                (r["di_plus"]-r["di_minus"])>=p["di_min_gap"] and r["ret1"]>=p["surge_pct"] and
                r.get("btc_rsi14",99)<=p["btc_rsi_max"] and
                (not p.get("btc_above_ma20",True) or r.get("btc_regime",0)>=1) and
                p["vix_min"]<=r.get("vix",20)<=p["vix_max"] and
                r["pct_ma20"]>=p["pct_ma20_min"] and r["vol_ratio"]>=p["vol_ratio_min"] and
                r["rsi14"]>=p["rsi14_min"])

    elif strategy == "vix_di_surge":
        if p.get("original"):
            def fn(r): return (18<=r.get("vix",0)<=25 and r["di_plus"]>r["di_minus"] and r["ret1"]>3.0)
        else:
            def fn(r): return (
                p["vix_min"]<=r.get("vix",20)<=p["vix_max"] and
                (r["di_plus"]-r["di_minus"])>=p["di_min_gap"] and r["ret1"]>=p["surge_pct"] and
                r.get("btc_rsi14",99)<=p["btc_rsi_max"] and
                r["pct_ma20"]>=p["pct_ma20_min"] and r["vol_ratio"]>=p["vol_ratio_min"] and
                r["rsi14"]>=p["rsi14_min"])

    elif strategy == "macd_vol":
        if p.get("original"):
            def fn(r): return (r["macd_hist"]>0 and r["vol_ratio"]>1.4 and r.get("btc_rsi14",99)<70 and r.get("vix",0)>16)
        else:
            def fn(r): return (
                r["macd_hist"]>=p["macd_threshold"] and r["vol_ratio"]>=p["vol_ratio_min"] and
                r.get("btc_rsi14",99)<=p["btc_rsi_max"] and
                p["vix_min"]<=r.get("vix",20)<=p["vix_max"] and
                r["rsi14"]>=p["rsi14_min"] and r["pct_ma20"]>=p["pct_ma20_min"] and
                r["ret1"]>=p["surge_pct"])

    elif strategy == "btc_rsi_e":
        if p.get("original"):
            def fn(r): return (r.get("btc_rsi14",99)<70 and r.get("vix",0)>16 and r["di_plus"]>r["di_minus"] and r["rsi14"]>55)
        else:
            def fn(r): return (
                r.get("btc_rsi14",99)<=p["btc_rsi_max"] and
                p["vix_min"]<=r.get("vix",20)<=p["vix_max"] and
                (r["di_plus"]-r["di_minus"])>=p["di_min_gap"] and
                r["rsi14"]>=p["rsi14_min"] and r["vol_ratio"]>=p["vol_ratio_min"] and
                r["pct_ma20"]>=p["pct_ma20_min"])

    elif strategy == "strong25_macd":
        if p.get("original"):
            def fn(r): return (r["pct_ma20"]>25 and r["macd_hist"]>0 and r["di_plus"]>r["di_minus"] and r.get("btc_rsi14",99)<70)
        else:
            def fn(r): return (
                r["pct_ma20"]>=p["pct_ma20_min"] and r["macd_hist"]>=p["macd_threshold"] and
                (r["di_plus"]-r["di_minus"])>=p["di_min_gap"] and
                r.get("btc_rsi14",99)<=p["btc_rsi_max"] and
                p["vix_min"]<=r.get("vix",20)<=p["vix_max"] and
                r["vol_ratio"]>=p["vol_ratio_min"] and r["rsi14"]>=p["rsi14_min"])
    return fn


# ─────────────────────────────────────────────────────────────────────
# Backtest Engine
# ─────────────────────────────────────────────────────────────────────
def run_bt(df, entry_fn, p, fx=1350.0):
    unit_usd = 740*p["unit_mul"]
    tgt, stp = p["target_pct"], p["stop_pct"]
    hld, pyr  = p["hold_days"], p["max_pyramid"]
    pyr_add   = p["pyramid_add_pct"]
    trail     = p.get("trailing_pct", None)
    trades, pos, peak = [], [], None
    for _, row in df.iterrows():
        if pd.isna(row.get("ma20")): continue
        price, d = row["Close"], row["Date"]
        if bool(pos):
            tq = sum(x[1] for x in pos); tc = sum(x[1]*x[2] for x in pos)
            avg = tc/tq; pp = (price-avg)/avg*100; held = (d-pos[0][0]).days
            if peak is None or price>peak: peak=price
            trail_hit = trail and peak and (price-peak)/peak*100<=trail
            if pp>=tgt or pp<=stp or held>=hld or trail_hit or row["pct_ma20"]<-35:
                trades.append({"date":str(d.date()),"entry":round(avg,2),"exit":round(price,2),
                               "pnl_pct":round(pp,2),"held":held,"layers":len(pos),
                               "pnl_krw":round((tq*price-tc)*fx)})
                pos,peak=[],None; continue
            if len(pos)<pyr and price>pos[-1][2]*(1+pyr_add/100) and entry_fn(row):
                pos.append((d,unit_usd*0.7/price,price))
        elif entry_fn(row):
            pos.append((d,unit_usd/price,price)); peak=price
    if not trades: return {"n":0,"pnl":0,"wr":0,"avg":0,"trades":[]}
    tdf=pd.DataFrame(trades)
    return {"n":len(tdf),"pnl":int(tdf["pnl_krw"].sum()),
            "wr":round((tdf["pnl_krw"]>0).mean()*100,1),
            "avg":round(tdf["pnl_pct"].mean(),2),
            "trades":trades}


# ─────────────────────────────────────────────────────────────────────
# 파라미터 설정
# ─────────────────────────────────────────────────────────────────────
ORIGINAL_PARAMS = {
    "di_surge":      {"original":True,"target_pct":80,"stop_pct":-25,"hold_days":90,"unit_mul":3,"max_pyramid":3,"pyramid_add_pct":7,"trailing_pct":None},
    "vix_di_surge":  {"original":True,"target_pct":80,"stop_pct":-25,"hold_days":90,"unit_mul":3,"max_pyramid":3,"pyramid_add_pct":7,"trailing_pct":None},
    "macd_vol":      {"original":True,"target_pct":80,"stop_pct":-25,"hold_days":90,"unit_mul":3,"max_pyramid":3,"pyramid_add_pct":7,"trailing_pct":None},
    "btc_rsi_e":     {"original":True,"target_pct":80,"stop_pct":-25,"hold_days":90,"unit_mul":3,"max_pyramid":3,"pyramid_add_pct":7,"trailing_pct":None},
    "strong25_macd": {"original":True,"target_pct":80,"stop_pct":-25,"hold_days":90,"unit_mul":3,"max_pyramid":3,"pyramid_add_pct":7,"trailing_pct":None},
}

CONSERVATIVE_PARAMS = {
    "di_surge":      {"di_min_gap":3.59,"surge_pct":2.84,"btc_rsi_max":76.83,"btc_above_ma20":False,
                      "vix_min":18.02,"vix_max":38.60,"pct_ma20_min":2.78,"vol_ratio_min":0.63,"rsi14_min":10.97,
                      "target_pct":54.2,"stop_pct":-21.6,"hold_days":43,"unit_mul":2.9,"max_pyramid":2,"pyramid_add_pct":10.5,"trailing_pct":None},
    "vix_di_surge":  {"di_min_gap":10.09,"surge_pct":2.05,"btc_rsi_max":76.75,
                      "vix_min":12.36,"vix_max":38.45,"pct_ma20_min":1.59,"vol_ratio_min":2.33,"rsi14_min":54.90,
                      "target_pct":41.2,"stop_pct":-21.6,"hold_days":52,"unit_mul":2.9,"max_pyramid":3,"pyramid_add_pct":9.3,"trailing_pct":None},
    "macd_vol":      {"macd_threshold":-0.02,"surge_pct":-0.19,"btc_rsi_max":69.67,
                      "vix_min":12.14,"vix_max":32.26,"pct_ma20_min":-7.23,"vol_ratio_min":1.43,"rsi14_min":68.78,
                      "target_pct":36.3,"stop_pct":-14.0,"hold_days":51,"unit_mul":2.9,"max_pyramid":3,"pyramid_add_pct":7.7,"trailing_pct":-12.0},
    "btc_rsi_e":     {"di_min_gap":7.82,"surge_pct":5.96,"btc_rsi_max":64.62,
                      "vix_min":12.40,"vix_max":34.74,"pct_ma20_min":12.76,"vol_ratio_min":1.42,"rsi14_min":37.16,
                      "target_pct":54.4,"stop_pct":-11.0,"hold_days":50,"unit_mul":3.0,"max_pyramid":2,"pyramid_add_pct":5.7,"trailing_pct":-8.0},
    "strong25_macd": {"di_min_gap":7.04,"macd_threshold":-0.26,"btc_rsi_max":76.89,
                      "vix_min":12.09,"vix_max":36.79,"pct_ma20_min":-7.20,"vol_ratio_min":1.24,"rsi14_min":3.73,
                      "target_pct":51.9,"stop_pct":-21.0,"hold_days":35,"unit_mul":2.8,"max_pyramid":3,"pyramid_add_pct":7.2,"trailing_pct":None},
}

STRATEGY_LABELS = {
    "di_surge":      "DI크로스+급등 (E챔피언)",
    "vix_di_surge":  "VIX중립+DI급등",
    "macd_vol":      "MACD+거래량폭발",
    "btc_rsi_e":     "BTC-RSI+DI복합",
    "strong25_macd": "강추세+MACD+DI",
}


def main():
    print("데이터 로드 중...")
    ohlcv  = pd.read_parquet(OHLCV_PATH); ohlcv["Date"] = pd.to_datetime(ohlcv["Date"])
    extra  = pd.read_parquet(EXTRA_PATH); extra["Date"] = pd.to_datetime(extra["Date"])
    btc_df = pd.read_parquet(BTC_PATH);  btc_df["Date"] = pd.to_datetime(btc_df["Date"])
    macro  = build_macro(btc_df, extra)

    def prep(t, d0, d1):
        sub = ohlcv[ohlcv["ticker"]==t].copy()
        if len(sub)<30: return None
        sub = add_features(sub)
        sub = sub.merge(macro.reset_index().rename(columns={"index":"Date"}), on="Date", how="left")
        for col in macro.columns:
            if col in sub.columns: sub[col]=sub[col].ffill()
        sub["ticker"]=t
        return sub[(sub["Date"]>=d0)&(sub["Date"]<=d1)].copy()

    # 데이터 준비
    data = {}
    for t in TICKERS:
        data[(t,"train")] = prep(t, TRAIN_START, TRAIN_END)
        data[(t,"oos")]   = prep(t, OOS_START, OOS_END)
        if data[(t,"train")] is not None:
            oos_df = data[(t,"oos")]
            oos_len = 0 if oos_df is None or oos_df.empty else len(oos_df)
            print(f"  {t}: train={len(data[(t,'train')])}일  oos={oos_len}일")

    # 전체 결과 수집
    results = {}
    strategies = list(ORIGINAL_PARAMS.keys())

    for strat in strategies:
        results[strat] = {}
        for param_type, params in [("original", ORIGINAL_PARAMS[strat]),
                                    ("conservative", CONSERVATIVE_PARAMS[strat])]:
            entry_fn = make_entry(strat, params)
            results[strat][param_type] = {}

            for period, d0, d1 in [("train", TRAIN_START, TRAIN_END),
                                    ("oos",   OOS_START,   OOS_END)]:
                period_pnl, period_n, period_wins = 0, 0, 0
                ticker_results = {}

                for t in TICKERS:
                    df = data.get((t, period))
                    if df is None or len(df)==0:
                        ticker_results[t] = {"n":0,"pnl":0,"wr":0,"avg":0,"trades":[]}
                        continue
                    r = run_bt(df, entry_fn, params)
                    period_pnl  += r["pnl"]
                    period_n    += r["n"]
                    period_wins += int(r["n"]*r["wr"]/100) if r["n"]>0 else 0
                    ticker_results[t] = r

                wr_all = round(period_wins/period_n*100,1) if period_n>0 else 0
                results[strat][param_type][period] = {
                    "total_pnl": period_pnl,
                    "total_n":   period_n,
                    "total_wr":  wr_all,
                    "tickers":   ticker_results
                }

            label = STRATEGY_LABELS[strat]
            tr = results[strat][param_type]["train"]
            os = results[strat][param_type]["oos"]
            tag = "원본" if param_type=="original" else "보수적"
            print(f"  [{tag}] {label}: train={tr['total_pnl']/10000:.0f}만원 WR={tr['total_wr']}% N={tr['total_n']} | oos={os['total_pnl']/10000:.1f}만원 WR={os['total_wr']}% N={os['total_n']}")

    # JSON 저장
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n데이터 저장: {OUT_JSON}")
    return results


if __name__ == "__main__":
    main()

"""
exp_I_new_tickers.py — 새 거래 종목 백테스트
============================================
새로 fetch한 종목들을 거래 대상으로 테스트:
- TQQQ: QQQ 3x (나스닥 3배)
- BITX: BTC 2x ETF
- TSLL: TSLA 2x ETF
- SOXL: 반도체 3x (신호용 겸 거래 대상)
기존 최강 신호(E실험 기준)를 새 종목에 적용
"""
import sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

OUT       = ROOT / "experiments" / "results" / "I_new_tickers.csv"
BTC_PATH  = ROOT / "data" / "market" / "daily" / "btc_mstr_daily.parquet"
EXTRA_PATH= ROOT / "data" / "market" / "daily" / "extra_signals_daily.parquet"

# 새 거래 대상 + 기존 최강 종목 비교
TRADE_TICKERS = ["TQQQ","BITX","SOXL","TSLL","IREN","CONL"]


def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def add_features(df):
    df = df.copy().sort_values("Date").reset_index(drop=True)
    c = df["Close"]
    df["ma5"]   = c.rolling(5).mean()
    df["ma10"]  = c.rolling(10).mean()
    df["ma20"]  = c.rolling(20).mean()
    df["ma60"]  = c.rolling(60).mean()
    df["pct_ma20"] = (c - df["ma20"]) / df["ma20"] * 100
    df["rsi7"]  = rsi(c, 7)
    df["rsi14"] = rsi(c, 14)
    df["rsi21"] = rsi(c, 21)
    df["ret3"]  = c.pct_change(3) * 100
    df["ret5"]  = c.pct_change(5) * 100
    df["ret10"] = c.pct_change(10) * 100

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd_hist"] = (ema12-ema26) - (ema12-ema26).ewm(span=9,adjust=False).mean()

    # ADX / DI
    high, low = df["High"], df["Low"]
    tr = pd.concat([high-low,(high-c.shift(1)).abs(),(low-c.shift(1)).abs()],axis=1).max(axis=1)
    dmp = (high-high.shift(1)).clip(lower=0).where((high-high.shift(1))>(low.shift(1)-low),0)
    dmm = (low.shift(1)-low).clip(lower=0).where((low.shift(1)-low)>(high-high.shift(1)),0)
    atr14 = tr.ewm(span=14,adjust=False).mean()
    df["di_plus"]  = dmp.ewm(span=14,adjust=False).mean() / atr14 * 100
    df["di_minus"] = dmm.ewm(span=14,adjust=False).mean() / atr14 * 100

    # 거래량
    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # 브레이크아웃
    df["high10"] = df["High"].rolling(10).max().shift(1)
    df["bo10"]   = (c > df["high10"]).astype(int)

    # RSI 가속
    df["rsi_accel"] = ((df["rsi14"]>df["rsi21"]) & (df["rsi14"]>60)).astype(int)

    # 연속 상승
    up = (c > c.shift(1)).astype(int)
    df["consec_up"] = up.groupby(((up==0)|(up!=up.shift(1))).cumsum()).cumsum() * up

    # 급등
    df["surge1"]  = ((c/c.shift(1)-1)*100 > 3).astype(int)
    df["surge3d"] = (df["ret3"] > 8).astype(int)

    # MA 정렬
    df["ma_aligned60"] = ((df["ma10"]>df["ma20"]) & (df["ma20"]>df["ma60"])).astype(int)

    return df


def build_btc_macro(btc_df, extra_df):
    btc = btc_df[btc_df["ticker"]=="BTC-USD"].copy().sort_values("Date")
    c = btc["Close"]
    btc["btc_ma20"]  = c.rolling(20).mean()
    btc["btc_ma60"]  = c.rolling(60).mean()
    btc["btc_rsi"]   = rsi(c, 14)
    btc["btc_ret3"]  = c.pct_change(3) * 100
    btc["btc_ret5"]  = c.pct_change(5) * 100
    btc["btc_regime"] = 0
    btc.loc[c > btc["btc_ma60"], "btc_regime"] = 1
    btc.loc[(c > btc["btc_ma20"]) & (btc["btc_rsi"] > 55), "btc_regime"] = 2
    btc.loc[(c > btc["btc_ma20"]) & (btc["btc_rsi"] > 65) & (btc["btc_ret5"] > 8), "btc_regime"] = 3
    btc["btc_strong3d"] = (btc["btc_ret3"] > 8).astype(int)
    btc_sig = btc.set_index("Date")[["btc_regime","btc_ret3","btc_ret5","btc_rsi","btc_strong3d"]]

    qqq = extra_df[extra_df["ticker"]=="QQQ"].copy().sort_values("Date").set_index("Date")
    qqq["qqq_ma20"]   = qqq["Close"].rolling(20).mean()
    qqq["qqq_ma60"]   = qqq["Close"].rolling(60).mean()
    qqq["qqq_rsi"]    = rsi(qqq["Close"], 14)
    qqq["qqq_bull"]   = (qqq["Close"] > qqq["qqq_ma60"]).astype(int)
    qqq["qqq_strong"] = ((qqq["Close"] > qqq["qqq_ma20"]) & (qqq["qqq_rsi"] > 55)).astype(int)
    qqq["qqq_ret10"]  = qqq["Close"].pct_change(10) * 100
    qqq_sig = qqq[["qqq_bull","qqq_strong","qqq_ret10"]]

    vix = extra_df[extra_df["ticker"]=="VIX"].copy().sort_values("Date").set_index("Date")
    vix["vix_low"]     = (vix["Close"] < 18).astype(int)
    vix["vix_mid"]     = (vix["Close"] < 25).astype(int)
    vix["vix_falling"] = (vix["Close"] < vix["Close"].shift(3)).astype(int)
    vix_sig = vix[["vix_low","vix_mid","vix_falling"]]

    mstr = btc_df[btc_df["ticker"]=="MSTR"].copy().sort_values("Date")
    mc = mstr["Close"]
    mstr["mstr_pct_ma20"] = (mc - mc.rolling(20).mean()) / mc.rolling(20).mean() * 100
    mstr["mstr_rsi"]      = rsi(mc, 14)
    mstr_sig = mstr.set_index("Date")[["mstr_pct_ma20","mstr_rsi"]]

    macro = btc_sig.join(qqq_sig, how="outer") \
                   .join(vix_sig, how="outer") \
                   .join(mstr_sig, how="outer")
    return macro


def backtest_with_pyramid(df, entry_fn, target_pct, stop_pct, hold_days,
                          unit_usd=1480.0, max_pyramid=3, pyramid_add_pct=7.0,
                          fx=1350.0):
    trades, pos = [], []
    peak = None
    for _, row in df.iterrows():
        if pd.isna(row.get("ma20")): continue
        price, d = row["Close"], row["Date"]
        pma20 = row["pct_ma20"]
        has_pos = bool(pos)

        if has_pos:
            tq = sum(p[1] for p in pos); tc = sum(p[1]*p[2] for p in pos)
            avg = tc/tq; pp = (price-avg)/avg*100
            held = (d-pos[0][0]).days
            if pp >= target_pct or pp <= stop_pct or held >= hold_days or pma20 < -35:
                trades.append({"Date":d,"PnL_KRW":(tq*price-tc)*fx,"PnL_pct":pp,"HeldDays":held,"Layers":len(pos)})
                pos, peak = [], None; continue
            if len(pos) < max_pyramid:
                if price > pos[-1][2]*(1+pyramid_add_pct/100) and entry_fn(row):
                    pos.append((d, unit_usd*0.7/price, price))

        if not has_pos and entry_fn(row):
            pos.append((d, unit_usd/price, price))
            peak = price

    if not trades: return {"n":0,"pnl":0,"wr":0,"avg":0,"avg_layers":0}
    tdf = pd.DataFrame(trades)
    return {"n":len(tdf),"pnl":round(tdf["PnL_KRW"].sum()),
            "wr":round((tdf["PnL_KRW"]>0).mean()*100,1),
            "avg":round(tdf["PnL_pct"].mean(),2),
            "avg_layers":round(tdf["Layers"].mean(),2)}


def main():
    print("[I] 새 거래 종목 백테스트 시작")
    btc_df = pd.read_parquet(BTC_PATH);  btc_df["Date"] = pd.to_datetime(btc_df["Date"])
    extra  = pd.read_parquet(EXTRA_PATH); extra["Date"]  = pd.to_datetime(extra["Date"])

    macro = build_btc_macro(btc_df, extra)

    # IREN/CONL은 profit_curve_ohlcv에서, 나머지는 extra_signals에서
    from pathlib import Path
    ohlcv_old = pd.read_parquet(ROOT / "data/market/daily/profit_curve_ohlcv.parquet")
    ohlcv_old["Date"] = pd.to_datetime(ohlcv_old["Date"])

    results = []

    for t in TRADE_TICKERS:
        if t in ohlcv_old["ticker"].unique():
            sub = ohlcv_old[ohlcv_old["ticker"]==t].copy()
        else:
            sub = extra[extra["ticker"]==t].copy()

        if len(sub) < 60: continue
        sub = add_features(sub)
        sub = sub.merge(macro.reset_index().rename(columns={"index":"Date"}),
                        on="Date", how="left")
        for col in macro.columns:
            if col in sub.columns:
                sub[col] = sub[col].ffill()

        print(f"  {t:6s}: {sub['Date'].min().date()} ~ {sub['Date'].max().date()}  ({len(sub)}일)")

        # E실험 최강 전략 + H실험 새 신호 조합
        STRATEGIES = {
            # E실험 챔피언 그대로 적용
            "di_cross_surge_btc2":  lambda r: (r["di_plus"]>r["di_minus"] and r["surge1"]==1 and r.get("btc_regime",0)==2),
            "bb_macd_adx_btc2":     lambda r: (r.get("rsi14",0)>55 and r["macd_hist"]>0 and r["di_plus"]>r["di_minus"] and r.get("btc_regime",0)==2),
            "rsi7_80_btc2":         lambda r: (r["rsi7"]>75 and r.get("btc_regime",0)>=2 and r["pct_ma20"]>5),
            "consec3_surge":        lambda r: (r.get("consec_up",0)>=3 and r["surge3d"]==1),
            "rsi_accel_btc2_vol":   lambda r: (r["rsi_accel"]==1 and r.get("btc_regime",0)==2 and r["vol_ratio"]>1.2),
            "consec4_btc1":         lambda r: (r.get("consec_up",0)>=4 and r.get("btc_regime",0)>=1),

            # H실험 새 신호 (QQQ + VIX)
            "triple_regime_all":    lambda r: (r.get("btc_regime",0)==2 and r.get("qqq_bull",0)==1 and r.get("vix_mid",0)==1 and r["rsi14"]>55),
            "triple_regime_strong": lambda r: (r.get("btc_regime",0)==2 and r.get("qqq_strong",0)==1 and r.get("vix_low",0)==1),
            "vix_falling_btc2":     lambda r: (r.get("vix_falling",0)==1 and r.get("btc_regime",0)==2),
            "btc2_qqq_strong_vol":  lambda r: (r.get("btc_regime",0)==2 and r.get("qqq_strong",0)==1 and r["vol_ratio"]>1.3),
            "qqq_strong_macd":      lambda r: (r.get("qqq_strong",0)==1 and r["macd_hist"]>0 and r.get("btc_regime",0)>=1),

            # F실험 구조 패턴
            "hh_hl_rsi_btc":        lambda r: (r.get("btc_regime",0)>=1 and r["rsi14"]>55 and r["pct_ma20"]>0 and r["ret5"]>0),
            "ma_aligned_btc2_vol":  lambda r: (r["ma_aligned60"]==1 and r.get("btc_regime",0)==2 and r["vol_ratio"]>1.2),
            "bo10_btc2_qqq":        lambda r: (r["bo10"]==1 and r.get("btc_regime",0)==2 and r.get("qqq_bull",0)==1),
            "btc3_strong_rsi_accel":lambda r: (r.get("btc_regime",0)==3 and r["rsi_accel"]==1),
            "mstr5_btc2_qqq":       lambda r: (r.get("mstr_pct_ma20",0)>5 and r.get("btc_regime",0)==2 and r.get("qqq_bull",0)==1),
        }

        PARAMS = [
            (25,-20,30, 1.0, 1),(30,-20,30, 1.0, 1),(40,-20,30, 1.0, 1),
            (50,-25,45, 1.0, 1),(60,-25,60, 1.0, 1),(80,-25,90, 1.0, 1),
            (30,-20,30, 2.0, 1),(40,-20,30, 2.0, 1),(50,-25,45, 2.0, 1),
            (40,-20,30, 3.0, 1),(50,-25,45, 3.0, 1),(60,-25,60, 3.0, 1),
            (40,-20,30, 2.0, 3),(50,-25,45, 2.0, 3),(80,-25,90, 2.0, 3),
            (50,-25,45, 3.0, 3),(80,-25,90, 3.0, 3),
        ]

        for strat_name, entry_fn in STRATEGIES.items():
            for (tgt, stp, hld, mul, max_pyr) in PARAMS:
                try:
                    r = backtest_with_pyramid(sub, entry_fn, tgt, stp, hld,
                                              unit_usd=740*mul, max_pyramid=max_pyr)
                    if r["n"] >= 3:
                        results.append({"ticker":t,"strategy":strat_name,
                                        "target":tgt,"stop":stp,"hold":hld,
                                        "unit_mul":mul,"max_pyr":max_pyr, **r})
                except Exception:
                    pass

    df_res = pd.DataFrame(results).sort_values("pnl", ascending=False)
    df_res.to_csv(OUT, index=False)
    print(f"\n[I] 결과 저장: {OUT}  ({len(df_res)}개 조합)")
    print("\nTOP 20:")
    print(df_res[["ticker","strategy","target","stop","hold","unit_mul","max_pyr",
                   "pnl","wr","n","avg"]].head(20).to_string(index=False))


ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    main()

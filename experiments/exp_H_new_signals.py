"""
exp_H_new_signals.py — 새 데이터 활용 실험
==========================================
새로 fetch한 데이터 사용:
- QQQ: 나스닥 레짐 (실제 데이터, C실험은 더미였음)
- VIX: 공포지수 레짐 (<20 탐욕, >30 공포)
- COIN: CONL 기초자산 직접 신호
- SOXL: 반도체 레버리지 (NVDL/AMDL 선행지표)
- BTC + QQQ + VIX 삼중 레짐 필터
"""
import sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

OUT       = ROOT / "experiments" / "results" / "H_new_signals.csv"
OHLCV_PATH  = ROOT / "data" / "market" / "daily" / "profit_curve_ohlcv.parquet"
BTC_PATH    = ROOT / "data" / "market" / "daily" / "btc_mstr_daily.parquet"
EXTRA_PATH  = ROOT / "data" / "market" / "daily" / "extra_signals_daily.parquet"

TICKERS = ["IREN", "PTIR", "CONL", "NVDL", "MSTU", "AMDL"]


def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def build_macro_signals(btc_df, extra_df):
    """QQQ + VIX + COIN + SOXL + BTC 통합 매크로 신호"""
    # BTC
    btc = btc_df[btc_df["ticker"]=="BTC-USD"].copy().sort_values("Date")
    c = btc["Close"]
    btc["btc_ma20"]  = c.rolling(20).mean()
    btc["btc_ma60"]  = c.rolling(60).mean()
    btc["btc_rsi"]   = rsi(c, 14)
    btc["btc_ret5"]  = c.pct_change(5) * 100
    btc["btc_ret10"] = c.pct_change(10) * 100
    btc["btc_regime"] = 0
    btc.loc[c > btc["btc_ma60"], "btc_regime"] = 1
    btc.loc[(c > btc["btc_ma20"]) & (btc["btc_rsi"] > 55), "btc_regime"] = 2
    btc_sig = btc.set_index("Date")[["btc_ma20","btc_ma60","btc_rsi","btc_ret5","btc_ret10","btc_regime"]]

    # QQQ
    qqq = extra_df[extra_df["ticker"]=="QQQ"].copy().sort_values("Date").set_index("Date")
    qqq["qqq_ma20"] = qqq["Close"].rolling(20).mean()
    qqq["qqq_ma60"] = qqq["Close"].rolling(60).mean()
    qqq["qqq_rsi"]  = rsi(qqq["Close"], 14)
    qqq["qqq_ret10"]= qqq["Close"].pct_change(10) * 100
    qqq["qqq_bull"] = (qqq["Close"] > qqq["qqq_ma60"]).astype(int)
    qqq["qqq_strong"] = ((qqq["Close"] > qqq["qqq_ma20"]) & (qqq["qqq_rsi"] > 55)).astype(int)
    qqq_sig = qqq[["qqq_ma20","qqq_ma60","qqq_rsi","qqq_ret10","qqq_bull","qqq_strong"]]

    # VIX (낮을수록 탐욕 = 강세 신호)
    vix = extra_df[extra_df["ticker"]=="VIX"].copy().sort_values("Date").set_index("Date")
    vix["vix_ma20"]   = vix["Close"].rolling(20).mean()
    vix["vix_low"]    = (vix["Close"] < 18).astype(int)    # 초탐욕 (<18)
    vix["vix_mid"]    = (vix["Close"] < 25).astype(int)    # 탐욕 (<25)
    vix["vix_high"]   = (vix["Close"] > 30).astype(int)    # 공포 (>30)
    vix["vix_spike"]  = (vix["Close"] > vix["vix_ma20"] * 1.3).astype(int)  # VIX 급등
    vix["vix_falling"]= (vix["Close"] < vix["Close"].shift(3)).astype(int)  # VIX 하락중
    vix_sig = vix[["Close","vix_low","vix_mid","vix_high","vix_spike","vix_falling"]].rename(
        columns={"Close":"vix"})

    # COIN (CONL 기초자산)
    coin = extra_df[extra_df["ticker"]=="COIN"].copy().sort_values("Date").set_index("Date")
    coin["coin_ma20"]  = coin["Close"].rolling(20).mean()
    coin["coin_rsi"]   = rsi(coin["Close"], 14)
    coin["coin_ret5"]  = coin["Close"].pct_change(5) * 100
    coin["coin_pct_ma20"] = (coin["Close"] - coin["coin_ma20"]) / coin["coin_ma20"] * 100
    coin["coin_bull"]  = (coin["Close"] > coin["coin_ma20"]).astype(int)
    coin_sig = coin[["coin_ma20","coin_rsi","coin_ret5","coin_pct_ma20","coin_bull"]]

    # SOXL (반도체 3x, NVDL/AMDL 선행)
    soxl = extra_df[extra_df["ticker"]=="SOXL"].copy().sort_values("Date").set_index("Date")
    soxl["soxl_ma20"]  = soxl["Close"].rolling(20).mean()
    soxl["soxl_rsi"]   = rsi(soxl["Close"], 14)
    soxl["soxl_ret5"]  = soxl["Close"].pct_change(5) * 100
    soxl["soxl_bull"]  = (soxl["Close"] > soxl["soxl_ma20"]).astype(int)
    soxl_sig = soxl[["soxl_ma20","soxl_rsi","soxl_ret5","soxl_bull"]]

    # 통합
    macro = btc_sig.join(qqq_sig, how="outer") \
                   .join(vix_sig, how="outer") \
                   .join(coin_sig, how="outer") \
                   .join(soxl_sig, how="outer")
    return macro


def add_ticker_features(df):
    df = df.copy().sort_values("Date").reset_index(drop=True)
    c = df["Close"]
    df["ma10"]     = c.rolling(10).mean()
    df["ma20"]     = c.rolling(20).mean()
    df["ma60"]     = c.rolling(60).mean()
    df["pct_ma20"] = (c - df["ma20"]) / df["ma20"] * 100
    df["rsi14"]    = rsi(c, 14)
    df["rsi7"]     = rsi(c, 7)
    df["ret3"]     = c.pct_change(3) * 100
    df["ret5"]     = c.pct_change(5) * 100
    df["vol_ratio"]= df["Volume"] / df["Volume"].rolling(20).mean()

    # 브레이크아웃
    df["high10"]   = df["High"].rolling(10).max().shift(1)
    df["bo10"]     = (c > df["high10"]).astype(int)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd_hist"] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()

    # 급등
    df["surge1"] = ((df["ret3"]/3 > 2) & (c > c.shift(1))).astype(int)

    # 연속 상승
    up = (c > c.shift(1)).astype(int)
    df["consec_up"] = up.groupby(((up==0)|(up!=up.shift(1))).cumsum()).cumsum() * up

    return df


def backtest(df, entry_fn, target_pct, stop_pct, hold_days,
             unit_usd=740.0, fx=1350.0):
    trades, pos = [], []
    for _, row in df.iterrows():
        if pd.isna(row.get("ma20")): continue
        price, d = row["Close"], row["Date"]
        pma20 = row["pct_ma20"]
        has_pos = bool(pos)

        if has_pos:
            tq = sum(p[1] for p in pos); tc = sum(p[1]*p[2] for p in pos)
            avg = tc/tq; pp = (price-avg)/avg*100
            held = (d - pos[0][0]).days
            if pp >= target_pct or pp <= stop_pct or held >= hold_days or pma20 < -30:
                trades.append({"Date":d,"PnL_KRW":(tq*price-tc)*fx,"PnL_pct":pp,"HeldDays":held})
                pos = []; continue

        if not has_pos and entry_fn(row):
            pos.append((d, unit_usd/price, price))

    if not trades: return {"n":0,"pnl":0,"wr":0,"avg":0}
    tdf = pd.DataFrame(trades)
    return {"n":len(tdf),"pnl":round(tdf["PnL_KRW"].sum()),
            "wr":round((tdf["PnL_KRW"]>0).mean()*100,1),
            "avg":round(tdf["PnL_pct"].mean(),2)}


def main():
    print("[H] 새 데이터 활용 실험 시작")
    ohlcv   = pd.read_parquet(OHLCV_PATH);  ohlcv["Date"]  = pd.to_datetime(ohlcv["Date"])
    btc_df  = pd.read_parquet(BTC_PATH);    btc_df["Date"] = pd.to_datetime(btc_df["Date"])
    extra   = pd.read_parquet(EXTRA_PATH);  extra["Date"]  = pd.to_datetime(extra["Date"])

    macro = build_macro_signals(btc_df, extra)
    print(f"  매크로 신호 컬럼: {list(macro.columns)}")

    results = []

    for t in TICKERS:
        sub = ohlcv[ohlcv["ticker"]==t].copy()
        if len(sub) < 60: continue
        sub = add_ticker_features(sub)
        sub = sub.merge(macro.reset_index().rename(columns={"index":"Date"}),
                        on="Date", how="left")

        # ffill
        macro_cols = [c for c in macro.columns if c in sub.columns]
        for col in macro_cols:
            sub[col] = sub[col].ffill()

        # 종목별 섹터 구분
        is_btc  = t in ["IREN","PTIR","CONL","MSTU"]
        is_semi = t in ["NVDL","AMDL"]

        STRATEGIES = {
            # ── BTC + QQQ 실제 듀얼 필터 (C실험 QQQ 더미 개선) ──
            "btc2_qqq_strong":     lambda r: r.get("btc_regime",0)==2 and r.get("qqq_strong",0)==1 and r["rsi14"]>55,
            "btc1_qqq_bull_rsi":   lambda r: r.get("btc_regime",0)>=1 and r.get("qqq_bull",0)==1 and r["rsi14"]>55,
            "btc2_qqq_bull_bo":    lambda r: r.get("btc_regime",0)==2 and r.get("qqq_bull",0)==1 and r["bo10"]==1,
            "btc2_qqq_strong_vol": lambda r: r.get("btc_regime",0)==2 and r.get("qqq_strong",0)==1 and r["vol_ratio"]>1.3,

            # ── VIX 공포지수 레짐 ──
            "vix_low_btc2":        lambda r: r.get("vix_low",0)==1 and r.get("btc_regime",0)==2 and r["rsi14"]>55,
            "vix_mid_btc1_rsi":    lambda r: r.get("vix_mid",0)==1 and r.get("btc_regime",0)>=1 and r["rsi14"]>60,
            "vix_falling_btc2":    lambda r: r.get("vix_falling",0)==1 and r.get("btc_regime",0)==2,
            "vix_fall_qqq_bull":   lambda r: r.get("vix_falling",0)==1 and r.get("qqq_bull",0)==1 and r.get("btc_regime",0)>=1,
            "vix_low_qqq_surge":   lambda r: r.get("vix_low",0)==1 and r.get("qqq_ret10",0)>5 and r["pct_ma20"]>0,

            # ── COIN 직접 신호 → CONL/IREN ──
            "coin_bull_btc2":      lambda r: r.get("coin_bull",0)==1 and r.get("btc_regime",0)==2 and r["rsi14"]>55,
            "coin_ret5_btc2":      lambda r: r.get("coin_ret5",0)>5 and r.get("btc_regime",0)==2,
            "coin_rsi65_btc2":     lambda r: r.get("coin_rsi",0)>65 and r.get("btc_regime",0)==2 and r["pct_ma20"]>0,
            "coin_pct10_qqq":      lambda r: r.get("coin_pct_ma20",0)>10 and r.get("qqq_bull",0)==1,

            # ── SOXL 선행 신호 → NVDL/AMDL ──
            "soxl_bull_qqq":       lambda r: r.get("soxl_bull",0)==1 and r.get("qqq_bull",0)==1 and r["rsi14"]>55,
            "soxl_ret5_pos":       lambda r: r.get("soxl_ret5",0)>5 and r["pct_ma20"]>0,
            "soxl_rsi60_qqq":      lambda r: r.get("soxl_rsi",0)>60 and r.get("qqq_strong",0)==1,

            # ── 삼중 레짐 필터 (BTC + QQQ + VIX) ──
            "triple_regime_all":   lambda r: (r.get("btc_regime",0)==2
                                               and r.get("qqq_bull",0)==1
                                               and r.get("vix_mid",0)==1
                                               and r["rsi14"]>55),
            "triple_regime_strong":lambda r: (r.get("btc_regime",0)==2
                                               and r.get("qqq_strong",0)==1
                                               and r.get("vix_low",0)==1),
            "triple_surge":        lambda r: (r.get("btc_regime",0)==2
                                               and r.get("qqq_bull",0)==1
                                               and r.get("vix_falling",0)==1
                                               and r["surge1"]==1),

            # ── QQQ 모멘텀 리드 ──
            "qqq_ret10_btc2":      lambda r: r.get("qqq_ret10",0)>5 and r.get("btc_regime",0)==2 and r["rsi14"]>55,
            "qqq_strong_consec":   lambda r: r.get("qqq_strong",0)==1 and r.get("consec_up",0)>=2 and r.get("btc_regime",0)>=1,
            "qqq_strong_macd":     lambda r: r.get("qqq_strong",0)==1 and r["macd_hist"]>0 and r.get("btc_regime",0)>=1,

            # ── VIX 스파이크 후 역발상 (VIX 고점 = 매수 기회) ──
            "vix_spike_recover":   lambda r: r.get("vix_spike",0)==0 and r.get("btc_regime",0)>=1
                                               and r.get("vix_high",0)==0 and r["rsi14"]>50,
            "vix_spike_next_btc":  lambda r: r.get("vix_spike",0)==1 and r.get("btc_regime",0)>=1,  # 스파이크 당일 진입

            # ── BTC + QQQ + COIN 트리플 크로스에셋 ──
            "btc_qqq_coin_all":    lambda r: (r.get("btc_regime",0)==2
                                               and r.get("qqq_bull",0)==1
                                               and r.get("coin_bull",0)==1
                                               and r["rsi14"]>55),
            "btc_qqq_coin_surge":  lambda r: (r.get("btc_ret5",0)>5
                                               and r.get("qqq_ret10",0)>3
                                               and r.get("coin_ret5",0)>5
                                               and r["pct_ma20"]>0),
        }

        PARAMS = [
            (15,-15,20),(20,-15,20),(25,-15,20),
            (20,-20,30),(25,-20,30),(30,-20,30),(40,-20,30),
            (30,-20,45),(40,-20,45),(50,-25,45),
            (50,-25,60),(60,-25,60),(80,-25,90),
        ]

        for strat_name, entry_fn in STRATEGIES.items():
            for (tgt, stp, hld) in PARAMS:
                for mul in [1.0, 2.0, 3.0]:
                    try:
                        r = backtest(sub, entry_fn, tgt, stp, hld, unit_usd=740*mul)
                        if r["n"] >= 3:
                            results.append({"ticker":t,"strategy":strat_name,
                                            "target":tgt,"stop":stp,"hold":hld,
                                            "unit_mul":mul, **r})
                    except Exception:
                        pass

    df_res = pd.DataFrame(results).sort_values("pnl", ascending=False)
    df_res.to_csv(OUT, index=False)
    print(f"\n[H] 결과 저장: {OUT}  ({len(df_res)}개 조합)")
    print("\nTOP 20:")
    print(df_res[["ticker","strategy","target","stop","hold","unit_mul",
                   "pnl","wr","n","avg"]].head(20).to_string(index=False))

if __name__ == "__main__":
    main()

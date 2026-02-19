"""
exp_A_cross_asset.py — 크로스-에셋 상관관계 신호 실험
=======================================================
MSTR 가격 → MSTU 진입 신호
COIN 가격  → CONL 진입 신호  (COIN은 market_daily에 없으므로 BTC 대용)
BTC N일 수익률/모멘텀 → 모든 종목 리드 신호
"""
import sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import itertools
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

OUT = ROOT / "experiments" / "results" / "A_cross_asset.csv"
OHLCV_PATH  = ROOT / "data" / "market" / "daily" / "profit_curve_ohlcv.parquet"
BTC_PATH    = ROOT / "data" / "market" / "daily" / "btc_mstr_daily.parquet"
FETCH_START = "2023-09-01"

TICKERS = ["MSTU", "CONL", "PTIR", "IREN", "NVDL", "AMDL", "ROBN"]


def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def add_base(df):
    df = df.copy().sort_values("Date").reset_index(drop=True)
    df["ma20"]    = df["Close"].rolling(20).mean()
    df["rsi14"]   = rsi(df["Close"], 14)
    df["ret1"]    = df["Close"].pct_change(1) * 100
    df["ret3"]    = df["Close"].pct_change(3) * 100
    df["ret5"]    = df["Close"].pct_change(5) * 100
    df["pct_ma20"] = (df["Close"] - df["ma20"]) / df["ma20"] * 100
    return df


def run(ticker_df, signal_col, signal_thresh, signal_dir,
        target_pct, stop_pct, hold_days, unit_usd=740.0, fx=1350.0):
    """
    signal_col  : 진입 신호 컬럼명 (cross-asset에서 가져온 값)
    signal_dir  : "above" = 신호 > 임계값, "below" = 신호 < 임계값
    """
    df = ticker_df.copy()
    trades, pos = [], []
    last_buy = None

    for _, row in df.iterrows():
        if pd.isna(row.get("ma20")): continue
        price, d = row["Close"], row["Date"]
        sig = row.get(signal_col, np.nan)
        if pd.isna(sig): continue
        pma20 = row["pct_ma20"]
        has_pos = bool(pos)

        # 청산
        if has_pos:
            tq = sum(p[1] for p in pos); tc = sum(p[1]*p[2] for p in pos)
            avg = tc/tq; pp = (price-avg)/avg*100
            held = (d - pos[0][0]).days
            if pp >= target_pct or pp <= stop_pct or held >= hold_days or pma20 < -25:
                pnl_usd = tq*price - tc
                trades.append({"Date":d,"PnL_KRW":pnl_usd*fx,"PnL_pct":pp,
                                "HeldDays":held,"Reason":"T" if pp>=target_pct else "S" if pp<=stop_pct else "X"})
                pos, last_buy = [], None
                continue

        # 진입
        if not has_pos:
            cond_sig = (sig > signal_thresh) if signal_dir == "above" else (sig < signal_thresh)
            if cond_sig:
                qty = unit_usd / price
                pos.append((d, qty, price))
                last_buy = price

    if not trades: return {"n":0,"pnl":0,"wr":0,"avg":0}
    tdf = pd.DataFrame(trades)
    return {"n":len(tdf),"pnl":round(tdf["PnL_KRW"].sum()),
            "wr":round((tdf["PnL_KRW"]>0).mean()*100,1),
            "avg":round(tdf["PnL_pct"].mean(),2)}


def main():
    print("[A] 크로스-에셋 신호 실험 시작")
    ohlcv = pd.read_parquet(OHLCV_PATH); ohlcv["Date"] = pd.to_datetime(ohlcv["Date"])
    btc_df = pd.read_parquet(BTC_PATH);  btc_df["Date"] = pd.to_datetime(btc_df["Date"])

    # BTC / MSTR 지표 계산
    btc  = add_base(btc_df[btc_df["ticker"]=="BTC-USD"].copy())
    mstr = add_base(btc_df[btc_df["ticker"]=="MSTR"].copy())

    # 크로스-에셋 신호 컬럼 생성
    btc_sig = btc[["Date","ret1","ret3","ret5","rsi14","pct_ma20"]].rename(
        columns={"ret1":"btc_ret1","ret3":"btc_ret3","ret5":"btc_ret5",
                 "rsi14":"btc_rsi","pct_ma20":"btc_pct_ma20"})
    mstr_sig = mstr[["Date","ret1","ret3","ret5","rsi14","pct_ma20"]].rename(
        columns={"ret1":"mstr_ret1","ret3":"mstr_ret3","ret5":"mstr_ret5",
                 "rsi14":"mstr_rsi","pct_ma20":"mstr_pct_ma20"})

    results = []
    for t in TICKERS:
        sub = ohlcv[ohlcv["ticker"]==t].copy()
        if len(sub) < 40: continue
        sub = add_base(sub)
        # 크로스 신호 병합
        sub = sub.merge(btc_sig, on="Date", how="left")
        sub = sub.merge(mstr_sig, on="Date", how="left")
        sub["btc_rsi"]    = sub["btc_rsi"].fillna(method="ffill")
        sub["mstr_rsi"]   = sub["mstr_rsi"].fillna(method="ffill")
        sub["btc_ret1"]   = sub["btc_ret1"].fillna(method="ffill")
        sub["mstr_ret1"]  = sub["mstr_ret1"].fillna(method="ffill")
        sub["btc_pct_ma20"]  = sub["btc_pct_ma20"].fillna(method="ffill")
        sub["mstr_pct_ma20"] = sub["mstr_pct_ma20"].fillna(method="ffill")

        # 크로스-에셋 조합 신호 추가
        sub["btc_mstr_combo"] = (sub["btc_rsi"] + sub.get("mstr_rsi", sub["btc_rsi"])) / 2
        sub["btc_momentum3"]  = sub["btc_ret3"].fillna(0)

        # 파라미터 그리드
        signal_configs = [
            # (signal_col, thresh, dir) — BTC/MSTR 지표를 진입 신호로
            ("btc_rsi",       50, "above"),
            ("btc_rsi",       60, "above"),
            ("btc_rsi",       40, "above"),
            ("mstr_rsi",      55, "above"),
            ("mstr_rsi",      60, "above"),
            ("btc_pct_ma20",   0, "above"),
            ("btc_pct_ma20",  -5, "above"),
            ("btc_pct_ma20",   5, "above"),
            ("mstr_pct_ma20",  0, "above"),
            ("mstr_pct_ma20",  5, "above"),
            ("btc_ret3",       3, "above"),
            ("btc_ret3",       5, "above"),
            ("btc_momentum3",  0, "above"),
            ("btc_mstr_combo",55, "above"),
            ("btc_mstr_combo",60, "above"),
            # 역발상: 낙폭 후 진입
            ("btc_rsi",       30, "below"),
            ("btc_rsi",       35, "below"),
            ("mstr_rsi",      35, "below"),
            ("btc_pct_ma20", -10, "below"),
        ]
        for sig_col, thresh, direction in signal_configs:
            for (tgt, stp, hld) in [(15,-20,30),(20,-20,30),(25,-20,30),
                                     (15,-15,20),(20,-15,20),(25,-15,20),
                                     (30,-20,45),(12,-10,20)]:
                r = run(sub, sig_col, thresh, direction, tgt, stp, hld)
                if r["n"] >= 3:
                    results.append({
                        "ticker":t, "signal":sig_col, "thresh":thresh,
                        "direction":direction, "target":tgt, "stop":stp, "hold":hld,
                        **r
                    })

    df_res = pd.DataFrame(results).sort_values("pnl", ascending=False)
    df_res.to_csv(OUT, index=False)

    print(f"\n[A] 결과 저장: {OUT}  ({len(df_res)}개 조합)")
    print("\nTOP 15:")
    print(df_res[["ticker","signal","thresh","direction","target","stop","hold",
                   "pnl","wr","n","avg"]].head(15).to_string(index=False))

if __name__ == "__main__": main()

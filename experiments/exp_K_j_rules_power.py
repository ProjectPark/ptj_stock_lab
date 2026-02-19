"""
exp_K_j_rules_power.py — J 실험 발견 기반 강력한 백테스트
===========================================================
J 실험에서 추출한 실제 수익 공통 패턴:

✅ BTC RSI < 61 (BTC 과열 아님)
✅ VIX 18~25 중립 구간
✅ 종목이 MA20 대비 +15% 이상 (추세 확인)
✅ 종목 5일 수익률 < 0 (눌림목 — 추세 내 되돌림)

핵심 반전:
- BTC 레짐보다 BTC RSI가 더 중요
- VIX 극탐욕(<16)은 오히려 위험
- MSTR이 MA20 아래일 때 CONL 진입 유리
- 종목이 이미 많이 오른 상태(+15%)에서 일시 눌림목이 최적 진입
"""
import sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

OUT = ROOT / "experiments" / "results" / "K_j_rules_power.csv"
OHLCV_PATH = ROOT / "data" / "market" / "daily" / "profit_curve_ohlcv.parquet"
BTC_PATH   = ROOT / "data" / "market" / "daily" / "btc_mstr_daily.parquet"
EXTRA_PATH = ROOT / "data" / "market" / "daily" / "extra_signals_daily.parquet"

# 실제 거래에서 검증된 종목 + 신규 추가
TICKERS = ["MSTU", "CONL", "IREN", "PTIR", "MSTX", "NVDL", "AMDL", "ROBN"]


def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def add_features(df):
    df = df.copy().sort_values("Date").reset_index(drop=True)
    c = df["Close"]

    df["ma5"]  = c.rolling(5).mean()
    df["ma10"] = c.rolling(10).mean()
    df["ma20"] = c.rolling(20).mean()
    df["ma60"] = c.rolling(60).mean()
    df["pct_ma20"] = (c - df["ma20"]) / df["ma20"] * 100
    df["pct_ma5"]  = (c - df["ma5"])  / df["ma5"]  * 100

    df["rsi7"]  = rsi(c, 7)
    df["rsi14"] = rsi(c, 14)
    df["rsi21"] = rsi(c, 21)

    df["ret1"]  = c.pct_change(1)  * 100
    df["ret3"]  = c.pct_change(3)  * 100
    df["ret5"]  = c.pct_change(5)  * 100
    df["ret10"] = c.pct_change(10) * 100

    # Bollinger Band
    std20 = c.rolling(20).std()
    df["bb_upper"] = df["ma20"] + 2*std20
    df["bb_lower"] = df["ma20"] - 2*std20
    df["bb_pct"]   = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]) * 100

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]      = ema12 - ema26
    df["macd_sig"]  = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]

    # ADX / DI
    high, low = df["High"], df["Low"]
    tr = pd.concat([high-low,(high-c.shift(1)).abs(),(low-c.shift(1)).abs()],axis=1).max(axis=1)
    dmp = (high-high.shift(1)).clip(lower=0).where((high-high.shift(1))>(low.shift(1)-low), 0)
    dmm = (low.shift(1)-low).clip(lower=0).where((low.shift(1)-low)>(high-high.shift(1)), 0)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    df["di_plus"]  = dmp.ewm(span=14, adjust=False).mean() / atr14 * 100
    df["di_minus"] = dmm.ewm(span=14, adjust=False).mean() / atr14 * 100
    df["adx"]      = ((dmp.ewm(span=14,adjust=False).mean()/atr14 -
                       dmm.ewm(span=14,adjust=False).mean()/atr14).abs()
                      .ewm(span=14,adjust=False).mean() * 100)

    # 거래량
    df["vol_ma20"]  = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_ma20"]

    # 브레이크아웃
    df["high10"] = df["High"].rolling(10).max().shift(1)
    df["high20"] = df["High"].rolling(20).max().shift(1)
    df["bo10"]   = (c > df["high10"]).astype(int)
    df["bo20"]   = (c > df["high20"]).astype(int)

    # RSI 가속
    df["rsi_accel"] = ((df["rsi14"] > df["rsi21"]) & (df["rsi14"] > 55)).astype(int)

    # 연속 상승
    up = (c > c.shift(1)).astype(int)
    df["consec_up"] = up.groupby(((up==0)|(up!=up.shift(1))).cumsum()).cumsum() * up

    # J 발견 핵심: 눌림목 (추세 내 단기 하락)
    df["pullback_in_trend"] = (
        (df["pct_ma20"] > 10) &   # MA20 대비 +10% 이상 (추세 확인)
        (df["ret5"] < 0)          # 5일 수익률 마이너스 (눌림목)
    ).astype(int)

    df["pullback_strong_trend"] = (
        (df["pct_ma20"] > 20) &   # MA20 대비 +20% 이상
        (df["ret5"] < 0)
    ).astype(int)

    df["pullback_deep"] = (
        (df["pct_ma20"] > 10) &
        (df["ret5"] < -5)         # 5일 -5% 이상 눌림
    ).astype(int)

    # VCP 수축 (F실험 베스트)
    df["range5"]  = df["High"].rolling(5).max()  - df["Low"].rolling(5).min()
    df["range20"] = df["High"].rolling(20).max() - df["Low"].rolling(20).min()
    df["vcp_ratio"] = df["range5"] / df["range20"]

    return df


def build_macro(btc_df, extra_df):
    # BTC
    btc = btc_df[btc_df["ticker"]=="BTC-USD"].copy().sort_values("Date")
    c = btc["Close"]
    btc["btc_ma20"]  = c.rolling(20).mean()
    btc["btc_ma60"]  = c.rolling(60).mean()
    btc["btc_rsi14"] = rsi(c, 14)
    btc["btc_rsi7"]  = rsi(c, 7)
    btc["btc_ret5"]  = c.pct_change(5) * 100
    btc["btc_ret10"] = c.pct_change(10) * 100
    btc["btc_regime"] = 0
    btc.loc[c > btc["btc_ma60"], "btc_regime"] = 1
    btc.loc[(c > btc["btc_ma20"]) & (btc["btc_rsi14"] > 55), "btc_regime"] = 2

    # MSTR
    mstr = btc_df[btc_df["ticker"]=="MSTR"].copy().sort_values("Date")
    mc = mstr["Close"]
    mstr["mstr_ma20"]     = mc.rolling(20).mean()
    mstr["mstr_rsi14"]    = rsi(mc, 14)
    mstr["mstr_pct_ma20"] = (mc - mstr["mstr_ma20"]) / mstr["mstr_ma20"] * 100

    # VIX
    vix = extra_df[extra_df["ticker"]=="VIX"].copy().sort_values("Date")
    vix["vix_ma10"]    = vix["Close"].rolling(10).mean()
    vix["vix_falling"] = (vix["Close"] < vix["Close"].shift(3)).astype(int)

    # QQQ
    qqq = extra_df[extra_df["ticker"]=="QQQ"].copy().sort_values("Date")
    qc = qqq["Close"]
    qqq["qqq_ma20"]   = qc.rolling(20).mean()
    qqq["qqq_ma60"]   = qc.rolling(60).mean()
    qqq["qqq_rsi14"]  = rsi(qc, 14)
    qqq["qqq_ret10"]  = qc.pct_change(10) * 100
    qqq["qqq_pct_ma20"] = (qc - qqq["qqq_ma20"]) / qqq["qqq_ma20"] * 100
    qqq["qqq_bull"]   = (qc > qqq["qqq_ma60"]).astype(int)
    qqq["qqq_strong"] = ((qc > qqq["qqq_ma20"]) & (qqq["qqq_rsi14"] > 55)).astype(int)

    macro = (btc.set_index("Date")[["btc_ma20","btc_ma60","btc_rsi14","btc_rsi7",
                                     "btc_ret5","btc_ret10","btc_regime"]]
             .join(mstr.set_index("Date")[["mstr_rsi14","mstr_pct_ma20"]], how="outer")
             .join(vix.set_index("Date")[["Close","vix_falling"]].rename(
                   columns={"Close":"vix"}), how="outer")
             .join(qqq.set_index("Date")[["qqq_rsi14","qqq_pct_ma20","qqq_ret10",
                                          "qqq_bull","qqq_strong"]], how="outer")
            )
    return macro.ffill()


def run_backtest(df, entry_fn, target_pct, stop_pct, hold_days,
                 unit_usd=2220.0, max_pyramid=3, pyramid_add_pct=7.0,
                 trailing_pct=None, fx=1350.0):
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

            if peak is None or price > peak: peak = price
            trail_hit = (trailing_pct is not None and peak is not None and
                         (price-peak)/peak*100 <= trailing_pct)

            if pp >= target_pct or pp <= stop_pct or held >= hold_days or trail_hit or pma20 < -35:
                trades.append({"Date":d,"PnL_KRW":(tq*price-tc)*fx,
                               "PnL_pct":pp,"HeldDays":held,"Layers":len(pos)})
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
    print("[K] J 규칙 기반 강력한 실험 시작")
    ohlcv  = pd.read_parquet(OHLCV_PATH); ohlcv["Date"]  = pd.to_datetime(ohlcv["Date"])
    extra  = pd.read_parquet(EXTRA_PATH); extra["Date"]  = pd.to_datetime(extra["Date"])
    btc_df = pd.read_parquet(BTC_PATH);  btc_df["Date"] = pd.to_datetime(btc_df["Date"])

    macro = build_macro(btc_df, extra)

    # OHLCV 소스 통합 (profit_curve + extra_signals)
    ohlcv_all = pd.concat([
        ohlcv,
        extra[extra["ticker"].isin(["MSTX","SOXL","TQQQ","TSLL"])].copy()
    ], ignore_index=True)

    results = []

    for t in TICKERS:
        sub = ohlcv_all[ohlcv_all["ticker"]==t].copy()
        if len(sub) < 60: continue
        sub = add_features(sub)
        sub = sub.merge(macro.reset_index().rename(columns={"index":"Date"}),
                        on="Date", how="left")
        for col in macro.columns:
            if col in sub.columns:
                sub[col] = sub[col].ffill()

        print(f"  {t:6s}: {sub['Date'].min().date()} ~ {sub['Date'].max().date()}  ({len(sub)}일)")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # J 규칙 기반 진입 전략 (실제 수익 패턴에서 추출)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        J_STRATEGIES = {

            # ── Core J Rule (4개 조건 모두) ──────────────────────
            # BTC RSI<61 + VIX 18~25 + 종목 MA20+15% + 5일 눌림
            "j_core_4cond": lambda r: (
                r.get("btc_rsi14", 99) < 61 and
                18 <= r.get("vix", 25) <= 25 and
                r["pct_ma20"] > 15 and
                r["ret5"] < 0
            ),

            # ── J Rule 강화 버전 ────────────────────────────────
            "j_core_strict": lambda r: (
                r.get("btc_rsi14", 99) < 55 and
                20 <= r.get("vix", 25) <= 25 and
                r["pct_ma20"] > 20 and
                r["ret5"] < -2
            ),

            # ── VIX 최적구간 + BTC RSI 조건 ──────────────────────
            "j_vix_optimal": lambda r: (
                18 <= r.get("vix", 0) <= 25 and
                r.get("btc_rsi14", 99) < 65 and
                r["pct_ma20"] > 10
            ),

            "j_vix_neutral_pullback": lambda r: (
                20 <= r.get("vix", 0) <= 25 and
                r["pullback_in_trend"] == 1
            ),

            # ── BTC RSI <61 (단일 최강 신호) 강화 ────────────────
            "j_btc_rsi_low_trend": lambda r: (
                r.get("btc_rsi14", 99) < 61 and
                r["pct_ma20"] > 15 and
                r["rsi14"] > 55
            ),

            "j_btc_rsi_low_pull": lambda r: (
                r.get("btc_rsi14", 99) < 61 and
                r["pullback_in_trend"] == 1 and
                r["rsi14"] > 50
            ),

            "j_btc_rsi_vix": lambda r: (
                r.get("btc_rsi14", 99) < 61 and
                r.get("vix", 0) > 18 and
                r["pct_ma20"] > 10
            ),

            # ── 눌림목 집중 (J 핵심 발견) ────────────────────────
            "j_pullback_strong": lambda r: (
                r["pullback_strong_trend"] == 1 and
                r.get("btc_rsi14", 99) < 65 and
                r.get("vix", 0) > 17
            ),

            "j_pullback_deep_bounce": lambda r: (
                r["pullback_deep"] == 1 and
                r["rsi14"] > 45 and
                r.get("btc_rsi14", 99) < 70
            ),

            "j_pullback_macd": lambda r: (
                r["pullback_in_trend"] == 1 and
                r["macd_hist"] > 0 and
                r.get("btc_rsi14", 99) < 65
            ),

            # ── MSTR 역발상 (MA20 아래일 때 CONL 유리) ───────────
            "j_mstr_below_vix": lambda r: (
                r.get("mstr_pct_ma20", 0) < 0 and
                18 <= r.get("vix", 0) <= 25 and
                r["pct_ma20"] > 5
            ),

            "j_mstr_below_btc_low": lambda r: (
                r.get("mstr_pct_ma20", 0) < 2 and
                r.get("btc_rsi14", 99) < 61 and
                r["rsi14"] > 50
            ),

            # ── VIX 중립 + QQQ 강세 + 눌림 ──────────────────────
            "j_vix_qqq_pullback": lambda r: (
                18 <= r.get("vix", 0) <= 25 and
                r.get("qqq_bull", 0) == 1 and
                r["pullback_in_trend"] == 1
            ),

            "j_triple_neutral": lambda r: (
                r.get("btc_rsi14", 99) < 65 and
                18 <= r.get("vix", 0) <= 25 and
                r.get("qqq_pct_ma20", 0) > 0 and
                r["pct_ma20"] > 10
            ),

            # ── J + F 융합 (VCP 눌림목) ───────────────────────────
            "j_vcp_pullback": lambda r: (
                r.get("vcp_ratio", 1) < 0.5 and
                r["pct_ma20"] > 10 and
                r.get("btc_rsi14", 99) < 65 and
                r.get("vix", 0) > 17
            ),

            # ── J + E 융합 (DI크로스 + 눌림) ─────────────────────
            "j_di_pullback": lambda r: (
                r["di_plus"] > r["di_minus"] and
                r["pullback_in_trend"] == 1 and
                r.get("btc_rsi14", 99) < 65
            ),

            "j_di_vix_rsi": lambda r: (
                r["di_plus"] > r["di_minus"] and
                r.get("vix", 0) > 18 and
                r.get("btc_rsi14", 99) < 65 and
                r["rsi14"] > 55
            ),

            # ── 조건 완화 (J 핵심 2~3개만) ───────────────────────
            "j_2cond_rsi_vix": lambda r: (
                r.get("btc_rsi14", 99) < 65 and
                r.get("vix", 0) > 18
            ),

            "j_2cond_pull_vix": lambda r: (
                r["pullback_in_trend"] == 1 and
                r.get("vix", 0) > 17
            ),

            "j_2cond_pull_btcrsi": lambda r: (
                r["pullback_in_trend"] == 1 and
                r.get("btc_rsi14", 99) < 65
            ),

            # ── 매우 공격적: J 조건 + 포지션 최대화 ──────────────
            "j_max_conviction": lambda r: (
                r.get("btc_rsi14", 99) < 58 and
                20 <= r.get("vix", 0) <= 25 and
                r["pct_ma20"] > 20 and
                r["ret5"] < -3 and
                r.get("qqq_bull", 0) == 1
            ),

            # ── 반전 확인 (눌림 후 반등 신호 출현) ───────────────
            "j_bounce_confirm": lambda r: (
                r["pct_ma20"] > 15 and
                r["ret5"] < 0 and
                r["ret1"] > 0 and     # 당일 반등 시작
                r["rsi_accel"] == 1 and
                r.get("btc_rsi14", 99) < 65
            ),

            "j_bounce_vol": lambda r: (
                r["pullback_in_trend"] == 1 and
                r["ret1"] > 1 and       # 당일 1%+ 반등
                r["vol_ratio"] > 1.3 and
                r.get("btc_rsi14", 99) < 65
            ),
        }

        # ── 파라미터 그리드 (E실험 최강 파라미터 + 다양한 변형) ──
        PARAMS = [
            # (target, stop, hold, unit_mul, max_pyr, trailing)
            # E 챔피언급
            (80, -25,  90, 3.0, 3, None),
            (60, -25,  60, 3.0, 3, None),
            (50, -25,  45, 3.0, 3, None),
            (80, -25,  90, 2.0, 5, None),
            # 5x 공격
            (50, -25,  45, 5.0, 1, None),
            (40, -20,  30, 5.0, 1, None),
            # 트레일링
            (80, -25,  90, 3.0, 3, -12.0),
            (60, -25,  60, 3.0, 3, -10.0),
            # 안정형 (높은 N 목표)
            (30, -20,  30, 3.0, 1, None),
            (40, -20,  30, 3.0, 1, None),
            (25, -15,  20, 3.0, 1, None),
            (30, -20,  45, 2.0, 2, None),
            # 중간형
            (40, -20,  30, 2.0, 3, None),
            (50, -25,  45, 2.0, 3, None),
            (60, -25,  60, 2.0, 5, None),
            # 단기 고수익
            (30, -15,  15, 4.0, 1, None),
            (40, -20,  20, 4.0, 1, None),
        ]

        for strat_name, entry_fn in J_STRATEGIES.items():
            for (tgt, stp, hld, mul, max_pyr, trail) in PARAMS:
                try:
                    r = run_backtest(sub, entry_fn, tgt, stp, hld,
                                     unit_usd=740*mul, max_pyramid=max_pyr,
                                     trailing_pct=trail)
                    if r["n"] >= 3:
                        results.append({
                            "ticker": t, "strategy": strat_name,
                            "target": tgt, "stop": stp, "hold": hld,
                            "unit_mul": mul, "max_pyr": max_pyr,
                            "trailing": trail is not None,
                            **r
                        })
                except Exception:
                    pass

    df_res = pd.DataFrame(results).sort_values("pnl", ascending=False)
    df_res.to_csv(OUT, index=False)
    print(f"\n[K] 결과 저장: {OUT}  ({len(df_res)}개 조합)")
    print("\nTOP 25:")
    print(df_res[[
        "ticker","strategy","target","stop","hold","unit_mul","max_pyr","trailing",
        "pnl","wr","n","avg","avg_layers"
    ]].head(25).to_string(index=False))

    # 전략별 최강 요약
    print("\n\n=== 전략별 MAX PnL (J 규칙 버전) ===")
    best = df_res.groupby("strategy").agg(
        max_pnl=("pnl","max"), avg_wr=("wr","mean"), count=("n","sum")
    ).sort_values("max_pnl", ascending=False)
    print(best.head(20).to_string())


if __name__ == "__main__":
    main()

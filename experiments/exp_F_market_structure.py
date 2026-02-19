"""
exp_F_market_structure.py — 시장 구조 패턴 실험
================================================
지금까지 안 쓴 새로운 접근법:
1. VCP (Volatility Contraction Pattern): 변동성 수축 → 폭발
2. 오버나잇 갭업 (갭 방향성 추종)
3. 가격 구조: 고점/저점 상승 (HH+HL 패턴)
4. 공매도 대용: 거래량 급감 후 급등 (volume dry-up)
5. 주가 탄성: 하락 후 반등 강도 측정
6. 캔들 패턴: 망치형, 상승 장악형, 도지 탈출
7. 52주 신고가 근접도
8. 내부봉 (Inside Bar) 브레이크아웃
"""
import sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

OUT = ROOT / "experiments" / "results" / "F_market_structure.csv"
OHLCV_PATH = ROOT / "data" / "market" / "daily" / "profit_curve_ohlcv.parquet"
BTC_PATH   = ROOT / "data" / "market" / "daily" / "btc_mstr_daily.parquet"

TICKERS = ["IREN", "PTIR", "CONL", "NVDL", "MSTU", "AMDL"]


def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def add_structure_features(df):
    df = df.copy().sort_values("Date").reset_index(drop=True)
    c, h, l, o, v = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"]

    df["ma10"] = c.rolling(10).mean()
    df["ma20"] = c.rolling(20).mean()
    df["ma60"] = c.rolling(60).mean()
    df["pct_ma20"] = (c - df["ma20"]) / df["ma20"] * 100
    df["rsi14"] = rsi(c, 14)

    # ── 1. VCP (Volatility Contraction Pattern) ──────────────────────────────
    # 고/저 범위가 점점 좁아지는 패턴 감지
    # 5일 범위 / 20일 범위 → 수축비
    df["range5"]  = h.rolling(5).max()  - l.rolling(5).min()
    df["range20"] = h.rolling(20).max() - l.rolling(20).min()
    df["vcp_ratio"] = df["range5"] / df["range20"]  # 낮을수록 수축
    df["vcp_squeeze"] = (df["vcp_ratio"] < 0.35).astype(int)
    # 수축 후 당일 강한 양봉 = VCP 브레이크아웃
    df["vcp_breakout"] = (
        (df["vcp_squeeze"].shift(1) == 1) &
        (c > c.shift(1) * 1.02) &
        (c > df["ma20"])
    ).astype(int)

    # ── 2. 오버나잇 갭 ──────────────────────────────────────────────────────
    # 전일 종가 대비 당일 시가 갭
    df["gap_pct"] = (o - c.shift(1)) / c.shift(1) * 100
    df["gap_up"]  = (df["gap_pct"] > 2).astype(int)    # 2%+ 갭업
    df["gap_up3"] = (df["gap_pct"] > 3).astype(int)    # 3%+ 갭업
    df["gap_fill_up"] = (
        (df["gap_up"].shift(1) == 1) &   # 전일 갭업
        (c > o)                           # 당일 갭 채우지 않고 상승
    ).astype(int)

    # ── 3. 고점/저점 상승 (HH + HL 구조) ────────────────────────────────────
    # 10일 전 고점, 저점과 비교
    df["hh"] = (h > h.shift(5).rolling(5).max()).astype(int)  # Higher High
    df["hl"] = (l > l.shift(5).rolling(5).min()).astype(int)  # Higher Low
    df["hh_hl"] = ((df["hh"] == 1) & (df["hl"] == 1)).astype(int)  # 추세 확인

    # ── 4. Volume Dry-Up → 급등 ─────────────────────────────────────────────
    # 거래량 극도 감소 (5일 평균의 50% 미만) 후 급등
    df["vol_ma5"]   = v.rolling(5).mean()
    df["vol_ma20"]  = v.rolling(20).mean()
    df["vol_ratio"] = v / df["vol_ma20"]
    df["vol_dry"]   = (v < df["vol_ma5"] * 0.5).astype(int)    # 거래량 반토막
    df["vol_dry_breakout"] = (
        (df["vol_dry"].rolling(3).sum() >= 2) &   # 최근 3일 중 2일 거래량 급감
        (df["vol_ratio"] > 1.5) &                  # 당일 거래량 폭발
        (c > c.shift(1))                           # 상승
    ).astype(int)

    # ── 5. 탄성 강도 (Elasticity) ─────────────────────────────────────────
    # 최근 N일 최저점 대비 반등 강도
    df["low10"] = l.rolling(10).min()
    df["elasticity10"] = (c - df["low10"]) / df["low10"] * 100
    df["strong_bounce"] = (
        (df["elasticity10"] > 15) &       # 10일 저점 대비 15%+ 반등
        (c > df["ma20"]) &                # MA20 위
        (df["vol_ratio"] > 1.2)
    ).astype(int)

    # ── 6. 캔들 패턴 ──────────────────────────────────────────────────────
    body = c - o
    body_abs = body.abs()
    upper_wick = h - c.where(c > o, o)
    lower_wick = o.where(c > o, c) - l
    total_range = h - l

    # 망치형 (Hammer): 아래꼬리 길고, 몸통 작고, 위꼬리 없음
    df["hammer"] = (
        (lower_wick > body_abs * 2) &
        (upper_wick < body_abs * 0.5) &
        (total_range > 0) &
        (c > df["ma20"])     # MA20 위에서만
    ).astype(int)

    # 상승 장악형 (Bullish Engulfing): 전일 음봉을 당일 양봉이 완전히 감쌈
    df["engulfing"] = (
        (o.shift(1) > c.shift(1)) &   # 전일 음봉
        (c > o) &                      # 당일 양봉
        (o < c.shift(1)) &             # 당일 시가 < 전일 종가
        (c > o.shift(1))               # 당일 종가 > 전일 시가
    ).astype(int)

    # 도지 후 상승 (Doji Escape): 도지 다음날 강한 양봉
    doji = (body_abs / total_range.replace(0, np.nan) < 0.1)
    df["doji_escape"] = (
        doji.shift(1) &
        (c > o) &
        (body_abs > total_range * 0.5)
    ).astype(int)

    # ── 7. 52주 신고가 근접 ──────────────────────────────────────────────
    df["high52w"]     = h.rolling(252).max().shift(1)
    df["high52w_pct"] = (c - df["high52w"]) / df["high52w"] * 100  # 0이면 신고가
    df["near_52w_high"]  = (df["high52w_pct"] >= -5).astype(int)    # 신고가 5% 이내
    df["break_52w_high"] = (c > df["high52w"]).astype(int)           # 신고가 돌파

    # ── 8. 내부봉 브레이크아웃 (Inside Bar) ─────────────────────────────
    # 전일 고/저 범위 내에 당일이 포함 = 내부봉
    df["inside_bar"] = ((h < h.shift(1)) & (l > l.shift(1))).astype(int)
    # 내부봉 다음날 전일 고점 돌파 = 브레이크아웃
    df["inside_bar_bo"] = (
        (df["inside_bar"].shift(1) == 1) &
        (c > h.shift(2)) &    # 내부봉 전날 고점 돌파
        (v > df["vol_ma20"])
    ).astype(int)

    # ── 9. 이동평균 되돌림 (MA Pullback) ────────────────────────────────
    # 추세 중 MA10/MA20으로 되돌아왔다가 반등
    df["ma10_pullback"] = (
        (l < df["ma10"] * 1.01) &     # MA10 근처까지 내려옴
        (c > df["ma10"]) &            # 종가는 MA10 위
        (c > o) &                     # 양봉
        (df["pct_ma20"] > 0)          # 전체 추세 상승
    ).astype(int)

    df["ma20_pullback"] = (
        (l < df["ma20"] * 1.02) &     # MA20 근처
        (c > df["ma20"]) &
        (c > o) &
        (df["rsi14"] > 45)
    ).astype(int)

    # ── 10. 가격 압축 후 방향 돌파 ─────────────────────────────────────
    # 5일 최고 - 최저 범위가 좁아진 후 돌파
    df["narrow_range5"] = (df["range5"] < df["range5"].rolling(20).quantile(0.2)).astype(int)
    df["nr5_breakout"]  = (
        (df["narrow_range5"].shift(1) == 1) &
        (c > h.shift(1)) &    # 전일 고점 돌파
        (v > df["vol_ma20"])
    ).astype(int)

    return df


def build_btc_regime(btc_df):
    btc = btc_df[btc_df["ticker"]=="BTC-USD"].copy().sort_values("Date")
    c = btc["Close"]
    btc["btc_ma20"]  = c.rolling(20).mean()
    btc["btc_ma60"]  = c.rolling(60).mean()
    btc["btc_rsi14"] = rsi(c, 14)
    btc["btc_ret5"]  = c.pct_change(5) * 100
    btc["btc_regime"] = 0
    btc.loc[c > btc["btc_ma60"], "btc_regime"] = 1
    btc.loc[(c > btc["btc_ma20"]) & (btc["btc_rsi14"] > 55), "btc_regime"] = 2
    return btc.set_index("Date")[["btc_regime","btc_ret5"]]


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
                pnl = tq*price - tc
                trades.append({"Date":d, "PnL_KRW":pnl*fx, "PnL_pct":pp, "HeldDays":held})
                pos = []; continue

        if not has_pos and entry_fn(row):
            pos.append((d, unit_usd/price, price))

    if not trades: return {"n":0,"pnl":0,"wr":0,"avg":0}
    tdf = pd.DataFrame(trades)
    return {"n":len(tdf), "pnl":round(tdf["PnL_KRW"].sum()),
            "wr":round((tdf["PnL_KRW"]>0).mean()*100,1),
            "avg":round(tdf["PnL_pct"].mean(),2)}


def main():
    print("[F] 시장 구조 패턴 실험 시작")
    ohlcv  = pd.read_parquet(OHLCV_PATH); ohlcv["Date"]  = pd.to_datetime(ohlcv["Date"])
    btc_df = pd.read_parquet(BTC_PATH);   btc_df["Date"] = pd.to_datetime(btc_df["Date"])
    btc_reg = build_btc_regime(btc_df)

    results = []
    for t in TICKERS:
        sub = ohlcv[ohlcv["ticker"]==t].copy()
        if len(sub) < 80: continue
        sub = add_structure_features(sub)
        sub = sub.merge(btc_reg.reset_index(), on="Date", how="left")
        sub["btc_regime"] = sub["btc_regime"].ffill().fillna(0)
        sub["btc_ret5"]   = sub["btc_ret5"].ffill().fillna(0)

        STRATEGIES = {
            # VCP
            "vcp_breakout_btc1":    lambda r: r["vcp_breakout"]==1 and r.get("btc_regime",0)>=1,
            "vcp_breakout_btc2":    lambda r: r["vcp_breakout"]==1 and r.get("btc_regime",0)==2,
            "vcp_breakout_vol":     lambda r: r["vcp_breakout"]==1 and r["vol_ratio"]>1.5 and r.get("btc_regime",0)>=1,
            "vcp_squeeze_bo":       lambda r: r["vcp_squeeze"]==0 and r.get("vcp_ratio",1)<0.5 and r["rsi14"]>55 and r.get("btc_regime",0)>=1,

            # 갭업
            "gap_up_btc2":          lambda r: r["gap_up"]==1 and r.get("btc_regime",0)==2,
            "gap_up3_btc1":         lambda r: r["gap_up3"]==1 and r.get("btc_regime",0)>=1,
            "gap_fill_up_btc2":     lambda r: r["gap_fill_up"]==1 and r.get("btc_regime",0)==2,
            "gap_up3_vol_btc":      lambda r: r["gap_up3"]==1 and r["vol_ratio"]>1.3 and r.get("btc_regime",0)>=1,

            # HH+HL 구조
            "hh_hl_btc2":           lambda r: r["hh_hl"]==1 and r.get("btc_regime",0)==2,
            "hh_hl_rsi_btc":        lambda r: r["hh_hl"]==1 and r["rsi14"]>55 and r.get("btc_regime",0)>=1,
            "hh_hl_ma_btc2":        lambda r: r["hh_hl"]==1 and r["pct_ma20"]>0 and r.get("btc_regime",0)==2 and r["rsi14"]>50,

            # Volume Dry-Up → 폭발
            "vol_dry_bo_btc1":      lambda r: r["vol_dry_breakout"]==1 and r.get("btc_regime",0)>=1,
            "vol_dry_bo_btc2":      lambda r: r["vol_dry_breakout"]==1 and r.get("btc_regime",0)==2,
            "vol_dry_bo_rsi":       lambda r: r["vol_dry_breakout"]==1 and r["rsi14"]>50 and r.get("btc_regime",0)>=1,

            # 탄성 반등
            "bounce_btc2":          lambda r: r["strong_bounce"]==1 and r.get("btc_regime",0)==2,
            "bounce_btc1_rsi":      lambda r: r["strong_bounce"]==1 and r["rsi14"]>50 and r.get("btc_regime",0)>=1,

            # 캔들 패턴
            "hammer_btc2":          lambda r: r["hammer"]==1 and r.get("btc_regime",0)==2,
            "engulf_btc2":          lambda r: r["engulfing"]==1 and r.get("btc_regime",0)==2,
            "engulf_vol_btc":       lambda r: r["engulfing"]==1 and r["vol_ratio"]>1.3 and r.get("btc_regime",0)>=1,
            "doji_escape_btc2":     lambda r: r["doji_escape"]==1 and r.get("btc_regime",0)==2,
            "doji_escape_btc_rsi":  lambda r: r["doji_escape"]==1 and r["rsi14"]>50 and r.get("btc_regime",0)>=1,

            # 52주 신고가
            "near_52w_btc2":        lambda r: r["near_52w_high"]==1 and r.get("btc_regime",0)==2 and r["rsi14"]>60,
            "break_52w_btc2":       lambda r: r["break_52w_high"]==1 and r.get("btc_regime",0)>=1,
            "break_52w_vol_btc":    lambda r: r["break_52w_high"]==1 and r["vol_ratio"]>1.3 and r.get("btc_regime",0)>=1,

            # 내부봉 브레이크아웃
            "inside_bar_bo_btc2":   lambda r: r["inside_bar_bo"]==1 and r.get("btc_regime",0)==2,
            "inside_bar_bo_btc1":   lambda r: r["inside_bar_bo"]==1 and r.get("btc_regime",0)>=1,
            "inside_bar_rsi_vol":   lambda r: r["inside_bar_bo"]==1 and r["rsi14"]>55 and r["vol_ratio"]>1.2,

            # MA 되돌림
            "ma10_pull_btc2":       lambda r: r["ma10_pullback"]==1 and r.get("btc_regime",0)==2,
            "ma20_pull_btc2":       lambda r: r["ma20_pullback"]==1 and r.get("btc_regime",0)==2,
            "ma10_pull_vol":        lambda r: r["ma10_pullback"]==1 and r["vol_ratio"]>1.2 and r.get("btc_regime",0)>=1,

            # NR5 브레이크아웃
            "nr5_bo_btc2":          lambda r: r["nr5_breakout"]==1 and r.get("btc_regime",0)==2,
            "nr5_bo_btc1_rsi":      lambda r: r["nr5_breakout"]==1 and r["rsi14"]>55 and r.get("btc_regime",0)>=1,

            # 복합 구조 패턴
            "vcp_engulf_btc2":      lambda r: (r["vcp_squeeze"]==1 or r["vcp_breakout"]==1) and r["engulfing"]==1 and r.get("btc_regime",0)==2,
            "gap_hh_hl_btc2":       lambda r: r["gap_up"]==1 and r["hh_hl"]==1 and r.get("btc_regime",0)==2,
            "nr5_vol_52w_btc":      lambda r: r["nr5_breakout"]==1 and r["near_52w_high"]==1 and r.get("btc_regime",0)>=1,
        }

        PARAMS = [
            (15,-20,30),(20,-20,30),(25,-20,30),(30,-20,30),
            (20,-15,20),(25,-15,20),(30,-15,20),
            (40,-20,30),(50,-25,45),(60,-25,60),
            (30,-20,45),(40,-20,45),(50,-20,45),
        ]

        for strat_name, entry_fn in STRATEGIES.items():
            for (tgt, stp, hld) in PARAMS:
                for mul in [1.0, 2.0, 3.0]:
                    try:
                        r = backtest(sub, entry_fn, tgt, stp, hld, unit_usd=740*mul)
                        if r["n"] >= 3:
                            results.append({"ticker":t, "strategy":strat_name,
                                            "target":tgt, "stop":stp, "hold":hld,
                                            "unit_mul":mul, **r})
                    except Exception:
                        pass

    df_res = pd.DataFrame(results).sort_values("pnl", ascending=False)
    df_res.to_csv(OUT, index=False)
    print(f"\n[F] 결과 저장: {OUT}  ({len(df_res)}개 조합)")
    print("\nTOP 20:")
    print(df_res[["ticker","strategy","target","stop","hold","unit_mul",
                   "pnl","wr","n","avg"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()

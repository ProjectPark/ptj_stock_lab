"""
exp_E_ultra_aggressive.py — 울트라 공격 전략
=============================================
이전 A~D 실험 베스트 신호 조합
타겟 60~100%, 포지션 3x~5x
4~5레이어 피라미딩
추세 추종 청산 (BTC 레짐 하락 시 탈출)
Kelly 기준 포지션 사이징
IREN · PTIR · CONL 집중
"""
import sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

OUT = ROOT / "experiments" / "results" / "E_ultra_aggressive.csv"
OHLCV_PATH = ROOT / "data" / "market" / "daily" / "profit_curve_ohlcv.parquet"
BTC_PATH   = ROOT / "data" / "market" / "daily" / "btc_mstr_daily.parquet"

TICKERS = ["IREN", "PTIR", "CONL", "NVDL", "MSTU", "AMDL"]


# ── 지표 계산 ─────────────────────────────────────────────────────────────────
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

    df["ret1"]  = c.pct_change(1) * 100
    df["ret3"]  = c.pct_change(3) * 100
    df["ret5"]  = c.pct_change(5) * 100
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
    df["macd_hist_prev"] = df["macd_hist"].shift(1)

    # ADX / DI
    high, low = df["High"], df["Low"]
    tr = pd.concat([high-low,
                    (high-c.shift(1)).abs(),
                    (low -c.shift(1)).abs()], axis=1).max(axis=1)
    dmp = (high-high.shift(1)).clip(lower=0).where(
          (high-high.shift(1)) > (low.shift(1)-low), 0)
    dmm = (low.shift(1)-low).clip(lower=0).where(
          (low.shift(1)-low) > (high-high.shift(1)), 0)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    df["di_plus"]  = dmp.ewm(span=14, adjust=False).mean() / atr14 * 100
    df["di_minus"] = dmm.ewm(span=14, adjust=False).mean() / atr14 * 100
    df["adx"] = (
        (dmp.ewm(span=14, adjust=False).mean() / atr14 -
         dmm.ewm(span=14, adjust=False).mean() / atr14).abs()
        .ewm(span=14, adjust=False).mean() * 100
    )

    # 거래량
    df["vol_ma20"]  = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_ma20"]

    # 브레이크아웃
    df["high5"]  = df["High"].rolling(5).max().shift(1)
    df["high10"] = df["High"].rolling(10).max().shift(1)
    df["bo5"]  = (c > df["high5"]).astype(int)
    df["bo10"] = (c > df["high10"]).astype(int)

    # MA 정렬
    df["ma_aligned"]   = ((df["ma5"]>df["ma10"]) & (df["ma10"]>df["ma20"])).astype(int)
    df["ma_aligned60"] = ((df["ma10"]>df["ma20"]) & (df["ma20"]>df["ma60"])).astype(int)

    # 급등 캔들
    df["is_bull_candle"] = (df["Close"] > df["Open"]).astype(int)
    df["surge3"]  = (df["ret3"] > 8).astype(int)   # 3일 8%+
    df["surge5"]  = (df["ret5"] > 12).astype(int)  # 5일 12%+
    df["surge1"]  = ((df["ret1"] > 3) & (df["is_bull_candle"] == 1)).astype(int)

    # RSI 가속 (D실험 베스트)
    df["rsi_accel"] = ((df["rsi14"] > df["rsi21"]) & (df["rsi14"] > 60)).astype(int)

    # 연속 상승
    up_day = (c > c.shift(1)).astype(int)
    df["consec_up"] = up_day.groupby(
        ((up_day == 0) | (up_day != up_day.shift(1))).cumsum()
    ).cumsum() * up_day

    return df


def build_btc_features(btc_df):
    btc = btc_df[btc_df["ticker"]=="BTC-USD"].copy().sort_values("Date")
    c = btc["Close"]
    btc["btc_ma20"]  = c.rolling(20).mean()
    btc["btc_ma60"]  = c.rolling(60).mean()
    btc["btc_ma120"] = c.rolling(120).mean()
    btc["btc_rsi14"] = rsi(c, 14)
    btc["btc_rsi7"]  = rsi(c, 7)
    btc["btc_ret3"]  = c.pct_change(3)  * 100
    btc["btc_ret5"]  = c.pct_change(5)  * 100
    btc["btc_ret10"] = c.pct_change(10) * 100
    btc["btc_vol_ratio"] = (btc["High"]-btc["Low"]) / (btc["High"]-btc["Low"]).rolling(14).mean()

    # MSTR 선행 신호 (A실험 베스트)
    mstr = btc_df[btc_df["ticker"]=="MSTR"].copy().sort_values("Date")
    mstr["mstr_pct_ma20"] = (mstr["Close"] - mstr["Close"].rolling(20).mean()) / mstr["Close"].rolling(20).mean() * 100
    mstr["mstr_rsi"]      = rsi(mstr["Close"], 14)
    mstr["mstr_ret3"]     = mstr["Close"].pct_change(3) * 100

    # 레짐 (0=약세, 1=보통, 2=강세, 3=초강세)
    btc["btc_regime"] = 0
    btc.loc[c > btc["btc_ma60"],  "btc_regime"] = 1
    btc.loc[(c > btc["btc_ma20"]) & (btc["btc_rsi14"] > 55), "btc_regime"] = 2
    btc.loc[(c > btc["btc_ma20"]) & (btc["btc_rsi14"] > 65) & (btc["btc_ret10"] > 15), "btc_regime"] = 3

    btc["btc_strong_3d"] = (btc["btc_ret3"] > 8).astype(int)   # 3일 8%+ 급등
    btc["btc_strong_5d"] = (btc["btc_ret5"] > 12).astype(int)  # 5일 12%+
    btc["btc_hot_rsi"]   = (btc["btc_rsi7"] > 70).astype(int)  # RSI7 > 70 과열

    result = btc.set_index("Date")[[
        "btc_ma20","btc_ma60","btc_ma120","btc_rsi14","btc_rsi7",
        "btc_ret3","btc_ret5","btc_ret10","btc_regime",
        "btc_strong_3d","btc_strong_5d","btc_hot_rsi"
    ]]

    # MSTR 신호 병합
    mstr_sig = mstr.set_index("Date")[["mstr_pct_ma20","mstr_rsi","mstr_ret3"]]
    result = result.join(mstr_sig, how="left")

    return result


def run_ultra_bt(df, entry_fn, exit_fn,
                 target_pct, stop_pct, hold_days,
                 unit_usd=2220.0,   # 기본 3x (740*3)
                 max_pyramid=5, pyramid_add_pct=7.0,
                 trailing_stop_pct=None,
                 btc_regime_exit=False,
                 fx=1350.0):
    """
    울트라 백테스트:
    - exit_fn: 추가 청산 조건 (BTC 레짐 하락 등)
    - trailing_stop_pct: 최고점 대비 낙폭 % (None이면 미사용)
    - btc_regime_exit: BTC 레짐이 0으로 떨어지면 강제 청산
    - max_pyramid 5레이어까지
    """
    trades, pos = [], []
    peak_price = None

    for _, row in df.iterrows():
        if pd.isna(row.get("ma20")):
            continue
        price, d = row["Close"], row["Date"]
        pma20 = row["pct_ma20"]
        btc_reg = row.get("btc_regime", 1)
        has_pos = bool(pos)

        if has_pos:
            tq = sum(p[1] for p in pos)
            tc = sum(p[1]*p[2] for p in pos)
            avg = tc / tq
            pp = (price - avg) / avg * 100
            held = (d - pos[0][0]).days

            # 최고가 업데이트 (트레일링 스탑용)
            if peak_price is None or price > peak_price:
                peak_price = price

            # 청산 조건
            trail_hit = False
            if trailing_stop_pct is not None and peak_price is not None:
                trail_pct = (price - peak_price) / peak_price * 100
                trail_hit = trail_pct <= trailing_stop_pct

            btc_exit = btc_regime_exit and btc_reg == 0

            should_exit = (
                pp >= target_pct
                or pp <= stop_pct
                or held >= hold_days
                or pma20 < -35
                or trail_hit
                or btc_exit
                or (exit_fn is not None and exit_fn(row, pos, avg))
            )

            if should_exit:
                pnl = tq*price - tc
                reason = ("T" if pp>=target_pct else
                          "Trail" if trail_hit else
                          "BTC" if btc_exit else
                          "S" if pp<=stop_pct else "X")
                trades.append({
                    "Date": d, "PnL_KRW": pnl*fx, "PnL_pct": pp,
                    "HeldDays": held, "Layers": len(pos), "Reason": reason
                })
                pos, peak_price = [], None
                continue

            # 피라미딩: 추가 진입
            if len(pos) < max_pyramid:
                last_price = pos[-1][2]
                if price > last_price * (1 + pyramid_add_pct/100) and entry_fn(row):
                    add_unit = unit_usd * (0.8 ** len(pos))  # 레이어마다 80%씩 축소
                    qty = add_unit / price
                    pos.append((d, qty, price))

        # 신규 진입
        if not has_pos and entry_fn(row):
            qty = unit_usd / price
            pos.append((d, qty, price))
            peak_price = price

    if not trades:
        return {"n":0,"pnl":0,"wr":0,"avg":0,"avg_layers":0}
    tdf = pd.DataFrame(trades)
    return {
        "n": len(tdf),
        "pnl": round(tdf["PnL_KRW"].sum()),
        "wr": round((tdf["PnL_KRW"]>0).mean()*100, 1),
        "avg": round(tdf["PnL_pct"].mean(), 2),
        "avg_layers": round(tdf["Layers"].mean(), 2)
    }


def main():
    print("[E] 울트라 공격 전략 실험 시작")
    ohlcv  = pd.read_parquet(OHLCV_PATH); ohlcv["Date"]  = pd.to_datetime(ohlcv["Date"])
    btc_df = pd.read_parquet(BTC_PATH);   btc_df["Date"] = pd.to_datetime(btc_df["Date"])

    btc_feat = build_btc_features(btc_df)

    results = []

    for t in TICKERS:
        sub = ohlcv[ohlcv["ticker"]==t].copy()
        if len(sub) < 60: continue
        sub = add_features(sub)
        sub = sub.merge(btc_feat.reset_index(), on="Date", how="left")
        for col in ["btc_ma20","btc_ma60","btc_rsi14","btc_rsi7","btc_ret3",
                    "btc_ret5","btc_ret10","btc_regime","btc_strong_3d",
                    "btc_strong_5d","btc_hot_rsi","mstr_pct_ma20","mstr_rsi","mstr_ret3"]:
            if col in sub.columns:
                sub[col] = sub[col].ffill()

        # ── 울트라 복합 신호 (A~D 베스트 조합) ──────────────────────────────────
        ULTRA_STRATEGIES = {
            # ── 최강 조합 (A+D 베스트) ──
            "mstr_over5_ret3_btc":   lambda r: (r.get("mstr_pct_ma20",0) > 5
                                                  and r.get("btc_ret3",0) > 5
                                                  and r.get("btc_regime",0) >= 1),

            "mstr_rsi_ret3_btc2":    lambda r: (r.get("mstr_rsi",0) > 60
                                                  and r.get("btc_ret3",0) > 5
                                                  and r.get("btc_regime",0) == 2),

            # ── 초강세 BTC 모멘텀 (D 베스트 강화) ──
            "btc_strong3d_surge":    lambda r: (r.get("btc_strong_3d",0)==1
                                                  and r["surge1"]==1
                                                  and r.get("btc_regime",0)>=1),

            "btc_strong3d_bo10":     lambda r: (r.get("btc_strong_3d",0)==1
                                                  and r["bo10"]==1
                                                  and r.get("btc_regime",0)>=1),

            "btc_strong5d_rsi_accel": lambda r: (r.get("btc_strong_5d",0)==1
                                                   and r["rsi_accel"]==1
                                                   and r["pct_ma20"]>0),

            "btc_regime3_all":       lambda r: (r.get("btc_regime",0)==3
                                                  and r["rsi14"]>55
                                                  and r["pct_ma20"]>0),

            "btc_regime3_surge":     lambda r: (r.get("btc_regime",0)==3
                                                  and r["surge3"]==1),

            "btc_regime3_bo10":      lambda r: (r.get("btc_regime",0)==3
                                                  and r["bo10"]==1
                                                  and r["vol_ratio"]>1.2),

            # ── RSI 가속 (D 1위) 강화 버전 ──
            "rsi_accel_btc2_vol":    lambda r: (r["rsi_accel"]==1
                                                  and r.get("btc_regime",0)==2
                                                  and r["vol_ratio"]>1.2),

            "rsi_accel_btc3":        lambda r: (r["rsi_accel"]==1
                                                  and r.get("btc_regime",0)==3),

            "rsi7_80_btc2":          lambda r: (r["rsi7"]>75
                                                  and r.get("btc_regime",0)>=2
                                                  and r["pct_ma20"]>5),

            # ── 연속 상승 (D 베스트) 강화 ──
            "consec3_btc2_vol":      lambda r: (r.get("consec_up",0)>=3
                                                  and r.get("btc_regime",0)==2
                                                  and r["vol_ratio"]>1.3),

            "consec4_btc1":          lambda r: (r.get("consec_up",0)>=4
                                                  and r.get("btc_regime",0)>=1),

            "consec3_surge":         lambda r: (r.get("consec_up",0)>=3
                                                  and r["surge3"]==1),

            # ── 브레이크아웃 + 볼륨 + BTC ──
            "bo5_vol_btc2_rsi":      lambda r: (r["bo5"]==1
                                                  and r["vol_ratio"]>1.5
                                                  and r.get("btc_regime",0)==2
                                                  and r["rsi14"]>60),

            "bo10_macd_btc2":        lambda r: (r["bo10"]==1
                                                  and r["macd_hist"]>0
                                                  and r.get("btc_regime",0)==2),

            # ── B실험 베스트 (MACD+ADX) 강화 ──
            "bb_macd_adx_btc2":      lambda r: (r["bb_pct"]>55
                                                  and r["macd_hist"]>0
                                                  and r["di_plus"]>r["di_minus"]
                                                  and r.get("btc_regime",0)==2),

            "macd_adx_vol_btc2":     lambda r: (r["macd_hist"]>0
                                                  and r["macd_hist"]>r["macd_hist_prev"]
                                                  and r["adx"]>25
                                                  and r["vol_ratio"]>1.3
                                                  and r.get("btc_regime",0)==2),

            "di_cross_surge_btc2":   lambda r: (r["di_plus"]>r["di_minus"]
                                                  and r["surge1"]==1
                                                  and r.get("btc_regime",0)==2),

            # ── MSTR 리드 + 자체 모멘텀 (A 베스트 강화) ──
            "mstr5_bo10_btc2":       lambda r: (r.get("mstr_pct_ma20",0)>5
                                                  and r["bo10"]==1
                                                  and r.get("btc_regime",0)==2),

            "mstr_rsi65_ma_btc2":    lambda r: (r.get("mstr_rsi",0)>65
                                                  and r["ma_aligned60"]==1
                                                  and r.get("btc_regime",0)==2),

            # ── 가변 레짐 + 최강 신호 복합 ──
            "all_green_ultra":       lambda r: (r.get("btc_regime",0)==2
                                                  and r["ma_aligned60"]==1
                                                  and r["rsi_accel"]==1
                                                  and r["macd_hist"]>0
                                                  and r["vol_ratio"]>1.2),

            "max_conviction":        lambda r: (r.get("btc_regime",0)>=2
                                                  and r["bo10"]==1
                                                  and r["rsi_accel"]==1
                                                  and r.get("mstr_pct_ma20",0)>0
                                                  and r["macd_hist"]>0),

            "triple_confirm":        lambda r: (r.get("btc_strong_3d",0)==1
                                                  and r["rsi_accel"]==1
                                                  and r["vol_ratio"]>1.3
                                                  and r["pct_ma20"]>0),
        }

        # ── 청산 함수 (BTC 레짐 하락 감지) ──
        def btc_regime_drop_exit(row, pos, avg):
            """BTC가 MA20 아래로 추락하면 조기 청산"""
            return row.get("btc_regime", 1) == 0 and (row["Close"] - avg) / avg * 100 > -5

        # ── 파라미터 그리드 (공격적) ──
        ULTRA_PARAMS = [
            # (target, stop, hold, unit_mul, max_pyr, pyramid_add, trail, btc_exit)
            # 기본 공격 (3x 포지션)
            (40, -20, 30,  3.0, 1, 7.0,  None,  False),
            (50, -20, 45,  3.0, 1, 7.0,  None,  False),
            (60, -25, 60,  3.0, 1, 7.0,  None,  False),
            (80, -25, 90,  3.0, 1, 7.0,  None,  False),

            # 4x 포지션
            (40, -20, 30,  4.0, 1, 7.0,  None,  False),
            (50, -25, 45,  4.0, 1, 7.0,  None,  False),
            (60, -25, 60,  4.0, 1, 7.0,  None,  False),

            # 5x 포지션 (극한)
            (30, -15, 20,  5.0, 1, 7.0,  None,  False),
            (40, -20, 30,  5.0, 1, 7.0,  None,  False),
            (50, -25, 45,  5.0, 1, 7.0,  None,  False),

            # 피라미딩 3레이어 (3x 기본)
            (50, -20, 45,  3.0, 3, 7.0,  None,  False),
            (60, -25, 60,  3.0, 3, 7.0,  None,  False),
            (80, -25, 90,  3.0, 3, 7.0,  None,  False),

            # 피라미딩 5레이어 (2x 기본)
            (60, -25, 60,  2.0, 5, 7.0,  None,  False),
            (80, -25, 90,  2.0, 5, 7.0,  None,  False),
            (100,-30, 120, 2.0, 5, 7.0,  None,  False),

            # 트레일링 스탑 (3x)
            (40, -20, 30,  3.0, 1, 7.0,  -10.0, False),
            (60, -25, 60,  3.0, 1, 7.0,  -12.0, False),
            (80, -30, 90,  3.0, 1, 7.0,  -15.0, False),

            # 트레일링 + 피라미딩 (2x, 3레이어)
            (50, -20, 45,  2.0, 3, 7.0,  -10.0, False),
            (80, -25, 90,  2.0, 3, 7.0,  -15.0, False),

            # BTC 레짐 탈출 (추세 추종 청산)
            (60, -25, 90,  3.0, 1, 7.0,  None,  True),
            (80, -25, 120, 3.0, 1, 7.0,  None,  True),
            (100,-30, 180, 2.0, 3, 7.0,  None,  True),

            # 피라미딩 강화 (5레이어) + BTC 청산
            (80, -25, 120, 2.0, 5, 5.0,  None,  True),
            (100,-30, 180, 2.0, 5, 5.0,  None,  True),
        ]

        for strat_name, entry_fn in ULTRA_STRATEGIES.items():
            for (tgt, stp, hld, mul, max_pyr, pyr_add, trail, btc_exit) in ULTRA_PARAMS:
                try:
                    unit = 740.0 * mul
                    exit_fn = btc_regime_drop_exit if btc_exit else None
                    r = run_ultra_bt(
                        sub, entry_fn, exit_fn,
                        tgt, stp, hld,
                        unit_usd=unit, max_pyramid=max_pyr,
                        pyramid_add_pct=pyr_add,
                        trailing_stop_pct=trail,
                        btc_regime_exit=btc_exit
                    )
                    if r["n"] >= 3:
                        results.append({
                            "ticker": t, "strategy": strat_name,
                            "target": tgt, "stop": stp, "hold": hld,
                            "unit_mul": mul, "max_pyr": max_pyr,
                            "trail": trail, "btc_exit": btc_exit,
                            **r
                        })
                except Exception:
                    pass

    df_res = pd.DataFrame(results).sort_values("pnl", ascending=False)
    df_res.to_csv(OUT, index=False)
    print(f"\n[E] 결과 저장: {OUT}  ({len(df_res)}개 조합)")
    print("\nTOP 20:")
    print(df_res[[
        "ticker","strategy","target","stop","hold",
        "unit_mul","max_pyr","trail","btc_exit",
        "pnl","wr","n","avg","avg_layers"
    ]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()

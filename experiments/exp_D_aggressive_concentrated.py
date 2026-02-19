"""
exp_D_aggressive_concentrated.py — 공격적 집중 전략 실험
=========================================================
CONL + IREN + PTIR 집중 (고변동성 종목)
단기 고수익 목표: 30~60% 타겟
포지션 크기 1.5x ~ 3x
단기 홀드 (5~15일) vs 장기 홀드 (45~90일)
연속 시그널 피라미딩 (최대 3 레이어)
급등 직후 추격 진입 vs 눌림목 진입
"""
import sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

OUT = ROOT / "experiments" / "results" / "D_aggressive_concentrated.csv"
OHLCV_PATH = ROOT / "data" / "market" / "daily" / "profit_curve_ohlcv.parquet"
BTC_PATH   = ROOT / "data" / "market" / "daily" / "btc_mstr_daily.parquet"

# D: 집중 대상 종목 (변동성 높은 레버리지 ETF)
FOCUS_TICKERS = ["CONL", "IREN", "PTIR", "MSTU", "NVDL", "AMDL"]


def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def add_features(df):
    df = df.copy().sort_values("Date").reset_index(drop=True)
    c = df["Close"]

    # 이동평균
    df["ma5"]   = c.rolling(5).mean()
    df["ma10"]  = c.rolling(10).mean()
    df["ma20"]  = c.rolling(20).mean()
    df["ma60"]  = c.rolling(60).mean()
    df["pct_ma20"] = (c - df["ma20"]) / df["ma20"] * 100
    df["pct_ma5"]  = (c - df["ma5"])  / df["ma5"]  * 100

    # RSI 다중 주기
    df["rsi7"]  = rsi(c, 7)
    df["rsi14"] = rsi(c, 14)
    df["rsi21"] = rsi(c, 21)

    # 수익률
    df["ret1"]  = c.pct_change(1) * 100
    df["ret3"]  = c.pct_change(3) * 100
    df["ret5"]  = c.pct_change(5) * 100
    df["ret10"] = c.pct_change(10) * 100

    # ATR (변동성)
    atr = (df["High"] - df["Low"]).rolling(14).mean()
    df["atr_pct"] = atr / c * 100

    # 볼린저밴드
    std20 = c.rolling(20).std()
    df["bb_upper"] = df["ma20"] + 2 * std20
    df["bb_lower"] = df["ma20"] - 2 * std20
    df["bb_pct"]   = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]) * 100

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]      = ema12 - ema26
    df["macd_sig"]  = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]

    # 신고가 브레이크아웃
    df["high5"]  = df["High"].rolling(5).max().shift(1)
    df["high10"] = df["High"].rolling(10).max().shift(1)
    df["high20"] = df["High"].rolling(20).max().shift(1)
    df["bo5"]    = (c > df["high5"]).astype(int)
    df["bo10"]   = (c > df["high10"]).astype(int)
    df["bo20"]   = (c > df["high20"]).astype(int)

    # 거래량
    df["vol_ma10"] = df["Volume"].rolling(10).mean()
    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    df["vol_ratio10"] = df["Volume"] / df["vol_ma10"]
    df["vol_ratio20"] = df["Volume"] / df["vol_ma20"]

    # 캔들 패턴
    df["body_pct"] = (df["Close"] - df["Open"]).abs() / df["Open"] * 100
    df["is_bull_candle"] = (df["Close"] > df["Open"]).astype(int)

    # 급등 감지 (단일봉 3% 이상 상승)
    df["surge_bull"] = ((df["ret1"] > 3) & df["is_bull_candle"].astype(bool)).astype(int)
    # 눌림목: 전일 급등 후 당일 소폭 하락
    df["pullback_after_surge"] = ((df["ret1"].shift(1) > 5) & (df["ret1"] < 0)).astype(int)

    # MA 정렬
    df["ma_aligned"]   = ((df["ma5"] > df["ma10"]) & (df["ma10"] > df["ma20"])).astype(int)
    df["ma_aligned60"] = ((df["ma10"] > df["ma20"]) & (df["ma20"] > df["ma60"])).astype(int)

    # 연속 상승일 수
    df["consec_up"] = (
        (c > c.shift(1)).astype(int)
        .groupby((c <= c.shift(1)).astype(int).cumsum())
        .cumsum()
    )

    return df


def build_btc_features(btc_df):
    btc = btc_df[btc_df["ticker"] == "BTC-USD"].copy().sort_values("Date")
    btc["btc_ma20"]  = btc["Close"].rolling(20).mean()
    btc["btc_ma60"]  = btc["Close"].rolling(60).mean()
    btc["btc_rsi14"] = rsi(btc["Close"], 14)
    btc["btc_ret3"]  = btc["Close"].pct_change(3) * 100
    btc["btc_ret7"]  = btc["Close"].pct_change(7) * 100
    btc["btc_atr_pct"] = (btc["High"] - btc["Low"]).rolling(14).mean() / btc["Close"] * 100

    # 레짐
    btc["btc_regime"] = 0
    btc.loc[btc["Close"] > btc["btc_ma60"],  "btc_regime"] = 1
    btc.loc[(btc["Close"] > btc["btc_ma20"]) & (btc["btc_rsi14"] > 55), "btc_regime"] = 2

    # 강한 상승 모멘텀 (3일 5% 이상)
    btc["btc_strong_bull"] = (btc["btc_ret3"] > 5).astype(int)

    return btc.set_index("Date")[[
        "btc_ma20", "btc_ma60", "btc_rsi14",
        "btc_ret3", "btc_ret7", "btc_atr_pct",
        "btc_regime", "btc_strong_bull"
    ]]


def run_aggressive_bt(df, entry_fn, target_pct, stop_pct, hold_days,
                      unit_usd=1110.0,  # 기본 1.5x (740*1.5)
                      max_pyramid=3, pyramid_add_pct=5.0,
                      trailing_stop=False, fx=1350.0):
    """
    공격적 백테스트:
    - 피라미딩 지원 (최대 3레이어)
    - 트레일링 스탑 옵션
    - 각 진입 레이어별 추가 진입 조건: 이전 진입가 대비 +pyramid_add_pct%
    """
    trades, pos = [], []
    peak_price = None

    for _, row in df.iterrows():
        if pd.isna(row.get("ma20")):
            continue
        price, d = row["Close"], row["Date"]
        pma20 = row["pct_ma20"]
        has_pos = bool(pos)

        if has_pos:
            tq = sum(p[1] for p in pos)
            tc = sum(p[1] * p[2] for p in pos)
            avg = tc / tq
            pp = (price - avg) / avg * 100
            held = (d - pos[0][0]).days

            # 트레일링 스탑 업데이트
            if trailing_stop:
                if peak_price is None or price > peak_price:
                    peak_price = price
                trail_pct = (price - peak_price) / peak_price * 100
                should_exit = (pp >= target_pct or trail_pct <= stop_pct / 2
                               or held >= hold_days or pma20 < -25)
            else:
                should_exit = (pp >= target_pct or pp <= stop_pct
                               or held >= hold_days or pma20 < -30)

            if should_exit:
                pnl = tq * price - tc
                reason = "T" if pp >= target_pct else "S" if pp <= stop_pct else "X"
                trades.append({
                    "Date": d, "PnL_KRW": pnl * fx, "PnL_pct": pp,
                    "HeldDays": held, "Layers": len(pos), "Reason": reason
                })
                pos, peak_price = [], None
                continue

            # 피라미딩: 이전 마지막 진입가 대비 상승 시 추가 진입
            if len(pos) < max_pyramid:
                last_entry_price = pos[-1][2]
                if price > last_entry_price * (1 + pyramid_add_pct / 100) and entry_fn(row):
                    qty = unit_usd / price
                    pos.append((d, qty, price))

        # 신규 진입
        if not has_pos and entry_fn(row):
            qty = unit_usd / price
            pos.append((d, qty, price))
            peak_price = price

    if not trades:
        return {"n": 0, "pnl": 0, "wr": 0, "avg": 0, "avg_layers": 0}
    tdf = pd.DataFrame(trades)
    return {
        "n": len(tdf),
        "pnl": round(tdf["PnL_KRW"].sum()),
        "wr": round((tdf["PnL_KRW"] > 0).mean() * 100, 1),
        "avg": round(tdf["PnL_pct"].mean(), 2),
        "avg_layers": round(tdf["Layers"].mean(), 2)
    }


def main():
    print("[D] 공격적 집중 전략 실험 시작")
    ohlcv  = pd.read_parquet(OHLCV_PATH); ohlcv["Date"]  = pd.to_datetime(ohlcv["Date"])
    btc_df = pd.read_parquet(BTC_PATH);   btc_df["Date"] = pd.to_datetime(btc_df["Date"])

    btc_feat = build_btc_features(btc_df)

    results = []

    for t in FOCUS_TICKERS:
        sub = ohlcv[ohlcv["ticker"] == t].copy()
        if len(sub) < 60:
            continue
        sub = add_features(sub)
        sub = sub.merge(btc_feat.reset_index(), on="Date", how="left")
        for col in ["btc_ma20", "btc_ma60", "btc_rsi14", "btc_ret3",
                    "btc_ret7", "btc_regime", "btc_strong_bull"]:
            if col in sub.columns:
                sub[col] = sub[col].ffill()

        # ── 집중 전략 목록 ──────────────────────────────────────────────────────
        AGGRESSIVE_STRATEGIES = {
            # ── 브레이크아웃 + BTC 강세 ──
            "bo5_btc2":          lambda r: r["bo5"] == 1 and r.get("btc_regime", 0) == 2,
            "bo10_btc1plus":     lambda r: r["bo10"] == 1 and r.get("btc_regime", 0) >= 1,
            "bo10_vol_btc":      lambda r: r["bo10"] == 1 and r["vol_ratio10"] > 1.5 and r.get("btc_regime", 0) >= 1,
            "bo20_btc_strong":   lambda r: r["bo20"] == 1 and r.get("btc_strong_bull", 0) == 1,
            "bo10_macd_btc":     lambda r: r["bo10"] == 1 and r["macd_hist"] > 0 and r.get("btc_regime", 0) >= 1,

            # ── 급등 추격 ──
            "surge_follow_btc":  lambda r: r["surge_bull"] == 1 and r.get("btc_regime", 0) >= 1,
            "surge_strong_btc2": lambda r: r["surge_bull"] == 1 and r.get("btc_regime", 0) == 2 and r["rsi14"] < 75,
            "ret3_strong_btc":   lambda r: r["ret3"] > 10 and r.get("btc_regime", 0) >= 1,
            "ret5_surge_btc":    lambda r: r["ret5"] > 15 and r.get("btc_regime", 0) == 2,

            # ── 눌림목 매수 (강세장 내) ──
            "pullback_btc2":     lambda r: r["pullback_after_surge"] == 1 and r.get("btc_regime", 0) == 2,
            "dip_rsi_btc2":      lambda r: r["rsi14"] < 45 and r["pct_ma20"] > -10 and r.get("btc_regime", 0) == 2,
            "dip_ma5_btc2":      lambda r: r["pct_ma5"] < -3 and r["pct_ma20"] > 0 and r.get("btc_regime", 0) == 2,
            "bb_mid_btc2":       lambda r: 30 < r["bb_pct"] < 60 and r.get("btc_regime", 0) == 2 and r["rsi14"] > 45,

            # ── MA 정렬 + 강세 BTC ──
            "ma_align5_btc2":    lambda r: r["ma_aligned"] == 1 and r.get("btc_regime", 0) == 2,
            "ma_align60_btc2":   lambda r: r["ma_aligned60"] == 1 and r.get("btc_regime", 0) == 2 and r["rsi14"] > 55,
            "ma_btc_ret7":       lambda r: r["ma_aligned"] == 1 and r.get("btc_ret7", 0) > 7,

            # ── RSI 고강도 모멘텀 ──
            "rsi7_hot_btc2":     lambda r: r["rsi7"] > 70 and r.get("btc_regime", 0) == 2,
            "rsi14_65_btc2":     lambda r: r["rsi14"] > 65 and r.get("btc_regime", 0) == 2 and r["pct_ma20"] > 5,
            "rsi_accel":         lambda r: r["rsi14"] > r["rsi21"] and r["rsi14"] > 60 and r.get("btc_regime", 0) >= 1,

            # ── BTC 단기 강세 리드 ──
            "btc_ret3_5pct":     lambda r: r.get("btc_ret3", 0) > 5 and r["pct_ma20"] > 0 and r["rsi14"] > 50,
            "btc_ret3_8pct":     lambda r: r.get("btc_ret3", 0) > 8 and r["rsi14"] > 50,
            "btc_all_green":     lambda r: (r.get("btc_regime", 0) == 2 and r.get("btc_strong_bull", 0) == 1
                                            and r["rsi14"] > 55 and r["pct_ma20"] > 0),

            # ── 연속 상승 모멘텀 ──
            "consec3_btc2":      lambda r: r.get("consec_up", 0) >= 3 and r.get("btc_regime", 0) == 2,
            "consec2_vol_btc":   lambda r: r.get("consec_up", 0) >= 2 and r["vol_ratio20"] > 1.3 and r.get("btc_regime", 0) >= 1,

            # ── 복합 고강도 ──
            "ultra_bull":        lambda r: (r["bo10"] == 1 and r["ma_aligned60"] == 1
                                            and r.get("btc_regime", 0) == 2 and r["vol_ratio20"] > 1.2),
            "max_momentum":      lambda r: (r["rsi14"] > 65 and r["macd_hist"] > 0
                                            and r.get("btc_strong_bull", 0) == 1 and r["bo5"] == 1),
        }

        # 파라미터 그리드: 공격적 범위
        PARAM_GRID = [
            # (target, stop, hold, unit_usd_multiplier, max_pyramid, trailing)
            (25, -15, 20, 1.0,  1, False),
            (30, -20, 30, 1.0,  1, False),
            (40, -20, 30, 1.0,  1, False),
            (50, -25, 45, 1.0,  1, False),
            (30, -15, 20, 1.5,  1, False),
            (40, -20, 30, 1.5,  1, False),
            (30, -20, 30, 2.0,  1, False),
            (40, -20, 30, 2.0,  1, False),
            (50, -25, 45, 2.0,  1, False),
            # 피라미딩 (2레이어)
            (30, -20, 30, 1.0,  2, False),
            (40, -20, 30, 1.0,  2, False),
            (30, -20, 30, 1.5,  2, False),
            # 피라미딩 (3레이어)
            (40, -20, 45, 1.0,  3, False),
            (50, -25, 45, 1.0,  3, False),
            # 트레일링 스탑
            (30, -15, 30, 1.0,  1, True),
            (40, -20, 30, 1.5,  1, True),
            (50, -20, 45, 1.0,  2, True),
            # 단기 스윙
            (15, -10, 10, 1.0,  1, False),
            (20, -10, 10, 1.5,  1, False),
            (20, -12, 15, 2.0,  1, False),
        ]

        for strat_name, entry_fn in AGGRESSIVE_STRATEGIES.items():
            for (tgt, stp, hld, mul, max_pyr, trail) in PARAM_GRID:
                try:
                    unit = 740.0 * mul
                    r = run_aggressive_bt(
                        sub, entry_fn, tgt, stp, hld,
                        unit_usd=unit, max_pyramid=max_pyr,
                        trailing_stop=trail
                    )
                    if r["n"] >= 3:
                        results.append({
                            "ticker": t, "strategy": strat_name,
                            "target": tgt, "stop": stp, "hold": hld,
                            "unit_mul": mul, "max_pyr": max_pyr, "trailing": trail,
                            **r
                        })
                except Exception:
                    pass

    df_res = pd.DataFrame(results).sort_values("pnl", ascending=False)
    df_res.to_csv(OUT, index=False)
    print(f"\n[D] 결과 저장: {OUT}  ({len(df_res)}개 조합)")
    print("\nTOP 15:")
    print(df_res[[
        "ticker", "strategy", "target", "stop", "hold",
        "unit_mul", "max_pyr", "trailing", "pnl", "wr", "n", "avg"
    ]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()

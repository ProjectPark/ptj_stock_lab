"""
exp_L_hybrid.py — J 매크로 필터 + E 기술적 신호 하이브리드
=============================================================
핵심 아이디어:
- E 실험: 기술적 진입 신호 (DI크로스+급등, RSI가속, 연속상승) → WR 66.7% 달성
- J 실험: 실제 매매 기반 매크로 필터 추출 → BTC RSI<61, VIX 18~25, 눌림목
- K 실험: J 규칙만으로 진입 → WR 40~47% (E보다 낮음, 모멘텀 신호 없어서)

L 전략: J 매크로 조건(환경 필터) + E 기술적 신호(진입 타이밍) 결합
→ "좋은 환경에서 좋은 신호로만 진입" = 더 높은 WR 기대

핵심 인사이트:
1. btc_regime==2 조건 → btc_rsi14<65 + vix>17 로 교체 (더 넓은 커버리지)
   - J 발견: btc_regime 0 (약세장)에서도 81.6% WR 달성
   - btc_rsi14<61이 regime 레벨보다 더 중요한 필터
2. ticker_pct_ma20 ≥ 25% 구간 = 89.9% WR (J 최강 발견)
   → 추세 내 강한 종목에 기술적 신호 겹칠 때 최강
3. VIX 18~25 구간 = 91.1% WR (극공포/극탐욕 모두 피하기)
4. 풀백 후 반등 신호 = 실제 수익 패턴 (ret5<0이지만 기술적으로 반등 시작)
"""
import sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

OUT        = ROOT / "experiments" / "results" / "L_hybrid.csv"
OHLCV_PATH = ROOT / "data" / "market" / "daily" / "profit_curve_ohlcv.parquet"
BTC_PATH   = ROOT / "data" / "market" / "daily" / "btc_mstr_daily.parquet"
EXTRA_PATH = ROOT / "data" / "market" / "daily" / "extra_signals_daily.parquet"

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

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]      = ema12 - ema26
    df["macd_sig"]  = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]

    # ADX / DI
    high, low = df["High"], df["Low"]
    tr  = pd.concat([high-low,
                     (high-c.shift(1)).abs(),
                     (low-c.shift(1)).abs()], axis=1).max(axis=1)
    dmp = (high-high.shift(1)).clip(lower=0).where(
          (high-high.shift(1)) > (low.shift(1)-low), 0)
    dmm = (low.shift(1)-low).clip(lower=0).where(
          (low.shift(1)-low) > (high-high.shift(1)), 0)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    df["di_plus"]  = dmp.ewm(span=14, adjust=False).mean() / atr14 * 100
    df["di_minus"] = dmm.ewm(span=14, adjust=False).mean() / atr14 * 100
    df["adx"] = ((dmp.ewm(span=14,adjust=False).mean()/atr14 -
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

    # RSI 가속 (14 > 21 & > 55)
    df["rsi_accel"] = ((df["rsi14"] > df["rsi21"]) & (df["rsi14"] > 55)).astype(int)

    # 연속 상승
    up = (c > c.shift(1)).astype(int)
    df["consec_up"] = up.groupby(((up==0)|(up!=up.shift(1))).cumsum()).cumsum() * up

    # 급등 신호
    df["surge1"]  = ((c/c.shift(1)-1)*100 > 3).astype(int)
    df["surge3d"] = (df["ret3"] > 8).astype(int)
    df["surge5d"] = (df["ret5"] > 12).astype(int)

    # MA 정렬
    df["ma_aligned"] = ((df["ma5"] > df["ma10"]) &
                        (df["ma10"] > df["ma20"])).astype(int)

    # J 발견: 눌림목 패턴
    df["pullback_in_trend"] = (
        (df["pct_ma20"] > 10) & (df["ret5"] < 0)
    ).astype(int)

    df["pullback_strong_trend"] = (
        (df["pct_ma20"] > 20) & (df["ret5"] < 0)
    ).astype(int)

    df["pullback_deep"] = (
        (df["pct_ma20"] > 10) & (df["ret5"] < -5)
    ).astype(int)

    # J 발견: 강한 추세 구간 (89.9% WR)
    df["strong_trend_25"] = (df["pct_ma20"] > 25).astype(int)
    df["strong_trend_15"] = (df["pct_ma20"] > 15).astype(int)

    # 반등 시작 (눌림 후 당일 반등)
    df["bounce_start"] = (
        (df["ret5"] < 0) &
        (df["ret1"] > 0) &
        (df["pct_ma20"] > 10)
    ).astype(int)

    # VCP 수축
    df["range5"]    = df["High"].rolling(5).max()  - df["Low"].rolling(5).min()
    df["range20"]   = df["High"].rolling(20).max() - df["Low"].rolling(20).min()
    df["vcp_ratio"] = df["range5"] / df["range20"]

    return df


def build_macro(btc_df, extra_df):
    # BTC
    btc = btc_df[btc_df["ticker"]=="BTC-USD"].copy().sort_values("Date")
    c = btc["Close"]
    btc["btc_ma20"]   = c.rolling(20).mean()
    btc["btc_ma60"]   = c.rolling(60).mean()
    btc["btc_rsi14"]  = rsi(c, 14)
    btc["btc_rsi7"]   = rsi(c, 7)
    btc["btc_ret5"]   = c.pct_change(5) * 100
    btc["btc_ret10"]  = c.pct_change(10) * 100
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
    vix["vix_falling"] = (vix["Close"] < vix["Close"].shift(3)).astype(int)
    vix["vix_rising"]  = (vix["Close"] > vix["Close"].shift(3)).astype(int)

    # QQQ
    qqq = extra_df[extra_df["ticker"]=="QQQ"].copy().sort_values("Date")
    qc = qqq["Close"]
    qqq["qqq_ma20"]     = qc.rolling(20).mean()
    qqq["qqq_ma60"]     = qc.rolling(60).mean()
    qqq["qqq_rsi14"]    = rsi(qc, 14)
    qqq["qqq_ret10"]    = qc.pct_change(10) * 100
    qqq["qqq_pct_ma20"] = (qc - qqq["qqq_ma20"]) / qqq["qqq_ma20"] * 100
    qqq["qqq_bull"]     = (qc > qqq["qqq_ma60"]).astype(int)
    qqq["qqq_strong"]   = ((qc > qqq["qqq_ma20"]) & (qqq["qqq_rsi14"] > 55)).astype(int)

    macro = (
        btc.set_index("Date")[["btc_ma20","btc_ma60","btc_rsi14","btc_rsi7",
                                "btc_ret5","btc_ret10","btc_regime"]]
        .join(mstr.set_index("Date")[["mstr_rsi14","mstr_pct_ma20"]], how="outer")
        .join(vix.set_index("Date")[["Close","vix_falling","vix_rising"]]
              .rename(columns={"Close":"vix"}), how="outer")
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
                trades.append({"Date":d, "PnL_KRW":(tq*price-tc)*fx,
                               "PnL_pct":pp, "HeldDays":held, "Layers":len(pos)})
                pos, peak = [], None
                continue

            if len(pos) < max_pyramid:
                if price > pos[-1][2]*(1+pyramid_add_pct/100) and entry_fn(row):
                    pos.append((d, unit_usd*0.7/price, price))

        if not has_pos and entry_fn(row):
            pos.append((d, unit_usd/price, price))
            peak = price

    if not trades: return {"n":0,"pnl":0,"wr":0,"avg":0,"avg_layers":0}
    tdf = pd.DataFrame(trades)
    return {"n": len(tdf),
            "pnl":  round(tdf["PnL_KRW"].sum()),
            "wr":   round((tdf["PnL_KRW"]>0).mean()*100, 1),
            "avg":  round(tdf["PnL_pct"].mean(), 2),
            "avg_layers": round(tdf["Layers"].mean(), 2)}


def main():
    print("[L] 하이브리드 실험 시작: J 매크로 필터 × E 기술 신호")
    ohlcv  = pd.read_parquet(OHLCV_PATH); ohlcv["Date"]  = pd.to_datetime(ohlcv["Date"])
    extra  = pd.read_parquet(EXTRA_PATH); extra["Date"]  = pd.to_datetime(extra["Date"])
    btc_df = pd.read_parquet(BTC_PATH);  btc_df["Date"] = pd.to_datetime(btc_df["Date"])

    macro = build_macro(btc_df, extra)

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

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # HYBRID STRATEGIES
        # 구조: E 기술 신호 × J 매크로 환경 필터
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        STRATEGIES = {

            # ── [Group 1] E 챔피언 신호 + J 매크로 교체 ──────────────
            # btc_regime==2 → btc_rsi14<65 + vix>17 (커버리지 확대)

            # E 1위: DI크로스 + 급등 + BTC강세
            "e1_di_surge_j_macro": lambda r: (
                r["di_plus"] > r["di_minus"] and
                r["surge1"] == 1 and
                r.get("btc_rsi14", 99) < 65 and
                r.get("vix", 0) > 17
            ),

            # E 2위: RSI+MACD+DI + BTC강세
            "e2_macd_di_j_macro": lambda r: (
                r["rsi14"] > 55 and
                r["macd_hist"] > 0 and
                r["di_plus"] > r["di_minus"] and
                r.get("btc_rsi14", 99) < 65 and
                r.get("vix", 0) > 17
            ),

            # E 3위: RSI7>75 + pct_ma20>5 + BTC
            "e3_rsi7_80_j_macro": lambda r: (
                r["rsi7"] > 75 and
                r["pct_ma20"] > 5 and
                r.get("btc_rsi14", 99) < 65 and
                18 <= r.get("vix", 0) <= 28
            ),

            # E 4위: 연속3일+ 급등
            "e4_consec3_surge_j": lambda r: (
                r.get("consec_up", 0) >= 3 and
                r["surge3d"] == 1 and
                r.get("btc_rsi14", 99) < 65 and
                r.get("vix", 0) > 17
            ),

            # E 5위: RSI가속 + 거래량
            "e5_rsi_accel_vol_j": lambda r: (
                r["rsi_accel"] == 1 and
                r["vol_ratio"] > 1.2 and
                r.get("btc_rsi14", 99) < 65 and
                r.get("vix", 0) > 17
            ),

            # E 6위: 연속4일
            "e6_consec4_j": lambda r: (
                r.get("consec_up", 0) >= 4 and
                r.get("btc_rsi14", 99) < 65 and
                r.get("vix", 0) > 16
            ),

            # ── [Group 2] J 강한 추세 구간(pct_ma20>25%) + E 신호 ────
            # J 발견: ticker_pct_ma20 ≥ 25% = 89.9% WR 최강 구간

            "j_strong25_di_surge": lambda r: (
                r["pct_ma20"] > 25 and
                r["di_plus"] > r["di_minus"] and
                r["surge1"] == 1 and
                r.get("btc_rsi14", 99) < 70
            ),

            "j_strong25_rsi_accel": lambda r: (
                r["pct_ma20"] > 25 and
                r["rsi_accel"] == 1 and
                r.get("btc_rsi14", 99) < 70 and
                r.get("vix", 0) > 16
            ),

            "j_strong25_macd_di": lambda r: (
                r["pct_ma20"] > 25 and
                r["macd_hist"] > 0 and
                r["di_plus"] > r["di_minus"] and
                r.get("btc_rsi14", 99) < 70
            ),

            "j_strong25_consec": lambda r: (
                r["pct_ma20"] > 25 and
                r.get("consec_up", 0) >= 3 and
                r.get("btc_rsi14", 99) < 70 and
                r.get("vix", 0) > 16
            ),

            "j_strong25_vol_surge": lambda r: (
                r["pct_ma20"] > 25 and
                r["vol_ratio"] > 1.5 and
                r["surge1"] == 1 and
                r.get("btc_rsi14", 99) < 70
            ),

            # ── [Group 3] VIX 18~25 + E 기술 신호 ───────────────────
            # J 발견: VIX 18~25 중립구간 = 91.1% WR

            "j_vix_neutral_di_surge": lambda r: (
                18 <= r.get("vix", 0) <= 25 and
                r["di_plus"] > r["di_minus"] and
                r["surge1"] == 1
            ),

            "j_vix_neutral_rsi80": lambda r: (
                18 <= r.get("vix", 0) <= 25 and
                r["rsi7"] > 75 and
                r["pct_ma20"] > 5
            ),

            "j_vix_neutral_macd_di": lambda r: (
                18 <= r.get("vix", 0) <= 25 and
                r["macd_hist"] > 0 and
                r["di_plus"] > r["di_minus"] and
                r["rsi14"] > 55
            ),

            "j_vix_neutral_consec": lambda r: (
                18 <= r.get("vix", 0) <= 25 and
                r.get("consec_up", 0) >= 3 and
                r["surge3d"] == 1
            ),

            "j_vix_neutral_rsi_accel": lambda r: (
                18 <= r.get("vix", 0) <= 25 and
                r["rsi_accel"] == 1 and
                r["vol_ratio"] > 1.2
            ),

            # ── [Group 4] 눌림 후 반등 × E 신호 ─────────────────────
            # J 발견: ret5<0이지만 기술적 반등 신호 겹침 = 최적 진입

            "l_pullback_di_surge": lambda r: (
                r["pullback_in_trend"] == 1 and
                r["di_plus"] > r["di_minus"] and
                r["surge1"] == 1 and
                r.get("btc_rsi14", 99) < 65
            ),

            "l_pullback_macd_accel": lambda r: (
                r["pullback_in_trend"] == 1 and
                r["macd_hist"] > 0 and
                r["rsi_accel"] == 1 and
                r.get("btc_rsi14", 99) < 65
            ),

            "l_bounce_di_j": lambda r: (
                r["bounce_start"] == 1 and
                r["di_plus"] > r["di_minus"] and
                r.get("btc_rsi14", 99) < 65 and
                r.get("vix", 0) > 17
            ),

            "l_bounce_vol_surge_j": lambda r: (
                r["bounce_start"] == 1 and
                r["vol_ratio"] > 1.5 and
                r["surge1"] == 1 and
                r.get("btc_rsi14", 99) < 65
            ),

            "l_pullback_strong_di": lambda r: (
                r["pullback_strong_trend"] == 1 and  # pct_ma20>20 + ret5<0
                r["di_plus"] > r["di_minus"] and
                r.get("btc_rsi14", 99) < 65
            ),

            # ── [Group 5] 최강 융합 (모든 조건 중 베스트) ───────────

            # E 챔피언 신호 + J 매크로 + J 강한 추세
            "l_champion_full": lambda r: (
                r["di_plus"] > r["di_minus"] and
                r["surge1"] == 1 and
                r["pct_ma20"] > 10 and            # 추세 확인
                r.get("btc_rsi14", 99) < 65 and
                r.get("vix", 0) > 17
            ),

            # VIX 18~25 + BTC RSI<61 + E 강한 신호
            "l_max_conviction": lambda r: (
                18 <= r.get("vix", 0) <= 25 and
                r.get("btc_rsi14", 99) < 61 and
                r["pct_ma20"] > 15 and
                r["di_plus"] > r["di_minus"] and
                r["rsi_accel"] == 1
            ),

            # VIX + BTC RSI + DI크로스 + 거래량 (조건 균형)
            "l_balanced_4way": lambda r: (
                r.get("vix", 0) > 18 and
                r.get("btc_rsi14", 99) < 65 and
                r["di_plus"] > r["di_minus"] and
                r["vol_ratio"] > 1.3 and
                r["rsi14"] > 55
            ),

            # QQQ 강세 + J 매크로 + E 신호 (전체 마켓 강세)
            "l_qqq_j_macro_e": lambda r: (
                r.get("qqq_bull", 0) == 1 and
                r.get("btc_rsi14", 99) < 65 and
                r.get("vix", 0) > 17 and
                r["di_plus"] > r["di_minus"] and
                r["surge1"] == 1
            ),

            # MSTR MA20 아래 + J 매크로 + E 강한 신호
            "l_mstr_below_e_signal": lambda r: (
                r.get("mstr_pct_ma20", 0) < 2 and
                r.get("btc_rsi14", 99) < 65 and
                r.get("vix", 0) > 17 and
                (r["di_plus"] > r["di_minus"] or r["rsi_accel"] == 1) and
                r["pct_ma20"] > 10
            ),

            # ── [Group 6] 완화 버전 (더 많은 거래 수) ───────────────
            # WR보다 N이 많아서 총 수익이 큰 버전

            "l_wide_btc_rsi_e": lambda r: (
                r.get("btc_rsi14", 99) < 70 and
                r.get("vix", 0) > 16 and
                r["di_plus"] > r["di_minus"] and
                r["rsi14"] > 55
            ),

            "l_wide_vix_consec": lambda r: (
                r.get("vix", 0) > 16 and
                r.get("btc_rsi14", 99) < 70 and
                r.get("consec_up", 0) >= 3 and
                r["pct_ma20"] > 5
            ),

            "l_wide_macd_vol": lambda r: (
                r["macd_hist"] > 0 and
                r["vol_ratio"] > 1.4 and
                r.get("btc_rsi14", 99) < 70 and
                r.get("vix", 0) > 16
            ),
        }

        # ── 파라미터 그리드 ────────────────────────────────────────────
        # E 챔피언 파라미터 중심 + 다양한 변형
        # (target, stop, hold, unit_mul, max_pyr, trailing)
        PARAMS = [
            # E 챔피언 (T80/S-25/H90 3x pyr3) — 기본
            (80, -25,  90, 3.0, 3, None),
            # E 변형
            (60, -25,  60, 3.0, 3, None),
            (50, -25,  45, 3.0, 3, None),
            (40, -20,  30, 3.0, 3, None),
            # 5x 단타 공격
            (50, -25,  45, 5.0, 1, None),
            (40, -20,  30, 5.0, 1, None),
            (30, -15,  20, 5.0, 1, None),
            # 피라미딩 강화
            (80, -25,  90, 2.0, 5, None),
            (60, -25,  60, 2.0, 5, None),
            # 트레일링 (수익 보호)
            (80, -25,  90, 3.0, 3, -12.0),
            (60, -25,  60, 3.0, 3, -10.0),
            (50, -25,  45, 3.0, 3, -10.0),
            # 중형 안정
            (40, -20,  30, 2.0, 3, None),
            (50, -25,  45, 2.0, 3, None),
            (30, -20,  30, 3.0, 1, None),
            # 장기 홀딩
            (100, -30, 120, 2.0, 3, None),
            (80, -25,  90, 2.0, 3, -15.0),
        ]

        for strat_name, entry_fn in STRATEGIES.items():
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
    print(f"\n[L] 결과 저장: {OUT}  ({len(df_res)}개 조합)")

    print("\nTOP 30:")
    print(df_res[[
        "ticker","strategy","target","stop","hold","unit_mul","max_pyr","trailing",
        "pnl","wr","n","avg","avg_layers"
    ]].head(30).to_string(index=False))

    # 전략 그룹별 베스트
    print("\n\n=== 전략별 MAX PnL ===")
    best_by_strat = df_res.groupby("strategy").agg(
        max_pnl=("pnl","max"),
        avg_wr=("wr","mean"),
        total_n=("n","sum")
    ).sort_values("max_pnl", ascending=False)
    print(best_by_strat.head(30).to_string())

    # 종목별 베스트
    print("\n\n=== 종목별 MAX PnL ===")
    best_by_ticker = df_res.groupby("ticker").agg(
        max_pnl=("pnl","max"),
        best_strategy=("strategy","first"),
        avg_wr=("wr","mean")
    ).sort_values("max_pnl", ascending=False)
    print(best_by_ticker.to_string())

    # E 챔피언과 비교
    print("\n\n=== E 챔피언 (33.5M) 대비 L 실험 성과 ===")
    over_e = df_res[df_res["pnl"] > 33_500_000]
    print(f"  33.5M 초과 조합: {len(over_e)}개")
    if len(over_e) > 0:
        print(over_e[[
            "ticker","strategy","target","stop","hold","unit_mul","max_pyr",
            "pnl","wr","n","avg"
        ]].head(10).to_string(index=False))

    print(f"\n  전체 최고: {df_res['pnl'].max():,.0f}원 ({df_res['pnl'].max()/10000:.1f}만원)")


if __name__ == "__main__":
    main()

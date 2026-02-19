"""
exp_J_real_trade_analysis.py — 실제 거래 기반 신호 추출
=======================================================
실제 매수일의 시장 조건을 역추적:
- BTC 레짐/RSI/수익률
- 종목 RSI/MA위치
- VIX 수준
- QQQ/MSTR 상태

수익 거래 vs 손해 거래의 조건 분포를 비교해서
실제로 어떤 조건에서 진입했을 때 수익이 났는지 추출
"""
import sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

OUT_ANALYSIS = ROOT / "experiments" / "results" / "J_real_trade_conditions.csv"
OUT_RULES    = ROOT / "experiments" / "results" / "J_extracted_rules.csv"
OUT_BACKTEST = ROOT / "experiments" / "results" / "J_rule_backtest.csv"

OHLCV_PATH = ROOT / "data" / "market" / "daily" / "profit_curve_ohlcv.parquet"
BTC_PATH   = ROOT / "data" / "market" / "daily" / "btc_mstr_daily.parquet"
EXTRA_PATH = ROOT / "data" / "market" / "daily" / "extra_signals_daily.parquet"

# 한국어 종목명 → 영문 티커 매핑
TICKER_MAP = {
    "T-REX 스트래티지 타겟 2배 ETF":         "MSTU",
    "T-REX 스트래티지 타겟 2배 E":           "MSTU",
    "T-REX 스트래티지 타겟 2배 ET":          "MSTU",
    "T-REX 스트래티지 타겟":                 "MSTU",
    "T-REX 스트래티지 데일리 타겟":           "MSTU",
    "스트래티지":                             "MSTU",
    "그래닛셰어즈 코인베이스 데일리 2배 롱 ETF": "CONL",
    "그래닛셰어즈 코인베이스 데일리":           "CONL",
    "그래닛셰어즈 코인베이스 데일":            "CONL",
    "그래닛셰어즈 코인베이스":                 "CONL",
    "코인베이스":                             "CONL",
    "일드맥스 코인베이스 옵션":                "CONL",
    "일드맥스 코인베이스 옵션 배당 ETF":        "CONL",
    "T-REX 로빈후드 데일리 타겟 2배 ETF":     "ROBN",
    "그래닛셰어즈 엔비디아 데일리 2배 롱 ETF":  "NVDL",
    "디렉시온 미국 반도체 3배 ETF":            "SOXL",
    "디렉시온 미국 반도체":                    "SOXL",
    "그래닛셰어즈 AMD 데일리 2배 ETF":        "AMDL",
    "디렉시온 테슬라 2배 ETF":               "TSLL",
    "디렉시온 테슬라 2배 E":                 "TSLL",
    "디렉시온 테슬라 2배":                    "TSLL",
    "T-REX 테슬라 데일리":                   "TSLL",
    "일드맥스 테슬라 옵션 인컴":               "TSLL",
    "프로셰어즈 QQQ 3배 ETF":               "TQQQ",
    "프로셰어즈 QQQ 3배 ET":               "TQQQ",
    "프로셰어즈 QQQ 3배":                   "TQQQ",
    "디렉시온 데일리 팔란티어 2배 롱 ETF":     "PLTR2",
    "레버리지 셰어즈 팔란티어 데일리 2배 ETF":  "PLTR2",
    "디파이언스 스트래티지 2배 롱 ETF":        "MSTX",
    "디파이언스 스트래티지 2배 롱 ET":         "MSTX",
    "디파이언스 스트래티지 2배 롱":            "MSTX",
    "디파이언스 스트래티":                     "MSTX",
    "디파이언스 데일리 타겟 2배 롱 아이렌 ETF": "IREN",
    "마이크로섹터 미국 은행 3배 ETN":          "BNKU",
    "마이크로섹터 금광 3배 ETN":              "GDXU",
    "프로셰어즈 비트코인 선물 ETF":            "BITU",
    "프로셰어즈 비트코인 선물":               "BITU",
    "비트코인 전략 2배 ETF":                 "BITX",
    "비트코인 전략 2배 ETF(U":              "BITX",
    "비트코인 전략 2배 ET":                  "BITX",
    "이더리움 2배 ETF":                     "ETHU",
    "이더리움 2배 ETF(US92864":            "ETHU",
    "튜크리움 엑스알피(리플) 데일리 2배 롱 ETF": "XXRP",
    "디렉시온 TSMC 2배 ETF":               "TSM2",
    "프로셰어즈 울트라 써클 인터넷 그룹 2배 ETF": "FDL",
    "그래닛셰어즈 알리바바 데일리 2배 롱 ETF":   "BABX",
    "JP모건 커버드콜 옵션 ETF":              "JEPI",
    "JP모건 커버드콜 옵션 E":               "JEPI",
    "JP모건 나스닥 프리미엄":                "JEPQ",
    "디렉시온 세계 금광 2배 ETF":            "NUGT",
    "디렉시온 넷플릭스 2배 ETF":             "NFLX2",
    "디파이언스 아이온큐 2배 롱 ETF":         "IQNQ",
    "프로셰어즈 트러스트 울트라 비":            "UVIX",
    "프로셰어즈 트러스트":                    "UVIX",
    "디렉시온 테슬라 2배 E":               "TSLL",
    "그래닛셰어즈 애플 데일":               "AAPB",
}

# BTC 관련 종목 (크립토 레짐 적용)
BTC_RELATED = {"MSTU","CONL","ROBN","MSTX","IREN","BITX","BITU","ETHU","XXRP"}
# 반도체/기술 종목
SEMI_RELATED = {"NVDL","AMDL","SOXL","TQQQ","TSLL","PLTR2"}


def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def build_macro(btc_df, extra_df):
    """날짜 인덱스 기반 매크로 신호 테이블"""
    # BTC
    btc = btc_df[btc_df["ticker"]=="BTC-USD"].copy().sort_values("Date")
    c = btc["Close"]
    btc["btc_ma20"]   = c.rolling(20).mean()
    btc["btc_ma60"]   = c.rolling(60).mean()
    btc["btc_rsi14"]  = rsi(c, 14)
    btc["btc_ret3"]   = c.pct_change(3)  * 100
    btc["btc_ret10"]  = c.pct_change(10) * 100
    btc["btc_ret20"]  = c.pct_change(20) * 100
    btc["btc_pct_ma20"] = (c - btc["btc_ma20"]) / btc["btc_ma20"] * 100
    btc["btc_regime"] = 0
    btc.loc[c > btc["btc_ma60"], "btc_regime"] = 1
    btc.loc[(c > btc["btc_ma20"]) & (btc["btc_rsi14"] > 55), "btc_regime"] = 2
    btc.loc[(c > btc["btc_ma20"]) & (btc["btc_rsi14"] > 65) & (btc["btc_ret10"] > 15), "btc_regime"] = 3

    # MSTR
    mstr = btc_df[btc_df["ticker"]=="MSTR"].copy().sort_values("Date")
    mc = mstr["Close"]
    mstr["mstr_ma20"]     = mc.rolling(20).mean()
    mstr["mstr_rsi14"]    = rsi(mc, 14)
    mstr["mstr_pct_ma20"] = (mc - mstr["mstr_ma20"]) / mstr["mstr_ma20"] * 100
    mstr["mstr_ret5"]     = mc.pct_change(5) * 100

    # QQQ
    qqq = extra_df[extra_df["ticker"]=="QQQ"].copy().sort_values("Date")
    qc = qqq["Close"]
    qqq["qqq_ma20"]   = qc.rolling(20).mean()
    qqq["qqq_ma60"]   = qc.rolling(60).mean()
    qqq["qqq_rsi14"]  = rsi(qc, 14)
    qqq["qqq_ret10"]  = qc.pct_change(10) * 100
    qqq["qqq_pct_ma20"] = (qc - qqq["qqq_ma20"]) / qqq["qqq_ma20"] * 100
    qqq["qqq_bull"]   = (qc > qqq["qqq_ma60"]).astype(int)

    # VIX
    vix = extra_df[extra_df["ticker"]=="VIX"].copy().sort_values("Date")
    vix["vix_ma10"] = vix["Close"].rolling(10).mean()

    # 통합
    macro = (btc.set_index("Date")[["btc_ma20","btc_ma60","btc_rsi14","btc_ret3",
                                     "btc_ret10","btc_ret20","btc_pct_ma20","btc_regime"]]
             .join(mstr.set_index("Date")[["mstr_rsi14","mstr_pct_ma20","mstr_ret5"]], how="outer")
             .join(qqq.set_index("Date")[["qqq_rsi14","qqq_pct_ma20","qqq_ret10","qqq_bull"]], how="outer")
             .join(vix.set_index("Date")[["Close","vix_ma10"]].rename(columns={"Close":"vix"}), how="outer")
            )
    return macro.ffill()


def get_ticker_features_at_date(ohlcv_dict, ticker, date):
    """특정 종목의 특정 날짜 기술적 지표"""
    if ticker not in ohlcv_dict:
        return {}
    df = ohlcv_dict[ticker]
    # date 이전 데이터만 사용 (미래 데이터 오염 방지)
    sub = df[df["Date"] <= date].copy()
    if len(sub) < 20:
        return {}
    c = sub["Close"]
    ma20 = c.rolling(20).mean().iloc[-1]
    ma60 = c.rolling(60).mean().iloc[-1] if len(sub) >= 60 else np.nan
    rsi14 = rsi(c, 14).iloc[-1]
    pct_ma20 = (c.iloc[-1] - ma20) / ma20 * 100 if not pd.isna(ma20) else np.nan
    ret5  = c.pct_change(5).iloc[-1]  * 100 if len(sub) >= 5  else np.nan
    ret10 = c.pct_change(10).iloc[-1] * 100 if len(sub) >= 10 else np.nan
    vol_ratio = (sub["Volume"].iloc[-1] /
                 sub["Volume"].rolling(20).mean().iloc[-1]) if "Volume" in sub.columns else np.nan
    return {
        "ticker_price":    c.iloc[-1],
        "ticker_rsi14":    round(rsi14, 1),
        "ticker_pct_ma20": round(pct_ma20, 2),
        "ticker_ret5":     round(ret5, 2),
        "ticker_ret10":    round(ret10, 2),
        "ticker_vol_ratio":round(vol_ratio, 2),
    }


def main():
    print("[J] 실제 거래 기반 신호 추출 시작")

    # 데이터 로드
    profit = pd.read_csv(ROOT / "history" / "수익_거래내역.csv")
    loss   = pd.read_csv(ROOT / "history" / "손해_거래내역.csv")
    profit["result"] = "profit"; loss["result"] = "loss"
    trades = pd.concat([profit, loss], ignore_index=True)
    trades["매수일"] = pd.to_datetime(trades["최초매수일"])
    trades["매도일"] = pd.to_datetime(trades["판매일자"])

    # 티커 매핑
    trades["ticker"] = trades["종목명"].map(TICKER_MAP).fillna("UNKNOWN")
    trades["sector"] = trades["ticker"].apply(
        lambda t: "BTC" if t in BTC_RELATED else
                  "SEMI" if t in SEMI_RELATED else "OTHER"
    )

    # OHLCV 로드
    ohlcv_raw  = pd.read_parquet(OHLCV_PATH); ohlcv_raw["Date"]  = pd.to_datetime(ohlcv_raw["Date"])
    extra_raw  = pd.read_parquet(EXTRA_PATH); extra_raw["Date"]  = pd.to_datetime(extra_raw["Date"])
    btc_raw    = pd.read_parquet(BTC_PATH);   btc_raw["Date"]    = pd.to_datetime(btc_raw["Date"])

    # 종목별 OHLCV 딕셔너리
    ohlcv_dict = {}
    for t in ohlcv_raw["ticker"].unique():
        ohlcv_dict[t] = ohlcv_raw[ohlcv_raw["ticker"]==t].reset_index(drop=True)
    for t in extra_raw["ticker"].unique():
        ohlcv_dict[t] = extra_raw[extra_raw["ticker"]==t].reset_index(drop=True)

    # 매크로 신호 빌드
    macro = build_macro(btc_raw, extra_raw)
    print(f"  매크로 신호 준비 완료: {macro.shape}")

    # ── 각 거래별 매수 시점 조건 추출 ──────────────────────────────────────
    records = []
    for _, row in trades.iterrows():
        buy_date = row["매수일"]
        ticker   = row["ticker"]
        result   = row["result"]
        pnl_pct  = row["손익률_%"]
        held_days = (row["매도일"] - buy_date).days

        # 매크로 조건 (매수일 기준)
        macro_row = macro.loc[macro.index <= buy_date].iloc[-1] if len(macro.loc[macro.index <= buy_date]) > 0 else {}

        # 종목 기술적 조건
        ticker_feat = get_ticker_features_at_date(ohlcv_dict, ticker, buy_date)

        rec = {
            "매수일":       buy_date,
            "매도일":       row["매도일"],
            "ticker":      ticker,
            "sector":      row["sector"],
            "result":      result,
            "pnl_pct":     pnl_pct,
            "held_days":   held_days,
        }

        # 매크로 조건 추가
        if len(macro_row) > 0:
            rec.update({
                "btc_regime":      macro_row.get("btc_regime", np.nan),
                "btc_rsi14":       macro_row.get("btc_rsi14", np.nan),
                "btc_pct_ma20":    macro_row.get("btc_pct_ma20", np.nan),
                "btc_ret10":       macro_row.get("btc_ret10", np.nan),
                "mstr_rsi14":      macro_row.get("mstr_rsi14", np.nan),
                "mstr_pct_ma20":   macro_row.get("mstr_pct_ma20", np.nan),
                "qqq_rsi14":       macro_row.get("qqq_rsi14", np.nan),
                "qqq_pct_ma20":    macro_row.get("qqq_pct_ma20", np.nan),
                "qqq_bull":        macro_row.get("qqq_bull", np.nan),
                "vix":             macro_row.get("vix", np.nan),
            })

        # 종목 조건 추가
        rec.update(ticker_feat)
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(OUT_ANALYSIS, index=False)
    print(f"\n  거래별 조건 추출 완료: {len(df)}건")

    # ── 수익/손해 조건 분포 비교 ─────────────────────────────────────────
    profit_df = df[df["result"]=="profit"]
    loss_df   = df[df["result"]=="loss"]

    print("\n" + "="*70)
    print("수익 vs 손해 — 매수 시점 조건 비교")
    print("="*70)

    metrics = ["btc_regime","btc_rsi14","btc_pct_ma20","btc_ret10",
               "mstr_pct_ma20","qqq_rsi14","qqq_pct_ma20","vix",
               "ticker_rsi14","ticker_pct_ma20","ticker_ret5","ticker_vol_ratio"]

    comparison = []
    for m in metrics:
        if m not in df.columns: continue
        p_val = profit_df[m].dropna()
        l_val = loss_df[m].dropna()
        comparison.append({
            "지표":     m,
            "수익_중앙값": round(p_val.median(), 2),
            "손해_중앙값": round(l_val.median(), 2),
            "수익_평균":  round(p_val.mean(), 2),
            "손해_평균":  round(l_val.mean(), 2),
            "차이":      round(p_val.median() - l_val.median(), 2),
        })

    comp_df = pd.DataFrame(comparison).sort_values("차이", key=abs, ascending=False)
    print(comp_df.to_string(index=False))

    # ── BTC 레짐별 수익률 분포 ────────────────────────────────────────────
    print("\n" + "="*70)
    print("BTC 레짐별 수익/손해 비율")
    print("="*70)
    regime_stat = df.groupby(["btc_regime","result"]).size().unstack(fill_value=0)
    regime_stat["총계"] = regime_stat.sum(axis=1)
    if "profit" in regime_stat.columns:
        regime_stat["수익률%"] = (regime_stat["profit"] / regime_stat["총계"] * 100).round(1)
    print(regime_stat)

    # ── VIX 구간별 분석 ───────────────────────────────────────────────────
    print("\n" + "="*70)
    print("VIX 구간별 수익/손해")
    print("="*70)
    df["vix_band"] = pd.cut(df["vix"],
                             bins=[0, 15, 20, 25, 30, 40, 100],
                             labels=["<15(극탐욕)","15~20(탐욕)","20~25(중립)","25~30(불안)","30~40(공포)",">40(극공포)"])
    vix_stat = df.groupby(["vix_band","result"]).size().unstack(fill_value=0)
    vix_stat["총계"] = vix_stat.sum(axis=1)
    if "profit" in vix_stat.columns:
        vix_stat["수익률%"] = (vix_stat["profit"] / vix_stat["총계"] * 100).round(1)
    print(vix_stat)

    # ── 종목별 수익/손해 분포 ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("주요 종목별 수익/손해 비율")
    print("="*70)
    ticker_stat = df.groupby(["ticker","result"]).size().unstack(fill_value=0)
    ticker_stat["총계"] = ticker_stat.sum(axis=1)
    if "profit" in ticker_stat.columns:
        ticker_stat["수익률%"] = (ticker_stat["profit"] / ticker_stat["총계"] * 100).round(1)
        ticker_stat["평균손익%"] = df.groupby("ticker")["pnl_pct"].mean().round(2)
    ticker_stat = ticker_stat[ticker_stat["총계"] >= 3].sort_values("수익률%", ascending=False)
    print(ticker_stat)

    # ── 조건 임계값 탐색 (이진 분류) ─────────────────────────────────────
    print("\n" + "="*70)
    print("최적 진입 조건 임계값 탐색")
    print("="*70)

    rules = []
    for col in ["btc_regime","btc_rsi14","btc_pct_ma20","mstr_pct_ma20",
                "qqq_rsi14","vix","ticker_rsi14","ticker_pct_ma20"]:
        if col not in df.columns: continue
        col_data = df[col].dropna()
        for thresh in col_data.quantile([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]).values:
            for direction in ["above","below"]:
                mask = (df[col] >= thresh) if direction=="above" else (df[col] < thresh)
                sub  = df[mask]
                if len(sub) < 5: continue
                wr   = (sub["result"]=="profit").mean() * 100
                n    = len(sub)
                rules.append({
                    "조건": f"{col} {'≥' if direction=='above' else '<'} {thresh:.1f}",
                    "승률%": round(wr, 1),
                    "N": n,
                    "평균손익%": round(sub["pnl_pct"].mean(), 2),
                })

    rules_df = pd.DataFrame(rules).sort_values("승률%", ascending=False)
    rules_df.to_csv(OUT_RULES, index=False)
    print(f"\n  Top 20 조건 (승률 기준):")
    print(rules_df.head(20).to_string(index=False))

    # ── 복합 조건 (수익 공통 패턴) ────────────────────────────────────────
    print("\n" + "="*70)
    print("복합 조건 분석 — 수익 공통 패턴")
    print("="*70)

    combos = []
    for btc_reg in [1, 2, 3]:
        for vix_max in [20, 25, 30]:
            for mstr_min in [-5, 0, 5, 10]:
                mask = ((df["btc_regime"] >= btc_reg) &
                        (df["vix"] < vix_max) &
                        (df["mstr_pct_ma20"] >= mstr_min))
                sub = df[mask]
                if len(sub) < 5: continue
                wr  = (sub["result"]=="profit").mean() * 100
                combos.append({
                    "조건": f"BTC레짐≥{btc_reg} + VIX<{vix_max} + MSTR_MA20≥{mstr_min}%",
                    "승률%": round(wr, 1),
                    "N": len(sub),
                    "평균손익%": round(sub["pnl_pct"].mean(), 2),
                })

    # QQQ 추가
    for btc_reg in [1, 2]:
        for qqq_min in [0, 2, 5]:
            for vix_max in [20, 25]:
                mask = ((df["btc_regime"] >= btc_reg) &
                        (df["qqq_pct_ma20"] >= qqq_min) &
                        (df["vix"] < vix_max))
                sub = df[mask]
                if len(sub) < 5: continue
                wr = (sub["result"]=="profit").mean() * 100
                combos.append({
                    "조건": f"BTC레짐≥{btc_reg} + QQQ_MA20≥{qqq_min}% + VIX<{vix_max}",
                    "승률%": round(wr, 1),
                    "N": len(sub),
                    "평균손익%": round(sub["pnl_pct"].mean(), 2),
                })

    combos_df = pd.DataFrame(combos).sort_values("승률%", ascending=False).drop_duplicates()
    print(combos_df.head(20).to_string(index=False))

    # ── 연도/분기별 패턴 ─────────────────────────────────────────────────
    print("\n" + "="*70)
    print("연도별 수익률")
    print("="*70)
    df["연도"] = df["매수일"].dt.year
    df["분기"] = df["매수일"].dt.to_period("Q").astype(str)
    yr_stat = df.groupby(["연도","result"]).size().unstack(fill_value=0)
    yr_stat["총계"] = yr_stat.sum(axis=1)
    if "profit" in yr_stat.columns:
        yr_stat["수익률%"] = (yr_stat["profit"] / yr_stat["총계"] * 100).round(1)
        yr_stat["총손익원"] = df.groupby("연도")["pnl_pct"].sum().round(0)
    print(yr_stat)

    print(f"\n[J] 분석 완료")
    print(f"  조건 분석: {OUT_ANALYSIS}")
    print(f"  추출 규칙: {OUT_RULES}")


if __name__ == "__main__":
    main()

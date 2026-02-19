"""
exp_G_rotation_portfolio.py — 포트폴리오 회전 & 상대강도 전략
=============================================================
지금까지 안 한 새 접근법:
1. 상대강도(RS) 회전: 매주 가장 강한 종목 1개만 보유
2. 동적 포지션 스케일링: 확신도에 따라 0.5x ~ 3x
3. BTC 레짐 전환 시 즉시 포트 리밸런스
4. 섹터 모멘텀 (BTC 관련 vs 반도체 관련 분류)
5. Mean-Reversion vs Momentum 자동 전환
6. 최대 낙폭(MDD) 제한 기반 포지션 축소
"""
import sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

OUT = ROOT / "experiments" / "results" / "G_rotation_portfolio.csv"
OHLCV_PATH = ROOT / "data" / "market" / "daily" / "profit_curve_ohlcv.parquet"
BTC_PATH   = ROOT / "data" / "market" / "daily" / "btc_mstr_daily.parquet"

BTC_TICKERS  = ["IREN", "PTIR", "CONL", "MSTU"]       # BTC 연동 레버리지
SEMI_TICKERS = ["NVDL", "AMDL"]                        # 반도체 레버리지
ALL_TICKERS  = BTC_TICKERS + SEMI_TICKERS


def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def build_universe(ohlcv, btc_df):
    """모든 종목 일별 피처 계산 후 wide 포맷으로 반환"""
    btc = btc_df[btc_df["ticker"]=="BTC-USD"].copy().sort_values("Date")
    c = btc["Close"]
    btc["btc_ma20"]  = c.rolling(20).mean()
    btc["btc_ma60"]  = c.rolling(60).mean()
    btc["btc_rsi"]   = rsi(c, 14)
    btc["btc_ret10"] = c.pct_change(10) * 100
    btc["btc_regime"] = 0
    btc.loc[c > btc["btc_ma60"], "btc_regime"] = 1
    btc.loc[(c > btc["btc_ma20"]) & (btc["btc_rsi"] > 55), "btc_regime"] = 2
    btc_feat = btc.set_index("Date")[["btc_regime","btc_ret10","btc_rsi"]]

    records = {}
    for t in ALL_TICKERS:
        sub = ohlcv[ohlcv["ticker"]==t].copy().sort_values("Date")
        if len(sub) < 60: continue
        c2 = sub["Close"]
        sub["ma20"]     = c2.rolling(20).mean()
        sub["rsi14"]    = rsi(c2, 14)
        sub["pct_ma20"] = (c2 - sub["ma20"]) / sub["ma20"] * 100
        sub["ret5"]     = c2.pct_change(5) * 100
        sub["ret10"]    = c2.pct_change(10) * 100
        sub["ret20"]    = c2.pct_change(20) * 100
        sub["vol_ratio"]= sub["Volume"] / sub["Volume"].rolling(20).mean()

        # 상대강도 (RS): 20일 수익률 기준
        sub["rs_score"] = sub["ret20"].rank(pct=True) * 100

        sub = sub.merge(btc_feat.reset_index(), on="Date", how="left")
        sub["btc_regime"] = sub["btc_regime"].ffill().fillna(0)
        sub["btc_ret10"]  = sub["btc_ret10"].ffill().fillna(0)
        sub["btc_rsi"]    = sub["btc_rsi"].ffill().fillna(50)
        records[t] = sub.set_index("Date")

    return records


# ── 전략 1: 상대강도 회전 (주간) ─────────────────────────────────────────────
def run_rs_rotation(records, rebal_days=5, top_n=1,
                    target_pct=30, stop_pct=-20, unit_usd=1480.0,
                    require_btc_bull=True, fx=1350.0):
    """
    매 rebal_days마다 RS 가장 강한 top_n 종목으로 교체
    BTC 약세 시 전량 현금
    """
    # 공통 날짜 생성
    all_dates = sorted(set.union(*[set(records[t].index) for t in records]))
    all_dates = pd.DatetimeIndex(all_dates)

    trades = []
    pos = {}      # {ticker: (entry_date, qty, entry_price)}
    day_count = 0

    for d in all_dates:
        # BTC 레짐 확인
        btc_reg = 0
        for t in records:
            if d in records[t].index:
                btc_reg = records[t].loc[d, "btc_regime"]
                break

        # 현재 포지션 청산 체크
        to_close = []
        for t, (entry_d, qty, entry_p) in pos.items():
            if d not in records[t].index: continue
            price = records[t].loc[d, "Close"]
            pp = (price - entry_p) / entry_p * 100
            held = (d - entry_d).days

            # BTC 약세 → 청산 or 목표/손절
            should_exit = (pp >= target_pct or pp <= stop_pct
                           or (require_btc_bull and btc_reg == 0))
            if should_exit:
                pnl = qty * price - qty * entry_p
                trades.append({"Date":d, "Ticker":t, "PnL_KRW":pnl*fx,
                                "PnL_pct":pp, "HeldDays":held})
                to_close.append(t)

        for t in to_close:
            del pos[t]

        # 리밸런스 주기
        day_count += 1
        if day_count % rebal_days != 0:
            continue

        if require_btc_bull and btc_reg == 0:
            continue  # 약세장: 진입 없음

        # RS 스코어로 종목 랭킹
        scores = {}
        for t in records:
            if d not in records[t].index: continue
            row = records[t].loc[d]
            if pd.isna(row.get("ma20")): continue
            if require_btc_bull and row.get("btc_regime",0) < 1: continue
            scores[t] = row.get("ret20", 0)   # 20일 수익률 = RS 대용

        if not scores: continue
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        targets = [t for t, s in ranked[:top_n] if s > 0]  # 수익률 양수만

        # 새 포지션 진입
        for t in targets:
            if t in pos: continue   # 이미 보유
            if d not in records[t].index: continue
            price = records[t].loc[d, "Close"]
            qty = unit_usd / price
            pos[t] = (d, qty, price)

    if not trades: return {"n":0,"pnl":0,"wr":0,"avg":0}
    tdf = pd.DataFrame(trades)
    return {"n":len(tdf), "pnl":round(tdf["PnL_KRW"].sum()),
            "wr":round((tdf["PnL_KRW"]>0).mean()*100,1),
            "avg":round(tdf["PnL_pct"].mean(),2)}


# ── 전략 2: BTC 레짐별 섹터 전환 ─────────────────────────────────────────────
def run_sector_rotation(records, target_pct=30, stop_pct=-20,
                        unit_usd=1480.0, fx=1350.0):
    """
    BTC 레짐2 → BTC 관련 (IREN/CONL) 집중
    BTC 레짐1 → 혼합
    BTC 레짐0 → 반도체 (NVDL/AMDL) or 현금
    """
    all_dates = sorted(set.union(*[set(records[t].index) for t in records]))
    trades = []
    pos = {}

    for d in pd.DatetimeIndex(all_dates):
        btc_reg = 0
        for t in records:
            if d in records[t].index:
                btc_reg = int(records[t].loc[d, "btc_regime"])
                break

        # 레짐별 투자 대상
        if btc_reg == 2:
            target_pool = ["IREN","PTIR","CONL"]
        elif btc_reg == 1:
            target_pool = ["IREN","CONL","NVDL"]
        else:
            target_pool = []   # 현금

        # 포지션 청산
        to_close = []
        for t, (entry_d, qty, entry_p) in pos.items():
            if d not in records[t].index: continue
            price = records[t].loc[d, "Close"]
            pp = (price - entry_p) / entry_p * 100
            held = (d - entry_d).days
            if (pp >= target_pct or pp <= stop_pct
                    or t not in target_pool):  # 섹터 벗어나면 청산
                pnl = qty * price - qty * entry_p
                trades.append({"Date":d,"Ticker":t,"PnL_KRW":pnl*fx,
                                "PnL_pct":pp,"HeldDays":held})
                to_close.append(t)
        for t in to_close: del pos[t]

        # 신규 진입 (풀에서 RS 1위)
        if not target_pool: continue
        best_t, best_rs = None, -999
        for t in target_pool:
            if t not in records or d not in records[t].index: continue
            if t in pos: continue
            rs = records[t].loc[d, "ret20"] if not pd.isna(records[t].loc[d].get("ret20",np.nan)) else -999
            if rs > best_rs:
                best_rs, best_t = rs, t
        if best_t and best_rs > 0:
            price = records[best_t].loc[d, "Close"]
            pos[best_t] = (d, unit_usd/price, price)

    if not trades: return {"n":0,"pnl":0,"wr":0,"avg":0}
    tdf = pd.DataFrame(trades)
    return {"n":len(tdf), "pnl":round(tdf["PnL_KRW"].sum()),
            "wr":round((tdf["PnL_KRW"]>0).mean()*100,1),
            "avg":round(tdf["PnL_pct"].mean(),2)}


# ── 전략 3: 동적 확신도 사이징 ──────────────────────────────────────────────
def run_conviction_sizing(records, target_pct=30, stop_pct=-20,
                          hold_days=30, fx=1350.0):
    """
    BTC 레짐 + RSI + 거래량 → 확신도 점수 (0~10)
    점수에 비례해 포지션 크기 결정
    """
    all_dates = sorted(set.union(*[set(records[t].index) for t in records]))
    all_trades = []

    for t in records:
        sub_dates = [d for d in all_dates if d in records[t].index]
        pos = []
        for d in pd.DatetimeIndex(sub_dates):
            row = records[t].loc[d]
            if pd.isna(row.get("ma20")): continue
            price = row["Close"]
            pma20 = row.get("pct_ma20", 0)
            has_pos = bool(pos)

            if has_pos:
                tq = sum(p[1] for p in pos); tc = sum(p[1]*p[2] for p in pos)
                avg = tc/tq; pp = (price-avg)/avg*100
                held = (d - pos[0][0]).days
                if pp >= target_pct or pp <= stop_pct or held >= hold_days or pma20 < -30:
                    pnl = tq*price - tc
                    all_trades.append({"Date":d,"Ticker":t,"PnL_KRW":pnl*fx,
                                       "PnL_pct":pp,"HeldDays":held})
                    pos = []; continue

            if not has_pos:
                btc_reg = row.get("btc_regime", 0)
                rsi14   = row.get("rsi14", 50)
                vol_r   = row.get("vol_ratio", 1.0)
                pma20_v = pma20

                # 확신도 점수 계산
                score = 0
                score += btc_reg * 2           # 레짐: 0/2/4
                score += 1 if rsi14 > 60 else (0.5 if rsi14 > 50 else 0)
                score += 1 if vol_r > 1.5 else (0.5 if vol_r > 1.2 else 0)
                score += 1 if pma20_v > 10 else (0.5 if pma20_v > 0 else 0)

                if score >= 3:  # 최소 확신도
                    unit = 740.0 * (score / 4.0)  # 점수 비례 (3점→555, 7점→1295)
                    unit = min(unit, 740*3)        # 최대 3x 캡
                    pos.append((d, unit/price, price))

    if not all_trades: return {"n":0,"pnl":0,"wr":0,"avg":0}
    tdf = pd.DataFrame(all_trades)
    return {"n":len(tdf), "pnl":round(tdf["PnL_KRW"].sum()),
            "wr":round((tdf["PnL_KRW"]>0).mean()*100,1),
            "avg":round(tdf["PnL_pct"].mean(),2)}


def main():
    print("[G] 포트폴리오 회전 & 상대강도 전략 실험 시작")
    ohlcv  = pd.read_parquet(OHLCV_PATH); ohlcv["Date"]  = pd.to_datetime(ohlcv["Date"])
    btc_df = pd.read_parquet(BTC_PATH);   btc_df["Date"] = pd.to_datetime(btc_df["Date"])

    records = build_universe(ohlcv, btc_df)
    print(f"  로드된 종목: {list(records.keys())}")

    results = []

    # ── 전략 1: RS 회전 그리드 ──────────────────────────────────────────────
    print("  [1] RS 회전 그리드 탐색...")
    for rebal in [3, 5, 7, 10]:
        for top_n in [1, 2]:
            for (tgt, stp) in [(20,-15),(25,-20),(30,-20),(40,-20),(50,-25)]:
                for mul in [1.0, 2.0, 3.0]:
                    for btc_bull in [True, False]:
                        try:
                            r = run_rs_rotation(records, rebal_days=rebal, top_n=top_n,
                                                target_pct=tgt, stop_pct=stp,
                                                unit_usd=740*mul,
                                                require_btc_bull=btc_bull)
                            if r["n"] >= 3:
                                results.append({
                                    "strategy": "rs_rotation",
                                    "rebal_days": rebal, "top_n": top_n,
                                    "target": tgt, "stop": stp,
                                    "unit_mul": mul, "btc_bull": btc_bull, **r
                                })
                        except Exception: pass

    # ── 전략 2: 섹터 회전 그리드 ──────────────────────────────────────────
    print("  [2] 섹터 회전 그리드 탐색...")
    for (tgt, stp) in [(20,-15),(25,-20),(30,-20),(40,-20),(50,-25),(60,-25)]:
        for mul in [1.0, 1.5, 2.0, 3.0]:
            try:
                r = run_sector_rotation(records, target_pct=tgt, stop_pct=stp,
                                        unit_usd=740*mul)
                if r["n"] >= 3:
                    results.append({
                        "strategy": "sector_rotation",
                        "rebal_days": 0, "top_n": 1,
                        "target": tgt, "stop": stp,
                        "unit_mul": mul, "btc_bull": True, **r
                    })
            except Exception: pass

    # ── 전략 3: 확신도 사이징 그리드 ─────────────────────────────────────
    print("  [3] 확신도 동적 사이징...")
    for (tgt, stp, hld) in [(20,-15,20),(25,-20,30),(30,-20,30),(40,-20,30),(50,-25,45)]:
        try:
            r = run_conviction_sizing(records, target_pct=tgt, stop_pct=stp,
                                      hold_days=hld)
            if r["n"] >= 3:
                results.append({
                    "strategy": "conviction_sizing",
                    "rebal_days": hld, "top_n": 0,
                    "target": tgt, "stop": stp,
                    "unit_mul": "dynamic", "btc_bull": True, **r
                })
        except Exception: pass

    df_res = pd.DataFrame(results).sort_values("pnl", ascending=False)
    df_res.to_csv(OUT, index=False)
    print(f"\n[G] 결과 저장: {OUT}  ({len(df_res)}개 조합)")
    print("\nTOP 20:")
    print(df_res[["strategy","rebal_days","top_n","target","stop","unit_mul",
                   "btc_bull","pnl","wr","n","avg"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()

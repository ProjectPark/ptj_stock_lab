"""
optimize_di_surge_optuna.py — di_cross_surge 전략 Optuna 최적화 + OOS 검증
=========================================================================
Base strategy: di_cross_surge (DI크로스 + 급등 + 매크로 조건)
  - E 챔피언: DI+>DI- + surge1 + btc_regime==2 → IREN 33.5M, WR 66.7%

최적화 대상:
  1) 신호 파라미터 (DI 갭, 급등%, BTC 조건, VIX 범위, 거래량, pct_ma20)
  2) 청산 파라미터 (목표%, 손절%, 보유일)
  3) 포지션 사이징 (배율, 피라미딩 횟수/비율)

분할:
  - Train:  2023-09-01 ~ 2025-12-31 (최적화용)
  - Test:   2026-01-01 ~ 2026-02-19 (OOS 검증, 미사용 데이터)

최적화 목적함수:
  - 학습 구간 총 PnL 최대화
  - N < 5이면 페널티 (-1,000,000원)
  - WR < 40% 이면 추가 페널티 (저품질 고수익 방지)
"""
import sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import optuna
from datetime import datetime

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ── 데이터 경로 ─────────────────────────────────────────────────────────
OHLCV_PATH = ROOT / "data" / "market" / "daily" / "profit_curve_ohlcv.parquet"
BTC_PATH   = ROOT / "data" / "market" / "daily" / "btc_mstr_daily.parquet"
EXTRA_PATH = ROOT / "data" / "market" / "daily" / "extra_signals_daily.parquet"
OUT_CSV    = ROOT / "experiments" / "results" / "M_optuna_best.csv"

# ── 최적화 대상 종목 (E, L 실험 최강) ──────────────────────────────────
OPT_TICKERS  = ["IREN", "CONL", "PTIR"]   # 학습 구간 최적화
TRAIN_END    = pd.Timestamp("2025-12-31")
TEST_START   = pd.Timestamp("2026-01-01")
TEST_END     = pd.Timestamp("2026-02-19")
N_TRIALS     = 500                          # Optuna 시도 횟수


# ──────────────────────────────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────────────────────────────
def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def add_features(df):
    df = df.copy().sort_values("Date").reset_index(drop=True)
    c  = df["Close"]

    df["ma5"]  = c.rolling(5).mean()
    df["ma10"] = c.rolling(10).mean()
    df["ma20"] = c.rolling(20).mean()
    df["ma60"] = c.rolling(60).mean()
    df["pct_ma20"] = (c - df["ma20"]) / df["ma20"] * 100

    df["rsi7"]  = rsi(c, 7)
    df["rsi14"] = rsi(c, 14)
    df["rsi21"] = rsi(c, 21)
    df["ret1"]  = c.pct_change(1)  * 100
    df["ret3"]  = c.pct_change(3)  * 100
    df["ret5"]  = c.pct_change(5)  * 100

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd_hist"] = (ema12-ema26) - (ema12-ema26).ewm(span=9,adjust=False).mean()

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

    # 거래량
    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # 연속 상승
    up = (c > c.shift(1)).astype(int)
    df["consec_up"] = up.groupby(((up==0)|(up!=up.shift(1))).cumsum()).cumsum() * up

    # 눌림목
    df["pullback_in_trend"] = (
        (df["pct_ma20"] > 10) & (df["ret5"] < 0)
    ).astype(int)

    return df


def build_macro(btc_df, extra_df):
    # BTC
    btc = btc_df[btc_df["ticker"]=="BTC-USD"].copy().sort_values("Date")
    c   = btc["Close"]
    btc["btc_ma20"]   = c.rolling(20).mean()
    btc["btc_ma60"]   = c.rolling(60).mean()
    btc["btc_rsi14"]  = rsi(c, 14)
    btc["btc_ret5"]   = c.pct_change(5) * 100
    btc["btc_regime"] = 0
    btc.loc[c > btc["btc_ma60"], "btc_regime"] = 1
    btc.loc[(c > btc["btc_ma20"]) & (btc["btc_rsi14"] > 55), "btc_regime"] = 2

    # MSTR
    mstr = btc_df[btc_df["ticker"]=="MSTR"].copy().sort_values("Date")
    mc   = mstr["Close"]
    mstr["mstr_pct_ma20"] = (mc - mc.rolling(20).mean()) / mc.rolling(20).mean() * 100

    # VIX
    vix = extra_df[extra_df["ticker"]=="VIX"].copy().sort_values("Date")
    vix["vix_falling"] = (vix["Close"] < vix["Close"].shift(3)).astype(int)

    # QQQ
    qqq = extra_df[extra_df["ticker"]=="QQQ"].copy().sort_values("Date")
    qc  = qqq["Close"]
    qqq["qqq_ma20"]   = qc.rolling(20).mean()
    qqq["qqq_ma60"]   = qc.rolling(60).mean()
    qqq["qqq_rsi14"]  = rsi(qc, 14)
    qqq["qqq_pct_ma20"] = (qc - qqq["qqq_ma20"]) / qqq["qqq_ma20"] * 100
    qqq["qqq_bull"]   = (qc > qqq["qqq_ma60"]).astype(int)

    macro = (
        btc.set_index("Date")[["btc_ma20","btc_ma60","btc_rsi14","btc_ret5","btc_regime"]]
        .join(mstr.set_index("Date")[["mstr_pct_ma20"]], how="outer")
        .join(vix.set_index("Date")[["Close","vix_falling"]].rename(
              columns={"Close":"vix"}), how="outer")
        .join(qqq.set_index("Date")[["qqq_rsi14","qqq_pct_ma20","qqq_bull"]], how="outer")
    )
    return macro.ffill()


# ──────────────────────────────────────────────────────────────────────
# Backtest Engine (파라미터 주입 방식)
# ──────────────────────────────────────────────────────────────────────
def make_entry_fn(p: dict):
    """파라미터 딕셔너리 → 진입 함수 생성"""
    def entry(r):
        # 1) DI 크로스 (가장 중요한 기술 신호)
        if (r["di_plus"] - r["di_minus"]) < p["di_min_gap"]:
            return False
        # 2) 급등 (당일 상승률)
        if r["ret1"] < p["surge_pct"]:
            return False
        # 3) BTC RSI (과열 여부)
        if r.get("btc_rsi14", 99) > p["btc_rsi_max"]:
            return False
        # 4) BTC MA20 위 (선택적)
        if p["btc_above_ma20"] and r.get("btc_regime", 0) < 1:
            return False
        # 5) VIX 범위
        vix = r.get("vix", 20)
        if vix < p["vix_min"] or vix > p["vix_max"]:
            return False
        # 6) 종목 추세 강도
        if r["pct_ma20"] < p["pct_ma20_min"]:
            return False
        # 7) 거래량 (선택적)
        if r["vol_ratio"] < p["vol_ratio_min"]:
            return False
        # 8) RSI 추가 조건 (선택적)
        if p["rsi14_min"] > 0 and r["rsi14"] < p["rsi14_min"]:
            return False
        # 9) QQQ 강세 (선택적)
        if p["qqq_bull_required"] and r.get("qqq_bull", 0) == 0:
            return False
        return True
    return entry


def run_backtest(df, entry_fn, p, fx=1350.0):
    """백테스트 실행 — 파라미터 딕셔너리 기반"""
    unit_usd    = 740 * p["unit_mul"]
    target_pct  = p["target_pct"]
    stop_pct    = p["stop_pct"]
    hold_days   = p["hold_days"]
    max_pyramid = p["max_pyramid"]
    pyr_add_pct = p["pyramid_add_pct"]
    trailing    = p.get("trailing_pct", None)

    trades, pos, peak = [], [], None

    for _, row in df.iterrows():
        if pd.isna(row.get("ma20")): continue
        price, d = row["Close"], row["Date"]
        has_pos  = bool(pos)

        if has_pos:
            tq  = sum(x[1] for x in pos)
            tc  = sum(x[1]*x[2] for x in pos)
            avg = tc / tq
            pp  = (price - avg) / avg * 100
            held = (d - pos[0][0]).days

            if peak is None or price > peak: peak = price
            trail_hit = (trailing is not None and peak is not None and
                         (price - peak) / peak * 100 <= trailing)

            if pp >= target_pct or pp <= stop_pct or held >= hold_days or trail_hit or row["pct_ma20"] < -35:
                trades.append({"Date": d, "Entry": avg, "Exit": price,
                               "PnL_KRW": (tq*price - tc)*fx,
                               "PnL_pct": pp, "HeldDays": held,
                               "Layers": len(pos), "ticker": row.get("ticker","")})
                pos, peak = [], None
                continue

            if len(pos) < max_pyramid and price > pos[-1][2] * (1 + pyr_add_pct/100):
                if entry_fn(row):
                    pos.append((d, unit_usd * 0.7 / price, price))

        if not has_pos and entry_fn(row):
            pos.append((d, unit_usd / price, price))
            peak = price

    if not trades:
        return {"n": 0, "pnl": 0, "wr": 0, "avg": 0, "trades": pd.DataFrame()}
    tdf = pd.DataFrame(trades)
    return {"n":   len(tdf),
            "pnl": round(tdf["PnL_KRW"].sum()),
            "wr":  round((tdf["PnL_KRW"] > 0).mean() * 100, 1),
            "avg": round(tdf["PnL_pct"].mean(), 2),
            "trades": tdf}


# ──────────────────────────────────────────────────────────────────────
# Optuna Objective
# ──────────────────────────────────────────────────────────────────────
_TRAIN_DATA = {}    # 전역 캐시 (trial마다 재로드 방지)

def objective(trial, tickers):
    p = {
        # ── 신호 파라미터 ─────────────────────────────────────────────
        "di_min_gap":       trial.suggest_float("di_min_gap",       -3.0, 10.0),
        "surge_pct":        trial.suggest_float("surge_pct",         1.0,  6.0),
        "btc_rsi_max":      trial.suggest_float("btc_rsi_max",      45.0, 80.0),
        "btc_above_ma20":   trial.suggest_categorical("btc_above_ma20", [True, False]),
        "vix_min":          trial.suggest_float("vix_min",          12.0, 24.0),
        "vix_max":          trial.suggest_float("vix_max",          20.0, 40.0),
        "pct_ma20_min":     trial.suggest_float("pct_ma20_min",    -10.0, 25.0),
        "vol_ratio_min":    trial.suggest_float("vol_ratio_min",     0.5,  2.5),
        "rsi14_min":        trial.suggest_float("rsi14_min",         0.0, 70.0),
        "qqq_bull_required":trial.suggest_categorical("qqq_bull_required", [True, False]),
        # ── 청산 파라미터 ─────────────────────────────────────────────
        "target_pct":       trial.suggest_float("target_pct",       20.0, 100.0),
        "stop_pct":         trial.suggest_float("stop_pct",        -35.0, -10.0),
        "hold_days":        trial.suggest_int(  "hold_days",         10,   120),
        "trailing_pct":     trial.suggest_categorical(
                                "trailing_pct", [None, -8.0, -12.0, -15.0, -20.0]),
        # ── 포지션 사이징 ─────────────────────────────────────────────
        "unit_mul":         trial.suggest_float("unit_mul",          1.0,  5.0),
        "max_pyramid":      trial.suggest_int(  "max_pyramid",        1,    5),
        "pyramid_add_pct":  trial.suggest_float("pyramid_add_pct",   4.0, 15.0),
    }

    # VIX 범위 sanity check
    if p["vix_min"] >= p["vix_max"]:
        return -1_000_000

    entry_fn = make_entry_fn(p)
    total_pnl = 0
    total_n   = 0
    total_wins = 0

    for t in tickers:
        if t not in _TRAIN_DATA:
            continue
        df = _TRAIN_DATA[t]
        r  = run_backtest(df, entry_fn, p)
        total_pnl  += r["pnl"]
        total_n    += r["n"]
        total_wins += int(r["n"] * r["wr"] / 100) if r["n"] > 0 else 0

    # 페널티: 거래 횟수 너무 적음
    if total_n < 5:
        return -1_000_000

    # 페널티: WR 너무 낮음 (과최적화 방지)
    overall_wr = total_wins / total_n * 100 if total_n > 0 else 0
    if overall_wr < 35:
        return total_pnl * 0.3  # 강력 패널티

    return total_pnl


# ──────────────────────────────────────────────────────────────────────
# OOS Validation 상세 출력
# ──────────────────────────────────────────────────────────────────────
def oos_validation(best_params, oos_data):
    """2026-01-01 ~ 2026-02-19 OOS 검증 + 거래 상세"""
    print("\n" + "="*70)
    print("  OOS 검증: 2026-01-01 ~ 2026-02-19 (미사용 데이터)")
    print("="*70)

    entry_fn = make_entry_fn(best_params)
    all_trades = []

    for t, df in oos_data.items():
        r = run_backtest(df, entry_fn, best_params)
        status = "✅" if r["pnl"] > 0 else ("⚠️" if r["n"] == 0 else "❌")
        print(f"\n  {status} {t:6s}: {r['n']}거래, WR={r['wr']}%, avg={r['avg']}%, "
              f"PnL={r['pnl']:>12,.0f}원")

        if not r["trades"].empty:
            print(f"     {'날짜':12s} {'진입가':>8s} {'청산가':>8s} "
                  f"{'수익%':>7s} {'보유일':>5s} {'층수':>3s}")
            print(f"     {'-'*55}")
            for _, row in r["trades"].iterrows():
                sign = "+" if row["PnL_pct"] > 0 else ""
                print(f"     {str(row['Date'].date()):12s} "
                      f"${row['Entry']:>8.2f} ${row['Exit']:>8.2f} "
                      f"{sign}{row['PnL_pct']:>6.1f}% "
                      f"{int(row['HeldDays']):>5d}일 "
                      f"{int(row['Layers']):>3d}층")
            all_trades.append(r["trades"])

    if all_trades:
        combined = pd.concat(all_trades)
        total_pnl = combined["PnL_KRW"].sum()
        wr = (combined["PnL_KRW"] > 0).mean() * 100
        print(f"\n  ─── OOS 합계 ───────────────────────────────────────")
        print(f"  총 거래: {len(combined)}건  WR: {wr:.1f}%  총 PnL: {total_pnl:,.0f}원")
    else:
        print(f"\n  OOS 구간에서 진입 신호 없음 (조건이 너무 엄격하거나 기간이 짧음)")


# ──────────────────────────────────────────────────────────────────────
# 신호 발생 일자 추적 (미진입된 날 포함)
# ──────────────────────────────────────────────────────────────────────
def show_signal_dates(best_params, oos_data):
    """OOS 구간에서 신호가 발생한 모든 날짜 출력"""
    print("\n" + "="*70)
    print("  OOS 신호 발생 일자 (진입 조건 충족일)")
    print("="*70)
    entry_fn = make_entry_fn(best_params)

    for t, df in oos_data.items():
        signal_rows = df[df.apply(entry_fn, axis=1)]
        if signal_rows.empty:
            print(f"  {t:6s}: 신호 없음")
        else:
            dates = signal_rows["Date"].dt.date.tolist()
            print(f"  {t:6s}: {len(dates)}회 신호")
            for d in dates:
                row = signal_rows[signal_rows["Date"].dt.date == d].iloc[0]
                print(f"    {d}  Close={row['Close']:.2f}  DI+={row['di_plus']:.1f}/"
                      f"DI-={row['di_minus']:.1f}  ret1={row['ret1']:.1f}%  "
                      f"pct_ma20={row['pct_ma20']:.1f}%  "
                      f"btc_rsi={row.get('btc_rsi14',0):.1f}  "
                      f"vix={row.get('vix',0):.1f}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("[M] di_cross_surge Optuna 최적화 시작")
    print(f"    Train: 2023-09-01 ~ 2025-12-31")
    print(f"    OOS:   2026-01-01 ~ 2026-02-19")
    print(f"    Trials: {N_TRIALS}\n")

    # ── 데이터 로드 ────────────────────────────────────────────────────
    ohlcv  = pd.read_parquet(OHLCV_PATH); ohlcv["Date"]  = pd.to_datetime(ohlcv["Date"])
    extra  = pd.read_parquet(EXTRA_PATH); extra["Date"]  = pd.to_datetime(extra["Date"])
    btc_df = pd.read_parquet(BTC_PATH);  btc_df["Date"] = pd.to_datetime(btc_df["Date"])

    macro = build_macro(btc_df, extra)

    ohlcv_all = pd.concat([
        ohlcv,
        extra[extra["ticker"].isin(["MSTX"])].copy()
    ], ignore_index=True)

    def prep_ticker(t, date_start, date_end):
        sub = ohlcv_all[ohlcv_all["ticker"]==t].copy()
        if len(sub) < 30: return None
        sub = add_features(sub)
        sub = sub.merge(macro.reset_index().rename(columns={"index":"Date"}),
                        on="Date", how="left")
        for col in macro.columns:
            if col in sub.columns:
                sub[col] = sub[col].ffill()
        sub["ticker"] = t
        mask = (sub["Date"] >= date_start) & (sub["Date"] <= date_end)
        return sub[mask].copy()

    # ── 학습 데이터 준비 ─────────────────────────────────────────────
    TRAIN_START = pd.Timestamp("2023-09-01")
    print("=== 학습 데이터 준비 ===")
    for t in OPT_TICKERS:
        df = prep_ticker(t, TRAIN_START, TRAIN_END)
        if df is not None and len(df) >= 30:
            _TRAIN_DATA[t] = df
            print(f"  {t}: {df['Date'].min().date()} ~ {df['Date'].max().date()} ({len(df)}일)")

    # ── OOS 데이터 준비 ──────────────────────────────────────────────
    print("\n=== OOS 데이터 준비 (2026-01 ~ 02-19) ===")
    oos_data = {}
    for t in OPT_TICKERS:
        df = prep_ticker(t, TEST_START, TEST_END)
        if df is not None and len(df) > 0:
            oos_data[t] = df
            print(f"  {t}: {df['Date'].min().date()} ~ {df['Date'].max().date()} ({len(df)}일)")

    # ── E 챔피언 베이스라인 먼저 측정 ───────────────────────────────
    print("\n=== E 챔피언 베이스라인 (btc_regime==2 버전) ===")
    baseline_params = {
        "di_min_gap": 0.0, "surge_pct": 3.0, "btc_rsi_max": 80.0,
        "btc_above_ma20": True, "vix_min": 12.0, "vix_max": 40.0,
        "pct_ma20_min": -20.0, "vol_ratio_min": 0.5, "rsi14_min": 0.0,
        "qqq_bull_required": False,
        "target_pct": 80.0, "stop_pct": -25.0, "hold_days": 90,
        "trailing_pct": None, "unit_mul": 3.0, "max_pyramid": 3,
        "pyramid_add_pct": 7.0,
    }
    # btc_regime==2: BTC above MA20 + RSI>55 → btc_above_ma20=True + btc_rsi_max=80 근사
    # 실제 E 챔피언 신호 별도 체크
    bl_total = 0
    for t, df in _TRAIN_DATA.items():
        # E 오리지널 신호 직접 테스트
        def e_entry(r):
            return (r["di_plus"] > r["di_minus"] and
                    r["ret1"] > 3.0 and
                    r.get("btc_regime", 0) == 2)
        r = run_backtest(df, e_entry, baseline_params)
        bl_total += r["pnl"]
        print(f"  {t}: N={r['n']}, WR={r['wr']}%, avg={r['avg']}%, PnL={r['pnl']:,}원")
    print(f"  ─── 베이스라인 합계: {bl_total:,}원 ({bl_total/10000:.1f}만원) ───")

    # ── Optuna 최적화 ────────────────────────────────────────────────
    print(f"\n=== Optuna 최적화 시작 ({N_TRIALS} trials) ===")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=50)
    )

    study.optimize(
        lambda trial: objective(trial, list(_TRAIN_DATA.keys())),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )

    best  = study.best_trial
    bp    = best.params
    print(f"\n✅ 최적화 완료")
    print(f"   Best PnL (Train): {best.value:,}원 ({best.value/10000:.1f}만원)")

    # ── 최적 파라미터 출력 ───────────────────────────────────────────
    print("\n=== 최적 파라미터 ===")
    signal_params = ["di_min_gap","surge_pct","btc_rsi_max","btc_above_ma20",
                     "vix_min","vix_max","pct_ma20_min","vol_ratio_min",
                     "rsi14_min","qqq_bull_required"]
    exit_params   = ["target_pct","stop_pct","hold_days","trailing_pct"]
    pos_params    = ["unit_mul","max_pyramid","pyramid_add_pct"]

    print("  [신호]")
    for k in signal_params:
        print(f"    {k:25s}: {bp.get(k)}")
    print("  [청산]")
    for k in exit_params:
        print(f"    {k:25s}: {bp.get(k)}")
    print("  [포지션]")
    for k in pos_params:
        print(f"    {k:25s}: {bp.get(k)}")

    # ── 학습 구간 검증 ───────────────────────────────────────────────
    print("\n=== 학습 구간 최종 성과 (Train) ===")
    entry_fn = make_entry_fn(bp)
    train_total = 0
    train_trades_list = []
    for t, df in _TRAIN_DATA.items():
        r = run_backtest(df, entry_fn, bp)
        train_total += r["pnl"]
        print(f"  {t}: N={r['n']}, WR={r['wr']}%, avg={r['avg']}%, PnL={r['pnl']:,}원")
        if not r["trades"].empty:
            r["trades"]["ticker"] = t
            train_trades_list.append(r["trades"])
    print(f"  ─── Train 합계: {train_total:,}원 ({train_total/10000:.1f}만원) ───")
    print(f"  ─── 베이스라인 대비: {(train_total - bl_total)/10000:+.1f}만원 ───")

    # ── OOS 검증 (핵심!) ─────────────────────────────────────────────
    oos_validation(bp, oos_data)

    # ── OOS 신호 발생 일자 ───────────────────────────────────────────
    show_signal_dates(bp, oos_data)

    # ── 결과 저장 ────────────────────────────────────────────────────
    rows = []
    for t, df in {**_TRAIN_DATA, **oos_data}.items():
        period = "train" if t in _TRAIN_DATA else "oos"
        r = run_backtest(df, entry_fn, bp)
        rows.append({"period": period, "ticker": t, **r,
                     **{f"param_{k}": v for k,v in bp.items()}})
    df_out = pd.DataFrame([{k: v for k,v in row.items() if k != "trades"}
                            for row in rows])
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\n결과 저장: {OUT_CSV}")

    # ── Top trials 요약 ──────────────────────────────────────────────
    print("\n=== Top 10 Trials ===")
    top_trials = sorted(study.trials, key=lambda t: t.value or -9e9, reverse=True)[:10]
    for i, tr in enumerate(top_trials):
        if tr.value is None: continue
        print(f"  #{i+1:2d}  PnL={tr.value/10000:.1f}만원  "
              f"surge={tr.params.get('surge_pct',0):.1f}%  "
              f"di_gap={tr.params.get('di_min_gap',0):.1f}  "
              f"btc_rsi<{tr.params.get('btc_rsi_max',0):.0f}  "
              f"T{tr.params.get('target_pct',0):.0f}/"
              f"S{tr.params.get('stop_pct',0):.0f}/"
              f"H{tr.params.get('hold_days',0)}  "
              f"{tr.params.get('unit_mul',0):.1f}x  "
              f"pyr{tr.params.get('max_pyramid',0)}")


if __name__ == "__main__":
    main()

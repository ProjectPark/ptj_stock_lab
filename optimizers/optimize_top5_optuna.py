"""
optimize_top5_optuna.py — 상위 전략 Optuna 최적화 (전략별 실행)
================================================================
Usage:
  python optimize_top5_optuna.py --strategy vix_di_surge
  python optimize_top5_optuna.py --strategy macd_vol
  python optimize_top5_optuna.py --strategy btc_rsi_e
  python optimize_top5_optuna.py --strategy strong25_macd

각 전략:
  vix_di_surge   ← j_vix_neutral_di_surge  (21.4M, WR100%, N=3)
  macd_vol       ← l_wide_macd_vol          (19.8M, WR57%, N=7)
  btc_rsi_e      ← l_wide_btc_rsi_e         (18.9M, WR71%, N=7)
  strong25_macd  ← j_strong25_macd_di       (18.0M, WR44%, N=9)

Train: 2023-09-01 ~ 2025-12-31
OOS:   2026-01-01 ~ 2026-02-19
"""
import sys, warnings, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

OHLCV_PATH = ROOT / "data" / "market" / "daily" / "profit_curve_ohlcv.parquet"
BTC_PATH   = ROOT / "data" / "market" / "daily" / "btc_mstr_daily.parquet"
EXTRA_PATH = ROOT / "data" / "market" / "daily" / "extra_signals_daily.parquet"

OPT_TICKERS = ["IREN", "CONL", "PTIR"]
TRAIN_END   = pd.Timestamp("2025-12-31")
TEST_START  = pd.Timestamp("2026-01-01")
TEST_END    = pd.Timestamp("2026-02-19")
N_TRIALS    = 400


# ─────────────────────────────────────────────────────────────────────
# Feature Engineering (공통)
# ─────────────────────────────────────────────────────────────────────
def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def add_features(df):
    df = df.copy().sort_values("Date").reset_index(drop=True)
    c  = df["Close"]
    df["ma5"]  = c.rolling(5).mean()
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
    tr  = pd.concat([high-low,(high-c.shift(1)).abs(),(low-c.shift(1)).abs()],axis=1).max(axis=1)
    dmp = (high-high.shift(1)).clip(lower=0).where((high-high.shift(1))>(low.shift(1)-low), 0)
    dmm = (low.shift(1)-low).clip(lower=0).where((low.shift(1)-low)>(high-high.shift(1)), 0)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    df["di_plus"]  = dmp.ewm(span=14, adjust=False).mean() / atr14 * 100
    df["di_minus"] = dmm.ewm(span=14, adjust=False).mean() / atr14 * 100
    # 거래량
    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    # 연속 상승
    up = (c > c.shift(1)).astype(int)
    df["consec_up"] = up.groupby(((up==0)|(up!=up.shift(1))).cumsum()).cumsum() * up
    # 눌림목
    df["pullback_in_trend"] = ((df["pct_ma20"] > 10) & (df["ret5"] < 0)).astype(int)
    return df


def build_macro(btc_df, extra_df):
    btc = btc_df[btc_df["ticker"]=="BTC-USD"].copy().sort_values("Date")
    c   = btc["Close"]
    btc["btc_ma20"]   = c.rolling(20).mean()
    btc["btc_ma60"]   = c.rolling(60).mean()
    btc["btc_rsi14"]  = rsi(c, 14)
    btc["btc_regime"] = 0
    btc.loc[c > btc["btc_ma60"], "btc_regime"] = 1
    btc.loc[(c > btc["btc_ma20"]) & (btc["btc_rsi14"] > 55), "btc_regime"] = 2

    mstr = btc_df[btc_df["ticker"]=="MSTR"].copy().sort_values("Date")
    mc   = mstr["Close"]
    mstr["mstr_pct_ma20"] = (mc - mc.rolling(20).mean()) / mc.rolling(20).mean() * 100

    vix = extra_df[extra_df["ticker"]=="VIX"].copy().sort_values("Date")
    qqq = extra_df[extra_df["ticker"]=="QQQ"].copy().sort_values("Date")
    qc  = qqq["Close"]
    qqq["qqq_ma20"]  = qc.rolling(20).mean()
    qqq["qqq_ma60"]  = qc.rolling(60).mean()
    qqq["qqq_rsi14"] = rsi(qc, 14)
    qqq["qqq_bull"]  = (qc > qqq["qqq_ma60"]).astype(int)

    macro = (
        btc.set_index("Date")[["btc_ma20","btc_ma60","btc_rsi14","btc_regime"]]
        .join(mstr.set_index("Date")[["mstr_pct_ma20"]], how="outer")
        .join(vix.set_index("Date")[["Close"]].rename(columns={"Close":"vix"}), how="outer")
        .join(qqq.set_index("Date")[["qqq_rsi14","qqq_bull"]], how="outer")
    )
    return macro.ffill()


# ─────────────────────────────────────────────────────────────────────
# 전략별 진입 함수 팩토리
# ─────────────────────────────────────────────────────────────────────
def make_entry_fn(strategy: str, p: dict):
    """전략 이름과 파라미터로 진입 함수 생성"""

    if strategy == "vix_di_surge":
        # j_vix_neutral_di_surge: VIX 범위 + DI크로스 + 급등 (핵심)
        def entry(r):
            vix = r.get("vix", 20)
            return (
                p["vix_min"] <= vix <= p["vix_max"] and
                (r["di_plus"] - r["di_minus"]) >= p["di_min_gap"] and
                r["ret1"] >= p["surge_pct"] and
                r.get("btc_rsi14", 99) <= p["btc_rsi_max"] and
                r["pct_ma20"] >= p["pct_ma20_min"] and
                r["vol_ratio"] >= p["vol_ratio_min"] and
                r["rsi14"] >= p["rsi14_min"]
            )

    elif strategy == "macd_vol":
        # l_wide_macd_vol: MACD + 거래량폭발 + BTC RSI + VIX
        def entry(r):
            return (
                r["macd_hist"] >= p["macd_threshold"] and
                r["vol_ratio"] >= p["vol_ratio_min"] and
                r.get("btc_rsi14", 99) <= p["btc_rsi_max"] and
                r.get("vix", 0) >= p["vix_min"] and
                r.get("vix", 0) <= p["vix_max"] and
                r["rsi14"] >= p["rsi14_min"] and
                r["pct_ma20"] >= p["pct_ma20_min"] and
                r["ret1"] >= p["surge_pct"]
            )

    elif strategy == "btc_rsi_e":
        # l_wide_btc_rsi_e: BTC RSI + VIX + DI크로스 + RSI14 (완화 버전)
        def entry(r):
            return (
                r.get("btc_rsi14", 99) <= p["btc_rsi_max"] and
                r.get("vix", 0) >= p["vix_min"] and
                r.get("vix", 0) <= p["vix_max"] and
                (r["di_plus"] - r["di_minus"]) >= p["di_min_gap"] and
                r["rsi14"] >= p["rsi14_min"] and
                r["vol_ratio"] >= p["vol_ratio_min"] and
                r["pct_ma20"] >= p["pct_ma20_min"]
            )

    elif strategy == "strong25_macd":
        # j_strong25_macd_di: 강한 추세(pct_ma20>25%) + MACD + DI + BTC RSI
        def entry(r):
            return (
                r["pct_ma20"] >= p["pct_ma20_min"] and
                r["macd_hist"] >= p["macd_threshold"] and
                (r["di_plus"] - r["di_minus"]) >= p["di_min_gap"] and
                r.get("btc_rsi14", 99) <= p["btc_rsi_max"] and
                r.get("vix", 0) >= p["vix_min"] and
                r.get("vix", 0) <= p["vix_max"] and
                r["vol_ratio"] >= p["vol_ratio_min"] and
                r["rsi14"] >= p["rsi14_min"]
            )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return entry


def suggest_params(trial, strategy: str) -> dict:
    """전략별 파라미터 공간 정의"""
    # 공통 청산/포지션 파라미터
    p = {
        "target_pct":      trial.suggest_float("target_pct",     20.0, 100.0),
        "stop_pct":        trial.suggest_float("stop_pct",      -35.0,  -8.0),
        "hold_days":       trial.suggest_int(  "hold_days",        10,   120),
        "trailing_pct":    trial.suggest_categorical("trailing_pct",
                               [None, -8.0, -12.0, -15.0, -20.0]),
        "unit_mul":        trial.suggest_float("unit_mul",         1.0,   5.0),
        "max_pyramid":     trial.suggest_int(  "max_pyramid",       1,     5),
        "pyramid_add_pct": trial.suggest_float("pyramid_add_pct",  4.0,  15.0),
    }
    # 공통 신호 파라미터 (모든 전략에 포함)
    p["btc_rsi_max"]   = trial.suggest_float("btc_rsi_max",  45.0, 82.0)
    p["vix_min"]       = trial.suggest_float("vix_min",      12.0, 24.0)
    p["vix_max"]       = trial.suggest_float("vix_max",      20.0, 40.0)
    p["pct_ma20_min"]  = trial.suggest_float("pct_ma20_min",-15.0, 30.0)
    p["vol_ratio_min"] = trial.suggest_float("vol_ratio_min",  0.4,  2.5)
    p["rsi14_min"]     = trial.suggest_float("rsi14_min",     0.0, 70.0)

    # 전략별 추가 파라미터
    if strategy in ("vix_di_surge", "btc_rsi_e"):
        p["di_min_gap"]  = trial.suggest_float("di_min_gap",  -3.0, 12.0)
        p["surge_pct"]   = trial.suggest_float("surge_pct",    1.0,  6.0)

    if strategy == "macd_vol":
        p["macd_threshold"] = trial.suggest_float("macd_threshold", -0.5, 2.0)
        p["surge_pct"]      = trial.suggest_float("surge_pct",      -1.0, 4.0)

    if strategy == "strong25_macd":
        p["di_min_gap"]      = trial.suggest_float("di_min_gap",     -2.0, 10.0)
        p["macd_threshold"]  = trial.suggest_float("macd_threshold", -0.5,  2.0)
        p["rsi14_min"]       = trial.suggest_float("rsi14_min",       0.0, 65.0)

    if strategy == "vix_di_surge":
        p["surge_pct"] = trial.suggest_float("surge_pct", 1.0, 6.0)
        p["di_min_gap"] = trial.suggest_float("di_min_gap", -3.0, 12.0)

    return p


# ─────────────────────────────────────────────────────────────────────
# Backtest Engine
# ─────────────────────────────────────────────────────────────────────
def run_backtest(df, entry_fn, p, fx=1350.0):
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
                               "PnL_pct": pp, "HeldDays": held, "Layers": len(pos)})
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


# ─────────────────────────────────────────────────────────────────────
# Optuna Objective
# ─────────────────────────────────────────────────────────────────────
_CACHE = {}

def objective(trial, strategy, tickers):
    p = suggest_params(trial, strategy)
    if p["vix_min"] >= p["vix_max"]:
        return -1_000_000

    entry_fn = make_entry_fn(strategy, p)
    total_pnl, total_n, total_wins = 0, 0, 0

    for t in tickers:
        key = (t, "train")
        if key not in _CACHE: continue
        r = run_backtest(_CACHE[key], entry_fn, p)
        total_pnl  += r["pnl"]
        total_n    += r["n"]
        total_wins += int(r["n"] * r["wr"] / 100) if r["n"] > 0 else 0

    if total_n < 4:
        return -1_000_000

    overall_wr = total_wins / total_n * 100 if total_n > 0 else 0
    if overall_wr < 30:
        return total_pnl * 0.2

    return total_pnl


# ─────────────────────────────────────────────────────────────────────
# OOS Validation
# ─────────────────────────────────────────────────────────────────────
def oos_validation(strategy, best_params):
    print(f"\n{'='*65}")
    print(f"  OOS 검증: 2026-01-01 ~ 2026-02-19  [{strategy}]")
    print(f"{'='*65}")
    entry_fn = make_entry_fn(strategy, best_params)
    all_trades = []

    for t in OPT_TICKERS:
        key = (t, "oos")
        if key not in _CACHE: continue
        df = _CACHE[key]
        r  = run_backtest(df, entry_fn, best_params)
        status = "✅" if r["pnl"] > 0 else ("⚠️" if r["n"] == 0 else "❌")
        print(f"\n  {status} {t:6s}: {r['n']}거래, WR={r['wr']}%, avg={r['avg']}%, "
              f"PnL={r['pnl']:>12,.0f}원")
        if not r["trades"].empty:
            print(f"     {'날짜':12s} {'진입가':>8s} {'청산가':>8s} {'수익%':>7s} {'보유일':>5s} {'층':>2s}")
            for _, row in r["trades"].iterrows():
                sign = "+" if row["PnL_pct"] > 0 else ""
                print(f"     {str(row['Date'].date()):12s} "
                      f"${row['Entry']:>8.2f} ${row['Exit']:>8.2f} "
                      f"{sign}{row['PnL_pct']:>6.1f}% {int(row['HeldDays']):>4d}일 {int(row['Layers']):>2d}층")
            all_trades.append(r["trades"])

    # 신호 발생 일자
    print(f"\n  --- 신호 발생 일자 ---")
    for t in OPT_TICKERS:
        key = (t, "oos")
        if key not in _CACHE: continue
        df = _CACHE[key]
        sigs = df[df.apply(entry_fn, axis=1)]
        if sigs.empty:
            print(f"  {t:6s}: 신호 없음")
        else:
            print(f"  {t:6s}: {len(sigs)}회 신호")
            for _, row in sigs.iterrows():
                print(f"    {str(row['Date'].date())}  Close={row['Close']:.2f}  "
                      f"DI+={row['di_plus']:.1f}/DI-={row['di_minus']:.1f}  "
                      f"ret1={row['ret1']:.1f}%  pct_ma20={row['pct_ma20']:.1f}%  "
                      f"btc_rsi={row.get('btc_rsi14',0):.1f}  vix={row.get('vix',0):.1f}")

    if all_trades:
        combined = pd.concat(all_trades)
        print(f"\n  OOS 합계: {len(combined)}건  "
              f"WR={(combined['PnL_KRW']>0).mean()*100:.1f}%  "
              f"PnL={combined['PnL_KRW'].sum():,.0f}원")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True,
                        choices=["vix_di_surge","macd_vol","btc_rsi_e","strong25_macd"])
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    args = parser.parse_args()
    strategy = args.strategy
    n_trials = args.trials

    STRATEGY_NAMES = {
        "vix_di_surge":  "j_vix_neutral_di_surge  (21.4M)",
        "macd_vol":      "l_wide_macd_vol          (19.8M)",
        "btc_rsi_e":     "l_wide_btc_rsi_e         (18.9M)",
        "strong25_macd": "j_strong25_macd_di       (18.0M)",
    }
    print(f"[Optuna] {STRATEGY_NAMES[strategy]}")
    print(f"  Train: 2023-09-01 ~ 2025-12-31  |  OOS: 2026-01-01 ~ 2026-02-19")
    print(f"  Trials: {n_trials}\n")

    # ── 데이터 로드 ──────────────────────────────────────────────────
    ohlcv  = pd.read_parquet(OHLCV_PATH); ohlcv["Date"]  = pd.to_datetime(ohlcv["Date"])
    extra  = pd.read_parquet(EXTRA_PATH); extra["Date"]  = pd.to_datetime(extra["Date"])
    btc_df = pd.read_parquet(BTC_PATH);  btc_df["Date"] = pd.to_datetime(btc_df["Date"])

    macro = build_macro(btc_df, extra)
    ohlcv_all = pd.concat([ohlcv, extra[extra["ticker"].isin(["MSTX"])].copy()], ignore_index=True)

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
        return sub[(sub["Date"] >= date_start) & (sub["Date"] <= date_end)].copy()

    print("데이터 준비 중...")
    for t in OPT_TICKERS:
        df_train = prep_ticker(t, pd.Timestamp("2023-09-01"), TRAIN_END)
        df_oos   = prep_ticker(t, TEST_START, TEST_END)
        if df_train is not None and len(df_train) >= 20:
            _CACHE[(t, "train")] = df_train
            print(f"  {t} Train: {len(df_train)}일  OOS: {len(df_oos) if df_oos is not None else 0}일")
        if df_oos is not None and len(df_oos) > 0:
            _CACHE[(t, "oos")] = df_oos

    # ── 베이스라인 (원래 L 실험 파라미터 근사) ──────────────────────
    print(f"\n=== 베이스라인 (L 실험 파라미터) ===")
    bl_params = _baseline_params(strategy)
    bl_entry  = make_entry_fn(strategy, bl_params)
    bl_total  = 0
    for t in OPT_TICKERS:
        if (t, "train") not in _CACHE: continue
        r = run_backtest(_CACHE[(t, "train")], bl_entry, bl_params)
        bl_total += r["pnl"]
        print(f"  {t}: N={r['n']}, WR={r['wr']}%, avg={r['avg']}%, PnL={r['pnl']:,}원")
    print(f"  베이스라인 합계: {bl_total:,}원 ({bl_total/10000:.1f}만원)")

    # ── Optuna 최적화 ────────────────────────────────────────────────
    print(f"\n=== Optuna 최적화 ({n_trials} trials) ===")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=40)
    )
    study.optimize(
        lambda trial: objective(trial, strategy, OPT_TICKERS),
        n_trials=n_trials,
        show_progress_bar=True
    )

    bp = study.best_trial.params
    best_val = study.best_value
    print(f"\n✅ 최적화 완료  →  Train PnL: {best_val:,.0f}원 ({best_val/10000:.1f}만원)")
    print(f"   베이스라인 대비: {(best_val-bl_total)/10000:+.1f}만원\n")

    # ── 파라미터 출력 ────────────────────────────────────────────────
    print("=== 최적 파라미터 ===")
    signal_keys = [k for k in bp if k not in
                   ("target_pct","stop_pct","hold_days","trailing_pct",
                    "unit_mul","max_pyramid","pyramid_add_pct")]
    exit_keys   = ["target_pct","stop_pct","hold_days","trailing_pct"]
    pos_keys    = ["unit_mul","max_pyramid","pyramid_add_pct"]
    print("  [신호]")
    for k in signal_keys: print(f"    {k:25s}: {bp[k]}")
    print("  [청산]")
    for k in exit_keys:   print(f"    {k:25s}: {bp.get(k)}")
    print("  [포지션]")
    for k in pos_keys:    print(f"    {k:25s}: {bp.get(k)}")

    # ── Train 최종 성과 ──────────────────────────────────────────────
    print(f"\n=== Train 최종 성과 ({strategy}) ===")
    opt_entry = make_entry_fn(strategy, bp)
    train_total = 0
    for t in OPT_TICKERS:
        if (t, "train") not in _CACHE: continue
        r = run_backtest(_CACHE[(t, "train")], opt_entry, bp)
        train_total += r["pnl"]
        print(f"  {t}: N={r['n']}, WR={r['wr']}%, avg={r['avg']}%, PnL={r['pnl']:,}원")
    print(f"  Train 합계: {train_total:,}원 ({train_total/10000:.1f}만원)")

    # ── OOS 검증 ─────────────────────────────────────────────────────
    oos_validation(strategy, bp)

    # ── Top 10 trials ────────────────────────────────────────────────
    print(f"\n=== Top 10 Trials [{strategy}] ===")
    top = sorted(study.trials, key=lambda t: t.value or -9e9, reverse=True)[:10]
    for i, tr in enumerate(top):
        if tr.value is None: continue
        tgt = tr.params.get("target_pct", 0)
        stp = tr.params.get("stop_pct", 0)
        hld = tr.params.get("hold_days", 0)
        mul = tr.params.get("unit_mul", 0)
        pyr = tr.params.get("max_pyramid", 0)
        print(f"  #{i+1:2d}  {tr.value/10000:7.1f}만원  "
              f"T{tgt:.0f}/S{stp:.0f}/H{hld}  "
              f"{mul:.1f}x pyr{pyr}")

    # ── CSV 저장 ─────────────────────────────────────────────────────
    out_path = ROOT / "experiments" / "results" / f"optuna_{strategy}.csv"
    rows = []
    for k in ["train","oos"]:
        for t in OPT_TICKERS:
            if (t, k) not in _CACHE: continue
            r = run_backtest(_CACHE[(t, k)], opt_entry, bp)
            rows.append({"period": k, "ticker": t, "strategy": strategy,
                         "n": r["n"], "pnl": r["pnl"], "wr": r["wr"], "avg": r["avg"],
                         **{f"p_{k2}": v for k2, v in bp.items()}})
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n결과 저장: {out_path}")


def _baseline_params(strategy: str) -> dict:
    """각 전략의 L 실험 기본 파라미터 근사값"""
    base = {
        "target_pct": 80.0, "stop_pct": -25.0, "hold_days": 90,
        "trailing_pct": None, "unit_mul": 3.0, "max_pyramid": 3,
        "pyramid_add_pct": 7.0, "btc_rsi_max": 80.0,
        "vix_min": 12.0, "vix_max": 40.0, "pct_ma20_min": -10.0,
        "vol_ratio_min": 0.5, "rsi14_min": 0.0,
    }
    if strategy == "vix_di_surge":
        base.update({"vix_min": 18.0, "vix_max": 25.0,
                     "di_min_gap": 0.0, "surge_pct": 3.0})
    elif strategy == "macd_vol":
        base.update({"macd_threshold": 0.0, "surge_pct": 0.0,
                     "vol_ratio_min": 1.4, "vix_min": 16.0})
    elif strategy == "btc_rsi_e":
        base.update({"btc_rsi_max": 70.0, "vix_min": 16.0,
                     "di_min_gap": 0.0, "surge_pct": 0.0, "rsi14_min": 55.0})
    elif strategy == "strong25_macd":
        base.update({"pct_ma20_min": 25.0, "macd_threshold": 0.0,
                     "di_min_gap": 0.0, "btc_rsi_max": 70.0})
    return base


if __name__ == "__main__":
    main()

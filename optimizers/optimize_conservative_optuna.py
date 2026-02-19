"""
optimize_conservative_optuna.py — 보수적 파라미터로 신호 품질 최적화
====================================================================
목표:
  - 과적합 방지: exit/position 파라미터를 보수적 범위로 제한
  - 신호 품질 집중: 어떤 조건에서 진입해야 하는지만 최적화
  - WR > 50%, N >= 8 조건으로 통계적 신뢰도 확보

변경점 (vs 기존 Optuna):
  target_pct : 88~99%  → 20~55%    (단기 현실적 목표)
  stop_pct   : -30~-35% → -10~-22% (손절 강화)
  hold_days  : 82~120일 → 15~60일   (단기 보유)
  unit_mul   : 4.5~5x  → 1.5~3.0x  (레버리지 축소)
  max_pyramid: 3~5     → 1~3        (피라미딩 제한)
  WR penalty : <30%    → <50%       (신뢰도 강화)
  N minimum  : 4       → 8          (통계 유의성)

목적함수: PnL × (WR/100)² × min(N/12, 1.0)
  → WR과 거래수를 동시에 최대화하는 균형 지표

Usage:
  python optimize_conservative_optuna.py --strategy all
  python optimize_conservative_optuna.py --strategy vix_di_surge
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
N_TRIALS    = 500

ALL_STRATEGIES = ["di_surge", "vix_di_surge", "macd_vol", "btc_rsi_e", "strong25_macd"]


# ─────────────────────────────────────────────────────────────────────
# Feature Engineering
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
    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    up = (c > c.shift(1)).astype(int)
    df["consec_up"] = up.groupby(((up==0)|(up!=up.shift(1))).cumsum()).cumsum() * up
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
# 진입 함수 팩토리
# ─────────────────────────────────────────────────────────────────────
def make_entry_fn(strategy: str, p: dict):
    if strategy == "di_surge":
        def entry(r):
            return (
                (r["di_plus"] - r["di_minus"]) >= p["di_min_gap"] and
                r["ret1"] >= p["surge_pct"] and
                r.get("btc_rsi14", 99) <= p["btc_rsi_max"] and
                (not p["btc_above_ma20"] or r.get("btc_regime", 0) >= 1) and
                p["vix_min"] <= r.get("vix", 20) <= p["vix_max"] and
                r["pct_ma20"] >= p["pct_ma20_min"] and
                r["vol_ratio"] >= p["vol_ratio_min"] and
                r["rsi14"] >= p["rsi14_min"]
            )

    elif strategy == "vix_di_surge":
        def entry(r):
            return (
                p["vix_min"] <= r.get("vix", 20) <= p["vix_max"] and
                (r["di_plus"] - r["di_minus"]) >= p["di_min_gap"] and
                r["ret1"] >= p["surge_pct"] and
                r.get("btc_rsi14", 99) <= p["btc_rsi_max"] and
                r["pct_ma20"] >= p["pct_ma20_min"] and
                r["vol_ratio"] >= p["vol_ratio_min"] and
                r["rsi14"] >= p["rsi14_min"]
            )

    elif strategy == "macd_vol":
        def entry(r):
            return (
                r["macd_hist"] >= p["macd_threshold"] and
                r["vol_ratio"] >= p["vol_ratio_min"] and
                r.get("btc_rsi14", 99) <= p["btc_rsi_max"] and
                p["vix_min"] <= r.get("vix", 20) <= p["vix_max"] and
                r["rsi14"] >= p["rsi14_min"] and
                r["pct_ma20"] >= p["pct_ma20_min"] and
                r["ret1"] >= p["surge_pct"]
            )

    elif strategy == "btc_rsi_e":
        def entry(r):
            return (
                r.get("btc_rsi14", 99) <= p["btc_rsi_max"] and
                p["vix_min"] <= r.get("vix", 20) <= p["vix_max"] and
                (r["di_plus"] - r["di_minus"]) >= p["di_min_gap"] and
                r["rsi14"] >= p["rsi14_min"] and
                r["vol_ratio"] >= p["vol_ratio_min"] and
                r["pct_ma20"] >= p["pct_ma20_min"]
            )

    elif strategy == "strong25_macd":
        def entry(r):
            return (
                r["pct_ma20"] >= p["pct_ma20_min"] and
                r["macd_hist"] >= p["macd_threshold"] and
                (r["di_plus"] - r["di_minus"]) >= p["di_min_gap"] and
                r.get("btc_rsi14", 99) <= p["btc_rsi_max"] and
                p["vix_min"] <= r.get("vix", 20) <= p["vix_max"] and
                r["vol_ratio"] >= p["vol_ratio_min"] and
                r["rsi14"] >= p["rsi14_min"]
            )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return entry


def suggest_params_conservative(trial, strategy: str) -> dict:
    """보수적 Exit 파라미터 + 신호 파라미터 자유 탐색"""
    p = {
        # ── 보수적 청산/포지션 (제한된 범위) ──────────────────────
        "target_pct":      trial.suggest_float("target_pct",     20.0,  55.0),  # ← 제한
        "stop_pct":        trial.suggest_float("stop_pct",      -22.0, -10.0),  # ← 제한
        "hold_days":       trial.suggest_int(  "hold_days",        15,    60),  # ← 제한
        "trailing_pct":    trial.suggest_categorical("trailing_pct",
                               [None, -5.0, -8.0, -12.0]),
        "unit_mul":        trial.suggest_float("unit_mul",         1.5,   3.0), # ← 제한
        "max_pyramid":     trial.suggest_int(  "max_pyramid",       1,     3),  # ← 제한
        "pyramid_add_pct": trial.suggest_float("pyramid_add_pct",  5.0,  12.0),
        # ── 신호 파라미터 (자유 탐색) ─────────────────────────────
        "btc_rsi_max":     trial.suggest_float("btc_rsi_max",    45.0,  80.0),
        "vix_min":         trial.suggest_float("vix_min",        12.0,  24.0),
        "vix_max":         trial.suggest_float("vix_max",        20.0,  40.0),
        "pct_ma20_min":    trial.suggest_float("pct_ma20_min",  -15.0,  30.0),
        "vol_ratio_min":   trial.suggest_float("vol_ratio_min",   0.4,   2.5),
        "rsi14_min":       trial.suggest_float("rsi14_min",       0.0,  70.0),
    }

    if strategy in ("di_surge", "vix_di_surge", "btc_rsi_e"):
        p["di_min_gap"] = trial.suggest_float("di_min_gap", -3.0, 12.0)
        p["surge_pct"]  = trial.suggest_float("surge_pct",   1.0,  6.0)

    if strategy == "di_surge":
        p["btc_above_ma20"] = trial.suggest_categorical("btc_above_ma20", [True, False])

    if strategy == "macd_vol":
        p["macd_threshold"] = trial.suggest_float("macd_threshold", -0.5, 2.0)
        p["surge_pct"]      = trial.suggest_float("surge_pct",      -1.0, 4.0)

    if strategy == "strong25_macd":
        p["di_min_gap"]     = trial.suggest_float("di_min_gap",     -2.0, 10.0)
        p["macd_threshold"] = trial.suggest_float("macd_threshold", -0.5,  2.0)

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
# 목적함수 (균형 지표)
# ─────────────────────────────────────────────────────────────────────
_CACHE = {}

def objective(trial, strategy):
    p = suggest_params_conservative(trial, strategy)
    if p["vix_min"] >= p["vix_max"]:
        return -999_999

    entry_fn = make_entry_fn(strategy, p)
    total_pnl, total_n, total_wins = 0, 0, 0

    for t in OPT_TICKERS:
        key = (t, "train")
        if key not in _CACHE: continue
        r = run_backtest(_CACHE[key], entry_fn, p)
        total_pnl  += r["pnl"]
        total_n    += r["n"]
        total_wins += int(r["n"] * r["wr"] / 100) if r["n"] > 0 else 0

    # N 최소 조건 (통계 유의성)
    if total_n < 8:
        return -999_999

    overall_wr = total_wins / total_n * 100 if total_n > 0 else 0

    # 균형 목적함수: PnL × WR² × 거래수 보정
    # WR < 50% 강력 페널티
    wr_factor = (overall_wr / 100) ** 2
    n_factor  = min(total_n / 15, 1.0)   # 15거래 이상이면 보정 없음

    if overall_wr < 50:
        wr_factor *= 0.2  # 강력 페널티

    score = total_pnl * wr_factor * (0.5 + 0.5 * n_factor)
    return score


# ─────────────────────────────────────────────────────────────────────
# OOS 검증
# ─────────────────────────────────────────────────────────────────────
def oos_validation(strategy, bp):
    print(f"\n{'='*65}")
    print(f"  OOS 검증: 2026-01-01 ~ 2026-02-19  [{strategy}]")
    print(f"{'='*65}")
    entry_fn = make_entry_fn(strategy, bp)
    all_trades = []

    for t in OPT_TICKERS:
        key = (t, "oos")
        if key not in _CACHE: continue
        r  = run_backtest(_CACHE[key], entry_fn, bp)
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
    print(f"\n  --- 신호 발생 일자 (OOS) ---")
    for t in OPT_TICKERS:
        key = (t, "oos")
        if key not in _CACHE: continue
        df   = _CACHE[key]
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
    else:
        print(f"\n  OOS: 완료된 거래 없음")


# ─────────────────────────────────────────────────────────────────────
# 전략 최적화 실행
# ─────────────────────────────────────────────────────────────────────
def run_strategy(strategy, n_trials):
    print(f"\n{'━'*65}")
    print(f"  [{strategy.upper()}] 보수적 최적화 시작 ({n_trials} trials)")
    print(f"  target: 20~55%  |  stop: -10~-22%  |  hold: 15~60일  |  max 3x")
    print(f"{'━'*65}")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=50)
    )
    study.optimize(
        lambda trial: objective(trial, strategy),
        n_trials=n_trials,
        show_progress_bar=True
    )

    bp       = study.best_trial.params
    best_val = study.best_value

    # 실제 PnL 계산 (목적함수는 스코어이므로 재계산)
    entry_fn   = make_entry_fn(strategy, bp)
    train_total = 0
    train_n     = 0
    train_wins  = 0

    print(f"\n✅ {strategy} 완료  (score={best_val:.0f})")
    print(f"\n=== 최적 파라미터 [{strategy}] ===")
    signal_keys = [k for k in bp if k not in
                   ("target_pct","stop_pct","hold_days","trailing_pct",
                    "unit_mul","max_pyramid","pyramid_add_pct")]
    print(f"  [신호]  ", end="")
    print("  ".join(f"{k}={v:.2f}" if isinstance(v,float) else f"{k}={v}"
                    for k, v in bp.items() if k in signal_keys))
    print(f"  [청산]  target={bp['target_pct']:.1f}%  stop={bp['stop_pct']:.1f}%  "
          f"hold={bp['hold_days']}일  trailing={bp.get('trailing_pct')}")
    print(f"  [포지션] {bp['unit_mul']:.1f}x  pyr{bp['max_pyramid']}  "
          f"add{bp['pyramid_add_pct']:.1f}%")

    print(f"\n=== Train 성과 [{strategy}] ===")
    for t in OPT_TICKERS:
        if (t, "train") not in _CACHE: continue
        r = run_backtest(_CACHE[(t, "train")], entry_fn, bp)
        train_total += r["pnl"]
        train_n     += r["n"]
        train_wins  += int(r["n"] * r["wr"] / 100) if r["n"] > 0 else 0
        print(f"  {t}: N={r['n']}, WR={r['wr']}%, avg={r['avg']}%, PnL={r['pnl']:,}원")
    wr_all = train_wins / train_n * 100 if train_n > 0 else 0
    print(f"  합계: {train_total:,}원 ({train_total/10000:.1f}만원) | WR={wr_all:.1f}% | N={train_n}")

    # OOS
    oos_validation(strategy, bp)

    # Top 5 trials
    print(f"\n  Top 5 Trials [{strategy}]")
    top = sorted(study.trials, key=lambda t: t.value or -9e9, reverse=True)[:5]
    for i, tr in enumerate(top):
        if tr.value is None: continue
        tgt = tr.params.get("target_pct", 0)
        stp = tr.params.get("stop_pct", 0)
        hld = tr.params.get("hold_days", 0)
        mul = tr.params.get("unit_mul", 0)
        pyr = tr.params.get("max_pyramid", 0)
        print(f"    #{i+1}  score={tr.value:.0f}  T{tgt:.0f}/S{stp:.0f}/H{hld}  {mul:.1f}x pyr{pyr}")

    # 저장
    out_path = ROOT / "experiments" / "results" / f"conservative_{strategy}.csv"
    rows = []
    for period, ts, te in [("train", pd.Timestamp("2023-09-01"), TRAIN_END),
                            ("oos",   TEST_START, TEST_END)]:
        for t in OPT_TICKERS:
            key = (t, period)
            if key not in _CACHE: continue
            r = run_backtest(_CACHE[key], entry_fn, bp)
            rows.append({"period": period, "ticker": t, "strategy": strategy,
                         "n": r["n"], "pnl": r["pnl"], "wr": r["wr"], "avg": r["avg"],
                         **{f"p_{k}": v for k, v in bp.items()}})
    pd.DataFrame(rows).to_csv(out_path, index=False)

    return {"strategy": strategy, "train_pnl": train_total,
            "train_wr": round(wr_all,1), "train_n": train_n,
            "best_params": bp}


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="all",
                        choices=["all"] + ALL_STRATEGIES)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    args = parser.parse_args()

    strategies = ALL_STRATEGIES if args.strategy == "all" else [args.strategy]

    print(f"[보수적 Optuna] 전략: {strategies}")
    print(f"Train: 2023-09-01 ~ 2025-12-31  |  OOS: 2026-01-01 ~ 2026-02-19")
    print(f"목적함수: PnL × (WR/100)² × N보정  |  WR<50% = 80%패널티  |  N<8 = 제외")

    # ── 데이터 로드 ──────────────────────────────────────────────────
    ohlcv  = pd.read_parquet(OHLCV_PATH); ohlcv["Date"]  = pd.to_datetime(ohlcv["Date"])
    extra  = pd.read_parquet(EXTRA_PATH); extra["Date"]  = pd.to_datetime(extra["Date"])
    btc_df = pd.read_parquet(BTC_PATH);  btc_df["Date"] = pd.to_datetime(btc_df["Date"])
    macro  = build_macro(btc_df, extra)
    ohlcv_all = pd.concat([ohlcv, extra[extra["ticker"].isin(["MSTX"])].copy()], ignore_index=True)

    def prep(t, d0, d1):
        sub = ohlcv_all[ohlcv_all["ticker"]==t].copy()
        if len(sub) < 30: return None
        sub = add_features(sub)
        sub = sub.merge(macro.reset_index().rename(columns={"index":"Date"}),
                        on="Date", how="left")
        for col in macro.columns:
            if col in sub.columns: sub[col] = sub[col].ffill()
        sub["ticker"] = t
        return sub[(sub["Date"] >= d0) & (sub["Date"] <= d1)].copy()

    print("\n데이터 준비...")
    for t in OPT_TICKERS:
        df_tr = prep(t, pd.Timestamp("2023-09-01"), TRAIN_END)
        df_os = prep(t, TEST_START, TEST_END)
        if df_tr is not None and len(df_tr) >= 20:
            _CACHE[(t,"train")] = df_tr
        if df_os is not None and len(df_os) > 0:
            _CACHE[(t,"oos")] = df_os
        n_tr = len(df_tr) if df_tr is not None else 0
        n_os = len(df_os) if df_os is not None else 0
        print(f"  {t}: Train={n_tr}일  OOS={n_os}일")

    # ── 전략별 순차 실행 ─────────────────────────────────────────────
    summary = []
    for strat in strategies:
        res = run_strategy(strat, args.trials)
        summary.append(res)

    # ── 최종 종합 요약 ───────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print(f"  최종 종합 요약 (보수적 최적화)")
    print(f"{'='*65}")
    print(f"  {'전략':20s} {'Train PnL':>12s} {'WR':>6s} {'N':>4s}")
    print(f"  {'-'*50}")
    for r in sorted(summary, key=lambda x: x["train_pnl"], reverse=True):
        print(f"  {r['strategy']:20s} {r['train_pnl']/10000:>10.1f}만원 "
              f"{r['train_wr']:>5.1f}% {r['train_n']:>4d}거래")

    best = max(summary, key=lambda x: x["train_pnl"])
    print(f"\n  🏆 최강: {best['strategy']} ({best['train_pnl']/10000:.1f}만원, "
          f"WR={best['train_wr']}%)")


if __name__ == "__main__":
    main()

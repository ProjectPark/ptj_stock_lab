#!/usr/bin/env python3
"""
PTJ v4 Phase 1 Best Trial #388 — 매매 패턴 상세 분석
=====================================================
Best Trial #388 파라미터로 백테스트 재실행 후 trade log를 분석한다.

Usage:
    pyenv shell ptj_stock_lab && python experiments/analyze_v4_best.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT), str(_ROOT / "backtests"), str(_ROOT / "strategies")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config

# ── Best Trial #388 파라미터 ─────────────────────────────────────────
BEST_PARAMS = {
    "COIN_SELL_BEARISH_PCT": 0.70,
    "COIN_SELL_PROFIT_PCT": 2.50,
    "CONL_SELL_AVG_PCT": 0.75,
    "CONL_SELL_PROFIT_PCT": 4.50,
    "DCA_DROP_PCT": -1.35,
    "MAX_HOLD_HOURS": 6,
    "PAIR_GAP_SELL_THRESHOLD_V2": 6.60,
    "PAIR_SELL_FIRST_PCT": 0.70,
    "STOP_LOSS_BULLISH_PCT": -16.00,
    "STOP_LOSS_PCT": -4.25,
    "TAKE_PROFIT_PCT": 3.00,
    "V4_CB_BTC_CRASH_PCT": -3.50,
    "V4_CB_BTC_SURGE_PCT": 8.50,
    "V4_CB_GLD_COOLDOWN_DAYS": 1,
    "V4_CB_GLD_SPIKE_PCT": 4.00,
    "V4_CB_VIX_COOLDOWN_DAYS": 13,
    "V4_CB_VIX_SPIKE_PCT": 3.00,
    "V4_COIN_TRIGGER_PCT": 3.00,
    "V4_CONL_ADX_MIN": 10.00,
    "V4_CONL_EMA_SLOPE_MIN_PCT": 0.50,
    "V4_CONL_TRIGGER_PCT": 3.00,
    "V4_DCA_BUY": 750,
    "V4_DCA_MAX_COUNT": 1,
    "V4_ENTRY_CUTOFF_HOUR": 9,
    "V4_ENTRY_CUTOFF_MINUTE": 0,
    "V4_HIGH_VOL_HIT_COUNT": 3,
    "V4_HIGH_VOL_MOVE_PCT": 11.00,
    "V4_HIGH_VOL_STOP_LOSS_PCT": -3.00,
    "V4_INITIAL_BUY": 3500,
    "V4_MAX_PER_STOCK": 8500,
    "V4_PAIR_FIXED_TP_PCT": 7.50,
    "V4_PAIR_GAP_ENTRY_THRESHOLD": 2.00,
    "V4_PAIR_IMMEDIATE_SELL_PCT": 0.20,
    "V4_SIDEWAYS_ATR_DECLINE_PCT": 10.00,
    "V4_SIDEWAYS_EMA_SLOPE_MAX": 0.05,
    "V4_SIDEWAYS_GAP_FAIL_COUNT": 2,
    "V4_SIDEWAYS_GLD_THRESHOLD": 0.50,
    "V4_SIDEWAYS_INDEX_THRESHOLD": 0.50,
    "V4_SIDEWAYS_MIN_SIGNALS": 3,
    "V4_SIDEWAYS_POLY_HIGH": 0.60,
    "V4_SIDEWAYS_POLY_LOW": 0.30,
    "V4_SIDEWAYS_RANGE_MAX_PCT": 3.00,
    "V4_SIDEWAYS_RSI_HIGH": 65.00,
    "V4_SIDEWAYS_RSI_LOW": 35.00,
    "V4_SIDEWAYS_TRIGGER_FAIL_COUNT": 3,
    "V4_SIDEWAYS_VOLUME_DECLINE_PCT": 15.00,
    "V4_SPLIT_BUY_INTERVAL_MIN": 10,
}


def run_backtest_with_best() -> dict:
    """Best Trial #388 파라미터로 백테스트 실행 후 상세 trade log 반환."""
    import backtest_common
    from backtest_v4 import BacktestEngineV4

    # config 파라미터 임시 교체
    originals = {}
    for key, value in BEST_PARAMS.items():
        originals[key] = getattr(config, key)
        setattr(config, key, value)

    try:
        engine = BacktestEngineV4()
        engine.run(verbose=False)

        # trade log 직렬화
        trades_raw = []
        for t in engine.trades:
            entry_time = t.entry_time
            exit_time = t.exit_time
            trades_raw.append({
                "date": str(t.date),
                "ticker": t.ticker,
                "side": t.side,
                "price": t.price,
                "qty": t.qty,
                "amount": t.amount_krw,
                "pnl": t.pnl_krw,
                "pnl_pct": t.pnl_pct,
                "net_pnl": t.net_pnl_krw,
                "signal_type": t.signal_type,
                "exit_reason": t.exit_reason,
                "dca_level": t.dca_level,
                "fees": t.fees_krw,
                "entry_time": str(entry_time) if entry_time else None,
                "exit_time": str(exit_time) if exit_time else None,
            })

        equity = [{"date": str(d), "equity": v} for d, v in engine.equity_curve]
        initial = engine.initial_capital_krw
        final = engine.equity_curve[-1][1] if engine.equity_curve else initial

        return {
            "summary": {
                "initial": initial,
                "final": round(final, 2),
                "total_return_pct": round((final - initial) / initial * 100, 4),
                "mdd": backtest_common.calc_mdd(engine.equity_curve),
                "sharpe": backtest_common.calc_sharpe(engine.equity_curve),
                "total_buys": sum(1 for t in engine.trades if t.side == "BUY"),
                "total_sells": sum(1 for t in engine.trades if t.side == "SELL"),
                "sideways_days": engine.sideways_days,
                "cb_buy_blocks": getattr(engine, "cb_buy_blocks", 0),
                "entry_cutoff_blocks": engine.entry_cutoff_blocks,
            },
            "trades": trades_raw,
            "equity_curve": equity,
        }
    finally:
        for key, value in originals.items():
            setattr(config, key, value)


# ── 분석 함수들 ──────────────────────────────────────────────────────

def analyze_buy_patterns(trades: list[dict]) -> dict:
    """매수 패턴 분석."""
    buys = [t for t in trades if t["side"] == "BUY"]

    # 시그널별 매수 빈도
    sig_count = defaultdict(int)
    for t in buys:
        sig_count[t["signal_type"]] += 1

    # 매수 시각 분포 (ET 시간 기준, entry_time에서 추출)
    hour_count = defaultdict(int)
    for t in buys:
        if t["entry_time"]:
            try:
                dt = datetime.fromisoformat(t["entry_time"])
                hour_count[dt.hour] += 1
            except Exception:
                pass

    # DCA 레벨 분포
    dca_count = defaultdict(int)
    for t in buys:
        dca_count[t["dca_level"]] += 1

    # 종목별 매수 빈도
    ticker_count = defaultdict(int)
    for t in buys:
        ticker_count[t["ticker"]] += 1

    # 평균 매수 금액
    avg_amount = sum(t["amount"] for t in buys) / len(buys) if buys else 0

    return {
        "total_buys": len(buys),
        "signal_frequency": dict(sorted(sig_count.items(), key=lambda x: -x[1])),
        "hour_distribution": dict(sorted(hour_count.items())),
        "dca_level_distribution": dict(sorted(dca_count.items())),
        "ticker_frequency": dict(sorted(ticker_count.items(), key=lambda x: -x[1])),
        "avg_amount": round(avg_amount, 2),
    }


def analyze_sell_patterns(trades: list[dict]) -> dict:
    """매도 패턴 분석."""
    sells = [t for t in trades if t["side"] == "SELL"]

    # 매도 사유별 통계
    reason_stats = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0, "hold_hours": []})
    for t in sells:
        reason = t["exit_reason"] or "unknown"
        reason_stats[reason]["count"] += 1
        reason_stats[reason]["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            reason_stats[reason]["wins"] += 1

        # 보유 시간 계산
        if t["entry_time"] and t["exit_time"]:
            try:
                entry = datetime.fromisoformat(t["entry_time"])
                exit_ = datetime.fromisoformat(t["exit_time"])
                hold_hours = (exit_ - entry).total_seconds() / 3600
                reason_stats[reason]["hold_hours"].append(hold_hours)
            except Exception:
                pass

    # 보유 시간 집계
    reason_summary = {}
    for reason, s in reason_stats.items():
        hold_list = s["hold_hours"]
        reason_summary[reason] = {
            "count": s["count"],
            "pnl": round(s["pnl"], 2),
            "win_rate": round(s["wins"] / s["count"] * 100, 1) if s["count"] > 0 else 0,
            "avg_pnl": round(s["pnl"] / s["count"], 2) if s["count"] > 0 else 0,
            "avg_hold_hours": round(sum(hold_list) / len(hold_list), 2) if hold_list else None,
            "min_hold_hours": round(min(hold_list), 2) if hold_list else None,
            "max_hold_hours": round(max(hold_list), 2) if hold_list else None,
        }

    # 매도 시각 분포
    hour_count = defaultdict(int)
    for t in sells:
        if t["exit_time"]:
            try:
                dt = datetime.fromisoformat(t["exit_time"])
                hour_count[dt.hour] += 1
            except Exception:
                pass

    # 전체 보유 시간
    all_hold = []
    for t in sells:
        if t["entry_time"] and t["exit_time"]:
            try:
                entry = datetime.fromisoformat(t["entry_time"])
                exit_ = datetime.fromisoformat(t["exit_time"])
                all_hold.append((exit_ - entry).total_seconds() / 3600)
            except Exception:
                pass

    wins = [t for t in sells if t["pnl"] > 0]
    losses = [t for t in sells if t["pnl"] <= 0]

    return {
        "total_sells": len(sells),
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate": round(len(wins) / len(sells) * 100, 1) if sells else 0,
        "total_pnl": round(sum(t["pnl"] for t in sells), 2),
        "avg_win": round(sum(t["pnl"] for t in wins) / len(wins), 2) if wins else 0,
        "avg_loss": round(sum(t["pnl"] for t in losses) / len(losses), 2) if losses else 0,
        "reason_stats": dict(sorted(reason_summary.items(), key=lambda x: -abs(x[1]["pnl"]))),
        "sell_hour_distribution": dict(sorted(hour_count.items())),
        "avg_hold_hours": round(sum(all_hold) / len(all_hold), 2) if all_hold else None,
        "median_hold_hours": round(sorted(all_hold)[len(all_hold) // 2], 2) if all_hold else None,
        "max_hold_hours": round(max(all_hold), 2) if all_hold else None,
    }


def analyze_signal_lifecycle(trades: list[dict]) -> dict:
    """시그널별 진입→청산 라이프사이클 분석."""
    sells = [t for t in trades if t["side"] == "SELL"]

    sig_stats = defaultdict(lambda: {
        "count": 0, "pnl": 0.0, "wins": 0,
        "exit_reasons": defaultdict(int),
        "hold_hours": [], "tickers": defaultdict(int),
    })

    for t in sells:
        sig = t["signal_type"]
        reason = t["exit_reason"] or "unknown"
        sig_stats[sig]["count"] += 1
        sig_stats[sig]["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            sig_stats[sig]["wins"] += 1
        sig_stats[sig]["exit_reasons"][reason] += 1
        sig_stats[sig]["tickers"][t["ticker"]] += 1

        if t["entry_time"] and t["exit_time"]:
            try:
                entry = datetime.fromisoformat(t["entry_time"])
                exit_ = datetime.fromisoformat(t["exit_time"])
                hold_hours = (exit_ - entry).total_seconds() / 3600
                sig_stats[sig]["hold_hours"].append(hold_hours)
            except Exception:
                pass

    result = {}
    for sig, s in sig_stats.items():
        hold_list = s["hold_hours"]
        result[sig] = {
            "count": s["count"],
            "pnl": round(s["pnl"], 2),
            "win_rate": round(s["wins"] / s["count"] * 100, 1) if s["count"] > 0 else 0,
            "avg_pnl": round(s["pnl"] / s["count"], 2) if s["count"] > 0 else 0,
            "exit_reasons": dict(sorted(s["exit_reasons"].items(), key=lambda x: -x[1])),
            "tickers": dict(sorted(s["tickers"].items(), key=lambda x: -x[1])),
            "avg_hold_hours": round(sum(hold_list) / len(hold_list), 2) if hold_list else None,
            "max_hold_hours": round(max(hold_list), 2) if hold_list else None,
        }

    return dict(sorted(result.items(), key=lambda x: -abs(x[1]["pnl"])))


def analyze_entry_exit_mapping(trades: list[dict]) -> list[dict]:
    """각 매수와 대응하는 매도를 날짜/종목 기준으로 매핑 (요약)."""
    # 매도 레코드에는 entry_time도 있으므로 매도 기준으로 분석
    sells = [t for t in trades if t["side"] == "SELL"]

    records = []
    for t in sells:
        entry_dt = None
        exit_dt = None
        hold_hours = None

        if t["entry_time"]:
            try:
                entry_dt = datetime.fromisoformat(t["entry_time"])
            except Exception:
                pass
        if t["exit_time"]:
            try:
                exit_dt = datetime.fromisoformat(t["exit_time"])
            except Exception:
                pass
        if entry_dt and exit_dt:
            hold_hours = round((exit_dt - entry_dt).total_seconds() / 3600, 2)

        records.append({
            "ticker": t["ticker"],
            "signal": t["signal_type"],
            "entry_date": str(entry_dt.date()) if entry_dt else t["date"],
            "entry_hour": entry_dt.hour if entry_dt else None,
            "exit_date": str(exit_dt.date()) if exit_dt else t["date"],
            "exit_hour": exit_dt.hour if exit_dt else None,
            "hold_hours": hold_hours,
            "exit_reason": t["exit_reason"],
            "pnl": t["pnl"],
            "pnl_pct": t["pnl_pct"],
            "win": t["pnl"] > 0,
        })

    return sorted(records, key=lambda x: x["entry_date"])


def main():
    print("=" * 60)
    print("  PTJ v4 Best Trial #388 — 매매 패턴 분석")
    print("=" * 60)

    # ── Step 1: 백테스트 실행 ─────────────────────────────────────
    print("\n[1/4] Best Trial #388 백테스트 실행 중...")
    data = run_backtest_with_best()
    summary = data["summary"]
    trades = data["trades"]

    print(f"  수익률  : {summary['total_return_pct']:+.2f}%")
    print(f"  MDD     : -{summary['mdd']:.2f}%")
    print(f"  Sharpe  : {summary['sharpe']:.4f}")
    print(f"  매수    : {summary['total_buys']}회  매도: {summary['total_sells']}회")
    print(f"  횡보장  : {summary['sideways_days']}일  CB차단: {summary['cb_buy_blocks']}회")

    # ── Step 2: trade log 저장 ────────────────────────────────────
    print("\n[2/4] Trade log 저장 중...")
    trades_dir = _ROOT / "data" / "results" / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    out_path = trades_dir / "v4_phase1_best_trades.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  저장: {out_path}")
    print(f"  총 거래 레코드: {len(trades)}개")

    # ── Step 3: 분석 실행 ─────────────────────────────────────────
    print("\n[3/4] 분석 실행 중...")
    buy_analysis = analyze_buy_patterns(trades)
    sell_analysis = analyze_sell_patterns(trades)
    sig_lifecycle = analyze_signal_lifecycle(trades)
    trade_records = analyze_entry_exit_mapping(trades)

    analysis = {
        "summary": summary,
        "buy_analysis": buy_analysis,
        "sell_analysis": sell_analysis,
        "signal_lifecycle": sig_lifecycle,
        "trade_records": trade_records,
        "generated_at": datetime.now().isoformat(),
    }

    analysis_path = trades_dir / "v4_phase1_best_analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2, default=str)
    print(f"  저장: {analysis_path}")

    # ── Step 4: 콘솔 요약 출력 ────────────────────────────────────
    print("\n[4/4] 분석 요약")

    print("\n  [매수 패턴]")
    print(f"  총 매수: {buy_analysis['total_buys']}회  평균금액: ${buy_analysis['avg_amount']:,.0f}")
    print(f"  시그널별: {buy_analysis['signal_frequency']}")
    print(f"  DCA 레벨별: {buy_analysis['dca_level_distribution']}")
    print(f"  시각별(ET): {buy_analysis['hour_distribution']}")

    print("\n  [매도 패턴]")
    print(f"  총 매도: {sell_analysis['total_sells']}회  승률: {sell_analysis['win_rate']}%")
    print(f"  평균 보유: {sell_analysis['avg_hold_hours']}시간  중앙값: {sell_analysis['median_hold_hours']}시간")
    print(f"  평균 수익: ${sell_analysis['avg_win']:+,.2f}  평균 손실: ${sell_analysis['avg_loss']:+,.2f}")
    print("\n  매도 사유별:")
    for reason, s in sell_analysis["reason_stats"].items():
        print(f"    {reason:35s}: {s['count']:3d}회  P&L {s['pnl']:+8.2f}  승률 {s['win_rate']:5.1f}%  평균보유 {s['avg_hold_hours']}h")

    print("\n  [시그널별 라이프사이클]")
    for sig, s in sig_lifecycle.items():
        print(f"    {sig:25s}: {s['count']:3d}회  P&L {s['pnl']:+8.2f}  승률 {s['win_rate']:5.1f}%  avg보유 {s['avg_hold_hours']}h")
        print(f"      청산방식: {s['exit_reasons']}")

    print(f"\n  분석 완료. 결과: {analysis_path}")
    return analysis


if __name__ == "__main__":
    main()

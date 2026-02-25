"""
Line B 19개 전략 개별 성능 평가 스크립트
=========================================
각 전략을 registry에서 불러와 backtest_1min_v2.parquet 기반 일일 시그널을 생성하고,
전략별 신호 통계(BUY/SELL/HOLD/SKIP 분포, 진입 조건 통과율 등)를 산출한다.

출력: 마크다운 테이블 (stdout) + docs/reports/backtest/line_b_strategy_stats.md
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# ── 프로젝트 루트를 sys.path에 추가 ──────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Line B 전략 등록 (registry 채우기) ────────────────────────────
from simulation.strategies.line_b_taejun.common.base import (
    Action,
    MarketData,
    Position,
    Signal,
)
from simulation.strategies.line_b_taejun.common.registry import (
    _REGISTRY,
    get_strategy,
    list_strategies,
)

# import all strategy modules so @register decorators execute
import simulation.strategies.line_b_taejun.strategies.jab_soxl       # noqa: F401
import simulation.strategies.line_b_taejun.strategies.jab_bitu       # noqa: F401
import simulation.strategies.line_b_taejun.strategies.jab_tsll       # noqa: F401
import simulation.strategies.line_b_taejun.strategies.jab_etq        # noqa: F401
import simulation.strategies.line_b_taejun.strategies.vix_gold       # noqa: F401
import simulation.strategies.line_b_taejun.strategies.sp500_entry    # noqa: F401
import simulation.strategies.line_b_taejun.strategies.bargain_buy    # noqa: F401
import simulation.strategies.line_b_taejun.strategies.short_macro    # noqa: F401
import simulation.strategies.line_b_taejun.strategies.reit_risk      # noqa: F401
import simulation.strategies.line_b_taejun.strategies.sector_rotate  # noqa: F401
import simulation.strategies.line_b_taejun.strategies.bank_conditional  # noqa: F401
import simulation.strategies.line_b_taejun.strategies.emergency_mode    # noqa: F401
import simulation.strategies.line_b_taejun.strategies.crash_buy         # noqa: F401
import simulation.strategies.line_b_taejun.strategies.soxl_independent  # noqa: F401
import simulation.strategies.line_b_taejun.strategies.bear_regime       # noqa: F401
import simulation.strategies.line_b_taejun.strategies.conditional_coin  # noqa: F401
import simulation.strategies.line_b_taejun.strategies.conditional_conl  # noqa: F401
import simulation.strategies.line_b_taejun.strategies.twin_pair         # noqa: F401
import simulation.strategies.line_b_taejun.strategies.bearish_defense   # noqa: F401

# ── 데이터 경로 ──────────────────────────────────────────────────
DATA_DIR = ROOT / "data"
OHLCV_PATH = DATA_DIR / "market" / "ohlcv" / "backtest_1min_v2.parquet"
POLY_DIR = DATA_DIR / "polymarket"
REPORT_PATH = ROOT / "docs" / "reports" / "backtest" / "line_b_strategy_stats.md"


# ══════════════════════════════════════════════════════════════════
# 1. 데이터 로딩
# ══════════════════════════════════════════════════════════════════

def load_daily_data() -> pd.DataFrame:
    """1분봉 parquet → 일봉 OHLCV 요약."""
    print("[1/4] Loading OHLCV data ...")
    df = pd.read_parquet(OHLCV_PATH)
    df["date"] = pd.to_datetime(df["date"])

    daily = (
        df.groupby(["symbol", "date"])
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
    )
    print(f"  → {len(daily)} daily bars, {daily['symbol'].nunique()} symbols, "
          f"{daily['date'].nunique()} trading days")
    return daily


def compute_daily_changes(daily: pd.DataFrame) -> pd.DataFrame:
    """일봉에서 전일 대비 변동률(%) 계산."""
    daily = daily.sort_values(["symbol", "date"])
    daily["prev_close"] = daily.groupby("symbol")["close"].shift(1)
    daily["change_pct"] = (
        (daily["close"] - daily["prev_close"]) / daily["prev_close"] * 100
    )
    return daily


def load_polymarket_probs(date_str: str) -> dict[str, float] | None:
    """특정 날짜의 Polymarket 확률을 로딩한다.

    btc_up, ndx_up 형태로 반환. 데이터 없으면 None.
    """
    year = date_str[:4]
    fname = f"{date_str}_1m.json"
    fpath = POLY_DIR / year / fname
    if not fpath.exists():
        return None

    try:
        with open(fpath) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    indicators = data.get("indicators", {})
    probs: dict[str, float] = {}

    # btc_up: Bitcoin Up market의 마지막 확률
    btc = indicators.get("btc_up_down", {})
    btc_markets = btc.get("markets", [])
    for m in btc_markets:
        q = m.get("question", "").lower()
        if "up" in q and "down" not in q:
            ts = m.get("timeseries", [])
            if ts:
                probs["btc_up"] = ts[-1].get("price", 0.5)
                break
    # fallback: use first market, interpret price[0] as Up prob
    if "btc_up" not in probs and btc_markets:
        m0 = btc_markets[0]
        ts = m0.get("timeseries", [])
        if ts:
            # If first market has "Up" in question, use its price
            q = m0.get("question", "").lower()
            if "up" in q:
                probs["btc_up"] = ts[-1].get("price", 0.5)
            else:
                # "Down" market → invert
                probs["btc_up"] = 1.0 - ts[-1].get("price", 0.5)

    # ndx_up
    ndx = indicators.get("ndx_up_down", {})
    ndx_markets = ndx.get("markets", [])
    for m in ndx_markets:
        q = m.get("question", "").lower()
        if "up" in q and "down" not in q:
            ts = m.get("timeseries", [])
            if ts:
                probs["ndx_up"] = ts[-1].get("price", 0.5)
                break
    if "ndx_up" not in probs and ndx_markets:
        m0 = ndx_markets[0]
        ts = m0.get("timeseries", [])
        if ts:
            q = m0.get("question", "").lower()
            if "up" in q:
                probs["ndx_up"] = ts[-1].get("price", 0.5)
            else:
                probs["ndx_up"] = 1.0 - ts[-1].get("price", 0.5)

    return probs if probs else None


# ══════════════════════════════════════════════════════════════════
# 2. MarketData 어댑터
# ══════════════════════════════════════════════════════════════════

def build_market_data(
    date_row: pd.Timestamp,
    daily: pd.DataFrame,
    poly_probs: dict[str, float] | None,
) -> MarketData:
    """일봉 데이터 → Line B MarketData 변환.

    daily는 date==date_row 인 행만 전달.
    """
    day_data = daily[daily["date"] == date_row]

    changes: dict[str, float] = {}
    prices: dict[str, float] = {}
    volumes: dict[str, float] = {}

    for _, row in day_data.iterrows():
        sym = row["symbol"]
        changes[sym] = row["change_pct"] if pd.notna(row["change_pct"]) else 0.0
        prices[sym] = row["close"]
        volumes[sym] = row["volume"]

    # 시간: 17:30 KST (프리마켓 잽모드 윈도우 통과용)
    ts = datetime(date_row.year, date_row.month, date_row.day, 17, 35)

    return MarketData(
        changes=changes,
        prices=prices,
        poly=poly_probs,
        time=ts,
        volumes=volumes,
    )


# ══════════════════════════════════════════════════════════════════
# 3. 평가 루프
# ══════════════════════════════════════════════════════════════════

def evaluate_strategies(daily: pd.DataFrame) -> dict:
    """모든 전략에 대해 일별 generate_signal()을 호출하고 통계를 수집한다."""
    print("[2/4] Instantiating strategies ...")
    strategies: dict[str, object] = {}
    instantiation_errors: dict[str, str] = {}

    for name in list_strategies():
        try:
            s = get_strategy(name)
            strategies[name] = s
        except Exception as e:
            instantiation_errors[name] = str(e)
            print(f"  ✗ {name}: {e}")

    print(f"  → {len(strategies)} strategies loaded, {len(instantiation_errors)} failed")

    # 거래일 목록
    dates = sorted(daily["date"].unique())
    print(f"\n[3/4] Evaluating {len(strategies)} strategies × {len(dates)} days ...")

    # 결과 저장
    stats: dict[str, dict] = {}
    for name in strategies:
        stats[name] = {
            "BUY": 0,
            "SELL": 0,
            "HOLD": 0,
            "SKIP": 0,
            "ERROR": 0,
            "entry_checks": 0,
            "entry_pass": 0,
            "total_days": 0,
            "errors": [],
        }

    # evaluate()-only 전략 목록
    EVALUATE_ONLY = {"conditional_coin", "conditional_conl", "twin_pair", "bearish_defense"}

    for i, date in enumerate(dates):
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        poly = load_polymarket_probs(date_str)
        md = build_market_data(pd.Timestamp(date), daily, poly)

        if (i + 1) % 50 == 0:
            print(f"  ... day {i+1}/{len(dates)} ({date_str})")

        for name, strat in strategies.items():
            stats[name]["total_days"] += 1

            # ── evaluate()-only 전략은 별도 처리 ─────────────
            if name in EVALUATE_ONLY:
                try:
                    if name == "conditional_coin":
                        changes_dict = {
                            t: {"change_pct": md.changes.get(t, 0.0)}
                            for t in ["ETHU", "XXRP", "SOLT", "COIN"]
                        }
                        result = strat.evaluate(changes_dict)
                        if result.get("buy_signal"):
                            stats[name]["BUY"] += 1
                            stats[name]["entry_pass"] += 1
                        else:
                            stats[name]["SKIP"] += 1
                        stats[name]["entry_checks"] += 1

                    elif name == "conditional_conl":
                        changes_dict = {
                            t: {"change_pct": md.changes.get(t, 0.0)}
                            for t in ["ETHU", "XXRP", "SOLT", "CONL"]
                        }
                        result = strat.evaluate(changes_dict)
                        if result.get("buy_signal"):
                            stats[name]["BUY"] += 1
                            stats[name]["entry_pass"] += 1
                        else:
                            stats[name]["SKIP"] += 1
                        stats[name]["entry_checks"] += 1

                    elif name == "twin_pair":
                        changes_dict = {
                            t: {"change_pct": md.changes.get(t, 0.0)}
                            for t in ["COIN", "MSTU", "IRE", "CONL",
                                      "SOXL", "BITU", "ROBN", "ETHU"]
                        }
                        # Standard twin pairs
                        pairs = {
                            "coin_crypto": {
                                "lead": "COIN",
                                "follow": ["MSTU", "IRE"],
                                "label": "COIN 크립토",
                            },
                            "soxl_semi": {
                                "lead": "SOXL",
                                "follow": ["CONL"],
                                "label": "SOXL 반도체",
                            },
                        }
                        results = strat.evaluate(changes_dict, pairs)
                        entry_found = any(r.get("signal") == "ENTRY" for r in results)
                        sell_found = any(r.get("signal") == "SELL" for r in results)
                        if entry_found:
                            stats[name]["BUY"] += 1
                            stats[name]["entry_pass"] += 1
                        elif sell_found:
                            stats[name]["SELL"] += 1
                        else:
                            stats[name]["SKIP"] += 1
                        stats[name]["entry_checks"] += 1

                    elif name == "bearish_defense":
                        # Simple mode check (we don't have mode data, default normal)
                        result = strat.evaluate("normal")
                        if result.get("buy_brku"):
                            stats[name]["BUY"] += 1
                            stats[name]["entry_pass"] += 1
                        else:
                            stats[name]["SKIP"] += 1
                        stats[name]["entry_checks"] += 1

                except Exception as e:
                    stats[name]["ERROR"] += 1
                    if len(stats[name]["errors"]) < 3:
                        stats[name]["errors"].append(f"{date_str}: {e}")
                continue

            # ── generate_signal() 전략 ─────────────────────
            try:
                # check_entry (entry 가능 여부)
                entry_ok = strat.check_entry(md)
                stats[name]["entry_checks"] += 1
                if entry_ok:
                    stats[name]["entry_pass"] += 1

                # generate_signal (포지션 없음 → 진입 검토)
                sig = strat.generate_signal(md, None)
                action = sig.action.value  # BUY / SELL / HOLD / SKIP
                stats[name][action] += 1

            except Exception as e:
                stats[name]["ERROR"] += 1
                if len(stats[name]["errors"]) < 3:
                    stats[name]["errors"].append(f"{date_str}: {e}")

    return {
        "stats": stats,
        "instantiation_errors": instantiation_errors,
        "num_days": len(dates),
        "num_strategies": len(strategies),
        "date_range": (str(dates[0])[:10], str(dates[-1])[:10]),
    }


# ══════════════════════════════════════════════════════════════════
# 4. 리포트 생성
# ══════════════════════════════════════════════════════════════════

def generate_report(results: dict) -> str:
    """마크다운 리포트를 생성한다."""
    stats = results["stats"]
    inst_errs = results["instantiation_errors"]
    num_days = results["num_days"]
    date_range = results["date_range"]
    lines: list[str] = []

    lines.append("# Line B 전략별 시그널 통계 리포트")
    lines.append("")
    lines.append(f"- **평가 기간**: {date_range[0]} ~ {date_range[1]} ({num_days} 거래일)")
    lines.append(f"- **평가 전략 수**: {results['num_strategies']} / {results['num_strategies'] + len(inst_errs)}")
    lines.append(f"- **데이터 소스**: `data/market/ohlcv/backtest_1min_v2.parquet` (일봉 변환)")
    lines.append(f"- **Polymarket**: `data/polymarket/` JSON (btc_up, ndx_up)")
    lines.append(f"- **생성일**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # ── 전략 분류 ──────────────────────────────────────────
    lines.append("## 전략 분류")
    lines.append("")
    categories = {
        "잽모드 단타 (Jab)": ["jab_soxl", "jab_bitu", "jab_tsll", "jab_etq"],
        "이벤트 드리븐": ["vix_gold", "emergency_mode", "crash_buy"],
        "매크로/방어": ["short_macro", "reit_risk", "bear_regime_long", "bearish_defense"],
        "조건부 매매": ["bank_conditional", "sp500_entry", "conditional_coin",
                      "conditional_conl", "twin_pair"],
        "장기/로테이션": ["bargain_buy", "sector_rotate", "soxl_independent"],
    }
    for cat, strats in categories.items():
        registered = [s for s in strats if s in stats]
        lines.append(f"- **{cat}**: {', '.join(registered)}")
    lines.append("")

    # ── 메인 통계 테이블 ──────────────────────────────────
    lines.append("## 전략별 시그널 통계")
    lines.append("")
    lines.append("| 전략 | BUY | SELL | HOLD | SKIP | ERROR | 진입률 | 신호/일 | 비고 |")
    lines.append("|------|----:|-----:|-----:|-----:|------:|-------:|--------:|------|")

    # 정렬: BUY 빈도 내림차순
    sorted_names = sorted(stats.keys(), key=lambda n: stats[n]["BUY"], reverse=True)

    for name in sorted_names:
        s = stats[name]
        total_signals = s["BUY"] + s["SELL"] + s["HOLD"] + s["SKIP"]
        entry_rate = (
            f"{s['entry_pass'] / s['entry_checks'] * 100:.1f}%"
            if s["entry_checks"] > 0
            else "N/A"
        )
        signals_per_day = f"{(s['BUY'] + s['SELL']) / max(s['total_days'], 1):.3f}"
        notes = ""
        if s["ERROR"] > 0:
            notes = f"err: {s['errors'][0][:40]}" if s["errors"] else f"{s['ERROR']} errors"

        lines.append(
            f"| {name} | {s['BUY']} | {s['SELL']} | {s['HOLD']} | {s['SKIP']} "
            f"| {s['ERROR']} | {entry_rate} | {signals_per_day} | {notes} |"
        )

    lines.append("")

    # ── 활동도 순위 ─────────────────────────────────────
    lines.append("## 활동도 순위 (BUY 시그널 기준)")
    lines.append("")
    for rank, name in enumerate(sorted_names[:10], 1):
        s = stats[name]
        lines.append(f"{rank}. **{name}** — BUY {s['BUY']}회 "
                      f"({s['BUY'] / max(s['total_days'], 1) * 100:.1f}%일)")

    lines.append("")

    # ── 진입 조건 통과율 분석 ────────────────────────────
    lines.append("## 진입 조건 통과율 (check_entry)")
    lines.append("")
    lines.append("| 전략 | 검사일 | 통과 | 통과율 |")
    lines.append("|------|-------:|-----:|-------:|")
    for name in sorted_names:
        s = stats[name]
        if s["entry_checks"] > 0:
            rate = s["entry_pass"] / s["entry_checks"] * 100
            lines.append(
                f"| {name} | {s['entry_checks']} | {s['entry_pass']} | {rate:.1f}% |"
            )
    lines.append("")

    # ── 비활성 전략 ─────────────────────────────────────
    inactive = [n for n in sorted_names if stats[n]["BUY"] == 0 and stats[n]["SELL"] == 0]
    if inactive:
        lines.append("## 비활성 전략 (BUY/SELL 시그널 0회)")
        lines.append("")
        for name in inactive:
            s = stats[name]
            reason = ""
            if s["ERROR"] > 0:
                reason = f"오류 발생 ({s['ERROR']}회)"
            elif s["entry_pass"] == 0:
                reason = "진입 조건 미충족 (전 기간)"
            else:
                reason = "진입 조건 통과하나 generate_signal()에서 SKIP"
            lines.append(f"- **{name}**: {reason}")
        lines.append("")

    # ── 인스턴스화 실패 ──────────────────────────────────
    if inst_errs:
        lines.append("## 인스턴스화 실패 전략")
        lines.append("")
        for name, err in inst_errs.items():
            lines.append(f"- **{name}**: `{err}`")
        lines.append("")

    # ── 데이터 제약 사항 ──────────────────────────────────
    lines.append("## 데이터 제약 사항")
    lines.append("")
    lines.append("1. **시장 데이터 한계**: backtest_1min_v2.parquet에 포함된 16개 종목만 평가 가능")
    lines.append("   - 포함: BABX, BITU, BRKU, COIN, CONL, ETHU, GLD, IRE, MSTU, NVDL, QQQ, ROBN, SOLT, SOXL, SPY, XXRP")
    lines.append("   - 미포함: SOXX, TSLA, TSLL, VIX, GDXU, IAU, BAC, JPM, HSBC, WFC, RBC, C, ETQ, SOXS, BITI 등")
    lines.append("2. **Polymarket 데이터**: btc_up, ndx_up만 추출 (btc_monthly_dip 등 미포함)")
    lines.append("3. **history 데이터 없음**: 3년 최고가, 1Y 저가, VNQ 120일선 등 미제공 → bargain_buy, sector_rotate 등 제한")
    lines.append("4. **crypto 스팟 데이터 없음**: jab_bitu의 BTC/ETH/SOL/XRP 스팟 변동률 미제공")
    lines.append("5. **OHLCV 히스토리 없음**: ADX/EMA 기반 필터(conditional_conl, soxl_independent) 제한적")
    lines.append("6. **evaluate()-only 전략**: conditional_coin, conditional_conl, twin_pair, bearish_defense는 "
                  "evaluate() 직접 호출로 평가")
    lines.append("")

    # ── 전략 요약 해석 ──────────────────────────────────
    lines.append("## 해석 및 참고사항")
    lines.append("")
    lines.append("- **잽모드 전략 (jab_soxl/bitu/tsll/etq)**: 프리마켓 17:30 KST 이후, Polymarket + 개별종목 조건 ALL 충족 필요.")
    lines.append("  실제 데이터에서는 개별 반도체 11종목(NVDA, AMD 등)이나 크립토 스팟이 없어 조건 미충족이 대부분.")
    lines.append("- **bargain_buy**: 3년 최고가 대비 하락률 데이터(history) 필요. 미제공시 항상 SKIP.")
    lines.append("- **sector_rotate**: 1Y 저가 대비 상승률(history) 필요. 미제공시 항상 SKIP.")
    lines.append("- **vix_gold**: VIX 변동률 데이터 필요. backtest 데이터에 VIX 미포함 → SKIP.")
    lines.append("- **short_macro**: 나스닥/S&P500 ATH 데이터(history._macro) 필요. 미제공 → SKIP.")
    lines.append("- **crash_buy**: -30% 급락 + 15:55 ET 시간 필요. 시간 조건이 17:35 KST로 설정되어 미매칭.")
    lines.append("- **emergency_mode**: poly_prev(이전 확률) 필요. 미제공 → SKIP.")
    lines.append("- **bear_regime_long**: btc_up < 0.43 + btc_monthly_dip > 0.30 필요. dip 미제공.")
    lines.append("- **twin_pair**: 갭 분석으로 실제 ENTRY/SELL 빈도 측정 가능.")
    lines.append("- **conditional_coin/conl**: 트리거 종목(ETHU/XXRP/SOLT) 변동률 기반 직접 평가 가능.")
    lines.append("")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Line B Strategy Evaluation — Signal Statistics")
    print("=" * 60)

    # 데이터 로딩
    daily = load_daily_data()
    daily = compute_daily_changes(daily)

    # 전략 평가
    results = evaluate_strategies(daily)

    # 리포트 생성
    print("\n[4/4] Generating report ...")
    report = generate_report(results)

    # stdout 출력
    print("\n" + report)

    # 파일 저장
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\n✓ Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()

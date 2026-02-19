"""
쌍둥이 갭 백테스트 — 5분봉 기반 시뮬레이션
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import numpy as np
import config

# ── 설정 ──────────────────────────────────────────────
DATA_PATH = config.OHLCV_DIR / "backtest_5min.parquet"
OUTPUT_PATH = config.RESULTS_DIR / "backtests" / "backtest_twin_gap.csv"

PAIRS = {
    "coin": {"lead": "BITU", "follow": "MSTU"},
    "bank": {"lead": "ROBN", "follow": "CONL"},
    "semi": {"lead": "NVDL", "follow": "AMDL"},
}

PAIR_GAP_ENTRY_THRESHOLD = 1.5   # 갭 ±1.5% 이상 → 매수
PAIR_GAP_SELL_THRESHOLD = 0.3    # 갭 ±0.3% 이내 → 매도
STOP_LOSS_PCT = -3.0             # -3% 손절


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values(["symbol", "date", "timestamp"]).reset_index(drop=True)
    return df


def compute_intraday_pct(group: pd.DataFrame) -> pd.Series:
    """당일 시가 대비 변화율(%)."""
    open_of_day = group["open"].iloc[0]
    return (group["close"] - open_of_day) / open_of_day * 100


def simulate_pair(df: pd.DataFrame, pair_name: str, lead: str, follow: str) -> list[dict]:
    """한 페어에 대해 전 거래일 시뮬레이션."""
    lead_df = df[df["symbol"] == lead].copy()
    follow_df = df[df["symbol"] == follow].copy()

    dates = sorted(set(lead_df["date"]) & set(follow_df["date"]))
    trades = []

    for date in dates:
        ld = lead_df[lead_df["date"] == date].sort_values("timestamp").reset_index(drop=True)
        fd = follow_df[follow_df["date"] == date].sort_values("timestamp").reset_index(drop=True)

        if ld.empty or fd.empty:
            continue

        # 당일 시가 대비 변화율
        lead_open = ld["open"].iloc[0]
        follow_open = fd["open"].iloc[0]

        # 타임스탬프 기준 병합
        merged = pd.merge(
            ld[["timestamp", "close"]].rename(columns={"close": "lead_close"}),
            fd[["timestamp", "close", "low"]].rename(columns={"close": "follow_close", "low": "follow_low"}),
            on="timestamp",
            how="inner",
        )
        if merged.empty:
            continue

        merged["lead_pct"] = (merged["lead_close"] - lead_open) / lead_open * 100
        merged["follow_pct"] = (merged["follow_close"] - follow_open) / follow_open * 100
        merged["gap"] = merged["lead_pct"] - merged["follow_pct"]

        # ── 시뮬레이션 ──
        in_position = False
        entry_time = entry_price = gap_at_entry = None
        gap_direction = 0  # +1: lead가 앞서감(follow 매수), -1: follow가 앞서감

        for i, row in merged.iterrows():
            ts = row["timestamp"]

            if not in_position:
                # 진입 조건: 갭이 ±threshold 이상
                if abs(row["gap"]) >= PAIR_GAP_ENTRY_THRESHOLD:
                    in_position = True
                    entry_time = ts
                    entry_price = row["follow_close"]
                    gap_at_entry = row["gap"]
                    gap_direction = 1 if row["gap"] > 0 else -1
            else:
                # 손절 체크 (봉 저가 기준)
                drawdown = (row["follow_low"] - entry_price) / entry_price * 100
                if drawdown <= STOP_LOSS_PCT:
                    exit_price = entry_price * (1 + STOP_LOSS_PCT / 100)
                    trades.append(_make_trade(
                        date, pair_name, lead, follow,
                        entry_time, entry_price,
                        ts, exit_price,
                        gap_at_entry, row["gap"], "stop_loss",
                    ))
                    in_position = False
                    continue

                # 수렴 매도: 갭이 ±sell_threshold 이내
                if abs(row["gap"]) <= PAIR_GAP_SELL_THRESHOLD:
                    trades.append(_make_trade(
                        date, pair_name, lead, follow,
                        entry_time, entry_price,
                        ts, row["follow_close"],
                        gap_at_entry, row["gap"], "converge",
                    ))
                    in_position = False
                    continue

        # 장마감 강제 청산
        if in_position:
            last = merged.iloc[-1]
            trades.append(_make_trade(
                date, pair_name, lead, follow,
                entry_time, entry_price,
                last["timestamp"], last["follow_close"],
                gap_at_entry, last["gap"], "eod_close",
            ))

    return trades


def _make_trade(date, pair, lead, follow,
                entry_time, entry_price, exit_time, exit_price,
                gap_at_entry, gap_at_exit, exit_reason) -> dict:
    ret_pct = (exit_price - entry_price) / entry_price * 100
    hold_minutes = (exit_time - entry_time).total_seconds() / 60
    return {
        "date": date,
        "pair": pair,
        "lead": lead,
        "follow": follow,
        "entry_time": entry_time,
        "entry_price": round(entry_price, 4),
        "exit_time": exit_time,
        "exit_price": round(exit_price, 4),
        "return_pct": round(ret_pct, 4),
        "hold_minutes": int(hold_minutes),
        "exit_reason": exit_reason,
        "gap_at_entry": round(gap_at_entry, 4),
        "gap_at_exit": round(gap_at_exit, 4),
    }


def print_summary(trades_df: pd.DataFrame):
    """페어별 + 전체 요약 통계 출력."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("  쌍둥이 갭 백테스트 결과 요약")
    print(f"{sep}\n")

    for pair in trades_df["pair"].unique():
        pdf = trades_df[trades_df["pair"] == pair]
        _print_pair_stats(pair, pdf)

    print(f"\n{sep}")
    print("  전체 (All Pairs)")
    print(f"{sep}")
    _print_pair_stats("ALL", trades_df)


def _print_pair_stats(label: str, df: pd.DataFrame):
    n = len(df)
    if n == 0:
        print(f"\n[{label}] 거래 없음")
        return

    wins = df[df["return_pct"] > 0]
    losses = df[df["return_pct"] <= 0]
    win_rate = len(wins) / n * 100

    by_reason = df["exit_reason"].value_counts()

    print(f"\n── {label} ({df['lead'].iloc[0]} → {df['follow'].iloc[0]}) ──" if label != "ALL" else "")
    print(f"  총 거래 수      : {n}")
    print(f"  승률            : {win_rate:.1f}%  ({len(wins)}승 / {len(losses)}패)")
    print(f"  평균 수익률     : {df['return_pct'].mean():.2f}%")
    print(f"  중앙값 수익률   : {df['return_pct'].median():.2f}%")
    print(f"  최대 수익       : {df['return_pct'].max():.2f}%")
    print(f"  최대 손실       : {df['return_pct'].min():.2f}%")
    print(f"  수익률 표준편차 : {df['return_pct'].std():.2f}%")
    print(f"  누적 수익률     : {df['return_pct'].sum():.2f}%")
    print(f"  평균 보유시간   : {df['hold_minutes'].mean():.0f}분")
    print(f"  매도 사유       : {by_reason.to_dict()}")


def main():
    print("데이터 로드 중...")
    df = load_data()

    all_trades = []
    for pair_name, pair_info in PAIRS.items():
        print(f"  시뮬레이션: {pair_name} ({pair_info['lead']} → {pair_info['follow']})")
        trades = simulate_pair(df, pair_name, pair_info["lead"], pair_info["follow"])
        all_trades.extend(trades)
        print(f"    → {len(trades)} 거래 생성")

    trades_df = pd.DataFrame(all_trades)
    trades_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n결과 저장: {OUTPUT_PATH} ({len(trades_df)} rows)")

    print_summary(trades_df)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize stock buys and holdings in a date range."
    )
    parser.add_argument(
        "--input",
        default="toss/csv/stock_analysis_ready.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--start",
        default="2024-02-01",
        help="Start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        default="2026-02-28",
        help="End date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--out-dir",
        default="toss/csv",
        help="Output directory.",
    )
    return parser.parse_args()


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_symbol_name(value: object) -> str:
    text = clean_text(value)
    if not text:
        return ""

    # Remove full ISIN suffix like "(US1234567890)"
    text = re.sub(r"\([A-Z]{2}[A-Z0-9]{10}\)", "", text)
    # Remove dangling/incomplete code suffix like "(US921"
    text = re.sub(r"\([A-Z]{2}[A-Z0-9]*$", "", text)
    # Remove any unmatched "(" tail.
    text = re.sub(r"\([^)]*$", "", text)
    text = text.strip(" -_/")
    text = re.sub(r"\s+", " ", text)
    # Common truncation fix.
    text = re.sub(r"\sET$", " ETF", text)
    return text.strip()


def event_delta(tx_class: str, tx_type: str) -> int:
    tx_class = clean_text(tx_class)
    tx_type = clean_text(tx_type)
    if tx_class == "stock_buy":
        return 1
    if tx_class == "stock_sell":
        return -1
    if tx_class == "position_move":
        if "입고" in tx_type:
            return 1
        if "출고" in tx_type:
            return -1
    return 0


def first_nonempty(series: pd.Series) -> str:
    for v in series:
        text = clean_text(v)
        if text:
            return text
    return ""


def can_merge_suffix(suffix: str) -> bool:
    if not suffix:
        return False
    # Avoid merging distinct products like "... 2배", "... ETF", "... ETN".
    if suffix.startswith(" "):
        return False
    # Keep distinct when numeric suffix appears.
    if re.search(r"\d", suffix):
        return False
    return len(suffix) <= 3


def build_truncation_map(counts: pd.Series) -> dict[str, str]:
    names = counts.index.tolist()
    direct_map: dict[str, str] = {n: n for n in names}

    names_asc = sorted(names, key=len)
    for name in names_asc:
        best = name
        best_count = int(counts.get(name, 0))
        for cand in names:
            if len(cand) <= len(name):
                continue
            if not cand.startswith(name):
                continue
            suffix = cand[len(name) :]
            if not can_merge_suffix(suffix):
                continue
            cand_count = int(counts.get(cand, 0))
            if len(cand) > len(best) or (len(cand) == len(best) and cand_count >= best_count):
                best = cand
                best_count = cand_count
        direct_map[name] = best

    def resolve(name: str) -> str:
        seen: set[str] = set()
        cur = name
        while cur in direct_map and direct_map[cur] != cur and cur not in seen:
            seen.add(cur)
            cur = direct_map[cur]
        return cur

    return {name: resolve(name) for name in names}


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)

    df = pd.read_csv(in_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["type"] = df["type"].map(clean_text)
    df["tx_class"] = df["tx_class"].map(clean_text)
    df["symbol_name"] = df["symbol_name"].map(clean_text)
    df["isin"] = df["isin"].map(clean_text)
    df["signed_amount"] = pd.to_numeric(df["signed_amount"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Work only on stock-related rows with a date.
    df = df[df["date"].notna() & (df["stock_related"] == True)].copy()
    if df.empty:
        print("No stock-related rows found.")
        return 0

    df["symbol_norm"] = df["symbol_name"].map(normalize_symbol_name)
    no_isin_counts = df.loc[df["isin"] == "", "symbol_norm"].value_counts()
    trunc_map = build_truncation_map(no_isin_counts)
    df["symbol_canon"] = np.where(
        df["isin"] != "",
        df["symbol_norm"],
        df["symbol_norm"].map(lambda s: trunc_map.get(s, s)),
    )
    df["symbol_key"] = np.where(df["isin"] != "", df["isin"], df["symbol_canon"])
    df = df[df["symbol_key"] != ""].copy()

    # Build display labels (prefer most common normalized name).
    name_map = (
        df.groupby("symbol_key")["symbol_canon"]
        .agg(lambda s: first_nonempty(s.value_counts().index.tolist()))
        .to_dict()
    )
    isin_map = (
        df.groupby("symbol_key")["isin"]
        .agg(lambda s: first_nonempty(s))
        .to_dict()
    )

    period = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    if period.empty:
        print("No rows in selected date range.")
        return 0

    # 1) Buy summary in period
    buys = period[period["tx_class"] == "stock_buy"].copy()
    sells = period[period["tx_class"] == "stock_sell"].copy()

    buy_agg = (
        buys.groupby("symbol_key")
        .agg(
            first_buy_date=("date", "min"),
            last_buy_date=("date", "max"),
            buy_count=("tx_class", "size"),
            buy_amount_sum=("amount", "sum"),
        )
        .reset_index()
    )
    sell_agg = (
        sells.groupby("symbol_key")
        .agg(
            sell_count=("tx_class", "size"),
            sell_amount_sum=("amount", "sum"),
        )
        .reset_index()
    )
    buy_summary = buy_agg.merge(sell_agg, on="symbol_key", how="left")
    buy_summary["sell_count"] = buy_summary["sell_count"].fillna(0).astype(int)
    buy_summary["sell_amount_sum"] = buy_summary["sell_amount_sum"].fillna(0.0)
    buy_summary["net_cash_flow"] = buy_summary["sell_amount_sum"] - buy_summary["buy_amount_sum"]
    buy_summary["symbol"] = buy_summary["symbol_key"].map(name_map)
    buy_summary["isin"] = buy_summary["symbol_key"].map(isin_map)
    buy_summary = buy_summary[
        [
            "symbol",
            "isin",
            "first_buy_date",
            "last_buy_date",
            "buy_count",
            "buy_amount_sum",
            "sell_count",
            "sell_amount_sum",
            "net_cash_flow",
        ]
    ].sort_values(["buy_amount_sum", "buy_count"], ascending=[False, False])
    buy_summary["first_buy_date"] = buy_summary["first_buy_date"].dt.strftime("%Y-%m-%d")
    buy_summary["last_buy_date"] = buy_summary["last_buy_date"].dt.strftime("%Y-%m-%d")

    start_tag = start.strftime("%Y%m")
    end_tag = end.strftime("%Y%m")
    buy_out = out_dir / f"stock_buy_summary_{start_tag}_{end_tag}.csv"
    buy_summary.to_csv(buy_out, index=False, encoding="utf-8-sig")
    buy_symbols_txt = out_dir / f"stock_buy_symbols_{start_tag}_{end_tag}.txt"
    with buy_symbols_txt.open("w", encoding="utf-8") as f:
        for s in sorted(buy_summary["symbol"].dropna().astype(str).unique()):
            s = clean_text(s)
            if s:
                f.write(s + "\n")

    # 2) Held symbols in period (transaction-based estimate)
    period["event_delta"] = period.apply(
        lambda r: event_delta(r["tx_class"], r["type"]), axis=1
    )
    held = (
        period.groupby("symbol_key")
        .agg(
            first_seen=("date", "min"),
            last_seen=("date", "max"),
            buy_count=("tx_class", lambda s: int((s == "stock_buy").sum())),
            sell_count=("tx_class", lambda s: int((s == "stock_sell").sum())),
            dividend_count=("tx_class", lambda s: int((s == "dividend").sum())),
            position_move_count=("tx_class", lambda s: int((s == "position_move").sum())),
            net_cash_flow=("signed_amount", "sum"),
            event_delta_sum=("event_delta", "sum"),
        )
        .reset_index()
    )
    held["symbol"] = held["symbol_key"].map(name_map)
    held["isin"] = held["symbol_key"].map(isin_map)
    held = held[
        [
            "symbol",
            "isin",
            "first_seen",
            "last_seen",
            "buy_count",
            "sell_count",
            "dividend_count",
            "position_move_count",
            "net_cash_flow",
            "event_delta_sum",
        ]
    ].sort_values(["symbol", "isin"])
    held["first_seen"] = held["first_seen"].dt.strftime("%Y-%m-%d")
    held["last_seen"] = held["last_seen"].dt.strftime("%Y-%m-%d")

    held_out = out_dir / f"held_symbols_{start_tag}_{end_tag}.csv"
    held.to_csv(held_out, index=False, encoding="utf-8-sig")

    symbols_txt = out_dir / f"held_symbols_{start_tag}_{end_tag}.txt"
    with symbols_txt.open("w", encoding="utf-8") as f:
        for s in sorted(held["symbol"].dropna().astype(str).unique()):
            s = clean_text(s)
            if s:
                f.write(s + "\n")

    print(f"[OK] buy summary -> {buy_out} ({len(buy_summary)} symbols)")
    print(f"[OK] buy list -> {buy_symbols_txt}")
    print(f"[OK] held symbols -> {held_out} ({len(held)} symbols)")
    print(f"[OK] held list -> {symbols_txt}")
    print(
        f"[INFO] data date range in input: {df['date'].min().date()} to {df['date'].max().date()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

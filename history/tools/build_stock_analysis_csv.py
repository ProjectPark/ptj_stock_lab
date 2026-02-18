#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

OUTFLOW_KEYWORDS = (
    "출금",
    "출고",
    "구매",
    "수수료",
    "외국납부세액",
    "세액출금",
)
INFLOW_KEYWORDS = (
    "입금",
    "입고",
    "판매",
    "배당",
    "환급",
    "페이백",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one stock-analysis-ready CSV from normalized transactions."
    )
    parser.add_argument(
        "--input",
        default="toss/csv/all_merged_normalized.csv",
        help="Input normalized CSV path.",
    )
    parser.add_argument(
        "--output",
        default="toss/csv/stock_analysis_ready.csv",
        help="Output single CSV path.",
    )
    return parser.parse_args()


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def extract_isin(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""
    m = re.search(r"\(([A-Z]{2}[A-Z0-9]{10})\)", text)
    return m.group(1) if m else ""


def extract_symbol_name(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""
    text = re.sub(r"\([A-Z]{2}[A-Z0-9]{10}\)", "", text)
    text = re.sub(r"\(\s*\)", "", text)
    return clean_text(text)


def classify_transaction(tx_type: str, desc: str, isin: str) -> str:
    t = clean_text(tx_type)
    d = clean_text(desc)
    if "구매" in t:
        return "stock_buy"
    if "판매" in t:
        return "stock_sell"
    if "배당" in t:
        return "dividend"
    if "환전" in t:
        return "fx"
    if "이체" in t:
        return "transfer"
    if "수수료" in t:
        return "fee"
    if "세액" in t or "세금" in t:
        return "tax"
    if "분할" in t or "대체" in t or "입고" in t or "출고" in t:
        return "position_move"
    if isin:
        return "stock_other"
    if "ETF" in d.upper():
        return "stock_other"
    return "other"


def cash_direction(tx_type: str) -> int:
    tx_type = clean_text(tx_type)
    if any(k in tx_type for k in OUTFLOW_KEYWORDS):
        return -1
    if any(k in tx_type for k in INFLOW_KEYWORDS):
        return 1
    return 0


def is_stock_related(tx_class: str, isin: str, desc: str) -> bool:
    tx_class = clean_text(tx_class)
    isin = clean_text(isin)
    desc = clean_text(desc)
    if tx_class.startswith("stock_"):
        return True
    if tx_class in {"dividend", "position_move"}:
        return True
    if isin:
        return True
    return "ETF" in desc.upper()


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    base = df.copy()
    base["type"] = base["type"].map(clean_text)
    base["description"] = base["description"].map(clean_text)
    base["type"] = base["type"].replace("", pd.NA)
    base["description"] = base["description"].replace("", pd.NA)
    base["date"] = pd.to_datetime(base["date"], errors="coerce")

    base["isin"] = base["description"].map(extract_isin)
    base["symbol_name"] = base["description"].map(extract_symbol_name)
    base["tx_class"] = base.apply(
        lambda r: classify_transaction(r["type"], r["description"], r["isin"]), axis=1
    )
    base["cash_sign"] = base["type"].map(cash_direction)
    base["amount"] = pd.to_numeric(base["amount"], errors="coerce")
    base["balance"] = pd.to_numeric(base["balance"], errors="coerce")
    base["signed_amount"] = base["amount"] * base["cash_sign"]
    base.loc[base["cash_sign"] == 0, "signed_amount"] = pd.NA
    base["stock_related"] = base.apply(
        lambda r: is_stock_related(r["tx_class"], r["isin"], r["description"]), axis=1
    )

    # Keep a single row per exact transaction key after normalization.
    base = base.drop_duplicates(
        subset=[
            "_source_file",
            "_page",
            "_table",
            "date",
            "type",
            "description",
            "amount",
            "balance",
        ]
    )
    base = base[base["date"].notna()]
    base = base[base["type"].notna()]

    cols = [
        "date",
        "type",
        "tx_class",
        "stock_related",
        "description",
        "symbol_name",
        "isin",
        "amount",
        "cash_sign",
        "signed_amount",
        "balance",
        "_source_file",
        "_page",
        "_table",
    ]
    final = base[cols].sort_values(["date", "_source_file", "_page", "_table"])
    final.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[OK] {in_path} -> {out_path} ({len(final)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

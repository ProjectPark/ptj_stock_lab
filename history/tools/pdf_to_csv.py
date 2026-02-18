#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import pdfplumber
from pypdf import PdfReader

TABLE_SETTINGS_TEXT = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 3,
    "min_words_vertical": 1,
    "min_words_horizontal": 1,
    "intersection_tolerance": 3,
    "text_tolerance": 3,
}

HEADER_ALIASES = {
    "거래일시": "date_time",
    "거래일자": "date",
    "거래일": "date",
    "일시": "date_time",
    "날짜": "date",
    "시간": "time",
    "거래시간": "time",
    "적요": "description",
    "내용": "description",
    "거래내용": "description",
    "상세내용": "description",
    "가맹점명": "description",
    "거래처": "counterparty",
    "상대방": "counterparty",
    "출금": "debit",
    "출금금액": "debit",
    "인출": "debit",
    "입금": "credit",
    "입금금액": "credit",
    "거래금액": "amount",
    "사용금액": "amount",
    "결제금액": "amount",
    "금액": "amount",
    "잔액": "balance",
    "거래후잔액": "balance",
    "거래후잔액금액": "balance",
    "비고": "note",
    "메모": "note",
    "구분": "type",
    "거래구분": "type",
    "종목명(종목코드)": "symbol",
    "종목명": "symbol",
    "종목": "symbol",
    "거래수량": "quantity",
    "거래대금": "trade_amount",
    "단가": "unit_price",
    "수수료": "fee",
    "거래세": "trade_tax",
    "제세금": "tax",
}

NUMERIC_COL_KEYS = {
    "amount",
    "balance",
    "debit",
    "credit",
    "출금",
    "입금",
    "금액",
    "잔액",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PDF transaction statements in a folder to CSV files."
    )
    parser.add_argument(
        "--input-dir",
        default="toss",
        help="Directory containing source PDF files (default: toss).",
    )
    parser.add_argument(
        "--pattern",
        default="*.pdf",
        help="File glob pattern under input directory (default: *.pdf).",
    )
    parser.add_argument(
        "--output-dir",
        default="toss/csv",
        help="Directory to write CSV outputs (default: toss/csv).",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="PDF password. If omitted and encrypted PDFs are found, prompt once.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Fail instead of prompting when password is missing.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="CSV encoding (default: utf-8-sig).",
    )
    return parser.parse_args()


def normalize_key(text: str) -> str:
    return re.sub(r"[\s_\-/:()]+", "", text.strip().lower())


NORMALIZED_HEADER_KEYS = {normalize_key(k) for k in HEADER_ALIASES}

HEADER_SCAN_ROWS = 30
HEADER_MIN_SCORE = 4
WEAK_HEADER_HINTS = (
    "거래",
    "잔액",
    "금액",
    "수량",
    "수수료",
    "거래세",
    "제세금",
    "종목",
    "단가",
    "환율",
    "구분",
)


def clean_cell(cell: object) -> str:
    if cell is None:
        return ""
    text = str(cell).replace("\r", " ").replace("\n", " ").strip()
    return re.sub(r"\s+", " ", text)


def make_unique(columns: Iterable[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique: list[str] = []
    for raw_name in columns:
        name = raw_name if raw_name else "col"
        count = seen.get(name, 0) + 1
        seen[name] = count
        unique.append(name if count == 1 else f"{name}_{count}")
    return unique


def canonicalize_header(cell: str) -> str:
    cleaned = clean_cell(cell)
    if not cleaned:
        return ""
    mapped = HEADER_ALIASES.get(normalize_key(cleaned), cleaned)
    return mapped


def header_score(row: list[str]) -> int:
    score = 0
    for cell in row:
        if not cell:
            continue
        key = normalize_key(cell)
        if key in NORMALIZED_HEADER_KEYS:
            score += 4
        elif (not re.search(r"\d", key)) and any(hint in key for hint in WEAK_HEADER_HINTS):
            score += 1
    return score


def split_header_and_rows(rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    if not rows:
        return [], []

    best_idx = -1
    best_score = -1
    for idx, row in enumerate(rows[:HEADER_SCAN_ROWS]):
        score = header_score(row)
        if score > best_score:
            best_score = score
            best_idx = idx

    max_cols = max(len(r) for r in rows)
    padded = [r + [""] * (max_cols - len(r)) for r in rows]

    if best_score >= HEADER_MIN_SCORE:
        header = [canonicalize_header(c) for c in padded[best_idx]]
        header = [c if c else f"col_{i + 1}" for i, c in enumerate(header)]
        data = padded[best_idx + 1 :]
    else:
        header = [f"col_{i + 1}" for i in range(max_cols)]
        data = padded

    return make_unique(header), data


def clean_numeric_value(value: object) -> object:
    if value is None:
        return None
    text = clean_cell(value)
    if not text:
        return None
    numeric = text
    numeric = re.sub(r"[,\s원₩]", "", numeric)
    numeric = re.sub(r"^krw", "", numeric, flags=re.IGNORECASE)
    numeric = re.sub(r"^\((.+)\)$", r"-\1", numeric)
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", numeric):
        if "." in numeric:
            return float(numeric)
        return int(numeric)
    return text


def maybe_clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        key = normalize_key(col)
        if any(token in key for token in NUMERIC_COL_KEYS):
            df[col] = df[col].map(clean_numeric_value)
    return df


def pick_first_nonempty(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    result = pd.Series(pd.NA, index=df.index, dtype="object")
    for col in candidates:
        if col not in df.columns:
            continue
        values = df[col].map(lambda v: clean_cell(v) if pd.notna(v) else pd.NA)
        values = values.replace("", pd.NA)
        mask = result.isna() & values.notna()
        result.loc[mask] = values.loc[mask]
    return result


def pick_first_numeric(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    result = pd.Series(pd.NA, index=df.index, dtype="object")
    for col in candidates:
        if col not in df.columns:
            continue
        values = df[col].map(clean_numeric_value)
        values = values.where(pd.notna(values), pd.NA)
        mask = result.isna() & values.notna()
        result.loc[mask] = values.loc[mask]
    return result


def normalize_date_text(value: object) -> object:
    if value is None or pd.isna(value):
        return pd.NA
    text = clean_cell(value)
    if not text:
        return pd.NA

    match = re.search(r"(\d{4})[./-]\s*(\d{1,2})[./-]\s*(\d{1,2})", text)
    if not match:
        match = re.search(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일", text)
    if match:
        y, m, d = match.groups()
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    return text


def build_normalized_transactions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
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

    normalized = pd.DataFrame(
        {
            "_source_file": df.get("_source_file", pd.Series(pd.NA, index=df.index)),
            "_page": df.get("_page", pd.Series(pd.NA, index=df.index)),
            "_table": df.get("_table", pd.Series(pd.NA, index=df.index)),
        }
    )

    date = pick_first_nonempty(df, ["date", "date_time", "거래일자", "거래일시"])
    typ = pick_first_nonempty(df, ["type", "거래구분", "거 래구분"])
    desc = pick_first_nonempty(
        df,
        [
            "description",
            "counterparty",
            "symbol",
            "종목명(종목코드)",
            "종 목명(종목코드)",
            "종목명",
        ],
    )

    combined_col = None
    for col in df.columns:
        key = normalize_key(str(col))
        if "거래일자" in key and "거래구분" in key:
            combined_col = col
            break

    if combined_col is not None:
        combined = df[combined_col].map(lambda v: clean_cell(v) if pd.notna(v) else "")
        combined_date = combined.str.extract(r"(\d{4}[./-]\d{1,2}[./-]\d{1,2})", expand=False)
        combined_type = combined.str.replace(
            r"^\s*\d{4}[./-]\d{1,2}[./-]\d{1,2}\s*", "", regex=True
        ).replace("", pd.NA)

        mask_date = date.isna() & combined_date.notna()
        date.loc[mask_date] = combined_date.loc[mask_date]
        mask_type = typ.isna() & combined_type.notna()
        typ.loc[mask_type] = combined_type.loc[mask_type]

    amount = pick_first_numeric(
        df,
        [
            "amount",
            "trade_amount",
            "거래대금",
            "대금",
            "래대금",
            "debit",
            "credit",
        ],
    )
    balance = pick_first_numeric(df, ["balance", "잔액", "잔고", "잔   액"])

    normalized["date"] = date.map(normalize_date_text)
    normalized["type"] = typ
    normalized["description"] = desc
    normalized["amount"] = amount
    normalized["balance"] = balance

    normalized = normalized.replace("", pd.NA)
    normalized = normalized.dropna(subset=["date"], how="any")
    normalized = normalized.drop_duplicates()
    return normalized


def extract_tables_from_page(page: pdfplumber.page.Page) -> list[list[list[str]]]:
    tables = page.extract_tables() or []
    if not tables:
        tables = page.extract_tables(TABLE_SETTINGS_TEXT) or []

    cleaned_tables: list[list[list[str]]] = []
    for table in tables:
        cleaned_rows: list[list[str]] = []
        for row in table:
            if row is None:
                continue
            cleaned = [clean_cell(cell) for cell in row]
            if any(cleaned):
                cleaned_rows.append(cleaned)
        if cleaned_rows:
            cleaned_tables.append(cleaned_rows)
    return cleaned_tables


def ensure_password_if_needed(
    pdf_paths: list[Path], password: str | None, no_prompt: bool
) -> str | None:
    encrypted_exists = False
    for path in pdf_paths:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            encrypted_exists = True
            break

    if not encrypted_exists:
        return None
    if password is not None:
        return password
    if no_prompt:
        raise RuntimeError("Encrypted PDFs detected, but --password is missing.")
    return getpass.getpass("Enter PDF password: ")


def can_decrypt(path: Path, password: str | None) -> bool:
    reader = PdfReader(str(path))
    if not reader.is_encrypted:
        return True
    if password is None:
        return False
    return bool(reader.decrypt(password))


def process_pdf(path: Path, password: str | None) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []
    frames: list[pd.DataFrame] = []

    with pdfplumber.open(str(path), password=password) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            try:
                tables = extract_tables_from_page(page)
            except Exception as exc:
                errors.append(f"{path.name},page={page_idx},error={exc}")
                continue

            if not tables:
                errors.append(f"{path.name},page={page_idx},error=no_table")
                continue

            for table_idx, rows in enumerate(tables, start=1):
                header, data_rows = split_header_and_rows(rows)
                if not data_rows:
                    continue

                table_df = pd.DataFrame(data_rows, columns=header)
                table_df = table_df.replace("", pd.NA).dropna(how="all")
                if table_df.empty:
                    continue

                table_df.insert(0, "_source_file", path.name)
                table_df.insert(1, "_page", page_idx)
                table_df.insert(2, "_table", table_idx)
                table_df = maybe_clean_numeric_columns(table_df)
                frames.append(table_df)

    if not frames:
        empty = pd.DataFrame(columns=["_source_file", "_page", "_table"])
        return empty, errors

    file_df = pd.concat(frames, ignore_index=True, sort=False)
    file_df = file_df.drop_duplicates()
    return file_df, errors


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(input_dir.glob(args.pattern))
    if not pdf_paths:
        print(f"No PDF files found in {input_dir} with pattern {args.pattern}", file=sys.stderr)
        return 1

    try:
        password = ensure_password_if_needed(pdf_paths, args.password, args.no_prompt)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    all_frames: list[pd.DataFrame] = []
    all_errors: list[str] = []

    for path in pdf_paths:
        if not can_decrypt(path, password):
            print(f"Wrong or missing password for {path.name}", file=sys.stderr)
            return 1

        try:
            file_df, errors = process_pdf(path, password)
        except Exception as exc:
            all_errors.append(f"{path.name},page=ALL,error={exc}")
            continue

        out_path = output_dir / f"{path.stem}.csv"
        file_df.to_csv(out_path, index=False, encoding=args.encoding)
        all_frames.append(file_df)
        all_errors.extend(errors)
        print(f"[OK] {path.name} -> {out_path} ({len(file_df)} rows)")

    merged_path = output_dir / "all_merged.csv"
    if all_frames:
        merged_df = pd.concat(all_frames, ignore_index=True, sort=False).drop_duplicates()
    else:
        merged_df = pd.DataFrame(columns=["_source_file", "_page", "_table"])
    merged_df.to_csv(merged_path, index=False, encoding=args.encoding)
    print(f"[OK] merged -> {merged_path} ({len(merged_df)} rows)")

    normalized_path = output_dir / "all_merged_normalized.csv"
    normalized_df = build_normalized_transactions(merged_df)
    normalized_df.to_csv(normalized_path, index=False, encoding=args.encoding)
    print(f"[OK] normalized -> {normalized_path} ({len(normalized_df)} rows)")

    log_path = output_dir / "extract_errors.log"
    with log_path.open("w", encoding="utf-8") as fp:
        for line in all_errors:
            fp.write(line + "\n")
    print(f"[OK] errors -> {log_path} ({len(all_errors)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

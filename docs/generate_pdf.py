#!/usr/bin/env python3
"""
Markdown → PDF 변환기
=====================
마크다운 파일을 읽어 한글 지원 PDF로 변환한다.

지원 문법:
  # H1          → 제목 (20pt, 가운데 정렬)
  ## H2         → 섹션 제목 (14pt, 파란 밑줄)
  ### H3        → 서브 제목 (11pt)
  > blockquote  → 회색 메타 정보
  - bullet      → 불릿 리스트
  | table |     → 테이블 (자동 열폭)
  ---           → 구분선 (여백)
  **bold**      → 굵은 텍스트 (본문 내)
  일반 텍스트     → 본문

사용법:
  python docs/generate_pdf.py docs/stoploss_optimization_report.md
  python docs/generate_pdf.py docs/stoploss_with_fees_report.md
  python docs/generate_pdf.py  # 기본: 모든 docs/*.md 변환
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from fpdf import FPDF

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
DOCS_DIR = Path(__file__).parent


class MarkdownPDF(FPDF):
    """마크다운 → PDF 렌더러."""

    def __init__(self, header_text: str = "PTJ Trading Strategy"):
        super().__init__()
        self.add_font("F", "", FONT_PATH)
        self.add_font("F", "B", FONT_PATH)
        self.set_auto_page_break(auto=True, margin=15)
        self._header_text = header_text

    def header(self):
        self.set_font("F", "B", 9)
        self.set_text_color(130, 130, 130)
        self.cell(0, 8, self._header_text, align="R",
                  new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("F", "", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    # ── 렌더링 메서드 ──

    def render_h1(self, text: str):
        self.set_font("F", "B", 20)
        self.set_text_color(30, 30, 30)
        self.cell(0, 14, text, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def render_h2(self, text: str):
        self.set_font("F", "B", 14)
        self.set_text_color(30, 30, 30)
        self.ln(4)
        self.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(66, 133, 244)
        self.set_line_width(0.6)
        self.line(10, self.get_y(), 80, self.get_y())
        self.set_line_width(0.2)
        self.ln(4)

    def render_h3(self, text: str):
        self.set_font("F", "B", 11)
        self.set_text_color(60, 60, 60)
        self.ln(2)
        self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def render_blockquote(self, text: str):
        self.set_font("F", "", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 7, text, align="C", new_x="LMARGIN", new_y="NEXT")

    def render_bullet(self, text: str):
        self.set_font("F", "", 10)
        self.set_text_color(50, 50, 50)
        text = _strip_bold_markers(text)
        self.cell(6, 6, "\u2022 ", new_x="END")
        self.multi_cell(0, 6, text)
        self.ln(1)

    def render_text(self, text: str):
        self.set_font("F", "", 10)
        self.set_text_color(50, 50, 50)
        text = _strip_bold_markers(text)
        self.multi_cell(0, 6, text)
        self.ln(1)

    def render_hr(self):
        self.ln(3)

    def render_table(self, headers: list[str], rows: list[list[str]],
                     highlight_row: int | None = None):
        """테이블 렌더링 — 열폭 자동 계산."""
        n_cols = len(headers)
        # 열폭: 내용 최대 길이 기반 비례 배분 (190mm 총폭)
        max_lens = []
        for c in range(n_cols):
            col_vals = [headers[c]] + [r[c] if c < len(r) else "" for r in rows]
            max_len = max(_text_width(v) for v in col_vals)
            max_lens.append(max(max_len, 4))
        total = sum(max_lens)
        col_widths = [round(l / total * 190, 1) for l in max_lens]

        # 헤더
        self.set_font("F", "B", 7.5)
        self.set_fill_color(66, 133, 244)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 6.5,
                      _strip_bold_markers(h), border=1, fill=True, align="C")
        self.ln()

        # 행
        self.set_font("F", "", 7.5)
        for r_idx, row in enumerate(rows):
            is_hl = (r_idx == highlight_row)
            if is_hl:
                self.set_fill_color(232, 245, 233)
                self.set_font("F", "B", 7.5)
            elif r_idx % 2 == 0:
                self.set_fill_color(250, 250, 250)
                self.set_font("F", "", 7.5)
            else:
                self.set_fill_color(255, 255, 255)
                self.set_font("F", "", 7.5)

            self.set_text_color(30, 30, 30)
            for i in range(n_cols):
                val = _strip_bold_markers(row[i]) if i < len(row) else ""
                align = "C" if i > 0 else "L"
                self.cell(col_widths[i], 5.5, val,
                          border=1, fill=True, align=align)
            self.ln()
        self.ln(3)


# ── 마크다운 파서 ──

def _strip_bold_markers(text: str) -> str:
    """**bold** → bold, `code` → code."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    return text


def _text_width(text: str) -> int:
    """텍스트 폭 추정 (한글=2, 영문/숫자=1)."""
    w = 0
    for ch in text:
        if ord(ch) > 0x7F:
            w += 2
        else:
            w += 1
    return w


def _find_bold_row(rows: list[list[str]]) -> int | None:
    """**bold** 마커가 있는 행 인덱스 반환."""
    for i, row in enumerate(rows):
        if any("**" in cell for cell in row):
            return i
    return None


def _parse_table_line(line: str) -> list[str]:
    """| a | b | c | → ['a', 'b', 'c']"""
    parts = line.strip().strip("|").split("|")
    return [p.strip() for p in parts]


def _is_separator_row(cells: list[str]) -> bool:
    """테이블 구분선(|---|---|) 여부."""
    return all(re.match(r"^[-:]+$", c) for c in cells)


def parse_and_render(md_path: Path, pdf: MarkdownPDF) -> None:
    """마크다운 파일을 파싱하여 PDF에 렌더링."""
    lines = md_path.read_text(encoding="utf-8").splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]

        # 빈 줄
        if not line.strip():
            i += 1
            continue

        # --- 수평선
        if re.match(r"^---+\s*$", line):
            pdf.render_hr()
            i += 1
            continue

        # # H1
        if line.startswith("# ") and not line.startswith("## "):
            pdf.render_h1(line[2:].strip())
            i += 1
            continue

        # ### H3 (## 보다 먼저 체크)
        if line.startswith("### "):
            pdf.render_h3(line[4:].strip())
            i += 1
            continue

        # ## H2
        if line.startswith("## "):
            pdf.render_h2(line[3:].strip())
            i += 1
            continue

        # > blockquote
        if line.startswith("> "):
            pdf.render_blockquote(line[2:].strip())
            i += 1
            continue

        # - bullet
        if re.match(r"^- ", line):
            pdf.render_bullet(line[2:].strip())
            i += 1
            continue

        # | table |
        if line.strip().startswith("|"):
            headers_cells = _parse_table_line(line)
            i += 1
            # 구분선 스킵
            if i < len(lines) and lines[i].strip().startswith("|"):
                maybe_sep = _parse_table_line(lines[i])
                if _is_separator_row(maybe_sep):
                    i += 1

            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = _parse_table_line(lines[i])
                if _is_separator_row(cells):
                    i += 1
                    continue
                rows.append(cells)
                i += 1

            highlight = _find_bold_row(rows)
            pdf.render_table(headers_cells, rows, highlight_row=highlight)
            continue

        # 일반 텍스트
        pdf.render_text(line.strip())
        i += 1


def convert_md_to_pdf(md_path: Path, out_path: Path | None = None) -> Path:
    """마크다운 파일 → PDF 변환."""
    if out_path is None:
        out_path = md_path.with_suffix(".pdf")

    pdf = MarkdownPDF(header_text="PTJ Trading Strategy")
    pdf.alias_nb_pages()
    pdf.add_page()

    parse_and_render(md_path, pdf)

    pdf.output(str(out_path))
    print(f"  PDF 저장: {out_path}")
    return out_path


def main():
    if len(sys.argv) > 1:
        # 인자로 지정된 파일만 변환
        for arg in sys.argv[1:]:
            md_file = Path(arg)
            if not md_file.exists():
                md_file = DOCS_DIR / arg
            if not md_file.exists():
                print(f"  파일 없음: {arg}")
                continue
            convert_md_to_pdf(md_file)
    else:
        # docs/ 아래 모든 .md 변환
        md_files = sorted(DOCS_DIR.glob("*.md"))
        if not md_files:
            print("  변환할 .md 파일 없음")
            return
        for md_file in md_files:
            convert_md_to_pdf(md_file)


if __name__ == "__main__":
    main()

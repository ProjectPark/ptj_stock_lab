"""
Worker Rules v1 문서 → PDF 변환
=================================
docs/notes/line_b/review/taejun_worker_rules_v1.md → docs/pdf/taejun_worker_rules_v1.pdf
"""
from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent.parent
MD_PATH = ROOT / "docs" / "rules" / "line_b" / "taejun_worker_rules_v1.md"
PDF_DIR = ROOT / "docs" / "pdf"
PDF_PATH = PDF_DIR / "taejun_worker_rules_v1.pdf"

_TTC_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
FONT_PATH = "/tmp/AppleSDGothicNeo_0.ttf"


def _ensure_font():
    if not Path(FONT_PATH).exists():
        from fontTools.ttLib import TTCollection
        ttc = TTCollection(_TTC_PATH)
        ttc[0].save(FONT_PATH)


_ensure_font()


class WorkerPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("SD", fname=FONT_PATH)
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("SD", size=8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 6, "Line B Worker Rules v1 — taejun_worker_rules_v1", align="R")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("SD", size=8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 10, f"- {self.page_no()} -", align="C")

    def _title(self, text: str):
        self.set_font("SD", "", 18)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 10, text)
        self.ln(2)

    def _h2(self, text: str):
        self.ln(4)
        self.set_font("SD", "", 14)
        self.set_text_color(30, 30, 120)
        self.multi_cell(0, 8, text)
        self.set_draw_color(30, 30, 120)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def _h3(self, text: str):
        self.ln(3)
        self.set_font("SD", "", 12)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 7, text)
        self.ln(2)

    def _body(self, text: str):
        self.set_font("SD", "", 9)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def _quote(self, text: str):
        self.set_font("SD", "", 9)
        self.set_text_color(80, 80, 80)
        x = self.get_x()
        self.set_fill_color(240, 240, 245)
        self.set_x(x + 4)
        self.multi_cell(self.w - self.l_margin - self.r_margin - 8, 5.5, text, fill=True)
        self.ln(2)

    def _hr(self):
        self.set_draw_color(200, 200, 200)
        y = self.get_y() + 2
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(6)

    def _table(self, headers: list[str], rows: list[list[str]]):
        usable = self.w - self.l_margin - self.r_margin
        n_cols = len(headers)
        if n_cols == 0:
            return
        col_widths = self._calc_col_widths(headers, rows, usable)
        row_h = 6

        self.set_font("SD", "", 8)
        self.set_fill_color(50, 50, 80)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], row_h, clean_md(h), border=1, fill=True, align="C")
        self.ln(row_h)

        self.set_font("SD", "", 8)
        self.set_text_color(30, 30, 30)
        for r_idx, row in enumerate(rows):
            if r_idx % 2 == 0:
                self.set_fill_color(248, 248, 252)
            else:
                self.set_fill_color(255, 255, 255)

            max_lines = 1
            for i, cell_text in enumerate(row):
                w = col_widths[i] if i < len(col_widths) else col_widths[-1]
                lines = self._count_lines(cell_text, w)
                max_lines = max(max_lines, lines)
            cell_h = row_h * max_lines

            if self.get_y() + cell_h > self.h - 25:
                self.add_page()
                self.set_font("SD", "", 8)
                self.set_fill_color(50, 50, 80)
                self.set_text_color(255, 255, 255)
                for i, h in enumerate(headers):
                    self.cell(col_widths[i], row_h, h, border=1, fill=True, align="C")
                self.ln(row_h)
                self.set_font("SD", "", 8)
                self.set_text_color(30, 30, 30)
                if r_idx % 2 == 0:
                    self.set_fill_color(248, 248, 252)
                else:
                    self.set_fill_color(255, 255, 255)

            x_start = self.get_x()
            y_start = self.get_y()
            for i in range(n_cols):
                cell_text = row[i] if i < len(row) else ""
                w = col_widths[i] if i < len(col_widths) else col_widths[-1]
                self.set_xy(x_start + sum(col_widths[:i]), y_start)
                self.rect(x_start + sum(col_widths[:i]), y_start, w, cell_h, style="DF")
                self.set_xy(x_start + sum(col_widths[:i]) + 1, y_start + 0.5)
                self.multi_cell(w - 2, row_h, clean_md(cell_text))
            self.set_xy(x_start, y_start + cell_h)
        self.ln(3)

    def _count_lines(self, text: str, width: float) -> int:
        if not text:
            return 1
        self.set_font("SD", "", 8)
        sw = self.get_string_width(text)
        effective_w = width - 2
        if effective_w <= 0:
            return 1
        return max(1, int(sw / effective_w) + 1)

    def _calc_col_widths(self, headers, rows, usable):
        n = len(headers)
        self.set_font("SD", "", 8)
        max_w = []
        for i in range(n):
            hw = self.get_string_width(headers[i]) + 4
            cw = hw
            for row in rows:
                if i < len(row):
                    rw = self.get_string_width(row[i]) + 4
                    cw = max(cw, rw)
            max_w.append(min(cw, usable * 0.45))
        total = sum(max_w)
        if total == 0:
            return [usable / n] * n
        return [w / total * usable for w in max_w]


def parse_table(lines):
    if len(lines) < 2:
        return [], []
    headers = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    return headers, rows


def clean_md(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    text = text.replace("⚠️", "[!]").replace("⚠", "[!]")
    text = text.replace("✅", "[OK]").replace("❌", "[X]")
    text = text.replace("⬜", "[ ]").replace("🟩", "[G]")
    text = text.replace("−", "-").replace("\ufe0f", "")
    return text.strip()


def build_pdf():
    md_text = MD_PATH.read_text(encoding="utf-8")
    md_text = md_text.replace("\u2212", "-")
    md_text = md_text.replace("\u26a0\ufe0f", "[!]").replace("\u26a0", "[!]")
    lines = md_text.split("\n")

    pdf = WorkerPDF()
    pdf.add_page()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped == "---":
            pdf._hr()
            i += 1
            continue

        if stripped.startswith("# ") and not stripped.startswith("## "):
            pdf._title(clean_md(stripped[2:]))
            i += 1
            continue

        if stripped.startswith("## "):
            pdf._h2(clean_md(stripped[3:]))
            i += 1
            continue

        if stripped.startswith("### "):
            pdf._h3(clean_md(stripped[4:]))
            i += 1
            continue

        if stripped.startswith("> "):
            quote_lines = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote_lines.append(clean_md(lines[i].strip().lstrip("> ")))
                i += 1
            pdf._quote("\n".join(quote_lines))
            continue

        if "|" in stripped and not stripped.startswith(">"):
            table_lines = []
            while i < len(lines) and "|" in lines[i]:
                table_lines.append(lines[i])
                i += 1
            headers, rows = parse_table(table_lines)
            if headers:
                pdf._table(headers, rows)
            continue

        if re.match(r"^[-*]\s", stripped) or re.match(r"^\d+\.\s", stripped):
            text = clean_md(re.sub(r"^[-*]\s+", "  • ", stripped))
            text = re.sub(r"^(\d+)\.\s+", r"  \1. ", text)
            pdf._body(text)
            i += 1
            continue

        para_lines = []
        while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(("#", ">", "---", "|")):
            if re.match(r"^[-*]\s", lines[i].strip()) or re.match(r"^\d+\.\s", lines[i].strip()):
                break
            para_lines.append(lines[i].strip())
            i += 1
        if para_lines:
            pdf._body(clean_md(" ".join(para_lines)))
            continue

        i += 1

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdf.output(str(PDF_PATH))
    print(f"PDF saved: {PDF_PATH}")
    print(f"Pages: {pdf.pages_count}")


if __name__ == "__main__":
    build_pdf()

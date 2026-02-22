"""
전략 리뷰 문서 → PDF 변환
===========================
docs/notes/taejun_strategy_review_2026-02-22_VNQ.md → docs/pdf/taejun_strategy_review_2026-02-22_VNQ.pdf
"""
from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF

# Paths
ROOT = Path(__file__).resolve().parent.parent.parent
MD_PATH = ROOT / "docs" / "notes" / "taejun_strategy_review_2026-02-22_VNQ.md"
PDF_DIR = ROOT / "docs" / "pdf"
PDF_PATH = PDF_DIR / "taejun_strategy_review_2026-02-22_VNQ.pdf"

# Font: AppleSDGothicNeo (한글+영문 지원, TTC에서 추출)
_TTC_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
FONT_PATH = "/tmp/AppleSDGothicNeo_0.ttf"

def _ensure_font():
    from pathlib import Path
    if not Path(FONT_PATH).exists():
        from fontTools.ttLib import TTCollection
        ttc = TTCollection(_TTC_PATH)
        ttc[0].save(FONT_PATH)

_ensure_font()


class StrategyPDF(FPDF):
    """한글 지원 PDF — NotoSansGothic."""

    def __init__(self):
        super().__init__()
        self.add_font("SD", fname=FONT_PATH)
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("SD", size=8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 6, "taejun_attach_pattern — 전략 리뷰 2026-02-22 (VNQ)", align="R")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("SD", size=8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 10, f"- {self.page_no()} -", align="C")

    # ── rendering helpers ──

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
        """Draw a simple table with auto column widths."""
        usable = self.w - self.l_margin - self.r_margin
        n_cols = len(headers)
        if n_cols == 0:
            return

        # Determine column widths based on content
        col_widths = self._calc_col_widths(headers, rows, usable)
        row_h = 6

        # Header
        self.set_font("SD", "", 8)
        self.set_fill_color(50, 50, 80)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], row_h, clean_md(h), border=1, fill=True, align="C")
        self.ln(row_h)

        # Rows
        self.set_font("SD", "", 8)
        self.set_text_color(30, 30, 30)
        for r_idx, row in enumerate(rows):
            if r_idx % 2 == 0:
                self.set_fill_color(248, 248, 252)
            else:
                self.set_fill_color(255, 255, 255)

            # Calculate max height for this row
            max_lines = 1
            for i, cell_text in enumerate(row):
                w = col_widths[i] if i < len(col_widths) else col_widths[-1]
                lines = self._count_lines(cell_text, w)
                max_lines = max(max_lines, lines)
            cell_h = row_h * max_lines

            # Check page break
            if self.get_y() + cell_h > self.h - 25:
                self.add_page()
                # Re-draw header
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
                # Draw cell border + fill
                self.rect(x_start + sum(col_widths[:i]), y_start, w, cell_h, style="DF")
                self.set_xy(x_start + sum(col_widths[:i]) + 1, y_start + 0.5)
                self.multi_cell(w - 2, row_h, clean_md(cell_text))
            self.set_xy(x_start, y_start + cell_h)
        self.ln(3)

    def _count_lines(self, text: str, width: float) -> int:
        """Estimate number of lines for text in given width."""
        if not text:
            return 1
        self.set_font("SD", "", 8)
        sw = self.get_string_width(text)
        effective_w = width - 2
        if effective_w <= 0:
            return 1
        return max(1, int(sw / effective_w) + 1)

    def _calc_col_widths(self, headers: list[str], rows: list[list[str]],
                         usable: float) -> list[float]:
        """Calculate column widths proportionally."""
        n = len(headers)
        # Measure max content width per column
        self.set_font("SD", "", 8)
        max_w = []
        for i in range(n):
            hw = self.get_string_width(headers[i]) + 4
            cw = hw
            for row in rows:
                if i < len(row):
                    rw = self.get_string_width(row[i]) + 4
                    cw = max(cw, rw)
            max_w.append(min(cw, usable * 0.45))  # Cap at 45% of usable
        total = sum(max_w)
        if total == 0:
            return [usable / n] * n
        return [w / total * usable for w in max_w]


def parse_table(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    """Parse markdown table lines into headers and rows."""
    if len(lines) < 2:
        return [], []
    headers = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:  # Skip separator
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    return headers, rows


def clean_md(text: str) -> str:
    """Strip markdown formatting for plain text."""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # bold
    text = re.sub(r"`(.*?)`", r"\1", text)  # code
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)  # links
    # 폰트 미지원 특수문자 치환
    text = text.replace("⚠️", "[!]").replace("⚠", "[!]")
    text = text.replace("✅", "[OK]").replace("❌", "[X]")
    text = text.replace("⬜", "[ ]").replace("🟩", "[G]")
    text = text.replace("−", "-").replace("\ufe0f", "")
    return text.strip()


def build_pdf():
    """Parse markdown and build PDF."""
    md_text = MD_PATH.read_text(encoding="utf-8")
    # 폰트 미지원 문자 전역 치환
    md_text = md_text.replace("\u2212", "-")   # − → -
    md_text = md_text.replace("\u26a0\ufe0f", "[!]").replace("\u26a0", "[!]")  # ⚠️
    lines = md_text.split("\n")

    pdf = StrategyPDF()
    pdf.add_page()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip empty
        if not stripped:
            i += 1
            continue

        # Horizontal rule
        if stripped == "---":
            pdf._hr()
            i += 1
            continue

        # H1
        if stripped.startswith("# ") and not stripped.startswith("## "):
            pdf._title(clean_md(stripped[2:]))
            i += 1
            continue

        # H2
        if stripped.startswith("## "):
            pdf._h2(clean_md(stripped[3:]))
            i += 1
            continue

        # H3
        if stripped.startswith("### "):
            pdf._h3(clean_md(stripped[4:]))
            i += 1
            continue

        # Blockquote (multi-line)
        if stripped.startswith("> "):
            quote_lines = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote_lines.append(clean_md(lines[i].strip().lstrip("> ")))
                i += 1
            pdf._quote("\n".join(quote_lines))
            continue

        # Table
        if "|" in stripped and not stripped.startswith(">"):
            table_lines = []
            while i < len(lines) and "|" in lines[i]:
                table_lines.append(lines[i])
                i += 1
            headers, rows = parse_table(table_lines)
            if headers:
                pdf._table(headers, rows)
            continue

        # List items
        if re.match(r"^[-*]\s", stripped) or re.match(r"^\d+\.\s", stripped):
            text = clean_md(re.sub(r"^[-*]\s+", "  • ", stripped))
            text = re.sub(r"^(\d+)\.\s+", r"  \1. ", text)
            pdf._body(text)
            i += 1
            continue

        # Regular paragraph
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

    # Save
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdf.output(str(PDF_PATH))
    print(f"PDF saved: {PDF_PATH}")
    print(f"Pages: {pdf.pages_count}")


if __name__ == "__main__":
    build_pdf()

"""trading_rules_v2.md → PDF 변환 (fpdf2 + Apple SD Gothic Neo)."""
from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF

# ── 경로 ──────────────────────────────────────────────────────
BASE = Path(__file__).parent
MD_PATH = BASE / "trading_rules_v2.md"
OUT_PATH = BASE / "trading_rules_v2.pdf"
FONT_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc"

# ── 색상 ──────────────────────────────────────────────────────
C_BLACK = (33, 33, 33)
C_GRAY = (100, 100, 100)
C_BLUE = (30, 100, 200)
C_RED = (200, 50, 50)
C_BG_HEADER = (235, 240, 250)
C_BG_QUOTE = (245, 245, 245)
C_BG_CODE = (240, 240, 240)
C_LINE = (200, 200, 200)
C_TABLE_BORDER = (180, 180, 190)
C_TABLE_HEADER_BG = (55, 70, 100)
C_TABLE_HEADER_FG = (255, 255, 255)
C_TABLE_ALT_ROW = (245, 248, 255)


class RulesPDF(FPDF):
    """Markdown → PDF 렌더러."""

    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=20)
        self.add_font("gothic", style="", fname=FONT_PATH)
        self.add_font("gothic", style="B", fname=FONT_PATH)
        self.alias_nb_pages()

    # ── 헤더 / 푸터 ──────────────────────────────────────────
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("gothic", "B", 8)
        self.set_text_color(*C_GRAY)
        self.cell(0, 6, "PTJ 매매법 v2 — 매매 규칙서", align="L")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("gothic", "", 8)
        self.set_text_color(*C_GRAY)
        self.cell(0, 10, f"- {self.page_no()} / {{nb}} -", align="C")

    # ── 유틸리티 ──────────────────────────────────────────────
    def _clean(self, text: str) -> str:
        """마크다운 인라인 서식 제거 (볼드/이탈릭/코드/취소선)."""
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        text = re.sub(r"`(.+?)`", r"\1", text)
        text = re.sub(r"~~(.+?)~~", r"\1", text)
        text = re.sub(r"\\\|", "|", text)
        return text.strip()

    def _check_space(self, needed_mm: float = 20):
        """페이지 하단 공간 부족 시 새 페이지."""
        if self.get_y() + needed_mm > self.h - self.b_margin:
            self.add_page()

    # ── 블록 렌더 ─────────────────────────────────────────────
    def render_title(self, text: str):
        """문서 제목 (# H1)."""
        self.set_font("gothic", "B", 22)
        self.set_text_color(*C_BLACK)
        self.ln(30)
        self.cell(0, 14, self._clean(text), align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    def render_subtitle(self, text: str):
        """부제 (> 인용문 첫 줄)."""
        self.set_font("gothic", "", 10)
        self.set_text_color(*C_GRAY)
        self.cell(0, 7, self._clean(text), align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(8)

    def render_h2(self, text: str):
        """## 섹션 제목."""
        self._check_space(25)
        self.ln(6)
        self.set_draw_color(*C_BLUE)
        self.set_line_width(0.6)
        y = self.get_y()
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(3)
        self.set_font("gothic", "B", 14)
        self.set_text_color(*C_BLUE)
        self.cell(0, 9, self._clean(text), new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def render_h3(self, text: str):
        """### 소제목."""
        self._check_space(15)
        self.ln(3)
        self.set_font("gothic", "B", 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 7, self._clean(text), new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def render_paragraph(self, text: str):
        """일반 본문."""
        self.set_font("gothic", "", 10)
        self.set_text_color(*C_BLACK)
        self.multi_cell(0, 6, self._clean(text))
        self.ln(2)

    def render_bullet(self, text: str):
        """- 항목."""
        self.set_font("gothic", "", 10)
        self.set_text_color(*C_BLACK)
        x = self.l_margin
        self.set_x(x + 4)
        self.cell(4, 6, "•")
        self.set_x(x + 10)
        self.multi_cell(self.w - self.r_margin - x - 10, 6, self._clean(text))
        self.ln(1)

    def render_quote(self, text: str):
        """인용 블록 (> ...)."""
        self._check_space(12)
        self.set_fill_color(*C_BG_QUOTE)
        self.set_draw_color(*C_BLUE)
        x = self.l_margin
        w = self.w - self.l_margin - self.r_margin
        y_start = self.get_y()
        self.set_x(x + 6)
        self.set_font("gothic", "", 9)
        self.set_text_color(*C_GRAY)
        self.multi_cell(w - 8, 5.5, self._clean(text), fill=True)
        y_end = self.get_y()
        self.set_line_width(0.5)
        self.line(x + 2, y_start, x + 2, y_end)
        self.ln(2)

    def render_code_block(self, lines: list[str]):
        """코드 블록."""
        self._check_space(len(lines) * 5 + 8)
        self.set_fill_color(*C_BG_CODE)
        self.set_font("gothic", "", 9)
        self.set_text_color(50, 50, 50)
        x = self.l_margin + 4
        w = self.w - self.l_margin - self.r_margin - 8
        self.ln(1)
        for line in lines:
            self.set_x(x)
            self.cell(w, 5.5, line, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def render_table(self, rows: list[list[str]]):
        """표 렌더."""
        if not rows:
            return
        self._check_space(len(rows) * 7 + 10)

        num_cols = len(rows[0])
        avail_w = self.w - self.l_margin - self.r_margin
        col_w = avail_w / num_cols

        # 행 열 수 통일
        for idx, row in enumerate(rows):
            while len(row) < num_cols:
                row.append("")
            rows[idx] = row[:num_cols]

        # 열 너비 계산: 내용 길이 비례
        max_lens = [0] * num_cols
        for row in rows:
            for i, cell in enumerate(row):
                clen = len(self._clean(cell))
                if clen > max_lens[i]:
                    max_lens[i] = clen
        total = sum(max_lens) or 1
        col_widths = [max(avail_w * (ml / total), 15) for ml in max_lens]
        # 정규화
        s = sum(col_widths)
        col_widths = [cw * avail_w / s for cw in col_widths]

        self.set_draw_color(*C_TABLE_BORDER)
        self.set_line_width(0.2)

        for row_idx, row in enumerate(rows):
            # 페이지 넘김 체크
            if self.get_y() + 8 > self.h - self.b_margin:
                self.add_page()

            if row_idx == 0:
                # 헤더 행
                self.set_fill_color(*C_TABLE_HEADER_BG)
                self.set_text_color(*C_TABLE_HEADER_FG)
                self.set_font("gothic", "B", 9)
            else:
                if row_idx % 2 == 0:
                    self.set_fill_color(*C_TABLE_ALT_ROW)
                else:
                    self.set_fill_color(255, 255, 255)
                self.set_text_color(*C_BLACK)
                self.set_font("gothic", "", 9)

            x = self.l_margin
            max_h = 7
            # 높이 사전 계산
            cell_texts = []
            for i, cell in enumerate(row):
                ct = self._clean(cell) if i < len(row) else ""
                cell_texts.append(ct)
                # 대략 줄 수 계산
                char_per_line = max(int(col_widths[i] / 2.2), 1)
                n_lines = max(1, -(-len(ct) // char_per_line))  # ceil div
                h = n_lines * 5.5
                if h > max_h:
                    max_h = h

            max_h = min(max_h, 25)  # 최대 높이 제한

            for i, ct in enumerate(cell_texts):
                self.set_xy(x, self.get_y())
                self.cell(col_widths[i], max_h, ct[:60], border=1, fill=True)
                x += col_widths[i]

            self.ln(max_h)

        self.ln(3)

    def render_hr(self):
        """수평선."""
        self.ln(2)
        self.set_draw_color(*C_LINE)
        self.set_line_width(0.3)
        y = self.get_y()
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(4)


def parse_and_render(pdf: RulesPDF, md_text: str):
    """마크다운 파싱 → PDF 렌더."""
    lines = md_text.split("\n")
    i = 0
    is_first_quote = True
    quote_buffer: list[str] = []
    in_code = False
    code_lines: list[str] = []

    while i < len(lines):
        line = lines[i]

        # 코드 블록
        if line.strip().startswith("```"):
            if in_code:
                pdf.render_code_block(code_lines)
                code_lines = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_lines.append(line)
            i += 1
            continue

        # 빈 줄
        if not line.strip():
            if quote_buffer:
                pdf.render_quote(" ".join(quote_buffer))
                quote_buffer = []
            i += 1
            continue

        # 수평선
        if line.strip() == "---":
            if quote_buffer:
                pdf.render_quote(" ".join(quote_buffer))
                quote_buffer = []
            pdf.render_hr()
            i += 1
            continue

        # H1
        if line.startswith("# ") and not line.startswith("## "):
            pdf.render_title(line[2:])
            i += 1
            continue

        # H2
        if line.startswith("## "):
            if quote_buffer:
                pdf.render_quote(" ".join(quote_buffer))
                quote_buffer = []
            pdf.render_h2(line[3:])
            i += 1
            continue

        # H3
        if line.startswith("### "):
            if quote_buffer:
                pdf.render_quote(" ".join(quote_buffer))
                quote_buffer = []
            pdf.render_h3(line[4:])
            i += 1
            continue

        # 인용문
        if line.startswith("> "):
            content = line[2:].strip()
            if is_first_quote and i < 3:
                pdf.render_subtitle(content)
                is_first_quote = False
            else:
                quote_buffer.append(content)
            i += 1
            continue

        # 표
        if line.strip().startswith("|"):
            if quote_buffer:
                pdf.render_quote(" ".join(quote_buffer))
                quote_buffer = []
            table_rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row_text = lines[i].strip()
                # 구분선 행 스킵
                if re.match(r"^\|[\s\-:|]+\|$", row_text):
                    i += 1
                    continue
                cells = [c.strip() for c in row_text.split("|")[1:-1]]
                table_rows.append(cells)
                i += 1
            pdf.render_table(table_rows)
            continue

        # 글머리 기호
        if line.strip().startswith("- "):
            if quote_buffer:
                pdf.render_quote(" ".join(quote_buffer))
                quote_buffer = []
            pdf.render_bullet(line.strip()[2:])
            i += 1
            continue

        # 일반 단락
        if quote_buffer:
            pdf.render_quote(" ".join(quote_buffer))
            quote_buffer = []
        pdf.render_paragraph(line)
        i += 1

    # 남은 인용문 플러시
    if quote_buffer:
        pdf.render_quote(" ".join(quote_buffer))


def main():
    md_text = MD_PATH.read_text(encoding="utf-8")

    pdf = RulesPDF()
    pdf.add_page()
    parse_and_render(pdf, md_text)
    pdf.output(str(OUT_PATH))
    print(f"PDF 생성 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()

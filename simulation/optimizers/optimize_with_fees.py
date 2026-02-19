#!/usr/bin/env python3
"""
손절 최적화 + KIS 수수료 반영 비교
===================================
수수료 없음 vs KIS 수수료 포함 → 1년 / 1개월 비교
"""
from __future__ import annotations

from datetime import date, timedelta
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

import config
from simulation.backtests.backtest import (
    BacktestEngine, fetch_backtest_data,
    KIS_COMMISSION_PCT, KIS_SEC_FEE_PCT, KIS_FX_SPREAD_PCT,
)

STOP_LOSS_VALUES = [
    -1.0, -1.5, -2.0, -2.5, -3.0, -3.5,
    -4.0, -5.0, -6.0, -8.0, -10.0, -999.0,
]
LABELS = {v: f"{v:.1f}%" for v in STOP_LOSS_VALUES}
LABELS[-999.0] = "없음"


def run_sweep(start_date, end_date, use_fees, data):
    results = []
    for sl in STOP_LOSS_VALUES:
        engine = BacktestEngine(
            initial_capital=1000.0,
            start_date=start_date,
            end_date=end_date,
            stop_loss_pct=sl,
            use_fees=use_fees,
        )
        engine.run(verbose=False, data=data)
        s = engine.summary()
        s["stop_loss"] = LABELS[sl]
        s["stop_loss_raw"] = sl
        results.append(s)
    return pd.DataFrame(results)


def print_comparison_table(df_no_fee, df_fee, title):
    """수수료 미포함/포함 나란히 비교."""
    print()
    print("=" * 100)
    print(f"  {title}")
    print("=" * 100)
    print(
        f"  {'손절':>6s}  │ {'수수료 없음':^28s} │ {'KIS 수수료 포함':^34s} │ {'수수료':>8s}"
    )
    print(
        f"  {'라인':>6s}  │ {'수익률':>8s}  {'MDD':>7s}  {'Sharpe':>7s}  │ "
        f"{'수익률':>8s}  {'MDD':>7s}  {'Sharpe':>7s}  {'총수수료':>9s}  │ {'차이':>8s}"
    )
    print("-" * 100)

    best_fee_idx = df_fee["total_return_pct"].idxmax()

    for i in range(len(df_no_fee)):
        nf = df_no_fee.iloc[i]
        wf = df_fee.iloc[i]
        diff = wf["total_return_pct"] - nf["total_return_pct"]
        marker = " <-- BEST" if i == best_fee_idx else ""
        print(
            f"  {nf['stop_loss']:>6s}  │ "
            f"{nf['total_return_pct']:>+7.2f}%  {nf['mdd_pct']:>6.2f}%  {nf['sharpe']:>7.2f}  │ "
            f"{wf['total_return_pct']:>+7.2f}%  {wf['mdd_pct']:>6.2f}%  {wf['sharpe']:>7.2f}  "
            f"${wf['total_fees']:>8.2f}  │ "
            f"{diff:>+7.2f}%{marker}"
        )
    print("=" * 100)


def print_fee_breakdown(df_fee, label):
    """수수료 상세 내역."""
    best = df_fee.loc[df_fee["total_return_pct"].idxmax()]
    print(f"\n  [{label} — 최적 손절 {best['stop_loss']}] 수수료 상세")
    print(f"    매매수수료 (0.25%×2):  ${best['commission']:>8.2f}")
    print(f"    SEC Fee (0.00278%):    ${best['sec_fee']:>8.2f}")
    print(f"    환전스프레드 (0.1%×2): ${best['fx_cost']:>8.2f}")
    print(f"    ─────────────────────────────")
    print(f"    총 수수료:             ${best['total_fees']:>8.2f}")
    print(f"    수수료 비율:           {best['total_fees'] / 10:.2f}% (초기자금 대비)")


def generate_report_pdf(df_year_nf, df_year_f, df_month_nf, df_month_f):
    """PDF 리포트 생성."""
    from fpdf import FPDF

    FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
    OUT = Path(__file__).parent / "docs" / "stoploss_with_fees_report.pdf"

    class PDF(FPDF):
        def __init__(self):
            super().__init__()
            self.add_font("F", "", FONT)
            self.add_font("F", "B", FONT)
            self.set_auto_page_break(auto=True, margin=15)

        def header(self):
            self.set_font("F", "B", 9)
            self.set_text_color(130, 130, 130)
            self.cell(0, 8, "PTJ Strategy — Stop Loss + KIS Fee Analysis", align="R",
                      new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(200, 200, 200)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)

        def footer(self):
            self.set_y(-15)
            self.set_font("F", "", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

        def title_block(self, text):
            self.set_font("F", "B", 18)
            self.set_text_color(30, 30, 30)
            self.cell(0, 14, text, align="C", new_x="LMARGIN", new_y="NEXT")

        def section(self, text):
            self.set_font("F", "B", 13)
            self.set_text_color(30, 30, 30)
            self.ln(4)
            self.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(66, 133, 244)
            self.set_line_width(0.6)
            self.line(10, self.get_y(), 75, self.get_y())
            self.set_line_width(0.2)
            self.ln(4)

        def sub(self, text):
            self.set_font("F", "B", 11)
            self.set_text_color(60, 60, 60)
            self.ln(2)
            self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
            self.ln(1)

        def text_line(self, text):
            self.set_font("F", "", 10)
            self.set_text_color(50, 50, 50)
            self.multi_cell(0, 6, text)
            self.ln(1)

        def bullet(self, text):
            self.set_font("F", "", 10)
            self.set_text_color(50, 50, 50)
            self.cell(6, 6, "\u2022 ", new_x="END")
            self.multi_cell(0, 6, text)
            self.ln(1)

        def table(self, headers, rows, widths=None, highlight=None):
            if widths is None:
                widths = [190 / len(headers)] * len(headers)
            self.set_font("F", "B", 7.5)
            self.set_fill_color(66, 133, 244)
            self.set_text_color(255, 255, 255)
            for i, h in enumerate(headers):
                self.cell(widths[i], 6.5, h, border=1, fill=True, align="C")
            self.ln()
            self.set_font("F", "", 7.5)
            for r_idx, row in enumerate(rows):
                is_hl = (r_idx == highlight)
                if is_hl:
                    self.set_fill_color(232, 245, 233)
                    self.set_font("F", "B", 7.5)
                else:
                    self.set_fill_color(250, 250, 250) if r_idx % 2 == 0 else self.set_fill_color(255, 255, 255)
                    self.set_font("F", "", 7.5)
                self.set_text_color(30, 30, 30)
                for i, v in enumerate(row):
                    self.cell(widths[i], 5.5, str(v), border=1, fill=True, align="C" if i > 0 else "L")
                self.ln()
            self.ln(3)

    def make_comparison_rows(df_nf, df_f):
        rows = []
        best_idx = int(df_f["total_return_pct"].idxmax())
        for i in range(len(df_nf)):
            nf = df_nf.iloc[i]
            wf = df_f.iloc[i]
            diff = wf["total_return_pct"] - nf["total_return_pct"]
            rows.append([
                nf["stop_loss"],
                f"{nf['total_return_pct']:+.2f}%",
                f"{nf['sharpe']:.2f}",
                f"{wf['total_return_pct']:+.2f}%",
                f"{wf['mdd_pct']:.2f}%",
                f"{wf['sharpe']:.2f}",
                f"${wf['total_fees']:.2f}",
                f"{diff:+.2f}%",
            ])
        return rows, best_idx

    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # Title
    pdf.title_block("손절 라인 최적화 + KIS 수수료 분석")
    pdf.ln(2)
    pdf.set_font("F", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7, "2026-02-18 | $1,000 | Alpaca 5분봉 | PTJ 매매법",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # KIS 수수료 구조
    pdf.section("KIS 수수료 구조 (미국주식)")
    pdf.table(
        ["항목", "비율", "적용 시점", "비고"],
        [
            ["매매수수료", "0.25%", "매수 + 매도", "온라인 기준"],
            ["SEC Fee", "0.00278%", "매도 시", "미국 SEC 부과"],
            ["환전스프레드", "~0.10%", "매수 + 매도", "USD/KRW 환전"],
            ["편도 합계", "~0.35%", "-", "매수 또는 매도 1회"],
            ["왕복 합계", "~0.70%", "-", "매수 + 매도 1세트"],
        ],
        widths=[35, 25, 30, 60],
    )

    # 요약
    pdf.section("요약")
    best_y_nf = df_year_nf.loc[df_year_nf["total_return_pct"].idxmax()]
    best_y_f = df_year_f.loc[df_year_f["total_return_pct"].idxmax()]
    best_m_nf = df_month_nf.loc[df_month_nf["total_return_pct"].idxmax()]
    best_m_f = df_month_f.loc[df_month_f["total_return_pct"].idxmax()]
    pdf.table(
        ["기간", "조건", "최적 손절", "수익률", "MDD", "Sharpe", "총 수수료"],
        [
            ["1년", "수수료 없음", best_y_nf["stop_loss"],
             f"{best_y_nf['total_return_pct']:+.2f}%", f"{best_y_nf['mdd_pct']:.2f}%",
             f"{best_y_nf['sharpe']:.2f}", "-"],
            ["1년", "KIS 수수료", best_y_f["stop_loss"],
             f"{best_y_f['total_return_pct']:+.2f}%", f"{best_y_f['mdd_pct']:.2f}%",
             f"{best_y_f['sharpe']:.2f}", f"${best_y_f['total_fees']:.2f}"],
            ["1개월", "수수료 없음", best_m_nf["stop_loss"],
             f"{best_m_nf['total_return_pct']:+.2f}%", f"{best_m_nf['mdd_pct']:.2f}%",
             f"{best_m_nf['sharpe']:.2f}", "-"],
            ["1개월", "KIS 수수료", best_m_f["stop_loss"],
             f"{best_m_f['total_return_pct']:+.2f}%", f"{best_m_f['mdd_pct']:.2f}%",
             f"{best_m_f['sharpe']:.2f}", f"${best_m_f['total_fees']:.2f}"],
        ],
        widths=[20, 28, 22, 22, 22, 20, 28],
        highlight=1,
    )

    # 1년 비교 테이블
    pdf.add_page()
    pdf.section("1년 백테스트 — 수수료 없음 vs KIS 수수료")
    headers = ["손절", "수익률(무)", "Sharpe(무)", "수익률(KIS)", "MDD(KIS)", "Sharpe(KIS)", "수수료", "차이"]
    widths = [18, 24, 22, 24, 22, 22, 24, 20]
    rows, best_idx = make_comparison_rows(df_year_nf, df_year_f)
    pdf.table(headers, rows, widths, highlight=best_idx)

    # 수수료 상세
    best_f = df_year_f.loc[df_year_f["total_return_pct"].idxmax()]
    pdf.sub(f"1년 수수료 상세 (손절 {best_f['stop_loss']})")
    pdf.bullet(f"매매수수료 (0.25% x 2): ${best_f['commission']:.2f}")
    pdf.bullet(f"SEC Fee (0.00278%): ${best_f['sec_fee']:.2f}")
    pdf.bullet(f"환전스프레드 (~0.1% x 2): ${best_f['fx_cost']:.2f}")
    pdf.bullet(f"총 수수료: ${best_f['total_fees']:.2f} ({best_f['total_fees']/10:.1f}% of 초기자금)")

    # 1개월 비교 테이블
    pdf.add_page()
    pdf.section("최근 1개월 — 수수료 없음 vs KIS 수수료")
    rows_m, best_m_idx = make_comparison_rows(df_month_nf, df_month_f)
    pdf.table(headers, rows_m, widths, highlight=best_m_idx)

    best_fm = df_month_f.loc[df_month_f["total_return_pct"].idxmax()]
    pdf.sub(f"1개월 수수료 상세 (손절 {best_fm['stop_loss']})")
    pdf.bullet(f"매매수수료: ${best_fm['commission']:.2f}")
    pdf.bullet(f"SEC Fee: ${best_fm['sec_fee']:.2f}")
    pdf.bullet(f"환전스프레드: ${best_fm['fx_cost']:.2f}")
    pdf.bullet(f"총 수수료: ${best_fm['total_fees']:.2f}")

    # 분석
    pdf.add_page()
    pdf.section("분석")

    fee_impact = best_y_nf["total_return_pct"] - best_y_f["total_return_pct"]

    pdf.sub("1. 수수료가 수익률에 미치는 영향")
    pdf.bullet(f"1년 최적 기준: 수수료로 인해 수익률 {fee_impact:+.2f}%p 감소")
    pdf.bullet(f"총 수수료 ${best_f['total_fees']:.2f} = 초기자금의 {best_f['total_fees']/10:.1f}%")
    pdf.bullet("매매 횟수가 많을수록 수수료 부담 증가 (타이트한 손절 = 잦은 거래)")

    pdf.sub("2. 수수료 반영 시 최적 손절이 달라지는가?")
    if best_y_nf["stop_loss"] == best_y_f["stop_loss"]:
        pdf.bullet(f"1년: 수수료 유무와 관계없이 동일 ({best_y_f['stop_loss']})")
    else:
        pdf.bullet(f"1년: 수수료 없음 {best_y_nf['stop_loss']} -> KIS {best_y_f['stop_loss']}")
    if best_m_nf["stop_loss"] == best_m_f["stop_loss"]:
        pdf.bullet(f"1개월: 수수료 유무와 관계없이 동일 ({best_m_f['stop_loss']})")
    else:
        pdf.bullet(f"1개월: 수수료 없음 {best_m_nf['stop_loss']} -> KIS {best_m_f['stop_loss']}")

    pdf.sub("3. 수수료 절감 전략")
    pdf.bullet("손절 라인을 너무 타이트하게 잡으면 잦은 매매로 수수료 증가")
    pdf.bullet("수수료 대비 수익이 큰 손절 라인을 선택하는 것이 핵심")
    pdf.bullet("KIS 이벤트(신규 3개월 수수료 면제) 활용 시 초기 비용 절감 가능")

    # 권장사항
    pdf.section("권장사항")
    pdf.table(
        ["항목", "현재", "권장 (수수료 포함)"],
        [["STOP_LOSS_PCT", "-3.0%", best_y_f["stop_loss"]]],
        widths=[55, 45, 50],
        highlight=0,
    )
    pdf.ln(2)
    pdf.text_line(f"KIS 수수료 반영 시 예상 효과 (1년):")
    pdf.bullet(f"수익률: +4.12% -> {best_y_f['total_return_pct']:+.2f}%")
    pdf.bullet(f"MDD: 9.43% -> {best_y_f['mdd_pct']:.2f}%")
    pdf.bullet(f"연간 수수료: ~${best_y_f['total_fees']:.2f}")

    # 참고
    pdf.section("참고")
    pdf.bullet("데이터: Alpaca IEX 5분봉 (2025-02-18 ~ 2026-02-13)")
    pdf.bullet("수수료: 한국투자증권 온라인 기준 (2026년)")
    pdf.bullet("양도소득세(22%, 250만원 공제) 미반영")
    pdf.bullet("배당소득세(15%) 미반영")
    pdf.bullet("환율 변동 리스크 미반영")

    pdf.output(str(OUT))
    print(f"\n  PDF 저장: {OUT}")
    return OUT


def main():
    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║  손절 최적화 + KIS 수수료 비교 분석             ║")
    print("  ╚══════════════════════════════════════════════════╝")

    end_date = date(2026, 2, 17)
    start_year = date(2025, 2, 18)
    start_month = end_date - timedelta(days=30)

    print("\n[1/5] 데이터 로드")
    data = fetch_backtest_data(start_year - timedelta(days=10), end_date)

    print(f"\n[2/5] 1년 — 수수료 없음 ({len(STOP_LOSS_VALUES)}개)", end="", flush=True)
    df_year_nf = run_sweep(start_year, end_date, False, data)
    print(" OK")

    print(f"[3/5] 1년 — KIS 수수료 ({len(STOP_LOSS_VALUES)}개)", end="", flush=True)
    df_year_f = run_sweep(start_year, end_date, True, data)
    print(" OK")

    print(f"[4/5] 1개월 — 양쪽 ({len(STOP_LOSS_VALUES)}x2개)", end="", flush=True)
    df_month_nf = run_sweep(start_month, end_date, False, data)
    df_month_f = run_sweep(start_month, end_date, True, data)
    print(" OK")

    # 콘솔 출력
    print_comparison_table(df_year_nf, df_year_f, "1년 백테스트 (2025-02-18 ~ 2026-02-17)")
    print_fee_breakdown(df_year_f, "1년")

    print_comparison_table(df_month_nf, df_month_f, "최근 1개월 백테스트")
    print_fee_breakdown(df_month_f, "1개월")

    # PDF
    print("\n[5/5] PDF 생성")
    generate_report_pdf(df_year_nf, df_year_f, df_month_nf, df_month_f)

    # CSV
    combined = pd.concat([
        df_year_nf.assign(period="1년", fees="없음"),
        df_year_f.assign(period="1년", fees="KIS"),
        df_month_nf.assign(period="1개월", fees="없음"),
        df_month_f.assign(period="1개월", fees="KIS"),
    ])
    (config.RESULTS_DIR / "optimization").mkdir(parents=True, exist_ok=True)
    csv_path = config.RESULTS_DIR / "optimization" / "stoploss_fees_comparison.csv"
    combined.to_csv(csv_path, index=False)
    print(f"  CSV 저장: {csv_path}")
    print()


if __name__ == "__main__":
    main()

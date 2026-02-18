"""
PTJ 매매법 - 인터랙티브 HTML 대시보드
=====================================
Plotly.js CDN을 사용한 자체 완결형 HTML 파일 생성.
Python plotly 패키지 불필요 — 순수 HTML/JS를 직접 조립.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


# ============================================================
# 색상 / 스타일 상수
# ============================================================
COLORS = {
    "bg": "#1a1a2e",
    "card": "#16213e",
    "card_border": "#0f3460",
    "accent": "#e94560",
    "green": "#00e676",
    "red": "#ff5252",
    "yellow": "#ffd740",
    "text": "#ffffff",
    "text_muted": "#a0a0b0",
    "grid": "#2a2a4a",
    "blue": "#448aff",
}

SIGNAL_COLOR_MAP = {
    "SELL": COLORS["red"],
    "ENTRY": COLORS["green"],
    "HOLD": COLORS["yellow"],
}

SIGNAL_LABEL_MAP = {
    "SELL": "매도 시그널",
    "ENTRY": "매수 검토",
    "HOLD": "관망",
}


# ============================================================
# HTML 템플릿
# ============================================================
def _html_template(
    title: str,
    date_str: str,
    body_content: str,
) -> str:
    """메인 HTML 프레임."""
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: 'Noto Sans KR', sans-serif;
    background: {COLORS['bg']};
    color: {COLORS['text']};
    line-height: 1.6;
    padding: 20px;
}}
.container {{ max-width: 1400px; margin: 0 auto; }}

/* Header */
.header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    padding: 20px 24px;
    background: {COLORS['card']};
    border: 1px solid {COLORS['card_border']};
    border-radius: 12px;
    margin-bottom: 20px;
}}
.header h1 {{ font-size: 1.6rem; font-weight: 700; }}
.header-date {{ color: {COLORS['text_muted']}; font-size: 0.95rem; }}
.badge {{
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
}}
.badge-green {{ background: {COLORS['green']}20; color: {COLORS['green']}; border: 1px solid {COLORS['green']}40; }}
.badge-red {{ background: {COLORS['red']}20; color: {COLORS['red']}; border: 1px solid {COLORS['red']}40; }}
.badge-yellow {{ background: {COLORS['yellow']}20; color: {COLORS['yellow']}; border: 1px solid {COLORS['yellow']}40; }}

/* Section */
.section {{
    background: {COLORS['card']};
    border: 1px solid {COLORS['card_border']};
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
}}
.section-title {{
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid {COLORS['card_border']};
}}

/* Cards grid */
.card-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 14px;
}}
.card {{
    background: {COLORS['bg']};
    border: 1px solid {COLORS['card_border']};
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}}
.card-label {{ font-size: 0.85rem; color: {COLORS['text_muted']}; margin-bottom: 6px; }}
.card-value {{ font-size: 1.6rem; font-weight: 700; }}
.card-sub {{ font-size: 0.8rem; margin-top: 4px; }}

/* Plotly container */
.plot-container {{ width: 100%; min-height: 320px; }}

/* Table */
.data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}}
.data-table th {{
    background: {COLORS['bg']};
    padding: 10px 12px;
    text-align: left;
    font-weight: 500;
    color: {COLORS['text_muted']};
    border-bottom: 2px solid {COLORS['card_border']};
    cursor: pointer;
    user-select: none;
}}
.data-table th:hover {{ color: {COLORS['text']}; }}
.data-table td {{
    padding: 8px 12px;
    border-bottom: 1px solid {COLORS['card_border']}30;
}}
.data-table tr:hover td {{ background: {COLORS['card_border']}30; }}

/* Conditional section */
.cond-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 14px;
    margin-bottom: 16px;
}}
.cond-card {{
    background: {COLORS['bg']};
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    border: 2px solid transparent;
}}
.cond-card.positive {{ border-color: {COLORS['green']}60; }}
.cond-card.negative {{ border-color: {COLORS['red']}60; }}
.cond-indicator {{
    width: 12px; height: 12px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}}

/* Message box */
.msg-box {{
    padding: 12px 18px;
    border-radius: 8px;
    font-size: 0.95rem;
    margin-top: 10px;
}}
.msg-info {{ background: {COLORS['blue']}15; border-left: 4px solid {COLORS['blue']}; }}
.msg-warn {{ background: {COLORS['red']}15; border-left: 4px solid {COLORS['red']}; }}
.msg-ok {{ background: {COLORS['green']}15; border-left: 4px solid {COLORS['green']}; }}

@media (max-width: 768px) {{
    body {{ padding: 10px; }}
    .header {{ flex-direction: column; text-align: center; }}
    .card-grid {{ grid-template-columns: 1fr 1fr; }}
}}
</style>
</head>
<body>
<div class="container">
{body_content}
</div>
<script>
// 테이블 정렬
function sortTable(tableId, colIdx) {{
    const table = document.getElementById(tableId);
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const th = table.querySelectorAll('th')[colIdx];
    const asc = th.dataset.sort !== 'asc';
    th.dataset.sort = asc ? 'asc' : 'desc';
    rows.sort((a, b) => {{
        let av = a.cells[colIdx].textContent.trim();
        let bv = b.cells[colIdx].textContent.trim();
        const an = parseFloat(av.replace(/[^\\d.\\-]/g, ''));
        const bn = parseFloat(bv.replace(/[^\\d.\\-]/g, ''));
        if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
        return asc ? av.localeCompare(bv) : bv.localeCompare(av);
    }});
    rows.forEach(r => tbody.appendChild(r));
}}
</script>
</body>
</html>"""


# ============================================================
# 섹션 빌더
# ============================================================
def _build_header(signals: dict, date_str: str) -> str:
    """헤더 + 금 경고 뱃지."""
    gold = signals.get("gold", {})
    gold_warning = gold.get("warning", False)
    gold_msg = gold.get("message", "")

    if gold_warning:
        status_badge = '<span class="badge badge-red">매매 금지</span>'
    else:
        status_badge = '<span class="badge badge-green">매매 가능</span>'

    gold_badge = ""
    if gold_warning:
        gold_badge = f'<span class="badge badge-yellow">금 경고: {gold_msg}</span>'

    return f"""
    <div class="header">
        <div>
            <h1>PTJ 매매법 대시보드</h1>
            <span class="header-date">{date_str} 기준</span>
        </div>
        <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
            {status_badge}
            {gold_badge}
        </div>
    </div>"""


def _build_twin_gap_chart(signals: dict) -> str:
    """쌍둥이 페어 갭 바 차트 (Plotly)."""
    pairs = signals.get("twin_pairs", [])
    if not pairs:
        return ""

    labels = [p["pair"] for p in pairs]
    gaps = [p["gap"] for p in pairs]
    colors = [SIGNAL_COLOR_MAP.get(p["signal"], COLORS["yellow"]) for p in pairs]
    hover_texts = [
        f"{p['pair']}<br>{p['lead']} {p['lead_pct']:+.2f}% | "
        f"{p['follow']} {p['follow_pct']:+.2f}%<br>갭: {p['gap']:+.2f}%<br>"
        f"{SIGNAL_LABEL_MAP.get(p['signal'], '')}"
        for p in pairs
    ]
    annotations_data = [
        {"x": g, "y": l, "text": f"{g:+.2f}%", "signal": SIGNAL_LABEL_MAP.get(p["signal"], "")}
        for l, g, p in zip(labels, gaps, pairs)
    ]

    data = [{
        "type": "bar",
        "orientation": "h",
        "y": labels,
        "x": gaps,
        "marker": {"color": colors, "cornerradius": 4},
        "hovertext": hover_texts,
        "hoverinfo": "text",
        "textposition": "outside",
    }]

    # 매도/매수 기준선 shapes
    shapes = [
        {
            "type": "line", "x0": config.PAIR_GAP_SELL_THRESHOLD,
            "x1": config.PAIR_GAP_SELL_THRESHOLD,
            "y0": -0.5, "y1": len(labels) - 0.5,
            "line": {"color": COLORS["red"], "width": 1, "dash": "dot"},
        },
        {
            "type": "line", "x0": -config.PAIR_GAP_SELL_THRESHOLD,
            "x1": -config.PAIR_GAP_SELL_THRESHOLD,
            "y0": -0.5, "y1": len(labels) - 0.5,
            "line": {"color": COLORS["red"], "width": 1, "dash": "dot"},
        },
        {
            "type": "line", "x0": config.PAIR_GAP_ENTRY_THRESHOLD,
            "x1": config.PAIR_GAP_ENTRY_THRESHOLD,
            "y0": -0.5, "y1": len(labels) - 0.5,
            "line": {"color": COLORS["green"], "width": 1, "dash": "dot"},
        },
        {
            "type": "line", "x0": -config.PAIR_GAP_ENTRY_THRESHOLD,
            "x1": -config.PAIR_GAP_ENTRY_THRESHOLD,
            "y0": -0.5, "y1": len(labels) - 0.5,
            "line": {"color": COLORS["green"], "width": 1, "dash": "dot"},
        },
    ]

    annotations = []
    for a in annotations_data:
        annotations.append({
            "x": a["x"],
            "y": a["y"],
            "text": f"  {a['text']} ({a['signal']})",
            "showarrow": False,
            "xanchor": "left" if a["x"] >= 0 else "right",
            "font": {"color": COLORS["text"], "size": 12},
        })

    layout = {
        "paper_bgcolor": COLORS["card"],
        "plot_bgcolor": COLORS["card"],
        "font": {"color": COLORS["text"], "family": "'Noto Sans KR', sans-serif"},
        "margin": {"l": 160, "r": 100, "t": 10, "b": 30},
        "height": max(200, len(labels) * 80 + 40),
        "xaxis": {
            "zeroline": True,
            "zerolinecolor": COLORS["text_muted"],
            "gridcolor": COLORS["grid"],
            "title": "페어 갭 (%)",
        },
        "yaxis": {"gridcolor": COLORS["grid"]},
        "shapes": shapes,
        "annotations": annotations,
    }

    data_json = json.dumps(data, ensure_ascii=False)
    layout_json = json.dumps(layout, ensure_ascii=False)
    div_id = "chart-twin-gap"

    return f"""
    <div class="section">
        <div class="section-title">쌍둥이 페어 갭 차트</div>
        <div class="plot-container" id="{div_id}"></div>
        <script>Plotly.newPlot('{div_id}', {data_json}, {layout_json}, {{responsive: true, displayModeBar: false}});</script>
    </div>"""


def _build_twin_trend_chart(data: pd.DataFrame, signals: dict) -> str:
    """쌍둥이 가격 추이 라인 차트 (최근 30일, 정규화)."""
    pairs = signals.get("twin_pairs", [])
    if not pairs or data.empty:
        return ""

    # 최근 30 거래일만
    all_dates = sorted(data["Date"].unique())
    cutoff = all_dates[-30] if len(all_dates) >= 30 else all_dates[0]
    recent = data[data["Date"] >= cutoff].copy()

    traces = []
    n_pairs = len(pairs)

    for i, p in enumerate(pairs):
        for ticker, dash, role in [
            (p["lead"], "solid", "선행"),
            (p["follow"], "dash", "후행"),
        ]:
            tdf = recent[recent["ticker"] == ticker].sort_values("Date")
            if tdf.empty:
                continue
            base = tdf["Close"].iloc[0]
            if base == 0:
                continue
            norm_prices = ((tdf["Close"] / base) * 100).round(2).tolist()
            dates = [str(d)[:10] for d in tdf["Date"].tolist()]

            traces.append({
                "type": "scatter",
                "mode": "lines",
                "x": dates,
                "y": norm_prices,
                "name": f"{ticker} ({role})",
                "line": {"dash": dash, "width": 2},
                "xaxis": f"x{i + 1}" if i > 0 else "x",
                "yaxis": f"y{i + 1}" if i > 0 else "y",
                "hovertemplate": f"{ticker}<br>%{{x}}<br>%{{y:.1f}}<extra></extra>",
            })

    if not traces:
        return ""

    # 서브플롯 레이아웃 계산
    gap = 0.08
    plot_height = 1.0 / n_pairs - gap / n_pairs
    domains = []
    for i in range(n_pairs):
        bottom = i * (plot_height + gap)
        top = bottom + plot_height
        domains.append([round(bottom, 3), round(top, 3)])
    domains.reverse()  # 첫 페어를 위에 배치

    layout = {
        "paper_bgcolor": COLORS["card"],
        "plot_bgcolor": COLORS["card"],
        "font": {"color": COLORS["text"], "family": "'Noto Sans KR', sans-serif", "size": 11},
        "margin": {"l": 60, "r": 20, "t": 20, "b": 40},
        "height": n_pairs * 220,
        "showlegend": True,
        "legend": {"orientation": "h", "y": -0.05, "x": 0.5, "xanchor": "center"},
    }

    for i in range(n_pairs):
        ax_suffix = str(i + 1) if i > 0 else ""
        layout[f"xaxis{ax_suffix}"] = {
            "gridcolor": COLORS["grid"],
            "domain": [0, 1],
            "anchor": f"y{ax_suffix}",
            "showticklabels": (i == 0),
        }
        layout[f"yaxis{ax_suffix}"] = {
            "gridcolor": COLORS["grid"],
            "domain": domains[i],
            "anchor": f"x{ax_suffix}",
            "title": pairs[n_pairs - 1 - i]["pair"][:6],
        }

    data_json = json.dumps(traces, ensure_ascii=False)
    layout_json = json.dumps(layout, ensure_ascii=False)
    div_id = "chart-twin-trend"

    return f"""
    <div class="section">
        <div class="section-title">쌍둥이 가격 추이 (30일 정규화)</div>
        <div class="plot-container" id="{div_id}"></div>
        <script>Plotly.newPlot('{div_id}', {data_json}, {layout_json}, {{responsive: true, displayModeBar: false}});</script>
    </div>"""


def _build_conditional_section(signals: dict, changes: dict) -> str:
    """조건부 매매 섹션."""
    cond = signals.get("conditional", {})
    triggers = cond.get("triggers", {})
    all_pos = cond.get("all_positive", False)
    target = cond.get("target", "COIN")
    target_pct = cond.get("target_pct", 0.0)

    cards_html = ""
    for ticker, info in triggers.items():
        pct = info.get("change_pct", 0.0)
        positive = info.get("positive", False)
        cls = "positive" if positive else "negative"
        indicator_color = COLORS["green"] if positive else COLORS["red"]
        check = "&#10003;" if positive else "&#10007;"
        name = config.TICKERS.get(ticker, {}).get("name", ticker)

        cards_html += f"""
        <div class="cond-card {cls}">
            <div class="card-label">{name}</div>
            <div class="card-value" style="color: {'#00e676' if positive else '#ff5252'};">
                {ticker}
            </div>
            <div style="font-size:1.2rem; margin-top:6px; color: {indicator_color};">
                {pct:+.2f}% <span style="font-size:1.4rem;">{check}</span>
            </div>
        </div>"""

    # Target card
    target_name = config.TICKERS.get(target, {}).get("name", target)
    target_color = COLORS["green"] if all_pos else COLORS["text_muted"]
    target_border = COLORS["green"] if all_pos else COLORS["card_border"]
    target_label = "매수 시그널" if all_pos else "대기"

    cards_html += f"""
    <div class="cond-card" style="border-color: {target_border}60;">
        <div class="card-label">{target_name} (타겟)</div>
        <div class="card-value" style="color: {target_color};">{target}</div>
        <div style="font-size:1.0rem; margin-top:6px; color: {target_color};">
            {target_pct:+.2f}% &mdash; {target_label}
        </div>
    </div>"""

    msg = cond.get("message", "")
    msg_cls = "msg-ok" if all_pos else "msg-warn"

    return f"""
    <div class="section">
        <div class="section-title">조건부 매매</div>
        <p style="color:{COLORS['text_muted']}; margin-bottom:12px;">
            ETHU + XXRP + SOLT 3종목 모두 양전 시 &rarr; COIN 매수 신호
        </p>
        <div class="cond-grid">{cards_html}</div>
        <div class="msg-box {msg_cls}">{msg}</div>
    </div>"""


def _build_bearish_section(signals: dict) -> str:
    """하락장 방어주 바 차트."""
    bearish = signals.get("bearish", {})
    picks = bearish.get("bearish_picks", [])
    msg = bearish.get("message", "")
    market_down = bearish.get("market_down", False)

    if not picks:
        return ""

    tickers = [p["ticker"] for p in picks]
    pcts = [p["change_pct"] for p in picks]
    names = [p["name"] for p in picks]
    bar_colors = [COLORS["green"] if v > 0 else COLORS["red"] for v in pcts]
    hover = [f"{t} ({n})<br>{p:+.2f}%" for t, n, p in zip(tickers, names, pcts)]

    data = [{
        "type": "bar",
        "x": tickers,
        "y": pcts,
        "marker": {"color": bar_colors, "cornerradius": 4},
        "hovertext": hover,
        "hoverinfo": "text",
        "text": [f"{p:+.2f}%" for p in pcts],
        "textposition": "outside",
        "textfont": {"color": COLORS["text"]},
    }]

    layout = {
        "paper_bgcolor": COLORS["card"],
        "plot_bgcolor": COLORS["card"],
        "font": {"color": COLORS["text"], "family": "'Noto Sans KR', sans-serif"},
        "margin": {"l": 50, "r": 20, "t": 10, "b": 50},
        "height": 280,
        "xaxis": {"gridcolor": COLORS["grid"]},
        "yaxis": {"gridcolor": COLORS["grid"], "title": "등락률 (%)", "zeroline": True, "zerolinecolor": COLORS["text_muted"]},
    }

    data_json = json.dumps(data, ensure_ascii=False)
    layout_json = json.dumps(layout, ensure_ascii=False)
    div_id = "chart-bearish"
    msg_cls = "msg-warn" if market_down else "msg-info"

    return f"""
    <div class="section">
        <div class="section-title">하락장 방어주</div>
        <div class="plot-container" id="{div_id}"></div>
        <script>Plotly.newPlot('{div_id}', {data_json}, {layout_json}, {{responsive: true, displayModeBar: false}});</script>
        <div class="msg-box {msg_cls}">{msg}</div>
    </div>"""


def _build_stop_loss_section(signals: dict) -> str:
    """손절 경고 섹션 (해당 종목 있을 때만 표시)."""
    alerts = signals.get("stop_loss", [])
    if not alerts:
        return ""

    rows = ""
    for a in alerts:
        rows += f"""
        <div class="card" style="border-color: {COLORS['red']}60;">
            <div class="card-label" style="color:{COLORS['red']};">손절 경고</div>
            <div class="card-value" style="color:{COLORS['red']};">{a['ticker']}</div>
            <div class="card-sub" style="color:{COLORS['red']};">{a['change_pct']:+.2f}%</div>
            <div class="card-sub">{a['message']}</div>
        </div>"""

    return f"""
    <div class="section">
        <div class="section-title" style="color:{COLORS['red']};">손절 경고</div>
        <div class="card-grid">{rows}</div>
    </div>"""


def _build_ticker_table(changes: dict) -> str:
    """전체 종목 테이블."""
    rows_html = ""
    for ticker in sorted(changes.keys()):
        info = changes[ticker]
        name = config.TICKERS.get(ticker, {}).get("name", "")
        close = info.get("close", 0.0)
        pct = info.get("change_pct", 0.0)
        pct_color = COLORS["green"] if pct > 0 else (COLORS["red"] if pct < 0 else COLORS["text_muted"])
        arrow = "&#9650;" if pct > 0 else ("&#9660;" if pct < 0 else "&#8212;")
        date = info.get("date", "")

        rows_html += f"""
            <tr>
                <td style="font-weight:500;">{ticker}</td>
                <td>{name}</td>
                <td>${close:,.2f}</td>
                <td style="color:{pct_color}; font-weight:500;">{arrow} {pct:+.2f}%</td>
                <td style="color:{COLORS['text_muted']};">{date}</td>
            </tr>"""

    headers = ["티커", "종목명", "종가", "등락률", "날짜"]
    th_html = "".join(
        f'<th onclick="sortTable(\'tickerTable\', {i})">{h} &#x25B4;&#x25BE;</th>'
        for i, h in enumerate(headers)
    )

    return f"""
    <div class="section">
        <div class="section-title">전체 종목 현황</div>
        <div style="overflow-x:auto;">
            <table class="data-table" id="tickerTable">
                <thead><tr>{th_html}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
    </div>"""


# ============================================================
# 메인 함수
# ============================================================
def generate_dashboard_html(
    data: pd.DataFrame,
    changes: dict,
    signals: dict,
) -> Path:
    """인터랙티브 HTML 대시보드 생성.

    Args:
        data: 3개월 일봉 DataFrame [Date, Open, High, Low, Close, Volume, ticker]
        changes: {ticker: {"close", "prev_close", "change_pct", "date"}}
        signals: generate_all_signals() 결과

    Returns:
        생성된 HTML 파일 경로
    """
    # 날짜 결정
    if changes:
        sample = next(iter(changes.values()))
        date_str = sample.get("date", datetime.now().strftime("%Y-%m-%d"))
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # 각 섹션 조립
    sections = [
        _build_header(signals, date_str),
        _build_stop_loss_section(signals),
        _build_twin_gap_chart(signals),
        _build_twin_trend_chart(data, signals),
        _build_conditional_section(signals, changes),
        _build_bearish_section(signals),
        _build_ticker_table(changes),
    ]

    body = "\n".join(s for s in sections if s)

    html = _html_template(
        title=f"PTJ 매매법 대시보드 — {date_str}",
        date_str=date_str,
        body_content=body,
    )

    # 파일 저장
    output_path = config.CHART_DIR / "ptj_dashboard.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"[HTML 대시보드] {output_path}")
    return output_path

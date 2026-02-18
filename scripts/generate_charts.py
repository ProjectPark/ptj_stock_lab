#!/usr/bin/env python3
"""PTJ 매매 규칙 분석 차트 생성"""

import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import json
import os


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'charts')
CSV_PATH = os.path.join(os.path.dirname(__file__), 'trade_analysis.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 데이터 로드 ──
df = pd.read_csv(CSV_PATH)
df['날짜'] = pd.to_datetime(df['날짜'], format='%Y.%m.%d')
df['월'] = df['날짜'].dt.to_period('M')

RULES = {
    'R1': 'R1_위반',
    'R2': 'R2_위반',
    'R3': 'R3_위반',
    'R4': 'R4_위반',
    'R5': 'R5_위반',
}

RULE_NAMES = {
    'R1': 'R1: GLD 추세',
    'R2': 'R2: 갭 필터',
    'R3': 'R3: 포트폴리오',
    'R4': 'R4: 종목 집중',
    'R5': 'R5: SPY/QQQ',
}


# ════════════════════════════════════════════
# Chart 1: 규칙별 준수율 막대 차트
# ════════════════════════════════════════════
def chart1_rule_compliance():
    fig, ax = plt.subplots(figsize=(10, 5))

    rules = list(RULES.keys())
    compliance = []
    for r in rules:
        col = RULES[r]
        total = df[col].notna().sum()
        if total == 0:
            compliance.append(100.0)
        else:
            violations = df[col].sum()
            compliance.append((1 - violations / total) * 100)

    colors = []
    for c in compliance:
        if c >= 80:
            colors.append('#2ecc71')
        elif c >= 60:
            colors.append('#f1c40f')
        else:
            colors.append('#e74c3c')

    labels = [RULE_NAMES[r] for r in rules]
    y_pos = range(len(rules))

    bars = ax.barh(y_pos, compliance, color=colors, height=0.6, edgecolor='white', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlim(0, 110)
    ax.set_xlabel('준수율 (%)', fontsize=12)
    ax.set_title('매매 규칙별 준수율', fontsize=16, fontweight='bold', pad=15)

    for bar, val in zip(bars, compliance):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=13, fontweight='bold')

    # 기준선
    ax.axvline(x=80, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.axvline(x=60, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)

    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'rule_compliance.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('✓ rule_compliance.png')
    return {r: c for r, c in zip(rules, compliance)}


# ════════════════════════════════════════════
# Chart 2: 월별 거래 활동 + 위반 추이
# ════════════════════════════════════════════
def chart2_monthly_activity():
    fig, ax1 = plt.subplots(figsize=(14, 6))

    months = sorted(df['월'].unique())
    month_labels = [str(m) for m in months]
    x = np.arange(len(months))

    buy_counts = []
    sell_counts = []
    violation_counts = []

    for m in months:
        mdf = df[df['월'] == m]
        buy_counts.append((mdf['구분'] == '구매').sum())
        sell_counts.append((mdf['구분'] == '판매').sum())
        violation_counts.append((mdf['위반_총수'] > 0).sum())

    width = 0.5
    ax1.bar(x, buy_counts, width, label='매수 건수', color='#3498db', alpha=0.85)
    ax1.bar(x, sell_counts, width, bottom=buy_counts, label='매도 건수', color='#e74c3c', alpha=0.85)
    ax1.set_ylabel('거래 건수', fontsize=12, color='#2c3e50')
    ax1.set_xlabel('월', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(month_labels, rotation=45, ha='right', fontsize=9)
    ax1.legend(loc='upper left', fontsize=10)

    ax2 = ax1.twinx()
    ax2.plot(x, violation_counts, color='#e67e22', marker='o', linewidth=2.5,
             markersize=7, label='위반 거래 건수', zorder=5)
    ax2.set_ylabel('위반 거래 건수', fontsize=12, color='#e67e22')
    ax2.legend(loc='upper right', fontsize=10)

    ax1.set_title('월별 거래 활동 및 규칙 위반 추이', fontsize=16, fontweight='bold', pad=15)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'monthly_activity.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('✓ monthly_activity.png')

    return {
        'months': month_labels,
        'buy': buy_counts,
        'sell': sell_counts,
        'violations': violation_counts,
    }


# ════════════════════════════════════════════
# Chart 3: 종목별 매수/매도 금액 분포
# ════════════════════════════════════════════
def chart3_stock_distribution():
    fig, ax = plt.subplots(figsize=(12, 7))

    buy_df = df[df['구분'] == '구매'].groupby('종목명')['거래대금_원'].sum()
    sell_df = df[df['구분'] == '판매'].groupby('종목명')['거래대금_원'].sum()

    all_stocks = set(buy_df.index) | set(sell_df.index)
    totals = {}
    for s in all_stocks:
        totals[s] = buy_df.get(s, 0) + sell_df.get(s, 0)

    top10 = sorted(totals, key=totals.get, reverse=True)[:10]

    buy_vals = [buy_df.get(s, 0) for s in top10]
    sell_vals = [sell_df.get(s, 0) for s in top10]

    y_pos = np.arange(len(top10))
    height = 0.35

    ax.barh(y_pos - height / 2, [v / 1e6 for v in buy_vals], height,
            label='매수', color='#3498db', alpha=0.85)
    ax.barh(y_pos + height / 2, [v / 1e6 for v in sell_vals], height,
            label='매도', color='#e74c3c', alpha=0.85)

    # 짧은 종목명 사용 (20자 제한)
    short_names = [s[:20] + '…' if len(s) > 20 else s for s in top10]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_names, fontsize=10)
    ax.set_xlabel('거래대금 (백만 원)', fontsize=12)
    ax.set_title('상위 10개 종목 매수/매도 금액 분포', fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=11)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'stock_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('✓ stock_distribution.png')

    return {
        'stocks': short_names,
        'buy': [v / 1e6 for v in buy_vals],
        'sell': [v / 1e6 for v in sell_vals],
    }


# ════════════════════════════════════════════
# Chart 4: 월별 × 규칙별 위반 히트맵
# ════════════════════════════════════════════
def chart4_violation_heatmap():
    months = sorted(df['월'].unique())
    month_labels = [str(m) for m in months]
    rule_labels = list(RULES.keys())

    matrix = []
    for m in months:
        mdf = df[df['월'] == m]
        row = []
        for r in rule_labels:
            col = RULES[r]
            row.append(mdf[col].sum() if mdf[col].notna().any() else 0)
        matrix.append(row)

    matrix = np.array(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(8, max(6, len(months) * 0.5)))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')

    ax.set_xticks(range(len(rule_labels)))
    ax.set_xticklabels([RULE_NAMES[r] for r in rule_labels], fontsize=10, rotation=30, ha='right')
    ax.set_yticks(range(len(month_labels)))
    ax.set_yticklabels(month_labels, fontsize=9)

    # 셀 값 표시
    for i in range(len(month_labels)):
        for j in range(len(rule_labels)):
            val = int(matrix[i, j])
            if val > 0:
                text_color = 'white' if matrix[i, j] > matrix.max() * 0.6 else 'black'
                ax.text(j, i, str(val), ha='center', va='center',
                        fontsize=9, color=text_color, fontweight='bold')

    ax.set_title('월별 규칙 위반 히트맵', fontsize=16, fontweight='bold', pad=15)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('위반 건수', fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'violation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('✓ violation_heatmap.png')

    return {
        'months': month_labels,
        'rules': rule_labels,
        'matrix': matrix.tolist(),
    }


# ════════════════════════════════════════════
# Chart 5: 누적 거래금액 타임라인
# ════════════════════════════════════════════
def chart5_cumulative_pnl():
    fig, ax = plt.subplots(figsize=(14, 6))

    sorted_df = df.sort_values('날짜')
    sorted_df['매수누적'] = sorted_df.apply(
        lambda r: r['거래대금_원'] if r['구분'] == '구매' else 0, axis=1
    ).cumsum()
    sorted_df['매도누적'] = sorted_df.apply(
        lambda r: r['거래대금_원'] if r['구분'] == '판매' else 0, axis=1
    ).cumsum()
    sorted_df['순차이'] = sorted_df['매도누적'] - sorted_df['매수누적']

    dates = sorted_df['날짜']
    ax.fill_between(dates, sorted_df['매수누적'] / 1e6, alpha=0.3, color='#3498db', label='매수 누적')
    ax.fill_between(dates, sorted_df['매도누적'] / 1e6, alpha=0.3, color='#e74c3c', label='매도 누적')
    ax.plot(dates, sorted_df['매수누적'] / 1e6, color='#3498db', linewidth=1.5)
    ax.plot(dates, sorted_df['매도누적'] / 1e6, color='#e74c3c', linewidth=1.5)
    ax.plot(dates, sorted_df['순차이'] / 1e6, color='#2ecc71', linewidth=2, linestyle='--', label='순 차이 (매도-매수)')

    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')
    ax.set_xlabel('날짜', fontsize=12)
    ax.set_ylabel('금액 (백만 원)', fontsize=12)
    ax.set_title('누적 매수/매도 거래금액 추이', fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'cumulative_pnl.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('✓ cumulative_pnl.png')

    return {
        'dates': [d.strftime('%Y-%m-%d') for d in dates],
        'buy_cum': (sorted_df['매수누적'] / 1e6).round(2).tolist(),
        'sell_cum': (sorted_df['매도누적'] / 1e6).round(2).tolist(),
        'net': (sorted_df['순차이'] / 1e6).round(2).tolist(),
    }


# ════════════════════════════════════════════
# Chart 6: Plotly.js HTML 대시보드
# ════════════════════════════════════════════
def chart6_html_dashboard(compliance_data, monthly_data, stock_data, heatmap_data, cumulative_data):
    # 요약 테이블 데이터
    total_trades = len(df)
    buy_trades = (df['구분'] == '구매').sum()
    sell_trades = (df['구분'] == '판매').sum()
    total_buy_amount = df[df['구분'] == '구매']['거래대금_원'].sum()
    total_sell_amount = df[df['구분'] == '판매']['거래대금_원'].sum()
    total_violations = (df['위반_총수'] > 0).sum()
    avg_violations = df['위반_총수'].mean()
    clean_trades = (df['위반_총수'] == 0).sum()
    clean_rate = clean_trades / total_trades * 100

    summary_json = json.dumps({
        'total_trades': int(total_trades),
        'buy_trades': int(buy_trades),
        'sell_trades': int(sell_trades),
        'total_buy_amount': round(float(total_buy_amount / 1e6), 1),
        'total_sell_amount': round(float(total_sell_amount / 1e6), 1),
        'total_violations': int(total_violations),
        'avg_violations': round(float(avg_violations), 2),
        'clean_trades': int(clean_trades),
        'clean_rate': round(float(clean_rate), 1),
    }, ensure_ascii=False, cls=NumpyEncoder)

    compliance_json = json.dumps(compliance_data, ensure_ascii=False, cls=NumpyEncoder)
    monthly_json = json.dumps(monthly_data, ensure_ascii=False, cls=NumpyEncoder)
    stock_json = json.dumps(stock_data, ensure_ascii=False, cls=NumpyEncoder)
    heatmap_json = json.dumps(heatmap_data, ensure_ascii=False, cls=NumpyEncoder)

    # cumulative_data can be large, sample every 5th point for HTML
    cum_sampled = {
        'dates': cumulative_data['dates'][::3],
        'buy_cum': cumulative_data['buy_cum'][::3],
        'sell_cum': cumulative_data['sell_cum'][::3],
        'net': cumulative_data['net'][::3],
    }
    cumulative_json = json.dumps(cum_sampled, ensure_ascii=False)

    rule_full_names_json = json.dumps(
        {k: v for k, v in RULE_NAMES.items()}, ensure_ascii=False
    )

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PTJ 매매 규칙 분석 대시보드</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #f5f6fa;
    color: #2c3e50;
    padding: 20px;
  }}
  h1 {{
    text-align: center;
    font-size: 28px;
    margin-bottom: 8px;
    color: #2c3e50;
  }}
  .subtitle {{
    text-align: center;
    color: #7f8c8d;
    margin-bottom: 24px;
    font-size: 14px;
  }}
  .grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    max-width: 1400px;
    margin: 0 auto;
  }}
  .card {{
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }}
  .card.full {{ grid-column: 1 / -1; }}
  .card h2 {{
    font-size: 16px;
    color: #34495e;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 2px solid #ecf0f1;
  }}
  .summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
  }}
  .stat-box {{
    background: #f8f9fa;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
  }}
  .stat-box .value {{
    font-size: 28px;
    font-weight: bold;
    color: #2c3e50;
  }}
  .stat-box .label {{
    font-size: 13px;
    color: #7f8c8d;
    margin-top: 4px;
  }}
  .stat-box.green .value {{ color: #27ae60; }}
  .stat-box.red .value {{ color: #e74c3c; }}
  .stat-box.blue .value {{ color: #3498db; }}
  .stat-box.orange .value {{ color: #e67e22; }}
  .chart-container {{ width: 100%; min-height: 350px; }}
  @media (max-width: 900px) {{
    .grid {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>

<h1>PTJ 매매 규칙 분석 대시보드</h1>
<p class="subtitle">분석 기간: {df['날짜'].min().strftime('%Y.%m.%d')} ~ {df['날짜'].max().strftime('%Y.%m.%d')} | 총 {total_trades}건</p>

<div class="grid">

  <!-- 요약 통계 -->
  <div class="card full">
    <h2>요약 통계</h2>
    <div class="summary-grid" id="summary-grid"></div>
  </div>

  <!-- 규칙별 준수율 -->
  <div class="card">
    <h2>규칙별 준수율</h2>
    <div id="chart-compliance" class="chart-container"></div>
  </div>

  <!-- 월별 거래활동 -->
  <div class="card">
    <h2>월별 거래 활동 및 위반 추이</h2>
    <div id="chart-monthly" class="chart-container"></div>
  </div>

  <!-- 종목별 분포 -->
  <div class="card">
    <h2>상위 10개 종목 거래금액</h2>
    <div id="chart-stocks" class="chart-container"></div>
  </div>

  <!-- 히트맵 -->
  <div class="card">
    <h2>월별 규칙 위반 히트맵</h2>
    <div id="chart-heatmap" class="chart-container"></div>
  </div>

  <!-- 누적 거래금액 -->
  <div class="card full">
    <h2>누적 매수/매도 거래금액 추이</h2>
    <div id="chart-cumulative" class="chart-container" style="min-height:400px;"></div>
  </div>

</div>

<script>
const compliance = {compliance_json};
const monthly = {monthly_json};
const stocks = {stock_json};
const heatmap = {heatmap_json};
const cumulative = {cumulative_json};
const summary = {summary_json};
const ruleNames = {rule_full_names_json};

const plotConfig = {{responsive: true, displayModeBar: false}};

// ─── 요약 통계 ───
(function() {{
  const grid = document.getElementById('summary-grid');
  const items = [
    {{label: '총 거래 건수', value: summary.total_trades.toLocaleString() + '건', cls: ''}},
    {{label: '매수 건수', value: summary.buy_trades.toLocaleString() + '건', cls: 'blue'}},
    {{label: '매도 건수', value: summary.sell_trades.toLocaleString() + '건', cls: 'red'}},
    {{label: '총 매수금액', value: summary.total_buy_amount.toLocaleString() + '백만원', cls: 'blue'}},
    {{label: '총 매도금액', value: summary.total_sell_amount.toLocaleString() + '백만원', cls: 'red'}},
    {{label: '위반 거래 수', value: summary.total_violations.toLocaleString() + '건', cls: 'orange'}},
    {{label: '평균 위반 수', value: summary.avg_violations + '개/건', cls: 'orange'}},
    {{label: '무위반 준수율', value: summary.clean_rate + '%', cls: 'green'}},
  ];
  grid.innerHTML = items.map(i =>
    `<div class="stat-box ${{i.cls}}"><div class="value">${{i.value}}</div><div class="label">${{i.label}}</div></div>`
  ).join('');
}})();

// ─── 규칙별 준수율 ───
(function() {{
  const rules = Object.keys(compliance);
  const vals = rules.map(r => compliance[r]);
  const colors = vals.map(v => v >= 80 ? '#2ecc71' : v >= 60 ? '#f1c40f' : '#e74c3c');
  const labels = rules.map(r => ruleNames[r] || r);

  Plotly.newPlot('chart-compliance', [{{
    type: 'bar',
    orientation: 'h',
    y: labels.reverse(),
    x: vals.reverse(),
    marker: {{ color: colors.reverse() }},
    text: vals.map(v => v.toFixed(1) + '%'),
    textposition: 'outside',
    hovertemplate: '%{{y}}: %{{x:.1f}}%<extra></extra>',
  }}], {{
    xaxis: {{ title: '준수율 (%)', range: [0, 110] }},
    margin: {{ l: 140, r: 40, t: 10, b: 40 }},
    height: 300,
    shapes: [
      {{ type: 'line', x0: 80, x1: 80, y0: -0.5, y1: 4.5, line: {{ color: 'gray', dash: 'dash', width: 1 }} }},
    ]
  }}, plotConfig);
}})();

// ─── 월별 거래활동 ───
(function() {{
  Plotly.newPlot('chart-monthly', [
    {{
      type: 'bar',
      name: '매수',
      x: monthly.months,
      y: monthly.buy,
      marker: {{ color: '#3498db' }},
    }},
    {{
      type: 'bar',
      name: '매도',
      x: monthly.months,
      y: monthly.sell,
      marker: {{ color: '#e74c3c' }},
    }},
    {{
      type: 'scatter',
      name: '위반 건수',
      x: monthly.months,
      y: monthly.violations,
      yaxis: 'y2',
      mode: 'lines+markers',
      line: {{ color: '#e67e22', width: 2.5 }},
      marker: {{ size: 7 }},
    }},
  ], {{
    barmode: 'stack',
    yaxis: {{ title: '거래 건수' }},
    yaxis2: {{ title: '위반 건수', overlaying: 'y', side: 'right', showgrid: false }},
    margin: {{ l: 60, r: 60, t: 10, b: 60 }},
    height: 350,
    legend: {{ orientation: 'h', y: 1.12 }},
    xaxis: {{ tickangle: -45 }},
  }}, plotConfig);
}})();

// ─── 종목별 분포 ───
(function() {{
  const s = stocks.stocks.slice().reverse();
  const b = stocks.buy.slice().reverse();
  const sl = stocks.sell.slice().reverse();

  Plotly.newPlot('chart-stocks', [
    {{
      type: 'bar',
      orientation: 'h',
      name: '매수',
      y: s,
      x: b,
      marker: {{ color: '#3498db' }},
    }},
    {{
      type: 'bar',
      orientation: 'h',
      name: '매도',
      y: s,
      x: sl,
      marker: {{ color: '#e74c3c' }},
    }},
  ], {{
    barmode: 'group',
    xaxis: {{ title: '거래대금 (백만 원)' }},
    margin: {{ l: 180, r: 20, t: 10, b: 40 }},
    height: 400,
    legend: {{ orientation: 'h', y: 1.08 }},
  }}, plotConfig);
}})();

// ─── 히트맵 ───
(function() {{
  const rLabels = heatmap.rules.map(r => ruleNames[r] || r);

  Plotly.newPlot('chart-heatmap', [{{
    type: 'heatmap',
    z: heatmap.matrix,
    x: rLabels,
    y: heatmap.months,
    colorscale: 'YlOrRd',
    hovertemplate: '%{{y}} %{{x}}: %{{z}}건<extra></extra>',
    text: heatmap.matrix.map(row => row.map(v => v > 0 ? String(Math.round(v)) : '')),
    texttemplate: '%{{text}}',
    textfont: {{ size: 11 }},
  }}], {{
    margin: {{ l: 80, r: 20, t: 10, b: 80 }},
    height: Math.max(350, heatmap.months.length * 30),
    xaxis: {{ tickangle: -30 }},
  }}, plotConfig);
}})();

// ─── 누적 거래금액 ───
(function() {{
  Plotly.newPlot('chart-cumulative', [
    {{
      type: 'scatter',
      name: '매수 누적',
      x: cumulative.dates,
      y: cumulative.buy_cum,
      fill: 'tozeroy',
      fillcolor: 'rgba(52,152,219,0.15)',
      line: {{ color: '#3498db', width: 1.5 }},
    }},
    {{
      type: 'scatter',
      name: '매도 누적',
      x: cumulative.dates,
      y: cumulative.sell_cum,
      fill: 'tozeroy',
      fillcolor: 'rgba(231,76,60,0.15)',
      line: {{ color: '#e74c3c', width: 1.5 }},
    }},
    {{
      type: 'scatter',
      name: '순 차이 (매도-매수)',
      x: cumulative.dates,
      y: cumulative.net,
      line: {{ color: '#2ecc71', width: 2, dash: 'dash' }},
    }},
  ], {{
    xaxis: {{ title: '날짜' }},
    yaxis: {{ title: '금액 (백만 원)', tickformat: ',.0f' }},
    margin: {{ l: 80, r: 40, t: 10, b: 50 }},
    height: 400,
    legend: {{ orientation: 'h', y: 1.08 }},
    hovermode: 'x unified',
  }}, plotConfig);
}})();
</script>

</body>
</html>"""

    path = os.path.join(OUTPUT_DIR, 'trade_review_dashboard.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print('✓ trade_review_dashboard.html')


# ════════════════════════════════════════════
# Main
# ════════════════════════════════════════════
if __name__ == '__main__':
    print(f'데이터: {len(df)}건 ({df["날짜"].min().strftime("%Y.%m.%d")} ~ {df["날짜"].max().strftime("%Y.%m.%d")})')
    print(f'출력: {OUTPUT_DIR}\n')

    compliance_data = chart1_rule_compliance()
    monthly_data = chart2_monthly_activity()
    stock_data = chart3_stock_distribution()
    heatmap_data = chart4_violation_heatmap()
    cumulative_data = chart5_cumulative_pnl()
    chart6_html_dashboard(compliance_data, monthly_data, stock_data, heatmap_data, cumulative_data)

    print(f'\n모든 차트 생성 완료!')

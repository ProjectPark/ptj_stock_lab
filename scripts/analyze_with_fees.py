"""
5분봉 백테스트 + KIS 수수료 반영 분석
- 표준 수수료 (0.25%) vs 실제 적용 수수료 (거래내역서 기준)
- 두 시나리오를 비교하여 docs/5min_backtest_fees_report.md 생성
"""
import pandas as pd
import numpy as np
from pathlib import Path
import math
import json

BASE = Path(__file__).parent
DOCS = BASE.parent / "docs"

# KIS 수수료 상수
KIS_SEC_FEE_PCT = 0.00278  # SEC Fee (매도 시에만)


def net_return_series(gross_pct, buy_fee, sell_fee):
    """수수료 적용 수익률 (pandas Series or scalar, %)"""
    return ((1 + gross_pct / 100) * (1 - sell_fee) / (1 + buy_fee) - 1) * 100


def approx_win_rate(mean, std, threshold):
    """정규분포 가정 양전 확률 근사"""
    if std <= 0:
        return 0.0
    z = (mean - threshold) / std
    t = 1 / (1 + 0.2316419 * abs(z))
    d = 0.3989422804014327 * math.exp(-z * z / 2)
    p = d * t * (0.319381530 + t * (-0.356563782 + t * (
        1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    return (1 - p if z >= 0 else p) * 100


# ============================================================
# 1. 실제 거래내역 수수료 분석
# ============================================================
print("=" * 60)
print("1. 실제 거래내역 수수료 분석")
print("=" * 60)

trades_csv = BASE.parent / "history" / "거래내역서_20250218_20260217_1.csv"
df_trades = pd.read_csv(trades_csv)

for col in ['거래대금_달러', '수수료_달러', '제세금_달러', '거래대금_원', '수수료_원', '제세금_원']:
    df_trades[col] = pd.to_numeric(
        df_trades[col].astype(str).str.replace(',', ''), errors='coerce'
    ).fillna(0)

buys = df_trades[df_trades['거래구분'] == '구매']
sells = df_trades[df_trades['거래구분'] == '판매']

# 수수료율 계산 (0이 아닌 거래만)
nonzero_buy = buys[buys['수수료_달러'] > 0]
nonzero_sell = sells[sells['수수료_달러'] > 0]

total_buy_amount = buys['거래대금_달러'].sum()
total_sell_amount = sells['거래대금_달러'].sum()
total_buy_fee = buys['수수료_달러'].sum()
total_sell_fee = sells['수수료_달러'].sum()
total_sell_tax = sells['제세금_달러'].sum()
total_fees_actual = total_buy_fee + total_sell_fee + total_sell_tax

# 실효 수수료율
eff_buy_rate = nonzero_buy['수수료_달러'].sum() / nonzero_buy['거래대금_달러'].sum() * 100
eff_sell_rate = nonzero_sell['수수료_달러'].sum() / nonzero_sell['거래대금_달러'].sum() * 100
eff_tax_rate = total_sell_tax / total_sell_amount * 100 if total_sell_amount > 0 else 0

# 소액 면제 분석
zero_fee_buys = len(buys[buys['수수료_달러'] == 0])
zero_fee_sells = len(sells[sells['수수료_달러'] == 0])

# 소액 면제 기준 추정
if len(nonzero_buy) > 0:
    min_fee_amount = nonzero_buy['거래대금_달러'].min()
    max_nofee_amount = buys[buys['수수료_달러'] == 0]['거래대금_달러'].max()

print(f"\n거래 건수: 매수 {len(buys)}건 / 매도 {len(sells)}건")
print(f"총 매수 금액: ${total_buy_amount:,.2f}")
print(f"총 매도 금액: ${total_sell_amount:,.2f}")
print(f"\n매수 수수료: ${total_buy_fee:,.2f}")
print(f"매도 수수료: ${total_sell_fee:,.2f}")
print(f"매도 제세금(SEC): ${total_sell_tax:,.2f}")
print(f"총 비용: ${total_fees_actual:,.2f}")
print(f"\n실효 수수료율:")
print(f"  매수: {eff_buy_rate:.4f}%")
print(f"  매도: {eff_sell_rate:.4f}%")
print(f"  SEC Fee: {eff_tax_rate:.5f}%")
print(f"  왕복 실효: {eff_buy_rate + eff_sell_rate + eff_tax_rate:.4f}%")
print(f"\n수수료 면제: 매수 {zero_fee_buys}건 / 매도 {zero_fee_sells}건")

ACTUAL_COMM = eff_buy_rate  # ~0.10%
ACTUAL_BUY_FEE = ACTUAL_COMM / 100
ACTUAL_SELL_FEE = (ACTUAL_COMM + eff_tax_rate) / 100
ACTUAL_RT = (ACTUAL_BUY_FEE + ACTUAL_SELL_FEE) * 100

# 표준 수수료
STD_COMM = 0.25
STD_FX = 0.10
STD_BUY_FEE = (STD_COMM + STD_FX) / 100    # 0.0035
STD_SELL_FEE = (STD_COMM + KIS_SEC_FEE_PCT + STD_FX) / 100  # 0.003528
STD_RT = (STD_BUY_FEE + STD_SELL_FEE) * 100  # ~0.7028%

print(f"\n수수료 시나리오 비교:")
print(f"  표준 (0.25%+FX): 왕복 {STD_RT:.4f}%")
print(f"  실제 적용:       왕복 {ACTUAL_RT:.4f}%")
print(f"  차이:           {STD_RT - ACTUAL_RT:.4f}%p")

# ============================================================
# 2. 갭 시뮬레이션 수수료 반영 (표준 + 실제)
# ============================================================
print("\n" + "=" * 60)
print("2. 쌍둥이 갭 시뮬레이션 + 수수료")
print("=" * 60)

df_gap = pd.read_csv(BASE / "backtest_twin_gap.csv")

df_gap['return_gross'] = df_gap['return_pct']
df_gap['return_std'] = net_return_series(df_gap['return_gross'], STD_BUY_FEE, STD_SELL_FEE)
df_gap['return_actual'] = net_return_series(df_gap['return_gross'], ACTUAL_BUY_FEE, ACTUAL_SELL_FEE)

TRADE_AMT = 200  # $200/건

print(f"\n{'페어':>6}  {'수수료없음':^18}  {'표준(0.25%+FX)':^18}  {'실제({ACTUAL_COMM:.2f}%)':^18}")
print(f"{'':>6}  {'평균':>7} {'승률':>5} {'누적':>6}  {'평균':>7} {'승률':>5} {'누적':>6}  {'평균':>7} {'승률':>5} {'누적':>6}")
print("-" * 80)

gap_results = {}
for pair in ['coin', 'bank', 'semi', '전체']:
    sub = df_gap if pair == '전체' else df_gap[df_gap['pair'] == pair]
    n = len(sub)

    g_mean = sub['return_gross'].mean()
    g_wr = (sub['return_gross'] > 0).mean() * 100
    g_cum = sub['return_gross'].sum()

    s_mean = sub['return_std'].mean()
    s_wr = (sub['return_std'] > 0).mean() * 100
    s_cum = sub['return_std'].sum()

    a_mean = sub['return_actual'].mean()
    a_wr = (sub['return_actual'] > 0).mean() * 100
    a_cum = sub['return_actual'].sum()

    gap_results[pair] = {
        'count': n,
        'gross': {'mean': g_mean, 'wr': g_wr, 'cum': g_cum},
        'std': {'mean': s_mean, 'wr': s_wr, 'cum': s_cum, 'fee': n * TRADE_AMT * STD_RT / 100},
        'actual': {'mean': a_mean, 'wr': a_wr, 'cum': a_cum, 'fee': n * TRADE_AMT * ACTUAL_RT / 100},
    }

    print(f"{pair:>6}  {g_mean:>+6.2f}% {g_wr:>4.1f}% {g_cum:>+5.0f}%  "
          f"{s_mean:>+6.2f}% {s_wr:>4.1f}% {s_cum:>+5.0f}%  "
          f"{a_mean:>+6.2f}% {a_wr:>4.1f}% {a_cum:>+5.0f}%")

# 매도 사유별
print(f"\n매도 사유별:")
exit_results = {}
for reason in ['stop_loss', 'eod_close', 'converge']:
    sub = df_gap[df_gap['exit_reason'] == reason]
    exit_results[reason] = {
        'count': len(sub),
        'gross_mean': sub['return_gross'].mean(),
        'std_mean': sub['return_std'].mean(),
        'actual_mean': sub['return_actual'].mean(),
        'std_wr': (sub['return_std'] > 0).mean() * 100,
        'actual_wr': (sub['return_actual'] > 0).mean() * 100,
    }
    e = exit_results[reason]
    print(f"  {reason:>12}: 무 {e['gross_mean']:>+6.3f}% | 표준 {e['std_mean']:>+6.3f}% | "
          f"실제 {e['actual_mean']:>+6.3f}% ({len(sub)}건)")

# ============================================================
# 3. 보유시간 수수료 반영
# ============================================================
print("\n" + "=" * 60)
print("3. 보유시간 분석 + 수수료")
print("=" * 60)

df_hold = pd.read_csv(BASE / "backtest_hold_time_analysis.csv")
hold_all = df_hold[df_hold['symbol'] == 'ALL'].iloc[0]

periods = ['30min', '1h', '2h', '3h', '5h', 'close']
print(f"\n{'보유':>6}  {'무수수료':>8}  {'표준':>8}  {'실제':>8}  {'양전(무)':>7} {'양전(표)':>7} {'양전(실)':>7}")
print("-" * 70)

hold_results = {}
for p in periods:
    g_mean = hold_all[f'{p}_mean']
    g_std = hold_all[f'{p}_std']
    g_wr = hold_all[f'{p}_win_rate']

    s_mean = net_return_series(g_mean, STD_BUY_FEE, STD_SELL_FEE)
    a_mean = net_return_series(g_mean, ACTUAL_BUY_FEE, ACTUAL_SELL_FEE)

    s_wr = approx_win_rate(g_mean, g_std, STD_RT)
    a_wr = approx_win_rate(g_mean, g_std, ACTUAL_RT)

    hold_results[p] = {
        'gross_mean': g_mean, 'std_mean': s_mean, 'actual_mean': a_mean,
        'gross_wr': g_wr, 'std_wr': s_wr, 'actual_wr': a_wr,
    }

    print(f"{p:>6}  {g_mean:>+7.3f}%  {s_mean:>+7.3f}%  {a_mean:>+7.3f}%  "
          f"{g_wr:>6.1f}% {s_wr:>6.1f}% {a_wr:>6.1f}%")

# 5시간 규칙 (수수료 차이는 매도 1회분이므로 5h vs close 비교는 동일)
print(f"\n5시간 vs 장마감 (수수료 포함 — 차이 불변, 수수료는 동일하게 적용됨)")
h5_results = {}
for _, row in df_hold.iterrows():
    if row['symbol'] == 'ALL':
        continue
    sym = row['symbol']
    h5 = net_return_series(row['5h_mean'], ACTUAL_BUY_FEE, ACTUAL_SELL_FEE)
    cl = net_return_series(row['close_mean'], ACTUAL_BUY_FEE, ACTUAL_SELL_FEE)
    diff = h5 - cl
    h5_results[sym] = {'h5': h5, 'close': cl, 'diff': diff}

# ============================================================
# 4. 실제 매수 건 수수료 반영
# ============================================================
print("\n" + "=" * 60)
print("4. 실제 매수 건 + 수수료")
print("=" * 60)

df_actual = pd.read_csv(BASE / "backtest_actual_trade_tracking.csv")
ret_cols = ['ret_30min', 'ret_1h', 'ret_2h', 'ret_3h', 'ret_5h', 'ret_close']
for col in ret_cols:
    df_actual[col] = pd.to_numeric(df_actual[col], errors='coerce')

print(f"\n{'보유':>6}  {'무수수료':>8}  {'표준':>8}  {'실제':>8}  {'양전(실)':>7}")
print("-" * 50)

actual_results = {}
for col in ret_cols:
    valid = df_actual[col].dropna()
    g_mean = valid.mean()
    g_med = valid.median()

    s_series = net_return_series(valid, STD_BUY_FEE, STD_SELL_FEE)
    a_series = net_return_series(valid, ACTUAL_BUY_FEE, ACTUAL_SELL_FEE)

    period = col.replace('ret_', '')
    actual_results[period] = {
        'count': len(valid),
        'gross_mean': g_mean, 'gross_median': g_med,
        'std_mean': s_series.mean(), 'std_median': s_series.median(),
        'actual_mean': a_series.mean(), 'actual_median': a_series.median(),
        'std_wr': (s_series > 0).mean() * 100,
        'actual_wr': (a_series > 0).mean() * 100,
    }
    a = actual_results[period]
    print(f"{period:>6}  {g_mean:>+7.2f}%  {a['std_mean']:>+7.2f}%  {a['actual_mean']:>+7.2f}%  "
          f"{a['actual_wr']:>6.1f}%")

# 종목별 (실제 수수료 기준)
print(f"\n종목별 (실제 수수료, 장마감 기준):")
ticker_results = {}
for ticker, grp in df_actual.groupby('ticker'):
    valid = grp['ret_close'].dropna()
    if len(valid) < 5:
        continue
    g_mean = valid.mean()
    a_series = net_return_series(valid, ACTUAL_BUY_FEE, ACTUAL_SELL_FEE)
    s_series = net_return_series(valid, STD_BUY_FEE, STD_SELL_FEE)
    ticker_results[ticker] = {
        'count': len(valid),
        'gross_mean': g_mean,
        'std_mean': s_series.mean(), 'std_wr': (s_series > 0).mean() * 100,
        'actual_mean': a_series.mean(), 'actual_wr': (a_series > 0).mean() * 100,
    }

# sort by actual_mean desc
for ticker in sorted(ticker_results, key=lambda t: ticker_results[t]['actual_mean'], reverse=True):
    t = ticker_results[ticker]
    print(f"  {ticker:>8} {t['count']:>4}건  무 {t['gross_mean']:>+7.2f}%  "
          f"표준 {t['std_mean']:>+7.2f}%  실제 {t['actual_mean']:>+7.2f}%  "
          f"승률 {t['actual_wr']:.1f}%")

# ============================================================
# 5. 손익분기 분석
# ============================================================
print("\n" + "=" * 60)
print("5. 손익분기 & 연간 수수료 추산")
print("=" * 60)

# 연간 거래 횟수 추산 (실제 거래내역 기준)
trading_days_actual = df_trades['거래일자'].nunique()
trades_per_day = len(df_trades) / trading_days_actual
print(f"\n실제 거래 패턴:")
print(f"  거래일수: {trading_days_actual}일")
print(f"  일평균 거래: {trades_per_day:.1f}건 (매수+매도)")
print(f"  총 거래대금: ${total_buy_amount + total_sell_amount:,.2f}")
print(f"  총 수수료+세금: ${total_fees_actual:,.2f}")

# 시뮬레이션 기준
sim_trades = len(df_gap)
print(f"\n갭 시뮬레이션 기준 (255거래일):")
print(f"  총 거래: {sim_trades}건")

for label, rt, bf, sf in [
    ("표준(0.25%+FX)", STD_RT, STD_BUY_FEE, STD_SELL_FEE),
    (f"실제({ACTUAL_COMM:.2f}%)", ACTUAL_RT, ACTUAL_BUY_FEE, ACTUAL_SELL_FEE),
]:
    annual_fee = sim_trades * TRADE_AMT * rt / 100
    daily_fee = annual_fee / 255
    break_even = rt
    print(f"\n  [{label}]")
    print(f"    왕복 수수료: {rt:.4f}%")
    print(f"    건당($200): ${TRADE_AMT * rt / 100:.2f}")
    print(f"    연간 추산: ${annual_fee:,.2f}")
    print(f"    일평균: ${daily_fee:.2f}")
    print(f"    손익분기 수익률: {break_even:.4f}%")

# ============================================================
# 6. 저장
# ============================================================
df_gap.to_csv(BASE / "backtest_twin_gap_with_fees.csv", index=False)

# 분석 결과를 JSON으로 저장 (리포트 생성용)
report_data = {
    'actual_fees': {
        'total_buy_amount': total_buy_amount,
        'total_sell_amount': total_sell_amount,
        'total_fees': total_fees_actual,
        'eff_buy_rate': eff_buy_rate,
        'eff_sell_rate': eff_sell_rate,
        'eff_tax_rate': eff_tax_rate,
        'zero_fee_buys': zero_fee_buys,
        'zero_fee_sells': zero_fee_sells,
        'buy_count': len(buys),
        'sell_count': len(sells),
    },
    'fee_scenarios': {
        'std': {'label': '표준(0.25%+FX)', 'rt': STD_RT, 'buy': STD_BUY_FEE * 100, 'sell': STD_SELL_FEE * 100},
        'actual': {'label': f'실제({ACTUAL_COMM:.2f}%)', 'rt': ACTUAL_RT, 'buy': ACTUAL_BUY_FEE * 100, 'sell': ACTUAL_SELL_FEE * 100},
    },
    'gap_results': gap_results,
    'exit_results': exit_results,
    'hold_results': hold_results,
    'actual_results': actual_results,
    'ticker_results': ticker_results,
    'h5_results': h5_results,
}

with open(BASE / "fee_analysis_data.json", 'w') as f:
    json.dump(report_data, f, indent=2, default=str, ensure_ascii=False)

print(f"\n저장 완료:")
print(f"  - backtest_twin_gap_with_fees.csv")
print(f"  - fee_analysis_data.json")
print("\n완료!")

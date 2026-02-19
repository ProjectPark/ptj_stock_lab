"""5분봉 시그널 타임라인 생성 스크립트"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import numpy as np
import config

# ── 데이터 로드 ──
df = pd.read_parquet(config.OHLCV_DIR / "backtest_5min.parquet")
df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

# ── 당일 시가 계산 ──
# 각 종목-날짜별 첫 봉(09:30)의 open을 당일 시가로 사용
day_open = df.groupby(["symbol", "date"])["open"].first().rename("day_open")
df = df.merge(day_open, on=["symbol", "date"], how="left")
df["pct"] = (df["close"] - df["day_open"]) / df["day_open"] * 100

# ── 종목별 피벗 (timestamp × symbol → pct) ──
pct_pivot = df.pivot_table(index=["date", "timestamp"], columns="symbol", values="pct")
pct_pivot = pct_pivot.sort_index()

# ── 결과 DataFrame 구성 ──
result = pd.DataFrame(index=pct_pivot.index)
result = result.reset_index()

# ── R1: 금 시황 ──
result["R1_GLD_pct"] = pct_pivot["GLD"].values
result["R1_warning"] = result["R1_GLD_pct"] > 0

# R1 전환 시점: 양전→음전 또는 음전→양전
gld_positive = result["R1_GLD_pct"] > 0
# 날짜 경계에서 전환 감지 방지를 위해 날짜별로 처리
result["R1_transition"] = ""
for date_val, grp in result.groupby("date"):
    idx = grp.index
    pos = gld_positive.loc[idx]
    shift = pos.shift(1)
    to_neg = pos.eq(False) & shift.eq(True)
    to_pos = pos.eq(True) & shift.eq(False)
    labels = pd.Series("", index=idx)
    labels[to_neg] = "양전→음전"
    labels[to_pos] = "음전→양전"
    result.loc[idx, "R1_transition"] = labels

# ── R2: 쌍둥이 갭 ──
pairs = {
    "coin": ("BITU", "MSTU"),
    "bank": ("ROBN", "CONL"),
    "semi": ("NVDL", "AMDL"),
}

for name, (lead, follow) in pairs.items():
    gap = pct_pivot[lead].values - pct_pivot[follow].values
    result[f"R2_{name}_gap"] = gap
    signal = np.where(np.abs(gap) >= 1.5, "ENTRY",
             np.where(np.abs(gap) <= 0.3, "SELL", ""))
    result[f"R2_{name}_signal"] = signal

# ── R3: 조건부 매매 ──
result["R3_ETHU_pct"] = pct_pivot["ETHU"].values
result["R3_XXRP_pct"] = pct_pivot["XXRP"].values
result["R3_SOLT_pct"] = pct_pivot["SOLT"].values
result["R3_all_positive"] = (
    (result["R3_ETHU_pct"] > 0) &
    (result["R3_XXRP_pct"] > 0) &
    (result["R3_SOLT_pct"] > 0)
)

# R3 COIN 시그널: 조건 충족 → BUY, 하나라도 음전 전환 → OFF
result["R3_COIN_signal"] = ""
for date_val, grp in result.groupby("date"):
    idx = grp.index
    all_pos = result.loc[idx, "R3_all_positive"]
    prev_pos = all_pos.shift(1).fillna(False)
    labels = pd.Series("", index=idx)
    # 조건 충족 구간
    labels[all_pos] = "BUY"
    # 음전 전환 시점 (이전에 True였는데 지금 False)
    labels[prev_pos & ~all_pos] = "OFF"
    result.loc[idx, "R3_COIN_signal"] = labels

# ── R4: 손절 ──
stoploss_symbols = [s for s in pct_pivot.columns if s not in ("GLD", "SPY", "QQQ")]
r4_alerts_list = []
for i, row in pct_pivot.iterrows():
    alerts = []
    for sym in stoploss_symbols:
        if pd.notna(row.get(sym)) and row[sym] <= -3.0:
            alerts.append(f"{sym}({row[sym]:.1f}%)")
    r4_alerts_list.append(",".join(alerts))
result["R4_alerts"] = r4_alerts_list

# ── R5: 하락장 ──
result["R5_SPY_pct"] = pct_pivot["SPY"].values
result["R5_QQQ_pct"] = pct_pivot["QQQ"].values
result["R5_market_down"] = (result["R5_SPY_pct"] < 0) & (result["R5_QQQ_pct"] < 0)
result["R5_gold_up"] = result["R5_market_down"] & (result["R1_GLD_pct"] > 0)

# ── CSV 저장 ──
output_path = config.RESULTS_DIR / "backtests" / "backtest_signals_5min.csv"
result.to_csv(output_path, index=False)
print(f"저장 완료: {output_path} ({len(result):,} rows)")

# ── 일별 요약 ──
print("\n" + "=" * 70)
print("일별 요약 통계")
print("=" * 70)

daily_summary = []
for date_val, grp in result.groupby("date"):
    n = len(grp)
    r1_warn_ratio = grp["R1_warning"].sum() / n * 100

    r2_entries = {}
    for name in ["coin", "bank", "semi"]:
        entry_count = (grp[f"R2_{name}_signal"] == "ENTRY").sum()
        r2_entries[name] = entry_count

    r3_ratio = grp["R3_all_positive"].sum() / n * 100

    r4_count = (grp["R4_alerts"] != "").sum()
    r5_down_ratio = grp["R5_market_down"].sum() / n * 100

    daily_summary.append({
        "date": date_val,
        "bars": n,
        "R1_warn%": r1_warn_ratio,
        "R2_coin_entry": r2_entries["coin"],
        "R2_bank_entry": r2_entries["bank"],
        "R2_semi_entry": r2_entries["semi"],
        "R3_cond%": r3_ratio,
        "R4_alerts": r4_count,
        "R5_down%": r5_down_ratio,
    })

summary_df = pd.DataFrame(daily_summary)
print(summary_df.to_string(index=False))

# ── 전체 요약 ──
print("\n" + "=" * 70)
print("전체 기간 요약")
print("=" * 70)
print(f"기간: {result['date'].min()} ~ {result['date'].max()}")
print(f"총 거래일: {result['date'].nunique()}일")
print(f"총 5분봉: {len(result):,}개")
print(f"\nR1 GLD 양전(warning) 비율: {result['R1_warning'].mean()*100:.1f}%")
print(f"R1 전환 횟수: {(result['R1_transition'] != '').sum()}")

for name in ["coin", "bank", "semi"]:
    entry_n = (result[f"R2_{name}_signal"] == "ENTRY").sum()
    sell_n = (result[f"R2_{name}_signal"] == "SELL").sum()
    print(f"R2 {name}: ENTRY {entry_n:,}건, SELL {sell_n:,}건")

print(f"\nR3 조건 충족 비율: {result['R3_all_positive'].mean()*100:.1f}%")
print(f"R3 COIN BUY 시그널: {(result['R3_COIN_signal']=='BUY').sum():,}건")

r4_total = (result["R4_alerts"] != "").sum()
print(f"\nR4 손절 알림 발생 봉 수: {r4_total:,}건")

print(f"\nR5 하락장 비율: {result['R5_market_down'].mean()*100:.1f}%")
print(f"R5 금상승+하락장 비율: {result['R5_gold_up'].mean()*100:.1f}%")

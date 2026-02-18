"""
PTJ 매매법 실시간 대시보드
===========================
실행: streamlit run ptj/app.py
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BEARISH_TICKERS,
    CONDITIONAL_TARGET,
    CONDITIONAL_TRIGGERS,
    GOLD_TICKER,
    PAIR_GAP_ENTRY_THRESHOLD,
    PAIR_GAP_SELL_THRESHOLD,
    REFRESH_INTERVAL_SEC,
    STOP_LOSS_PCT,
    TICKERS,
    TWIN_PAIRS,
)
from fetcher import fetch_intraday, get_current_snapshot, get_intraday_pct_series

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="PTJ 매매 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# 세션 상태
# ============================================================
if "prices" not in st.session_state:
    st.session_state.prices = {}
if "intraday_df" not in st.session_state:
    st.session_state.intraday_df = pd.DataFrame()
if "last_update" not in st.session_state:
    st.session_state.last_update = None


# ============================================================
# 헬퍼
# ============================================================
def fmt_pct(val: float) -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.2f}%"


def get(symbol: str) -> dict:
    return st.session_state.prices.get(symbol, {})


def price_str(d: dict) -> str:
    p = d.get("price", 0)
    return f"${p:,.2f}" if p else "—"


# ============================================================
# 데이터 자동 갱신
# ============================================================
@st.fragment(run_every=f"{REFRESH_INTERVAL_SEC}s")
def live_fetcher():
    try:
        intra = fetch_intraday()
        snap = get_current_snapshot(intra)
        if snap:
            st.session_state.prices = snap
            st.session_state.intraday_df = intra
            st.session_state.last_update = datetime.now()
    except Exception as e:
        st.error(f"시세 조회 실패: {e}")


# ============================================================
# 상단: 상태 바
# ============================================================
def render_status_bar():
    gold_pct = get(GOLD_TICKER).get("change_pct", 0)
    spy_pct = get("SPY").get("change_pct", 0)
    qqq_pct = get("QQQ").get("change_pct", 0)
    ts = st.session_state.last_update

    c1, c2, c3, c4, c5 = st.columns([1.5, 1, 1, 1, 1])

    with c1:
        if gold_pct > 0:
            st.error("🔴 **매매금지**")
        else:
            st.success("🟢 **매매가능**")
    with c2:
        st.metric("GLD", price_str(get(GOLD_TICKER)), fmt_pct(gold_pct))
    with c3:
        st.metric("SPY", price_str(get("SPY")), fmt_pct(spy_pct))
    with c4:
        st.metric("QQQ", price_str(get("QQQ")), fmt_pct(qqq_pct))
    with c5:
        if ts:
            st.metric("갱신", ts.strftime("%H:%M:%S"), f"{REFRESH_INTERVAL_SEC}초 주기")


# ============================================================
# 전체 요약 탭 — 쌍둥이 페어 카드
# ============================================================
def render_pair_card(pair: dict):
    """하나의 쌍둥이 페어를 컨테이너로 표시"""
    lead_sym, follow_sym = pair["lead"], pair["follow"]
    lead_d, follow_d = get(lead_sym), get(follow_sym)
    lead_pct = lead_d.get("change_pct", 0)
    follow_pct = follow_d.get("change_pct", 0)
    gap = abs(lead_pct - follow_pct)

    if gap <= PAIR_GAP_SELL_THRESHOLD:
        gap_label = "🟢 매도"
    elif gap >= PAIR_GAP_ENTRY_THRESHOLD:
        gap_label = "🟡 진입검토"
    else:
        gap_label = "⚪ 대기"

    st.caption(pair["label"])

    m1, m2 = st.columns(2)
    with m1:
        st.metric(lead_sym, price_str(lead_d), fmt_pct(lead_pct))
    with m2:
        st.metric(follow_sym, price_str(follow_d), fmt_pct(follow_pct))

    st.metric("페어 갭", f"{gap:.2f}%", gap_label)

    # 미니 갭 차트
    intra = st.session_state.intraday_df
    if not intra.empty:
        pct_df = get_intraday_pct_series(intra, [lead_sym, follow_sym])
        if not pct_df.empty and lead_sym in pct_df.columns and follow_sym in pct_df.columns:
            gap_s = (pct_df[lead_sym] - pct_df[follow_sym]).abs()
            st.area_chart(pd.DataFrame({"갭(%)": gap_s}), height=120, use_container_width=True)

    # 손절 경고
    for sym, pct in [(lead_sym, lead_pct), (follow_sym, follow_pct)]:
        if pct <= STOP_LOSS_PCT:
            st.warning(f"⚠️ {sym} {fmt_pct(pct)} — 손절라인 도달")


def render_overview():
    """전체 요약 탭"""

    # --- 쌍둥이 3페어 ---
    st.subheader("쌍둥이 매매")
    cols = st.columns(3, gap="medium")
    for col, (key, pair) in zip(cols, TWIN_PAIRS.items()):
        with col:
            with st.container(border=True):
                render_pair_card(pair)

    # --- 조건부 매매 ---
    st.subheader("조건부 매매")
    with st.container(border=True):
        cond_cols = st.columns(len(CONDITIONAL_TRIGGERS) + 1, gap="medium")
        pos_cnt = 0

        for i, sym in enumerate(CONDITIONAL_TRIGGERS):
            d = get(sym)
            pct = d.get("change_pct", 0)
            is_pos = pct > 0
            if is_pos:
                pos_cnt += 1
            with cond_cols[i]:
                dot = "🔴" if is_pos else "🔵"
                st.metric(f"{dot} {sym}", price_str(d), fmt_pct(pct))

        target_d = get(CONDITIONAL_TARGET)
        with cond_cols[-1]:
            if pos_cnt == len(CONDITIONAL_TRIGGERS):
                st.success(f"**COIN 매수!**  \n{price_str(target_d)}")
            else:
                st.info(f"**COIN 대기** ({pos_cnt}/{len(CONDITIONAL_TRIGGERS)})  \n{price_str(target_d)}")

    # --- 하락장 대안 ---
    st.subheader("하락장 대안")
    with st.container(border=True):
        bear_cols = st.columns(len(BEARISH_TICKERS), gap="medium")
        for col, sym in zip(bear_cols, BEARISH_TICKERS):
            d = get(sym)
            with col:
                st.metric(f"{sym} ({TICKERS[sym]['name']})", price_str(d), fmt_pct(d.get("change_pct", 0)))


# ============================================================
# 시황 탭
# ============================================================
def render_tab_market():
    c1, c2, c3 = st.columns(3)
    for col, sym, label in [
        (c1, GOLD_TICKER, "금 (GLD)"),
        (c2, "SPY", "S&P 500"),
        (c3, "QQQ", "나스닥 100"),
    ]:
        d = get(sym)
        with col:
            st.metric(label, price_str(d), fmt_pct(d.get("change_pct", 0)))

    gold_pct = get(GOLD_TICKER).get("change_pct", 0)
    if gold_pct > 0:
        st.error("**매매금지** — 금 양전. 장 시작 후 30분간 금 추이를 확인하세요.")
    else:
        st.success("**매매가능** — 금 음전 상태입니다.")

    intra = st.session_state.intraday_df
    if not intra.empty:
        pct_df = get_intraday_pct_series(intra, [GOLD_TICKER, "SPY", "QQQ"])
        if not pct_df.empty:
            st.subheader("장중 등락률 추이 (%)")
            st.line_chart(pct_df, height=400, use_container_width=True)


# ============================================================
# 쌍둥이 탭
# ============================================================
def render_tab_twins():
    intra = st.session_state.intraday_df

    for key, pair in TWIN_PAIRS.items():
        lead_sym, follow_sym = pair["lead"], pair["follow"]
        lead_d, follow_d = get(lead_sym), get(follow_sym)
        lead_pct = lead_d.get("change_pct", 0)
        follow_pct = follow_d.get("change_pct", 0)
        gap = abs(lead_pct - follow_pct)

        st.subheader(pair["label"])

        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            st.metric(f"{lead_sym} — 선행", price_str(lead_d), fmt_pct(lead_pct))
        with c2:
            st.metric(f"{follow_sym} — 후행", price_str(follow_d), fmt_pct(follow_pct))
        with c3:
            st.metric("페어 갭", f"{gap:.2f}%")

        if not intra.empty:
            pct_df = get_intraday_pct_series(intra, [lead_sym, follow_sym])
            if not pct_df.empty and lead_sym in pct_df.columns and follow_sym in pct_df.columns:
                cl, cr = st.columns(2)
                with cl:
                    st.caption("등락률 비교 (%)")
                    st.line_chart(pct_df[[lead_sym, follow_sym]], height=300, use_container_width=True)
                with cr:
                    st.caption("페어 갭 추이 (%)")
                    gap_s = (pct_df[lead_sym] - pct_df[follow_sym]).abs()
                    st.area_chart(pd.DataFrame({"갭": gap_s}), height=300, use_container_width=True)

        st.divider()


# ============================================================
# 조건부 탭
# ============================================================
def render_tab_conditional():
    cols = st.columns(len(CONDITIONAL_TRIGGERS) + 1)
    pos_cnt = 0

    for i, sym in enumerate(CONDITIONAL_TRIGGERS):
        d = get(sym)
        pct = d.get("change_pct", 0)
        is_pos = pct > 0
        if is_pos:
            pos_cnt += 1
        dot = "🔴" if is_pos else "🔵"
        with cols[i]:
            st.metric(f"{dot} {sym} ({TICKERS[sym]['name']})", price_str(d), fmt_pct(pct))

    target_d = get(CONDITIONAL_TARGET)
    with cols[-1]:
        if pos_cnt == len(CONDITIONAL_TRIGGERS):
            st.success(f"**COIN 매수 신호!** {price_str(target_d)} {fmt_pct(target_d.get('change_pct', 0))}")
        else:
            st.info(f"**COIN 대기** ({pos_cnt}/{len(CONDITIONAL_TRIGGERS)}) {price_str(target_d)}")

    intra = st.session_state.intraday_df
    if not intra.empty:
        pct_df = get_intraday_pct_series(intra, CONDITIONAL_TRIGGERS + [CONDITIONAL_TARGET])
        if not pct_df.empty:
            st.subheader("장중 등락률 (%)")
            st.line_chart(pct_df, height=400, use_container_width=True)


# ============================================================
# 하락장 탭
# ============================================================
def render_tab_bearish():
    cols = st.columns(len(BEARISH_TICKERS))
    for col, sym in zip(cols, BEARISH_TICKERS):
        d = get(sym)
        with col:
            st.metric(f"{sym} ({TICKERS[sym]['name']})", price_str(d), fmt_pct(d.get("change_pct", 0)))

    intra = st.session_state.intraday_df
    if not intra.empty:
        pct_df = get_intraday_pct_series(intra, BEARISH_TICKERS)
        if not pct_df.empty:
            st.subheader("장중 등락률 (%)")
            st.line_chart(pct_df, height=400, use_container_width=True)


# ============================================================
# 전체 종목 탭
# ============================================================
def render_tab_table():
    prices = st.session_state.prices
    if not prices:
        st.info("데이터 로딩 중...")
        return

    rows = []
    for sym, d in prices.items():
        rows.append({
            "종목": sym,
            "종목명": d.get("name", ""),
            "현재가($)": f"{d.get('price', 0):,.2f}",
            "등락률(%)": fmt_pct(d.get("change_pct", 0)),
            "시가($)": f"{d.get('open', 0):,.2f}",
            "고가($)": f"{d.get('high', 0):,.2f}",
            "저가($)": f"{d.get('low', 0):,.2f}",
            "거래량": f"{d.get('volume', 0):,}",
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)


# ============================================================
# 사이드바
# ============================================================
def render_sidebar():
    with st.sidebar:
        st.header("매매 규칙")

        st.subheader("시황 판단")
        st.markdown(
            "- 금(GLD) 양전 → **매매금지**\n"
            "- 장 시작 30분간 금 상승 → 당일 중단\n"
            "- 프리마켓~새벽5시 양전 → 금지"
        )

        st.subheader("쌍둥이 매매")
        st.markdown(
            "- 2배 ETF, 2,000만원 한도\n"
            "- 5분 간격 분할매수\n"
            f"- 갭 {PAIR_GAP_SELL_THRESHOLD}% 이내 → **매도**\n"
            f"- 손절: **{STOP_LOSS_PCT}%**\n"
            "- **당일 매도 원칙**"
        )

        st.subheader("조건부 매매")
        st.markdown("- ETHU+XXRP+SOLT 3종목 양전 → COIN 매수")

        st.subheader("하락장")
        st.markdown("- 금 2x ETF / HIMZ / BRKU / BABX")


# ============================================================
# 메인
# ============================================================
def main():
    live_fetcher()
    render_status_bar()
    st.divider()

    tabs = st.tabs(["전체 요약", "시황", "쌍둥이", "조건부", "하락장", "전체 종목"])

    with tabs[0]:
        render_overview()
    with tabs[1]:
        render_tab_market()
    with tabs[2]:
        render_tab_twins()
    with tabs[3]:
        render_tab_conditional()
    with tabs[4]:
        render_tab_bearish()
    with tabs[5]:
        render_tab_table()

    render_sidebar()


if __name__ == "__main__":
    main()

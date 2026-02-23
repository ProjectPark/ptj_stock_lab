#!/usr/bin/env python3
"""PTJ 매매법 대시보드 - 메인 실행"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
if str(_ROOT / "fetchers") not in sys.path:
    sys.path.insert(0, str(_ROOT / "fetchers"))

import config
import fetch_data
from simulation.strategies.line_a import signals
import dashboard_html


# ============================================================
# ANSI 색상
# ============================================================
class C:
    """ANSI escape codes."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    DIM = "\033[2m"


def _color_pct(pct: float) -> str:
    """등락률을 색상 포함 문자열로 반환."""
    if pct > 0:
        return f"{C.GREEN}{pct:+.2f}%{C.RESET}"
    elif pct < 0:
        return f"{C.RED}{pct:+.2f}%{C.RESET}"
    return f"{C.DIM}{pct:+.2f}%{C.RESET}"


def _check_mark(positive: bool) -> str:
    """양전/음전 표시."""
    if positive:
        return f"{C.GREEN}\u2713{C.RESET}"
    return f"{C.RED}\u2717{C.RESET}"


# ============================================================
# 콘솔 출력
# ============================================================
def print_banner():
    """시작 배너 출력."""
    print()
    print(f"{C.CYAN}{C.BOLD}")
    print("  ╔═══════════════════════════════════════╗")
    print("  ║       PTJ 매매법 대시보드 시스템       ║")
    print("  ╚═══════════════════════════════════════╝")
    print(f"{C.RESET}")


def print_signal_summary(sigs: dict):
    """콘솔 시그널 요약 출력."""
    print()
    print(f"{C.BOLD}═══════════════════════════════════════{C.RESET}")
    print(f"{C.BOLD}  PTJ 매매법 시그널 요약{C.RESET}")
    print(f"{C.BOLD}═══════════════════════════════════════{C.RESET}")

    # --- 시황: 금 ---
    gold = sigs.get("gold", {})
    gold_warning = gold.get("warning", False)
    gold_msg = gold.get("message", "")
    if gold_warning:
        print(f"\n{C.RED}[시황]{C.RESET} {gold_msg}")
    else:
        print(f"\n{C.GREEN}[시황]{C.RESET} {gold_msg}")

    # --- 쌍둥이 페어 ---
    pairs = sigs.get("twin_pairs", [])
    print(f"\n{C.CYAN}[쌍둥이 페어]{C.RESET}")
    if pairs:
        for p in pairs:
            lead_str = f"{p['lead']} {_color_pct(p['lead_pct'])}"
            follow_str = f"{p['follow']} {_color_pct(p['follow_pct'])}"
            gap = p["gap"]
            signal = p["signal"]
            if signal == "SELL":
                sig_str = f"{C.RED}매도 시그널{C.RESET}"
            elif signal == "ENTRY":
                sig_str = f"{C.GREEN}매수 검토{C.RESET}"
            else:
                sig_str = f"{C.YELLOW}관망{C.RESET}"
            print(f"  {p['pair']}: {lead_str} | {follow_str} | 갭 {_color_pct(gap)} — {sig_str}")
    else:
        print(f"  {C.DIM}(데이터 없음){C.RESET}")

    # --- 조건부 매매 ---
    cond = sigs.get("conditional", {})
    triggers = cond.get("triggers", {})
    all_pos = cond.get("all_positive", False)
    print(f"\n{C.CYAN}[조건부 매매]{C.RESET}")

    trigger_parts = []
    for t, info in triggers.items():
        pct = info.get("change_pct", 0.0)
        positive = info.get("positive", False)
        trigger_parts.append(f"{t} {_color_pct(pct)} {_check_mark(positive)}")
    print(f"  {' | '.join(trigger_parts)}")

    cond_msg = cond.get("message", "")
    if all_pos:
        print(f"  {C.GREEN}\u2192 {cond_msg}{C.RESET}")
    else:
        print(f"  {C.YELLOW}\u2192 {cond_msg}{C.RESET}")

    # --- 손절 경고 ---
    stop_loss = sigs.get("stop_loss", [])
    print(f"\n{C.CYAN}[손절 경고]{C.RESET}")
    if stop_loss:
        for sl in stop_loss:
            print(f"  {C.RED}{sl['message']}{C.RESET}")
    else:
        print(f"  {C.DIM}(없음){C.RESET}")

    # --- 하락장 ---
    bearish = sigs.get("bearish", {})
    bear_msg = bearish.get("message", "")
    market_down = bearish.get("market_down", False)
    print(f"\n{C.CYAN}[하락장]{C.RESET}")
    if market_down:
        print(f"  {C.RED}{bear_msg}{C.RESET}")
        picks = bearish.get("bearish_picks", [])
        if picks:
            for bp in picks:
                print(f"    {bp['ticker']} ({bp['name']}): {_color_pct(bp['change_pct'])}")
    else:
        print(f"  {C.GREEN}{bear_msg}{C.RESET}")

    print()


def print_completion(html_path: Path | None):
    """완료 메시지 출력."""
    print(f"{C.BOLD}═══════════════════════════════════════{C.RESET}")
    print(f"{C.GREEN}{C.BOLD}  생성 완료{C.RESET}")
    print(f"{C.BOLD}═══════════════════════════════════════{C.RESET}")
    if html_path:
        print(f"  HTML 대시보드: {html_path}")
    print()


# ============================================================
# 메인
# ============================================================
def main():
    print_banner()

    # 1. 데이터 수집
    print(f"{C.CYAN}[1/4] 데이터 수집 중...{C.RESET}")
    data = fetch_data.fetch_all()
    if data.empty:
        print(f"{C.RED}데이터 수집 실패 — 종료합니다.{C.RESET}")
        sys.exit(1)
    print(f"  총 {len(data):,} rows 수집 완료\n")

    # 2. 등락률 계산
    print(f"{C.CYAN}[2/4] 등락률 계산 중...{C.RESET}")
    changes = fetch_data.get_latest_changes(data)
    print(f"  {len(changes)}개 종목 분석 완료\n")

    # 3. 시그널 분석
    print(f"{C.CYAN}[3/4] 시그널 분석 중...{C.RESET}")
    sigs = signals.generate_all_signals(changes)
    print_signal_summary(sigs)

    # 4. 대시보드 생성
    print(f"{C.CYAN}[4/4] 대시보드 생성 중...{C.RESET}")

    # HTML 대시보드
    html_path = None
    try:
        html_path = dashboard_html.generate_dashboard_html(data, changes, sigs)
    except Exception as e:
        print(f"  {C.RED}[WARN] HTML 대시보드 생성 실패: {e}{C.RESET}")

    # PNG 대시보드 (dashboard 모듈이 있는 경우)
    try:
        import dashboard
        dashboard.generate_dashboard(data, changes, sigs)
    except ImportError:
        pass
    except Exception as e:
        print(f"  {C.RED}[WARN] PNG 대시보드 생성 실패: {e}{C.RESET}")

    print_completion(html_path)


if __name__ == "__main__":
    main()

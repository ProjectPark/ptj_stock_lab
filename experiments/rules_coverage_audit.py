"""
Rules Coverage Audit — rules_coverage_audit.py
=================================================
문서(docs/rules/, docs/notes/line_b/review/)와 엔진 코드(simulation/strategies/) 간
커버리지를 텍스트 분석으로 측정한다.

실행:
    pyenv shell ptj_stock_lab && python experiments/rules_coverage_audit.py

출력:
    - 터미널: 커버리지 매트릭스
    - docs/reports/backtest/rules_coverage_summary.md
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import NamedTuple

# ──────────────────────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DOCS_RULES_A = ROOT / "docs/rules/line_a"
DOCS_RULES_C = ROOT / "docs/rules/line_c"
DOCS_RULES_D = ROOT / "docs/rules/line_d"
DOCS_NOTES_B = ROOT / "docs/notes/line_b/review"

ENGINE_A = ROOT / "simulation/strategies/line_a"
ENGINE_B = ROOT / "simulation/strategies/line_b_taejun"
ENGINE_C = ROOT / "simulation/strategies/line_c_d2s"
ENGINE_D = ROOT / "simulation/strategies/line_d_history"


# ──────────────────────────────────────────────────────────────
# 데이터 구조
# ──────────────────────────────────────────────────────────────
class RuleEntry(NamedTuple):
    rule_id: str          # 예: "CB-1", "R1", "E-1"
    description: str      # 규칙 설명 (1줄)
    source_doc: str       # 출처 파일
    implemented: bool     # 구현 여부
    impl_file: str        # 구현 파일 (없으면 "")
    impl_note: str        # 구현 메모 / 부분 여부


# ──────────────────────────────────────────────────────────────
# 유틸: 파일 텍스트 읽기
# ──────────────────────────────────────────────────────────────
def read_all_text(directory: Path, extensions: tuple = (".py",)) -> str:
    """디렉터리의 모든 파일 텍스트를 합친다."""
    texts = []
    if not directory.exists():
        return ""
    for path in directory.rglob("*"):
        if path.suffix in extensions and path.is_file():
            try:
                texts.append(f"\n# === {path.name} ===\n" + path.read_text(encoding="utf-8"))
            except Exception:
                pass
    return "\n".join(texts)


def read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def search_in_code(code: str, patterns: list[str]) -> tuple[bool, str]:
    """코드에서 패턴 목록 중 하나라도 발견되면 (True, 발견 패턴) 반환."""
    for pat in patterns:
        if re.search(pat, code, re.IGNORECASE):
            return True, pat
    return False, ""


# ──────────────────────────────────────────────────────────────
# ── Line A 규칙 정의 + 커버리지 체크 ──────────────────────────
# ──────────────────────────────────────────────────────────────
def audit_line_a() -> list[RuleEntry]:
    """Line A (v1~v5 규칙서) vs 엔진 코드 (signals.py, signals_v2.py, signals_v5.py)."""
    code_a = read_all_text(ENGINE_A, (".py",))
    # line_b_taejun도 v5 규칙을 구현하므로 함께 검색
    code_b = read_all_text(ENGINE_B, (".py",))
    full_code = code_a + "\n" + code_b

    rules = [
        # ── v1 기본 5개 규칙 ──────────────────────────────────────
        {
            "id": "R1",
            "desc": "금(GLD) 양전 시 매매 금지",
            "patterns": ["check_gold_signal", "gld_pct.*>.*0", "GoldFilter", "gold.*warning",
                         "gld_suppress", "GLD.*양전", "gold_signal"],
        },
        {
            "id": "R2",
            "desc": "쌍둥이 페어 갭 매매 (ENTRY/SELL/HOLD)",
            "patterns": ["check_twin_pairs", "TwinPairStrategy", "twin_pair", "gap.*ENTRY",
                         "PAIR_GAP", "twin_gaps"],
        },
        {
            "id": "R3",
            "desc": "조건부 매매 (ETHU+XXRP+SOLT 모두 양전 → COIN 매수)",
            "patterns": ["ConditionalCoinStrategy", "conditional_coin", "ETHU.*XXRP.*SOLT",
                         "coin_condition", "check_conditional"],
        },
        {
            "id": "R4",
            "desc": "손절 체크 (보유 종목 -3% 이하 즉시 매도, 추가 매수 금지)",
            "patterns": ["stop_loss", "StopLoss", "ATR.*손절", "stop_pct", "atr_stop",
                         "pnl_pct.*<=-", "stoploss"],
        },
        {
            "id": "R5",
            "desc": "하락장 방어 매매 (SPY+QQQ < 0 → 비방어주 매수 금지)",
            "patterns": ["BearishDefenseStrategy", "bearish_defense", "BRKU.*10", "bear.*mode",
                         "market_down.*spy.*qqq", "spy_change.*<.*0.*qqq"],
        },
        # ── v2 신규 ─────────────────────────────────────────────
        {
            "id": "V2-BULLISH",
            "desc": "Polymarket 강세장 모드 (BTC up >= 70%)",
            "patterns": ["determine_market_mode", "bullish.*0.70", "btc_up.*0.7",
                         "MarketModeFilter", "market_mode.*bullish"],
        },
        {
            "id": "V2-BEARISH",
            "desc": "Polymarket 하락장 모드 (3종 <= 20%)",
            "patterns": ["bearish.*0.20", "btc_up.*0.20.*ndx_up", "3.*조건.*20",
                         "market_mode.*bearish", "determine_market_mode"],
        },
        {
            "id": "V2-IRE",
            "desc": "코인 후행 종목 IRE 추가 (MSTU vs IRE 당일 선택)",
            "patterns": [r"\bIRE\b", "mstu.*ire", "coin.*follow.*mstu.*ire",
                         "MSTU.*IRE", "follow.*select"],
        },
        # ── v5 신규 ─────────────────────────────────────────────
        {
            "id": "CB-1",
            "desc": "VIX 급등 (+6%) → 7거래일 신규 매수 금지",
            "patterns": ["CB-1", "cb1", "vix.*6", "vix_spike.*6", "VIX.*급등",
                         "cb1_active", "cb1_remaining"],
        },
        {
            "id": "CB-2",
            "desc": "GLD 급등 (+3%) → 3거래일 신규 매수 금지",
            "patterns": ["CB-2", "cb2", "gld.*3.*%", "GLD.*3.*percent",
                         "cb2_active", "cb2_remaining"],
        },
        {
            "id": "CB-3",
            "desc": "BTC 급락 (-5%) → 신규 매수 금지",
            "patterns": ["CB-3", "cb3", "btc.*-5", "BTC.*급락.*5",
                         "cb3_active"],
        },
        {
            "id": "CB-4",
            "desc": "BTC 급등 (+5%) → 추격매수 금지",
            "patterns": ["CB-4", "cb4", "btc.*\\+5", "chase.*block", "cb4_active",
                         "only_chase_blocked"],
        },
        {
            "id": "CB-5",
            "desc": "금리 상승 확률 50%+ → 모든 신규 매수 금지 + 레버리지 3일 추가 대기",
            "patterns": ["CB-5", "cb5", "interest.*rate.*50", "금리.*50",
                         "cb5_active", "cb5_lev_cooldown", "poly.*rate"],
        },
        {
            "id": "CB-6",
            "desc": "종목 +20% 과열 → 레버리지→비레버리지 전환",
            "patterns": ["CB-6", "cb6", "overheat.*20", "과열.*20", "SOXX.*전환",
                         "cb6_tickers", "hot_ticker"],
        },
        {
            "id": "SIDEWAYS",
            "desc": "횡보장 감지 (6개 지표 중 3개 이상 → 현금 100%)",
            "patterns": ["evaluate_sideways", "SidewaysDetector", "sideways.*mode",
                         "is_sideways", "sideways_active", "횡보장"],
        },
        {
            "id": "V5-SOXL",
            "desc": "SOXL 독립 매매 (SOXX +2% 이상 + ADX >= 20)",
            "patterns": ["SoxlIndependentStrategy", "soxl_independent", "SOXX.*2.*%",
                         "soxl.*independent", "SOXL.*independent"],
        },
        {
            "id": "V5-CRASH",
            "desc": "급락 역매수 (SOXL/CONL/IRE -30% 또는 LULD 3회 → 95% 매수)",
            "patterns": ["CrashBuyStrategy", "crash_buy", "LULD", "-30.*%.*매수",
                         "crash_entry", "급락.*역매수", "CrashBuy"],
        },
        {
            "id": "V5-CONL-60",
            "desc": "CONL·IRE 고정 익절 (40% 갭수렴 + 60% +5% 익절)",
            "patterns": ["conl.*60", "ire.*60", "40.*60.*split", "split_60",
                         "take_profit.*5", "fixed_profit_60"],
        },
        {
            "id": "V5-ATR-STOP",
            "desc": "ATR 기반 손절 (1.5×ATR, 강세장 2.5×ATR)",
            "patterns": ["StopLossCalculator", "atr.*1.5", "ATR.*stop", "atr_stop",
                         "atr.*multiplier", "stop_loss_calc"],
        },
        {
            "id": "V5-VIX-GOLD",
            "desc": "VIX 방어모드 (IAU 40% + GDXU 30%)",
            "patterns": ["VixGold", "vix_gold", "IAU.*40", "GDXU.*30", "iau_pct",
                         "gdxu_pct", "vix.*gold.*defense"],
        },
        {
            "id": "V5-SWING",
            "desc": "급등 스윙 모드 (13절)",
            "patterns": ["SwingModeManager", "swing_mode", "swing.*entry", "스윙.*모드",
                         "swing_active"],
        },
        {
            "id": "V5-ASSET-MODE",
            "desc": "자산 모드 시스템 (이머전시>공격>방어>조심)",
            "patterns": ["asset_mode", "AssetMode", "emergency.*attack.*defense",
                         "이머전시.*공격.*방어", "M201", "m201_mode"],
        },
        {
            "id": "V5-POLY-QUALITY",
            "desc": "Polymarket 품질 필터 (극단값/5h 미갱신 제외)",
            "patterns": ["poly_quality", "PolyQuality", "poly.*stale", "poly.*extreme",
                         "ndx_stop", "poly_quality"],
        },
        {
            "id": "V5-NDX-STOP",
            "desc": "NDX 미갱신 시 NDX 의존 조건 정지",
            "patterns": ["ndx_stop", "poly.*NDX.*stale", "ndx.*미갱신", "poly_quality.*ndx"],
        },
        {
            "id": "V5-EMERGENCY",
            "desc": "이머전시 모드 (Polymarket 30pp+ 급변 → 수익매도+방향성매수)",
            "patterns": ["emergency_mode", "EmergencyMode", "poly.*30pp", "30.*급변",
                         "EMERGENCY_MODE", "emergency.*30"],
        },
    ]

    entries = []
    for r in rules:
        found, pat = search_in_code(full_code, r["patterns"])
        entries.append(RuleEntry(
            rule_id=r["id"],
            description=r["desc"],
            source_doc="trading_rules_v1~v5.md",
            implemented=found,
            impl_file="signals.py / signals_v2.py / signals_v5.py / line_b_taejun/" if found else "",
            impl_note=f"found: {pat}" if found else "NOT FOUND",
        ))
    return entries


# ──────────────────────────────────────────────────────────────
# ── Line B 커버리지 체크 ───────────────────────────────────────
# ──────────────────────────────────────────────────────────────
def audit_line_b() -> list[RuleEntry]:
    """Line B (notes/line_b/review/) vs line_b_taejun 엔진."""
    code_b = read_all_text(ENGINE_B, (".py",))

    # 3개 리뷰 노트에서 전략 아이디/모듈 식별
    # taejun_strategy_review_2026-02-23_VNQ.md 기준 M0~M40, M80, M200, M201, M28, M300 + 12개 전략
    rules = [
        # ── 마스터 플랜 인프라 ──────────────────────────────────
        {
            "id": "M1-LIMIT-ORDER",
            "desc": "지정가 주문 전용 (시장가 금지), TTL 2분",
            "patterns": ["limit_order", "LimitOrder", "order_ttl", "TTL.*120", "LIMIT.*only"],
        },
        {
            "id": "M5-WEIGHT",
            "desc": "종목별 진입 비율 가중치 관리",
            "patterns": ["m5_weight", "WeightManager", "weight_manager", "진입.*비율",
                         "entry.*weight", "M5"],
        },
        {
            "id": "M28-POLY-GATE",
            "desc": "Polymarket 게이트 (NDX 상승 확률 기반 매매 허용)",
            "patterns": ["m28_poly_gate", "PolyGate", "poly_gate", "M28", "ndx_up.*gate"],
        },
        {
            "id": "M200-STOP",
            "desc": "M200 원금손실 중단 시스템",
            "patterns": ["m200_stop", "M200Stop", "M200", "원금.*손실.*중단", "stop_loss.*capital"],
        },
        {
            "id": "M201-IMMEDIATE",
            "desc": "M201 즉시모드 (조건 달성 즉시 실행)",
            "patterns": ["m201_mode", "M201ImmediateMode", "M201", "immediate.*mode", "즉시.*모드"],
        },
        {
            "id": "MASTER-SCHD",
            "desc": "MASTER SCHD (전략 스케줄 관리)",
            "patterns": ["schd_master", "SchdMaster", "MASTER.*SCHD", "master_schedule",
                         "strategy.*schedule"],
        },
        {
            "id": "PROFIT-DIST",
            "desc": "수익금 분배 규칙",
            "patterns": ["profit_distributor", "ProfitDistributor", "수익금.*분배",
                         "profit.*distribute", "reinvest"],
        },
        {
            "id": "ASSET-MODE",
            "desc": "이머전시/공격/방어/조심 4단계 모드",
            "patterns": ["asset_mode", "AssetMode", "emergency.*attack.*caution",
                         "이머전시", "공격.*방어.*조심", "ASSET_MODE"],
        },
        {
            "id": "CIRCUIT-BREAKER",
            "desc": "CB-1~CB-6 서킷 브레이커 시스템",
            "patterns": ["CircuitBreaker", "circuit_breaker", "CB-1.*CB-2", "cb1.*cb2",
                         "CBStatus"],
        },
        # ── 12개 전략 모듈 ──────────────────────────────────────
        {
            "id": "TWIN-PAIR",
            "desc": "쌍둥이 페어 갭 매매 전략",
            "patterns": ["TwinPairStrategy", "twin_pair", "쌍둥이.*갭"],
        },
        {
            "id": "CONDITIONAL-COIN",
            "desc": "조건부 COIN 매매 (ETHU+XXRP+SOLT → COIN)",
            "patterns": ["ConditionalCoinStrategy", "conditional_coin", "COIN.*ETHU.*XXRP"],
        },
        {
            "id": "CONDITIONAL-CONL",
            "desc": "조건부 CONL 매매",
            "patterns": ["ConditionalConlStrategy", "conditional_conl", "CONL.*conditional"],
        },
        {
            "id": "BEARISH-DEFENSE",
            "desc": "하락장 방어 매매 (BRKU)",
            "patterns": ["BearishDefenseStrategy", "bearish_defense", "BRKU.*방어"],
        },
        {
            "id": "BARGAIN-BUY",
            "desc": "저가매수 (3년 최고가 대비 폭락 진입)",
            "patterns": ["bargain_buy", "BargainBuy", "BARGAIN_BUY", "저가매수.*폭락"],
        },
        {
            "id": "CRASH-BUY",
            "desc": "급락 역매수 (SOXL/CONL/IRE -30%+)",
            "patterns": ["CrashBuyStrategy", "crash_buy", "CRASH_BUY", "급락.*역매수"],
        },
        {
            "id": "VIX-GOLD",
            "desc": "VIX 방어모드 (IAU+GDXU)",
            "patterns": ["VixGold", "vix_gold", "VIX_GOLD", "IAU.*GDXU"],
        },
        {
            "id": "SOXL-INDEPENDENT",
            "desc": "SOXL 독립 매매 (4-7절)",
            "patterns": ["SoxlIndependentStrategy", "soxl_independent", "SOXL.*independent"],
        },
        {
            "id": "REIT-RISK",
            "desc": "리츠 리스크 (조심모드 트리거)",
            "patterns": ["reit_risk", "ReitRisk", "REIT_RISK", "리츠.*리스크", "VNQ.*risk"],
        },
        {
            "id": "SECTOR-ROTATE",
            "desc": "섹터 로테이션 (BTC→반도체→은행→금)",
            "patterns": ["sector_rotate", "SectorRotate", "SECTOR_ROTATE", "섹터.*로테이션"],
        },
        {
            "id": "SWING-MODE",
            "desc": "급등 스윙 모드",
            "patterns": ["swing_mode", "SwingModeManager", "SwingMode", "스윙.*모드"],
        },
        {
            "id": "EMERGENCY-MODE",
            "desc": "이머전시 모드 (30pp 급변 대응)",
            "patterns": ["emergency_mode", "EmergencyMode", "EMERGENCY_MODE"],
        },
        {
            "id": "JAB-SOXL",
            "desc": "잽모드 SOXL (프리마켓 단타)",
            "patterns": ["jab_soxl", "JabSoxl", "JAB_SOXL"],
        },
        {
            "id": "JAB-BITU",
            "desc": "잽모드 BITU (BTC 레버리지 프리마켓 단타)",
            "patterns": ["jab_bitu", "JabBitu", "JAB_BITU"],
        },
        {
            "id": "JAB-TSLL",
            "desc": "잽모드 TSLL (테슬라 레버리지 단타)",
            "patterns": ["jab_tsll", "JabTsll", "JAB_TSLL"],
        },
        {
            "id": "JAB-ETQ",
            "desc": "잽모드 ETQ (ETH 인버스 단타)",
            "patterns": ["jab_etq", "JabEtq", "JAB_ETQ", "JAB_SETH"],
        },
        {
            "id": "BEAR-REGIME",
            "desc": "하락 레짐 전략 (SHORT_MACRO 연계)",
            "patterns": ["bear_regime", "BearRegime", "short_macro", "ShortMacro"],
        },
        {
            "id": "SP500-ENTRY",
            "desc": "S&P500 편입 다음날 매수",
            "patterns": ["sp500_entry", "Sp500Entry", "SP500_ENTRY", "편입.*매수"],
        },
        {
            "id": "BANK-CONDITIONAL",
            "desc": "은행주 조건부 매매",
            "patterns": ["bank_conditional", "BankConditional", "은행.*조건부"],
        },
        {
            "id": "STOP-LOSS",
            "desc": "손절 시스템 (ATR 기반 + 레버리지 차등)",
            "patterns": ["StopLossCalculator", "stop_loss", "ATR.*stop", "leverage.*stop"],
        },
        {
            "id": "POLY-QUALITY",
            "desc": "Polymarket 품질 필터",
            "patterns": ["poly_quality", "PolyQuality", "poly.*stale"],
        },
        {
            "id": "ORCHESTRATOR",
            "desc": "전략 오케스트레이터",
            "patterns": ["orchestrator", "Orchestrator", "CompositeSignalEngine"],
        },
    ]

    entries = []
    for r in rules:
        found, pat = search_in_code(code_b, r["patterns"])
        entries.append(RuleEntry(
            rule_id=r["id"],
            description=r["desc"],
            source_doc="taejun_strategy_review_*.md (notes)",
            implemented=found,
            impl_file="line_b_taejun/" if found else "",
            impl_note=f"found: {pat}" if found else "NOT FOUND",
        ))
    return entries


# ──────────────────────────────────────────────────────────────
# ── Line C 커버리지 체크 ───────────────────────────────────────
# ──────────────────────────────────────────────────────────────
def audit_line_c() -> list[RuleEntry]:
    """Line C (attach v1~v2 규칙서) vs D2S 엔진."""
    code_c = read_all_text(ENGINE_C, (".py",))

    rules = [
        # attach v1 규칙 (R1~R16)
        {"id": "R1-GLD-FILTER", "desc": "GLD 시황 필터 (GLD >= 1.0% → 매수 억제)",
         "patterns": ["gld_suppress_threshold", "R1.*GLD", "gld.*1.0", "gld_pct.*1.0"]},
        {"id": "R2-TWIN-GAP", "desc": "쌍둥이 갭 진입 (페어별 차별화)",
         "patterns": ["check_twin_gaps", "twin_pairs", "gap_bank_conl", "R2"]},
        {"id": "R3-BTC-FILTER", "desc": "BTC up 확률 필터 (> 0.75 → 매수 억제)",
         "patterns": ["btc_up_max", "poly_btc_up", "R3.*BTC", "0.75"]},
        {"id": "R4-TAKE-PROFIT", "desc": "이익실현 (+5.9% 중앙값)",
         "patterns": ["take_profit_pct", "5.9", "R4.*profit", "익절.*5.9"]},
        {"id": "R5-HOLD-DAYS", "desc": "최적 보유 기간 (4~7거래일)",
         "patterns": ["optimal_hold_days", "hold_days_min.*4", "hold_days_max.*7", "R5"]},
        {"id": "R6-DCA-LIMIT", "desc": "일일 동일종목 매수 상한 (5회)",
         "patterns": ["dca_max_daily", "dca_max", "R6.*DCA", "5.*daily"]},
        {"id": "R7-RSI-FILTER", "desc": "RSI 진입 금지 (RSI > 80)",
         "patterns": ["rsi_danger_zone", "R7.*RSI", "RSI.*80", "rsi.*80"]},
        {"id": "R8-BB-FILTER", "desc": "볼린저밴드 진입 금지 (%B > 1.0)",
         "patterns": ["bb_danger_zone", "R8.*BB", "bb.*1.0", "pct_b.*1.0"]},
        {"id": "R9-VOL-FILTER", "desc": "거래량 필터 (1.2 ~ 2.0 상대 거래량)",
         "patterns": ["vol_entry_min", "vol_entry_max", "R9.*vol", "rel_volume"]},
        {"id": "R13-SPY-STREAK", "desc": "SPY 3일 연속 상승 후 매수 금지 (승률 27.3%)",
         "patterns": ["spy_streak_max", "spy_streak.*3", "R13", "연속.*상승.*금지"]},
        {"id": "R14-RISKOFF-BOOST", "desc": "GLD↑+SPY↓ 리스크오프 역발상 매수 (86.4% 승률)",
         "patterns": ["riskoff_gld_up_spy_down", "R14", "riskoff.*boost", "GLD.*SPY.*down"]},
        {"id": "R15-FRIDAY", "desc": "금요일 진입 우대 (88.3% 승률)",
         "patterns": ["friday_boost", "R15.*friday", "weekday.*4", "금요일"]},
        {"id": "R16-ATR-BOOST", "desc": "ATR Q4 진입 우대 (85.3% 승률)",
         "patterns": ["atr_high_quantile", "R16.*ATR", "atr_q.*0.75", "atr.*quantile"]},
        # attach v2 신규 (R17~R18)
        {"id": "R17-VBOUNCE", "desc": "충격 V-바운스 포지션 2배 확대 (%B < 0.15 + -10% 급락)",
         "patterns": ["vbounce_bb_threshold", "vbounce_crash_threshold", "R17", "V.*바운스"]},
        {"id": "R18-EARLY-STOPLOSS", "desc": "BB 하단 돌파 후 3일 비회복 시 조기 손절",
         "patterns": ["early_stoploss_days", "early_stoploss_recovery", "R18", "early.*stop"]},
        {"id": "DCA-LAYERS", "desc": "DCA 레이어 제한 (v2: 최대 2레이어, 3레이어+ 강력 억제)",
         "patterns": ["dca_max_layers", "dca.*2.*layer", "layer.*2", "3.*레이어.*금지"]},
    ]

    entries = []
    for r in rules:
        found, pat = search_in_code(code_c, r["patterns"])
        entries.append(RuleEntry(
            rule_id=r["id"],
            description=r["desc"],
            source_doc="trading_rules_attach_v1~v2.md",
            implemented=found,
            impl_file="line_c_d2s/" if found else "",
            impl_note=f"found: {pat}" if found else "NOT FOUND",
        ))
    return entries


# ──────────────────────────────────────────────────────────────
# ── Line D 커버리지 체크 ───────────────────────────────────────
# ──────────────────────────────────────────────────────────────
def audit_line_d() -> list[RuleEntry]:
    """Line D (jun_trade_2023_v1.md) vs line_d_history 엔진."""
    code_d = read_all_text(ENGINE_D, (".py",))

    # line_d_history에 코드가 있는지 확인
    d_files = list(ENGINE_D.glob("*.py"))
    d_has_impl = any(f.stat().st_size > 100 for f in d_files if f.name != "__init__.py")

    rules = [
        {"id": "E-1", "desc": "모멘텀 추세 확인 후 진입 (가격 > MA20, RSI >= 55, 낙폭 -15% 이내)",
         "patterns": ["E.1", "pct_from_ma20", "rsi.*55", "MA20.*entry", "momentum.*entry"]},
        {"id": "E-2", "desc": "BTC 과열 시 크립토 종목 진입 금지 (BTC RSI >= 75)",
         "patterns": ["E.2", "btc.*rsi.*75", "BTC.*과열", "btc_rsi.*75", "crypto.*block.*75"]},
        {"id": "E-3", "desc": "급등 추격 매수 금지 (5일 수익률 >= +8%)",
         "patterns": ["E.3", "5day.*8", "5일.*수익률.*8", "chase.*8", "streak.*+8"]},
        {"id": "VIX-REGIME", "desc": "VIX 구간별 매매 전략 (VIX 20~25 최고 승률)",
         "patterns": ["vix.*regime", "vix.*20.*25", "VIX.*구간", "vix_range"]},
        {"id": "BTC-REGIME", "desc": "BTC 레짐 판단 (0~3단계)",
         "patterns": ["btc_regime", "BTC.*레짐", "btc.*ma60", "btc.*ma20.*rsi"]},
        {"id": "X-1", "desc": "목표 수익 청산 (평균단가 대비 +15%)",
         "patterns": ["X.1", "target.*15", "target_pct.*15", "\\+15.*exit"]},
        {"id": "X-2", "desc": "추세 붕괴 청산 (MA20 대비 -15% 이탈)",
         "patterns": ["X.2", "ma20.*-15", "MA20.*이탈.*15", "trend.*break.*15"]},
        {"id": "X-3", "desc": "손절 (평균단가 대비 -20%)",
         "patterns": ["X.3", "stop.*-20", "손절.*20", "stop_pct.*20", "-0.20"]},
        {"id": "X-4", "desc": "시간 청산 (보유 45거래일 초과)",
         "patterns": ["X.4", "45.*trading.*days", "hold.*45", "time.*exit.*45"]},
    ]

    entries = []
    for r in rules:
        if not d_has_impl:
            # 엔진 코드가 없으면 모두 미구현
            entries.append(RuleEntry(
                rule_id=r["id"],
                description=r["desc"],
                source_doc="jun_trade_2023_v1.md",
                implemented=False,
                impl_file="",
                impl_note="line_d_history/ 엔진 코드 없음 (빈 디렉토리)",
            ))
        else:
            found, pat = search_in_code(code_d, r["patterns"])
            entries.append(RuleEntry(
                rule_id=r["id"],
                description=r["desc"],
                source_doc="jun_trade_2023_v1.md",
                implemented=found,
                impl_file="line_d_history/" if found else "",
                impl_note=f"found: {pat}" if found else "NOT FOUND",
            ))
    return entries


# ──────────────────────────────────────────────────────────────
# 출력 포맷터
# ──────────────────────────────────────────────────────────────
def format_table(entries: list[RuleEntry]) -> str:
    lines = []
    lines.append(f"| {'Rule ID':<20} | {'Impl':<5} | {'Description':<55} | {'Code Location':<35} | Note |")
    lines.append(f"|{'-'*21}|{'-'*7}|{'-'*57}|{'-'*37}|{'-'*30}|")
    for e in entries:
        impl = "Y" if e.implemented else "N"
        desc = e.description[:53] + ".." if len(e.description) > 55 else e.description
        loc = e.impl_file[:33] + ".." if len(e.impl_file) > 35 else e.impl_file
        note = e.impl_note[:28] + ".." if len(e.impl_note) > 30 else e.impl_note
        lines.append(f"| {e.rule_id:<20} | {impl:<5} | {desc:<55} | {loc:<35} | {note} |")
    return "\n".join(lines)


def coverage_pct(entries: list[RuleEntry]) -> float:
    if not entries:
        return 0.0
    return sum(1 for e in entries if e.implemented) / len(entries) * 100


def print_section(title: str, entries: list[RuleEntry]):
    impl = sum(1 for e in entries if e.implemented)
    total = len(entries)
    pct = coverage_pct(entries)
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"  커버리지: {impl}/{total} ({pct:.1f}%)")
    print(f"{'='*80}")
    for e in entries:
        status = "[Y]" if e.implemented else "[N]"
        print(f"  {status} {e.rule_id:<20} {e.description[:50]:<52} | {e.impl_note[:40]}")


# ──────────────────────────────────────────────────────────────
# Markdown 리포트 생성
# ──────────────────────────────────────────────────────────────
def generate_report(
    a_entries: list[RuleEntry],
    b_entries: list[RuleEntry],
    c_entries: list[RuleEntry],
    d_entries: list[RuleEntry],
) -> str:
    lines = [
        "# Rules ↔ Engine 커버리지 감사 보고서",
        "",
        "> 생성일: 2026-02-23  ",
        "> 분석: `experiments/rules_coverage_audit.py` 텍스트 패턴 분석",
        "",
        "---",
        "",
        "## 1. 요약 (Coverage Summary)",
        "",
        "| Line | 문서 | 규칙 수 | 구현 | 미구현 | 커버리지 |",
        "|------|------|:---:|:---:|:---:|:---:|",
    ]

    for label, entries, doc in [
        ("A (설계기반)", a_entries, "trading_rules_v1~v5.md"),
        ("B (태준수기)", b_entries, "taejun_strategy_review_*.md (notes)"),
        ("C (D2S행동추출)", c_entries, "trading_rules_attach_v1~v2.md"),
        ("D (장기거래)", d_entries, "jun_trade_2023_v1.md"),
    ]:
        impl = sum(1 for e in entries if e.implemented)
        total = len(entries)
        missing = total - impl
        pct = coverage_pct(entries)
        lines.append(f"| {label} | {doc} | {total} | {impl} | {missing} | **{pct:.1f}%** |")

    all_entries = a_entries + b_entries + c_entries + d_entries
    all_impl = sum(1 for e in all_entries if e.implemented)
    all_total = len(all_entries)
    all_pct = coverage_pct(all_entries)
    lines.append(f"| **전체** | — | **{all_total}** | **{all_impl}** | **{all_total - all_impl}** | **{all_pct:.1f}%** |")

    lines += ["", "---", ""]

    # 각 Line 상세
    for label, entries in [
        ("Line A — 설계기반 (v1~v5)", a_entries),
        ("Line B — 태준수기", b_entries),
        ("Line C — D2S 행동추출", c_entries),
        ("Line D — 장기거래", d_entries),
    ]:
        impl = sum(1 for e in entries if e.implemented)
        total = len(entries)
        pct = coverage_pct(entries)
        lines += [
            f"## {label}",
            "",
            f"커버리지: **{impl}/{total} ({pct:.1f}%)**",
            "",
            format_table(entries),
            "",
        ]

    # 갭 분석
    lines += ["---", "", "## 갭 분석 (Gaps)", ""]

    # 미구현 규칙
    not_impl = [e for e in all_entries if not e.implemented]
    if not_impl:
        lines += ["### 규칙은 있으나 코드가 없는 항목 (Rules Without Code)", ""]
        lines.append("| Line | Rule ID | 설명 | 출처 |")
        lines.append("|------|---------|------|------|")
        for e in not_impl:
            # Line 판별
            if e.source_doc.startswith("trading_rules_v"):
                ln = "A"
            elif e.source_doc.startswith("taejun"):
                ln = "B"
            elif e.source_doc.startswith("trading_rules_attach"):
                ln = "C"
            else:
                ln = "D"
            lines.append(f"| {ln} | {e.rule_id} | {e.description[:60]} | {e.source_doc} |")
        lines.append("")

    # 코드는 있으나 rules 문서 없는 항목 (알려진 것)
    lines += [
        "### 코드는 있으나 Rules 문서가 없는 항목 (Code Without Rules)",
        "",
        "| Line | 모듈/파일 | 설명 |",
        "|------|-----------|------|",
        "| B | `line_b_taejun/` 전체 | **FROZEN** 상태 — `docs/rules/line_b/` 규칙서 미존재 |",
        "| D | `line_d_history/` | 엔진 코드 없음 (빈 디렉토리) |",
        "",
        "---",
        "",
        "## 결론 및 권고사항",
        "",
        "### Line A",
        "- v1~v5 규칙의 핵심 로직(CB-1~6, 횡보장, ATR 손절, SOXL 독립)이 `line_b_taejun/`에 구현됨",
        "- `line_a/signals.py`는 DEPRECATED 상태 (v1 5개 규칙만)",
        "- **권고**: v6 규칙서 기반 signals_v6.py 또는 line_b_taejun 완전 대체 확인 필요",
        "",
        "### Line B",
        "- 12개 전략 + 인프라 모듈이 `line_b_taejun/`에 구현됨",
        "- **FROZEN 상태**: `docs/rules/line_b/` 없음 → 코드 수정 금지",
        "- **권고**: VNQ 기반 리뷰 노트(2026-02-23)를 rules 문서로 격상 필요",
        "",
        "### Line C",
        "- D2S 엔진(`d2s_engine.py`)이 attach v1 R1~R16 규칙 대부분 구현",
        "- attach v2 R17(V-바운스), R18(조기 손절)은 `params_d2s.py`에 파라미터 정의",
        "- **권고**: `backtest_d2s_v2.py` 검증 후 D2S_ENGINE_V2 활성화 확인 필요",
        "",
        "### Line D",
        "- `line_d_history/__init__.py`만 존재, 실제 엔진 코드 없음",
        "- 규칙서(jun_trade_2023_v1.md)에 E-1~3, X-1~4, VIX/BTC 레짐 규칙 정의됨",
        "- **권고**: Line D 엔진 구현 또는 Line D 통합 계획 수립 필요",
    ]

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    print("Rules ↔ Engine 커버리지 감사 시작...")
    print(f"  프로젝트 루트: {ROOT}")

    a_entries = audit_line_a()
    b_entries = audit_line_b()
    c_entries = audit_line_c()
    d_entries = audit_line_d()

    print_section("Line A — 설계기반 (v1~v5)", a_entries)
    print_section("Line B — 태준수기", b_entries)
    print_section("Line C — D2S 행동추출", c_entries)
    print_section("Line D — 장기거래", d_entries)

    all_entries = a_entries + b_entries + c_entries + d_entries
    all_impl = sum(1 for e in all_entries if e.implemented)
    all_total = len(all_entries)
    all_pct = coverage_pct(all_entries)

    print(f"\n{'='*80}")
    print(f"  전체 커버리지: {all_impl}/{all_total} ({all_pct:.1f}%)")
    print(f"{'='*80}")

    # 리포트 저장
    report_path = ROOT / "docs/reports/backtest/rules_coverage_summary.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = generate_report(a_entries, b_entries, c_entries, d_entries)
    report_path.write_text(report, encoding="utf-8")
    print(f"\n리포트 저장: {report_path}")


if __name__ == "__main__":
    main()

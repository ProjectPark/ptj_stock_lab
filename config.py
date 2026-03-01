"""
PTJ 매매법 대시보드 - 설정
=========================
한국투자증권 API (실시간) + yfinance (복기용)
"""
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs): pass  # noqa: E731 (SLURM 환경 — dotenv 미설치)
import os

# ============================================================
# 경로
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CHART_DIR = PROJECT_ROOT / "charts"

# ── 데이터 하위 디렉토리 ──────────────────────────────────────
MARKET_DIR    = DATA_DIR / "market"
CACHE_DIR     = MARKET_DIR / "cache"       # 종목별 일봉 캐시 (자동 갱신)
OHLCV_DIR          = MARKET_DIR / "ohlcv"       # 백테스트용 합산 분봉
OHLCV_1MIN_3Y      = MARKET_DIR / "ohlcv" / "backtest_1min_3y.parquet"  # 2023~ (기본)
OHLCV_1MIN_DEFAULT = OHLCV_1MIN_3Y                                       # alias
DAILY_DIR     = MARKET_DIR / "daily"       # 일봉 데이터
FX_DIR        = MARKET_DIR / "fx"          # 환율
POLY_DATA_DIR = DATA_DIR / "polymarket"    # Polymarket 확률 (연도별)
OPTUNA_DIR    = DATA_DIR / "optuna"        # Optuna DB
RESULTS_DIR   = DATA_DIR / "results"       # 출력 결과
META_DIR      = DATA_DIR / "meta"          # 메타데이터

for _d in [DATA_DIR, CHART_DIR, CACHE_DIR, OHLCV_DIR, DAILY_DIR, FX_DIR,
           POLY_DATA_DIR, OPTUNA_DIR, RESULTS_DIR,
           RESULTS_DIR / "backtests", RESULTS_DIR / "events",
           META_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ============================================================
# 한국투자증권 API 인증 (.env 파일에서 로드)
# ============================================================
load_dotenv(PROJECT_ROOT / ".env")

KIS_APP_KEY = os.getenv("KIS_APP_KEY", "")
KIS_APP_SECRET = os.getenv("KIS_APP_SECRET", "")
KIS_ACCOUNT_NO = os.getenv("KIS_ACCOUNT_NO", "")  # 예: "50123456-01"

# 실전투자 URL
KIS_BASE_URL = "https://openapi.koreainvestment.com:9443"

# ============================================================
# 종목 Registry
# exchange: NAS(나스닥), NYS(뉴욕), AMS(아멕스/NYSE Arca)
# ============================================================
TICKERS = {
    # --- 시황 판단: 금 ---
    "GLD":  {"name": "금 ETF",          "category": "gold",               "exchange": "AMS"},

    # --- 쌍둥이 매매: 코인 ---
    "BITU": {"name": "비트코인 2x",     "category": "twin_coin",          "exchange": "AMS"},
    "MSTU": {"name": "스트래티지 2x",   "category": "twin_coin",          "exchange": "AMS"},

    # --- 쌍둥이 매매: 은행/거래소 ---
    "ROBN": {"name": "로빈후드 2x",     "category": "twin_bank",          "exchange": "AMS"},
    "CONL": {"name": "코인베이스 2x",   "category": "twin_bank",          "exchange": "AMS"},

    # --- 쌍둥이 매매: 반도체 ---
    "NVDL": {"name": "엔비디아 2x",     "category": "twin_semi",          "exchange": "AMS"},
    "AMDL": {"name": "AMD 2x",          "category": "twin_semi",          "exchange": "AMS"},

    # --- 조건부 매매 ---
    "ETHU": {"name": "이더리움 2x",     "category": "conditional",        "exchange": "AMS"},
    "XXRP": {"name": "리플 2x",         "category": "conditional",        "exchange": "AMS"},
    "SOLT": {"name": "솔라나 2x",       "category": "conditional",        "exchange": "AMS"},
    "COIN": {"name": "코인베이스",       "category": "conditional_target", "exchange": "NAS"},

    # --- 하락장 매매 ---
    "HIMZ": {"name": "힘스앤허스 2x",   "category": "bearish",            "exchange": "AMS"},
    "BRKU": {"name": "버크셔 2x",       "category": "bearish",            "exchange": "AMS"},
    "BABX": {"name": "알리바바 2x",     "category": "bearish",            "exchange": "AMS"},

    # --- 시장 지표 ---
    "SPY":  {"name": "S&P 500 ETF",     "category": "market",             "exchange": "AMS"},
    "QQQ":  {"name": "나스닥 100 ETF",  "category": "market",             "exchange": "NAS"},
}

# ============================================================
# 쌍둥이 페어 정의  (선행 → 후행)
# ============================================================
TWIN_PAIRS = {
    "coin": {"lead": "BITU", "follow": "MSTU", "label": "코인 (BTC ↔ MSTR)"},
    "bank": {"lead": "ROBN", "follow": "CONL", "label": "은행 (HOOD ↔ COIN)"},
    "semi": {"lead": "NVDL", "follow": "AMDL", "label": "반도체 (NVDA ↔ AMD)"},
}

# ============================================================
# 조건부 매매
# ============================================================
CONDITIONAL_TRIGGERS = ["ETHU", "XXRP", "SOLT"]
CONDITIONAL_TARGET = "COIN"

# ============================================================
# 하락장 매매
# ============================================================
BEARISH_TICKERS = ["HIMZ", "BRKU", "BABX"]
GOLD_TICKER = "GLD"

# ============================================================
# 매매 파라미터
# ============================================================
PAIR_GAP_SELL_THRESHOLD = 0.3     # 페어갭 매도 기준 (%) — 이 이내로 수렴하면 매도
PAIR_GAP_ENTRY_THRESHOLD = 1.5    # 페어갭 진입 기준 (%) — 이 이상 벌어지면 매수 검토
STOP_LOSS_PCT = -3.5              # 손절 라인 (%)  # Optuna v5_s2 best  # was: -3.0
MAX_TRADE_AMOUNT_KRW = 20_000_000  # 최대 거래대금 (원)
SPLIT_BUY_INTERVAL_MIN = 5        # 분할매수 간격 (분)

# ============================================================
# 데이터 수집 설정
# ============================================================
LOOKBACK_PERIOD = "3mo"           # yfinance period 파라미터
CACHE_EXPIRE_HOURS = 1            # parquet 캐시 만료 (시간)
REFRESH_INTERVAL_SEC = 1          # 대시보드 갱신 주기 (초)

# ============================================================
# ▼▼▼  v2 설정 (기존 v1은 그대로 유지)  ▼▼▼
# ============================================================

# PTJ 티커 → Alpaca 티커 매핑 (다른 것만 기재)
ALPACA_TICKER_MAP = {
    "IRE": "IREN",   # PTJ에서 IRE, Alpaca에서 IREN
}
# 역매핑 (Alpaca → PTJ)
ALPACA_TICKER_REVERSE = {v: k for k, v in ALPACA_TICKER_MAP.items()}

# ── v2 종목 Registry ─────────────────────────────────────────
TICKERS_V2 = {
    # --- 시황 판단: 금 ---
    "GLD":  {"name": "금 ETF",          "category": "gold",               "exchange": "AMS"},

    # --- 쌍둥이 매매: 코인 ---
    "BITU": {"name": "비트코인 2x",     "category": "twin_coin",          "exchange": "AMS"},
    "MSTU": {"name": "스트래티지 2x",   "category": "twin_coin",          "exchange": "AMS"},
    "IRE":  {"name": "아이렌 2x",       "category": "twin_coin",          "exchange": "NAS"},  # v2 신규

    # --- 쌍둥이 매매: 은행/거래소 ---
    "ROBN": {"name": "로빈후드 2x",     "category": "twin_bank",          "exchange": "AMS"},
    "CONL": {"name": "코인베이스 2x",   "category": "twin_bank",          "exchange": "NAS"},

    # --- 조건부 매매 ---
    "ETHU": {"name": "이더리움 2x",     "category": "conditional",        "exchange": "AMS"},
    "XXRP": {"name": "리플 2x",         "category": "conditional",        "exchange": "AMS"},
    "SOLT": {"name": "솔라나 2x",       "category": "conditional",        "exchange": "NAS"},
    "COIN": {"name": "코인베이스",       "category": "conditional_target", "exchange": "NAS"},

    # --- 하락장 매매 (HIMZ 삭제) ---
    "BRKU": {"name": "버크셔 2x",       "category": "bearish",            "exchange": "NAS"},

    # --- 시장 지표 ---
    "SPY":  {"name": "S&P 500 ETF",     "category": "market",             "exchange": "AMS"},
    "QQQ":  {"name": "나스닥 100 ETF",  "category": "market",             "exchange": "NAS"},
}

# ── v2 쌍둥이 페어 (코인: 후행 2개, 반도체 폐지) ────────────
TWIN_PAIRS_V2 = {
    "coin":  {"lead": "BITU", "follow": ["MSTU", "IRE"], "label": "코인 (BTC ↔ MSTR/IREN)"},
    "bank":  {"lead": "ROBN", "follow": ["CONL"],        "label": "은행 (HOOD ↔ COIN)"},
}

# ── v2 조건부 매매 (변경 없음) ────────────────────────────────
CONDITIONAL_TRIGGERS_V2 = ["ETHU", "XXRP", "SOLT"]
CONDITIONAL_TARGET_V2 = "COIN"

# ── v2 하락장 방어 (HIMZ 삭제) ────────────────────────────────
BEARISH_TICKERS_V2 = ["BRKU"]

# ── v2 매매 파라미터 (USD 기준, 환전 없음) ─────────────────────
TOTAL_CAPITAL_USD = 15_000             # 총 투자금 $15,000
INITIAL_BUY_USD = 2_250                # 초기 진입 금액 (15%)
DCA_BUY_USD = 750                      # 물타기 1회 금액 (5%)
DCA_DROP_PCT = -1.8                    # 물타기 트리거 하락률 # Optuna v5_s2 best  # was: -1.35
DCA_MAX_COUNT = 7                      # 물타기 최대 횟수
MAX_PER_STOCK_USD = 7_500              # 종목당 최대 투입 (50%)
STOP_LOSS_BULLISH_PCT = -5.5           # 강세장 모드 손절 라인 # Optuna v5_s2 best  # was: -16.0
POLYMARKET_BULLISH_THRESHOLD = 70      # 강세장 모드 진입 기준 (%)
MAX_HOLD_HOURS = 2                     # 기본 시간 손절 (시간)  # Optuna v5_s2 best  # was: 5
TAKE_PROFIT_PCT = 4.5                  # 강세장 연장 시 즉시 매도 기준 (%) # Optuna v5_s2 best  # was: 4.0
BEARISH_DROP_THRESHOLD = -6.0          # 방어주 분할매수 진입 (%)
BEARISH_BUY_DAYS = 5                   # 방어주 분할매수 기간 (일)
BRKU_WEIGHT_PCT = 10                   # BRKU 포트폴리오 고정 비중 (%)

# ── v2 매도/매수 추가 상수 ──────────────────────────────────────
PAIR_GAP_SELL_THRESHOLD_V2 = 9.4     # 쌍둥이 매도 갭 기준 # Optuna v5_s2 best  # was: 9.0
PAIR_SELL_FIRST_PCT = 0.50           # 1차 매도 비율 (50%)  # Optuna v5_s2 best  # was: 0.80
PAIR_SELL_REMAINING_PCT = 0.30       # 잔여분 분할매도 비율 (30%)
PAIR_SELL_INTERVAL_MIN = 5           # 분할매도 간격 (분)
COIN_TRIGGER_PCT = 3.0               # COIN 매수 트리거 (ETHU/XXRP/SOLT 각각 +3%)
COIN_SELL_PROFIT_PCT = 3.5           # COIN 일반 매도 순수익 기준 # Optuna v5_s2 best  # was: 5.0
COIN_SELL_BEARISH_PCT = 0.3          # COIN 하락장 즉시 매도 순수익 기준 (+0.3%)
CONL_TRIGGER_PCT = 3.0               # CONL 매수 트리거 (ETHU/XXRP/SOLT 각각 +3%)
CONL_SELL_PROFIT_PCT = 1.5           # CONL 수익 실현 매도 (순수익 +1.5%)  # Optuna v5_s2 best  # was: 2.8
CONL_SELL_AVG_PCT = 1.0              # CONL 매도 — 트리거 평균 하한 (+1%)
COIN_FOLLOW_VOLATILITY_GAP = 0.5     # MSTU/IRE 선택 변동성 차이 기준 (%)
BEARISH_POLYMARKET_THRESHOLD = 20    # 하락장 진입 — Polymarket 3조건 기준 (%)
CONDITIONAL_START_KST = (17, 30)     # 조건부 매매 시작 시각 (한국시간 오후 5시 30분)

# ── 백테스트 엔진 공용 (KRW 기준) ─────────────────────────────
EXCHANGE_RATE_KRW = 1_350                 # 고정 환율 (KRW/USD)
TOTAL_CAPITAL_KRW = 20_000_000            # 총 투자금 (원)
INITIAL_BUY_KRW = 3_000_000              # 초기 진입 금액 (원)
DCA_BUY_KRW = 1_000_000                  # 물타기 1회 금액 (원)

# ============================================================
# ▼▼▼  v3 설정 (v2 기반 + 선별 매매형)  ▼▼▼
# ============================================================

# ── v3 종목/페어: v2와 동일 ────────────────────────────────────
TICKERS_V3 = TICKERS_V2
TWIN_PAIRS_V3 = TWIN_PAIRS_V2
CONDITIONAL_TRIGGERS_V3 = CONDITIONAL_TRIGGERS_V2
CONDITIONAL_TARGET_V3 = CONDITIONAL_TARGET_V2
BEARISH_TICKERS_V3 = BEARISH_TICKERS_V2

# ── v3 자금 (USD 기준, 환전 없음) ─────────────────────────────
V3_TOTAL_CAPITAL = 15_000             # 총 투자금 $15,000
V3_INITIAL_BUY = 2_250                # 초기 진입 금액 $2,250
V3_DCA_BUY = 750                      # 물타기 1회 금액 $750

# ── v3 매매 파라미터 ──────────────────────────────────────────
V3_PAIR_GAP_ENTRY_THRESHOLD = 2.2     # 쌍둥이 ENTRY 갭 기준: 1.5→2.2%
V3_DCA_MAX_COUNT = 4                  # 물타기 최대 횟수: 7→4회
V3_MAX_PER_STOCK = 5_250              # 종목당 최대 투입 $5,250
V3_MAX_PER_STOCK_KRW = 7_000_000      # 종목당 최대 투입 (원) — 백테스트 엔진용
V3_COIN_TRIGGER_PCT = 4.5             # COIN 매수 트리거: 3.0→4.5%
V3_CONL_TRIGGER_PCT = 4.5             # CONL 매수 트리거: 3.0→4.5%
V3_SPLIT_BUY_INTERVAL_MIN = 20        # 중복 주문 쿨타임: 5→20분

# ── v3 신규 파라미터 ────────────────────────────────────────────
V3_MAX_DAILY_TRADES_PER_STOCK = 1     # 종목당 일일 최대 트레이드
V3_ENTRY_CUTOFF_HOUR = 10             # 매수 마감 시간 (ET, hour)
V3_ENTRY_CUTOFF_MINUTE = 30           # 매수 마감 시간 (ET, minute)
V3_CONDITIONAL_EXEMPT_CUTOFF = True   # 조건부(COIN/CONL) 진입 마감 면제 (KST 17:30 프리마켓부터 가능 → 정규장 전체)

# ── v3 횡보장 감지 파라미터 ─────────────────────────────────────
V3_SIDEWAYS_ENABLED = True            # 횡보장 감지 기능 활성화
V3_SIDEWAYS_MIN_SIGNALS = 3           # 횡보장 판정 최소 충족 지표 수 (5개 중)
V3_SIDEWAYS_EVAL_INTERVAL_MIN = 30    # 횡보장 재평가 간격 (분)
V3_SIDEWAYS_POLY_LOW = 0.40           # 횡보장 Polymarket 확률 하한
V3_SIDEWAYS_POLY_HIGH = 0.60          # 횡보장 Polymarket 확률 상한
V3_SIDEWAYS_GLD_THRESHOLD = 0.3       # 횡보장 GLD |등락률| 기준 (%)
V3_SIDEWAYS_GAP_FAIL_COUNT = 2        # 횡보장 갭 수렴 실패 횟수 기준
V3_SIDEWAYS_TRIGGER_FAIL_COUNT = 2    # 횡보장 트리거 불발 횟수 기준
V3_SIDEWAYS_INDEX_THRESHOLD = 0.5     # 횡보장 SPY/QQQ |등락률| 기준 (%)

# ============================================================
# ▼▼▼  v4 설정 (v3 포팅용 초기값)  ▼▼▼
# ============================================================
#
# 참고:
# - v4/v5 규칙서는 docs/trading_rules_v4.md, docs/trading_rules_v5.md에 정의.
# - 아래 값은 "v3와 동일 구조로 코드/튜닝 파이프라인을 먼저 구동"하기 위한
#   부트스트랩 기본값이다. (구현 완료 후 버전별 로직에 맞춰 조정)

# ── v4 종목/페어: v3와 동일(초기 포팅) ─────────────────────────
TICKERS_V4 = TICKERS_V3
TWIN_PAIRS_V4 = TWIN_PAIRS_V3
CONDITIONAL_TRIGGERS_V4 = CONDITIONAL_TRIGGERS_V3
CONDITIONAL_TARGET_V4 = CONDITIONAL_TARGET_V3
BEARISH_TICKERS_V4 = BEARISH_TICKERS_V3

# ── v4 자금 (USD 기준) ───────────────────────────────────────
V4_TOTAL_CAPITAL = V3_TOTAL_CAPITAL
V4_INITIAL_BUY = V3_INITIAL_BUY
V4_DCA_BUY = V3_DCA_BUY

# ── v4 매매 파라미터 (초기값) ───────────────────────────────────
V4_PAIR_GAP_ENTRY_THRESHOLD = V3_PAIR_GAP_ENTRY_THRESHOLD
V4_DCA_MAX_COUNT = V3_DCA_MAX_COUNT
V4_MAX_PER_STOCK = V3_MAX_PER_STOCK
V4_MAX_PER_STOCK_KRW = V3_MAX_PER_STOCK_KRW
V4_COIN_TRIGGER_PCT = V3_COIN_TRIGGER_PCT
V4_CONL_TRIGGER_PCT = V3_CONL_TRIGGER_PCT
V4_SPLIT_BUY_INTERVAL_MIN = V3_SPLIT_BUY_INTERVAL_MIN

# ── v4 신규 파라미터 (초기값) ───────────────────────────────────
V4_MAX_DAILY_TRADES_PER_STOCK = V3_MAX_DAILY_TRADES_PER_STOCK
V4_ENTRY_CUTOFF_HOUR = 10            # 규칙서 기본 마감 (ET 10:00)
V4_ENTRY_CUTOFF_MINUTE = 0
V4_ENTRY_DEFAULT_START_HOUR = 8      # 기본 매수 시작 시간 (ET 08:00)
V4_ENTRY_DEFAULT_START_MINUTE = 0
V4_ENTRY_EARLY_START_HOUR = 3        # 조기 진입 시작 시간 (ET 03:30)
V4_ENTRY_EARLY_START_MINUTE = 30
V4_ENTRY_TREND_ADX_MIN = 25.0        # 조기 진입 ADX 기준
V4_ENTRY_VOLUME_RATIO_MIN = 1.5      # 조기 진입 거래량 배수 기준
V4_ENTRY_EMA_PERIOD = 20             # 조기 진입 EMA 기간
V4_ENTRY_EMA_SLOPE_LOOKBACK = 5      # 조기 진입 EMA 기울기 계산 lookback
V4_CONDITIONAL_EXEMPT_CUTOFF = V3_CONDITIONAL_EXEMPT_CUTOFF

# ── v4 횡보장 감지 파라미터 (초기값) ─────────────────────────────
V4_SIDEWAYS_ENABLED = V3_SIDEWAYS_ENABLED
V4_SIDEWAYS_MIN_SIGNALS = V3_SIDEWAYS_MIN_SIGNALS
V4_SIDEWAYS_EVAL_INTERVAL_MIN = V3_SIDEWAYS_EVAL_INTERVAL_MIN
V4_SIDEWAYS_POLY_LOW = V3_SIDEWAYS_POLY_LOW
V4_SIDEWAYS_POLY_HIGH = V3_SIDEWAYS_POLY_HIGH
V4_SIDEWAYS_GLD_THRESHOLD = V3_SIDEWAYS_GLD_THRESHOLD
V4_SIDEWAYS_GAP_FAIL_COUNT = V3_SIDEWAYS_GAP_FAIL_COUNT
V4_SIDEWAYS_TRIGGER_FAIL_COUNT = V3_SIDEWAYS_TRIGGER_FAIL_COUNT
V4_SIDEWAYS_INDEX_THRESHOLD = V3_SIDEWAYS_INDEX_THRESHOLD
V4_SIDEWAYS_ATR_DECLINE_PCT = 20.0
V4_SIDEWAYS_VOLUME_DECLINE_PCT = 30.0
V4_SIDEWAYS_EMA_SLOPE_MAX = 0.1
V4_SIDEWAYS_RSI_LOW = 45.0
V4_SIDEWAYS_RSI_HIGH = 55.0
V4_SIDEWAYS_BB_WIDTH_PERCENTILE = 20.0
V4_SIDEWAYS_RANGE_MAX_PCT = 2.0
V4_SIDEWAYS_EMA_SLOPE_LOOKBACK_DAYS = 5
V4_SIDEWAYS_BB_PERCENTILE_WINDOW_DAYS = 60

# ── v4 서킷브레이커 (문서 기준 기본값) ───────────────────────────
V4_CB_VIX_SPIKE_PCT = 3.0            # VIX 급등 기준 # Optuna v4_phase1 best  # was: 6.0
V4_CB_VIX_COOLDOWN_DAYS = 13         # VIX 급등 시 차단 기간 (거래일) # Optuna v4_phase1 best  # was: 7
V4_CB_GLD_SPIKE_PCT = 3.0            # GLD 급등 기준 (+3%)
V4_CB_GLD_COOLDOWN_DAYS = 3          # GLD 급등 시 차단 기간 (거래일)
V4_CB_BTC_CRASH_PCT = -6.0           # BTC 급락 기준 # Optuna v4_study5 best  # was: -5.0
V4_CB_BTC_SURGE_PCT = 13.5           # BTC 급등 기준 # Optuna v4_study5 best  # was: 5.0
V4_CB_RATE_HIKE_PROB_PCT = 50.0      # 금리 상승 우려 기준 (%)
V4_CB_OVERHEAT_PCT = 20.0            # 과열 전환 기준 (+20%)
V4_CB_OVERHEAT_RECOVERY_PCT = -10.0  # 과열 해제 기준 (고점 대비 조정 proxy)
V4_HIGH_VOL_MOVE_PCT = 10.0           # 고변동성 판정 변동률 기준 (|%|)
V4_HIGH_VOL_HIT_COUNT = 2             # 고변동성 판정 누적 횟수
V4_HIGH_VOL_STOP_LOSS_PCT = -4.0      # 고변동성 고정 손절
V4_CONL_ADX_MIN = 10.0                # CONL 조건부 진입 ADX 하한 # Optuna v4_phase1 best  # was: 18.0
V4_CONL_EMA_PERIOD = 20               # CONL EMA 기간
V4_CONL_EMA_SLOPE_LOOKBACK = 5        # EMA 기울기 계산용 lookback bars
V4_CONL_EMA_SLOPE_MIN_PCT = 0.0       # EMA 기울기 하한 (%)
V4_CONL_FIXED_BUY = 6_750             # CONL 조건부 고정 진입 금액 (USD)
V4_CONL_DCA_ENABLED = False           # CONL 물타기 허용 여부 (v4: 금지)
V4_PAIR_FIXED_TP_STOCKS = ["SOXL", "CONL", "IRE"]  # 40/60 고정 익절 대상
V4_PAIR_IMMEDIATE_SELL_PCT = 0.40     # 갭 수렴 즉시 매도 비율
V4_PAIR_FIXED_TP_PCT = 6.5            # 고정 익절 기준 # Optuna v4_study5 best  # was: 5.0
V4_CRASH_BUY_THRESHOLD_PCT = -40.0    # 급락 역매수 트리거
V4_CRASH_BUY_TIME_HOUR = 15           # 급락 역매수 시각 (ET 15:55)
V4_CRASH_BUY_TIME_MINUTE = 55
V4_CRASH_BUY_WEIGHT_PCT = 95.0        # 급락 역매수 비중
V4_CRASH_BUY_STOCKS = ["SOXL", "CONL", "IRE"]
V4_CRASH_BUY_FLAT_BAND_PCT = 0.1      # 다음날 시가 보합 판정 밴드 (%)
V4_CRASH_BUY_FLAT_OBSERVE_MIN = 30    # 보합 시 관찰 시간 (분)
V4_CB_OVERHEAT_SWITCH_MAP = {
    "SOXL": "SOXX",
    "CONL": "COIN",
    "IRE": "IREN",
    "MSTU": None,
}
V4_SWING_TRIGGER_PCT = 27.5          # Optuna v4_study2 best  # was: 15.0
V4_SWING_ELIGIBLE_TICKERS = ["SOXL", "SOXX", "CONL", "COIN", "IRE", "IREN"]
V4_SWING_STAGE1_WEIGHT_PCT = 90.0
V4_SWING_STAGE1_HOLD_DAYS = 21       # Optuna v4_study2 best  # was: 63
V4_SWING_STAGE1_ATR_MULT = 2.5       # Optuna v4_study2 best  # was: 1.5
V4_SWING_STAGE1_DRAWDOWN_PCT = -11.0  # Optuna v4_study2 best  # was: -15.0
V4_SWING_STAGE2_GLD_TICKER = "GLD"
V4_SWING_STAGE2_WEIGHT_PCT = 70.0
V4_SWING_STAGE2_HOLD_DAYS = 105
V4_SWING_STAGE2_STOP_PCT = -5.0
V4_SWING_VIX_STAGE1_WEIGHT_PCT = 80.0
V4_SWING_VIX_STAGE1_HOLD_DAYS = 105
V4_SWING_VIX_STAGE2_COOLDOWN_DAYS = 63

# ============================================================
# ▼▼▼  v5 설정 (v4 포팅용 초기값)  ▼▼▼
# ============================================================

# ── v5 종목/페어 ───────────────────────────────────────────────
TICKERS_V5 = {
    **TICKERS_V4,
    "IAU": {"name": "금 ETF", "category": "defense_core", "exchange": "AMS"},
    "GDXU": {"name": "금광주 3x", "category": "defense_tactical", "exchange": "AMS"},
}
TWIN_PAIRS_V5 = TWIN_PAIRS_V4
CONDITIONAL_TRIGGERS_V5 = CONDITIONAL_TRIGGERS_V4
CONDITIONAL_TARGET_V5 = CONDITIONAL_TARGET_V4
BEARISH_TICKERS_V5 = BEARISH_TICKERS_V4

# ── v5 자금 (USD 기준) ───────────────────────────────────────
V5_TOTAL_CAPITAL = V4_TOTAL_CAPITAL
V5_INITIAL_BUY = 1_000               # Optuna v5_s2 best  # was: V4_INITIAL_BUY (2250)
V5_DCA_BUY = 250                     # Optuna v5_s2 best  # was: V4_DCA_BUY (750)

# ── v5 매매 파라미터 (초기값) ───────────────────────────────────
V5_PAIR_GAP_ENTRY_THRESHOLD = 4.0    # Optuna v5_s2 best  # was: V4_PAIR_GAP_ENTRY_THRESHOLD (2.2)
V5_DCA_MAX_COUNT = 3                 # Optuna v5_s2 best  # was: V4_DCA_MAX_COUNT (4)
V5_MAX_PER_STOCK = 9_750             # Optuna v5_s2 best  # was: V4_MAX_PER_STOCK (5250)
V5_MAX_PER_STOCK_KRW = V4_MAX_PER_STOCK_KRW
V5_COIN_TRIGGER_PCT = 6.5            # Optuna v5_s2 best  # was: V4_COIN_TRIGGER_PCT (4.5)
V5_CONL_TRIGGER_PCT = 6.5            # Optuna v5_s2 best  # was: V4_CONL_TRIGGER_PCT (4.5)
V5_SPLIT_BUY_INTERVAL_MIN = 15      # Optuna v5_s2 best  # was: V4_SPLIT_BUY_INTERVAL_MIN (20)

# ── v5 신규 파라미터 (초기값) ───────────────────────────────────
V5_MAX_DAILY_TRADES_PER_STOCK = V4_MAX_DAILY_TRADES_PER_STOCK
V5_ENTRY_CUTOFF_HOUR = 12            # Optuna v5_s2 best  # was: 10
V5_ENTRY_CUTOFF_MINUTE = 0
V5_ENTRY_DEFAULT_START_HOUR = 8      # 기본 매수 시작 시간 (ET 08:00)
V5_ENTRY_DEFAULT_START_MINUTE = 0
V5_ENTRY_EARLY_START_HOUR = 4        # 조기 진입 시작 시간 (ET 04:00)
V5_ENTRY_EARLY_START_MINUTE = 0
V5_ENTRY_TREND_ADX_MIN = 25.0        # 조기 진입 ADX 기준
V5_ENTRY_VOLUME_RATIO_MIN = 1.5      # 조기 진입 거래량 배수 기준
V5_ENTRY_EMA_PERIOD = 20             # 조기 진입 EMA 기간
V5_ENTRY_EMA_SLOPE_LOOKBACK = 5      # 조기 진입 EMA 기울기 계산 lookback
V5_CONDITIONAL_EXEMPT_CUTOFF = False

# ── v5 횡보장 감지 파라미터 (초기값) ─────────────────────────────
V5_SIDEWAYS_ENABLED = V4_SIDEWAYS_ENABLED
V5_SIDEWAYS_MIN_SIGNALS = 4          # Optuna v5_s2 best  # was: V4_SIDEWAYS_MIN_SIGNALS (3)
V5_SIDEWAYS_EVAL_INTERVAL_MIN = V4_SIDEWAYS_EVAL_INTERVAL_MIN
V5_SIDEWAYS_POLY_LOW = 0.30          # Optuna v5_s2 best  # was: V4_SIDEWAYS_POLY_LOW (0.40)
V5_SIDEWAYS_POLY_HIGH = V4_SIDEWAYS_POLY_HIGH
V5_SIDEWAYS_GLD_THRESHOLD = 0.1     # Optuna v5_s2 best  # was: V4_SIDEWAYS_GLD_THRESHOLD (0.3)
V5_SIDEWAYS_GAP_FAIL_COUNT = V4_SIDEWAYS_GAP_FAIL_COUNT
V5_SIDEWAYS_TRIGGER_FAIL_COUNT = V4_SIDEWAYS_TRIGGER_FAIL_COUNT
V5_SIDEWAYS_INDEX_THRESHOLD = 0.7   # Optuna v5_s2 best  # was: V4_SIDEWAYS_INDEX_THRESHOLD (0.5)

# ── v5 서킷브레이커 (문서 기준 기본값) ───────────────────────────
V5_CB_VIX_SPIKE_PCT = V4_CB_VIX_SPIKE_PCT
V5_CB_VIX_COOLDOWN_DAYS = V4_CB_VIX_COOLDOWN_DAYS
V5_CB_GLD_SPIKE_PCT = 3.5            # Optuna v5_s2 best  # was: V4_CB_GLD_SPIKE_PCT (3.0)
V5_CB_GLD_COOLDOWN_DAYS = V4_CB_GLD_COOLDOWN_DAYS
V5_CB_BTC_CRASH_PCT = -4.5           # Optuna v5_s2 best  # was: V4_CB_BTC_CRASH_PCT (-6.0)
V5_CB_BTC_SURGE_PCT = 4.0            # Optuna v5_s2 best  # was: V4_CB_BTC_SURGE_PCT (13.5)
V5_CB_RATE_HIKE_PROB_PCT = V4_CB_RATE_HIKE_PROB_PCT
V5_CB_OVERHEAT_PCT = V4_CB_OVERHEAT_PCT
V5_CB_OVERHEAT_RECOVERY_PCT = V4_CB_OVERHEAT_RECOVERY_PCT
V5_CB_RATE_LEVERAGE_COOLDOWN_DAYS = 3
V5_HIGH_VOL_MOVE_PCT = 10.0
V5_HIGH_VOL_HIT_COUNT = 2
V5_HIGH_VOL_STOP_LOSS_1X_PCT = -4.0
V5_HIGH_VOL_STOP_LOSS_2X_PCT = -6.0
V5_HIGH_VOL_STOP_LOSS_3X_PCT = -8.0

# ── v5 Unix(VIX) 방어모드 ─────────────────────────────────────
V5_UNIX_DEFENSE_TRIGGER_PCT = 10.0
V5_UNIX_DEFENSE_IAU_WEIGHT_PCT = 40.0
V5_UNIX_DEFENSE_GDXU_WEIGHT_PCT = 30.0
V5_IAU_STOP_LOSS_PCT = -5.0
V5_IAU_BAN_TRIGGER_GDXU_DROP_PCT = -12.0
V5_IAU_BAN_DURATION_TRADING_DAYS = 40
V5_GDXU_HOLD_MIN_DAYS = 2
V5_GDXU_HOLD_MAX_DAYS = 3
V5_UNIX_REBALANCE_HOUR = 15
V5_UNIX_REBALANCE_MINUTE = 55

# ── 자금 유입 (기본값: 비활성) ────────────────────────────────
INJECTION_PCT = 0              # 투입원금 대비 입금 %
INJECTION_INTERVAL_DAYS = 0    # 거래일 간격 (0=비활성)
SIZE_BY_INVESTED = False       # 투입원금 기반 사이징

# CB 공통 대상 목록
CB_OVERHEAT_TICKERS = ["SOXL", "CONL", "IRE", "MSTU"]
LEVERAGED_TICKERS = ["BITU", "MSTU", "IRE", "ROBN", "CONL", "ETHU", "XXRP", "SOLT", "BRKU", "SOXL", "GDXU"]
LEVERAGE_MULTIPLIER = {
    "COIN": 1,
    "GLD": 1,
    "IAU": 1,
    "SPY": 1,
    "QQQ": 1,
    "BITU": 2,
    "MSTU": 2,
    "IRE": 2,
    "ROBN": 2,
    "CONL": 2,
    "ETHU": 2,
    "XXRP": 2,
    "SOLT": 2,
    "BRKU": 2,
    "SOXL": 3,
    "GDXU": 3,
}

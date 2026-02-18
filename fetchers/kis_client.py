"""
한국투자증권 Open API 클라이언트
================================
- OAuth 토큰 자동 발급/갱신
- 해외주식 시세 조회 API 래퍼
- Rate limiting (초당 20건 제한 준수)
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

import requests

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class KISClient:
    """한국투자증권 REST API 클라이언트."""

    def __init__(self):
        self._base_url = config.KIS_BASE_URL
        self._app_key = config.KIS_APP_KEY
        self._app_secret = config.KIS_APP_SECRET
        self._account_no = config.KIS_ACCOUNT_NO

        self._access_token: str = ""
        self._token_expires: datetime = datetime.min

        # Rate limiting: 최소 50ms 간격 (초당 20건)
        self._last_request_time: float = 0
        self._min_interval: float = 0.05

        self._validate_config()

    def _validate_config(self):
        """인증 정보 확인."""
        missing = []
        if not self._app_key:
            missing.append("KIS_APP_KEY")
        if not self._app_secret:
            missing.append("KIS_APP_SECRET")
        if not self._account_no:
            missing.append("KIS_ACCOUNT_NO")
        if missing:
            raise ValueError(
                f"ptj/.env 파일에 다음 항목을 입력하세요: {', '.join(missing)}\n"
                f"파일 위치: {config.PROJECT_ROOT / '.env'}"
            )

    # ── 토큰 관리 ────────────────────────────────────────────

    def _get_token(self) -> str:
        """OAuth 접근 토큰 발급. 유효하면 캐시 사용."""
        if self._access_token and datetime.now() < self._token_expires:
            return self._access_token

        url = f"{self._base_url}/oauth2/tokenP"
        body = {
            "grant_type": "client_credentials",
            "appkey": self._app_key,
            "appsecret": self._app_secret,
        }

        log.info("KIS 토큰 발급 요청...")
        resp = requests.post(url, json=body, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        self._access_token = data["access_token"]
        # 토큰 유효기간: 발급 후 약 24시간, 안전 마진 23시간
        self._token_expires = datetime.now() + timedelta(hours=23)
        log.info("KIS 토큰 발급 완료 (만료: %s)", self._token_expires.strftime("%H:%M:%S"))
        return self._access_token

    # ── HTTP 요청 ────────────────────────────────────────────

    def _throttle(self):
        """Rate limiting — 요청 간격 보장."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def _headers(self, tr_id: str) -> dict:
        """공통 요청 헤더."""
        token = self._get_token()
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {token}",
            "appkey": self._app_key,
            "appsecret": self._app_secret,
            "tr_id": tr_id,
        }

    def _get(self, path: str, tr_id: str, params: dict) -> dict:
        """GET 요청 + rate limiting."""
        self._throttle()
        url = f"{self._base_url}{path}"
        resp = requests.get(url, headers=self._headers(tr_id), params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # API 에러 체크
        rt_cd = data.get("rt_cd")
        if rt_cd and rt_cd != "0":
            msg = data.get("msg1", "Unknown error")
            log.error("KIS API 오류 [%s]: %s", rt_cd, msg)
            raise RuntimeError(f"KIS API error: {msg}")

        return data

    # ── 해외주식 현재가 ──────────────────────────────────────

    def get_overseas_price(self, symbol: str, exchange: str) -> dict:
        """해외주식 현재가 조회.

        Args:
            symbol: 종목코드 (예: "AAPL")
            exchange: 거래소코드 (NAS, NYS, AMS)

        Returns:
            {"symbol", "price", "open", "high", "low", "prev_close",
             "change_pct", "volume", "trade_date"}
        """
        data = self._get(
            path="/uapi/overseas-price/v1/quotations/price",
            tr_id="HHDFS00000300",
            params={
                "AUTH": "",
                "EXCD": exchange,
                "SYMB": symbol,
            },
        )

        output = data.get("output", {})
        last = _to_float(output.get("last", "0"))
        prev = _to_float(output.get("base", "0"))  # 전일 종가
        opn = _to_float(output.get("open", "0"))
        high = _to_float(output.get("high", "0"))
        low = _to_float(output.get("low", "0"))
        vol = _to_int(output.get("tvol", "0"))

        change_pct = 0.0
        if prev > 0:
            change_pct = round((last - prev) / prev * 100, 2)

        return {
            "symbol": symbol,
            "price": last,
            "open": opn,
            "high": high,
            "low": low,
            "prev_close": prev,
            "change_pct": change_pct,
            "volume": vol,
            "trade_date": output.get("ordy", ""),
        }

    # ── 해외주식 분봉 ────────────────────────────────────────

    def get_overseas_minutes(
        self, symbol: str, exchange: str, nmin: int = 1
    ) -> list[dict]:
        """해외주식 분봉 조회 (당일).

        Args:
            symbol: 종목코드
            exchange: 거래소코드
            nmin: 분봉 단위 (1, 5, 15, 30, 60)

        Returns:
            [{"time", "open", "high", "low", "close", "volume"}, ...]
        """
        # 분봉 단위 → API nmin 코드
        nmin_map = {1: "1", 5: "5", 15: "15", 30: "30", 60: "60"}
        nmin_str = nmin_map.get(nmin, "1")

        data = self._get(
            path="/uapi/overseas-price/v1/quotations/inquire-time-itemchartprice",
            tr_id="HHDFS76950200",
            params={
                "AUTH": "",
                "EXCD": exchange,
                "SYMB": symbol,
                "NMIN": nmin_str,
                "PINC": "1",
                "NEXT": "",
                "NREC": "120",
                "FILL": "",
                "KEYB": "",
            },
        )

        rows = []
        for item in data.get("output2", []):
            t = item.get("xymd", "") + item.get("xhms", "")
            rows.append({
                "time": t,
                "open": _to_float(item.get("open", "0")),
                "high": _to_float(item.get("high", "0")),
                "low": _to_float(item.get("low", "0")),
                "close": _to_float(item.get("clos", "0")),
                "volume": _to_int(item.get("tvol", "0")),
            })

        return rows

    # ── 해외주식 기간별 시세 (일봉) ──────────────────────────

    def get_overseas_daily(
        self, symbol: str, exchange: str, period: str = "D", count: int = 90
    ) -> list[dict]:
        """해외주식 기간별 시세 조회.

        Args:
            symbol: 종목코드
            exchange: 거래소코드
            period: "D"(일), "W"(주), "M"(월)
            count: 조회 건수

        Returns:
            [{"date", "open", "high", "low", "close", "volume"}, ...]
        """
        today = datetime.now().strftime("%Y%m%d")

        data = self._get(
            path="/uapi/overseas-price/v1/quotations/dailyprice",
            tr_id="HHDFS76240000",
            params={
                "AUTH": "",
                "EXCD": exchange,
                "SYMB": symbol,
                "GUBN": "0",  # 0: 일, 1: 주, 2: 월
                "BYMD": today,
                "MODP": "1",  # 수정주가 적용
            },
        )

        rows = []
        for item in data.get("output2", []):
            rows.append({
                "date": item.get("xymd", ""),
                "open": _to_float(item.get("open", "0")),
                "high": _to_float(item.get("high", "0")),
                "low": _to_float(item.get("low", "0")),
                "close": _to_float(item.get("clos", "0")),
                "volume": _to_int(item.get("tvol", "0")),
            })

        return rows[:count]


# ── 유틸리티 ─────────────────────────────────────────────────

def _to_float(val: str) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _to_int(val: str) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0

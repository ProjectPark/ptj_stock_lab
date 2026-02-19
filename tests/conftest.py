"""
PTJ 매매법 — 테스트 공통 fixture
================================
v1~v4 엔진 테스트에 사용하는 공유 fixture.
테스트 기간: 2026-01-02 ~ 2026-01-30 (1월 한 달, 데이터 있는 범위)
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

# 프로젝트 루트를 path에 추가
_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT), str(_ROOT / "backtests"), str(_ROOT / "strategies")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ============================================================
# 테스트 기간 (2026년 1월, 데이터 존재 범위)
# ============================================================
TEST_START = date(2026, 1, 2)
TEST_END = date(2026, 1, 30)


@pytest.fixture(scope="session")
def test_period():
    """테스트 기간 (start, end) 튜플."""
    return TEST_START, TEST_END

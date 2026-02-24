"""
엔진 이식: {엔진명}
소스: ptj_stock_lab/simulation/strategies/{경로}
대상: ptj_stock/backend/app/config.py
생성일: {날짜}

[삽입 위치]
  - Settings 클래스 내부에 필드 추가
  - 프리셋 dict에 해당 필드 추가 (있으면)
"""

from __future__ import annotations


# ── Settings 클래스에 아래 필드 추가 ────────────────────────
#
# # {엔진명} 파라미터
# xxx_threshold: float = 2.0    # 출처: lab V5_XXX_THRESHOLD
# xxx_enabled: bool = True      # 엔진 활성화 여부
#

# ── 프리셋에 아래 키 추가 (해당 시) ─────────────────────────
#
# PRESETS = {
#     "default": {
#         "xxx_threshold": 2.0,
#         "xxx_enabled": True,
#     },
# }

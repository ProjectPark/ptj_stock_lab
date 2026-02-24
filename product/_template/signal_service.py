"""
엔진 이식: {엔진명}
소스: ptj_stock_lab/simulation/strategies/{경로}
대상: ptj_stock/backend/app/services/signal_service.py
생성일: {날짜}

[삽입 위치]
  - compute_signals() 함수 내부에 파라미터 주입 코드 추가
"""

from __future__ import annotations


# ── compute_signals() 내부에 아래 블록 추가 ─────────────────
#
# # {엔진명} 파라미터 주입
# xxx_params = {
#     "key": settings.xxx_key,  # Settings 필드에서 읽기
# }
# # generate_all_signals() 호출 시 전달
# signals = generate_all_signals(
#     changes=changes,
#     xxx_params=xxx_params,  # ← 추가
# )

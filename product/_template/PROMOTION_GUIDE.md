# Product → ptj_stock 복사 가이드

## 엔진: {engine_name}
## 생성일: {날짜}
## 상태: draft

---

### 복사 순서

| # | 이 폴더 파일 | ptj_stock 대상 | 삽입 위치 |
|---|-------------|----------------|----------|
| 1 | `signals.py` | `backend/app/core/signals.py` | 상수 영역, 함수 영역, generate_all result dict |
| 2 | `config.py` | `backend/app/config.py` | Settings 클래스 |
| 3 | `signal_service.py` | `backend/app/services/signal_service.py` | compute_signals() 내부 |
| 4 | `auto_trader.py` | `backend/app/services/auto_trader.py` | 엔진 함수, evaluate_and_execute() |
| 5 | `execution_layer.py` | `backend/app/services/execution_layer.py` | 새 파일 또는 기존에 추가 |

### 파라미터 매핑

| Lab (config.py) | product/ 상수 | ptj_stock Settings 필드 |
|-----------------|--------------|----------------------|
| TODO | TODO | TODO |

### 검증

```bash
cd /Users/taehyunpark/project/ptj_stock
python -c "from backend.app.core.signals import generate_all_signals; print('OK')"
```

### 배포

```bash
ssh iMac "cd /path/to/ptj_stock && git pull && docker compose up -d --build"
```

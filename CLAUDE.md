# CLAUDE.md — ptj_stock_lab (실험/시뮬레이션 전용)

이 repo는 **실험/시뮬레이션 전용**입니다. 서빙/배포 코드는 `ptj_stock` repo에서 관리합니다.

## 관련 Repo
- **ptj_stock**: 배포 서버용 — backend, frontend, Docker
- **ptj_stock_lab** (여기): 실험용 — backtest, optimize, signals 연구

## 엔진 프로모션 워크플로우
1. 이 repo에서 새 전략 개발 & backtest
2. Optuna 최적화로 파라미터 확정
3. `strategies/signals_vN.py`를 파라미터 주입 방식으로 정리
4. `ptj_stock/backend/app/core/signals.py`에 복사 & 커밋
5. `ptj_stock`에서 `make dev`로 로컬 테스트
6. `git push` → iMac에 배포

## Python 환경
- pyenv 환경: `pyenv shell market`
- Python 실행 시 항상 `pyenv shell market && python script.py` 형태로 사용

## 구조

```
ptj_stock_lab/
├── strategies/          # 시그널 엔진 (v1~v5)
├── backtests/           # 백테스트 스크립트
├── optimizers/          # Optuna 최적화
├── experiments/         # 실험, 분석, compliance 평가
├── fetchers/            # 데이터 수집 (KIS, Polygon, Polymarket)
├── polymarket/          # Polymarket 연동 모듈
├── data/                # 데이터 (gitignore)
│   ├── parquet/         # 종목별 시세
│   ├── optuna/          # Optuna DB
│   └── results/         # 백테스트 결과, JSON
├── history/             # 거래내역
│   ├── 2024/            # 2024 거래내역 CSV
│   ├── 2025/            # 2025 거래내역
│   └── tools/           # PDF→CSV 변환 도구
├── docs/                # 리포트, 트레이딩 룰
├── scripts/             # 유틸리티 스크립트
├── config.py            # 공통 설정
├── dashboard.py         # Streamlit 대시보드
├── app.py               # Legacy Streamlit app
└── run.py               # 실행 진입점
```

## 실행 예시

### 백테스트
```bash
pyenv shell market && python backtests/backtest_v5.py
```

### Optuna 최적화
```bash
pyenv shell market && python optimizers/optimize_v5_optuna.py
```

### 대시보드
```bash
pyenv shell market && streamlit run dashboard.py
```

## 전략 버전 히스토리
| 버전 | 파일 | 핵심 변경 |
|---|---|---|
| v1 | signals.py | 기본 5개 규칙 |
| v2 | signals_v2.py | 파라미터화, 임계값 조정 |
| v3 | signals_v3.py | Optuna 최적화 반영 |
| v4 | signals_v4.py | 스윙 이벤트, CB 감지 추가 |
| v5 | signals_v5.py | 횡보장 감지, 진입 시간 제한 |

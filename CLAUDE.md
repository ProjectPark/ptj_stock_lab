# CLAUDE.md — ptj_stock_lab (실험/시뮬레이션 전용)

이 repo는 **실험/시뮬레이션 전용**입니다. 서빙/배포 코드는 `ptj_stock` repo에서 관리합니다.

## 관련 Repo
- **ptj_stock**: 배포 서버용 — backend, frontend, Docker
- **ptj_stock_lab** (여기): 실험용 — backtest, optimize, signals 연구

## 엔진 프로모션 워크플로우
1. 이 repo에서 새 전략 개발 & backtest
2. Optuna 최적화로 파라미터 확정
3. `simulation/strategies/signals_vN.py`를 파라미터 주입 방식으로 정리
4. `ptj_stock/backend/app/core/signals.py`에 복사 & 커밋
5. `ptj_stock`에서 `make dev`로 로컬 테스트
6. `git push` → iMac에 배포

## Python 환경
- pyenv 환경: `pyenv shell ptj_stock_lab`
- Python 실행 시 항상 `pyenv shell ptj_stock_lab && python script.py` 형태로 사용

## 구조

```
ptj_stock_lab/
├── simulation/          # 시뮬레이션 코어 패키지
│   ├── strategies/      # 시그널 엔진 (v1~v5)
│   ├── backtests/       # 백테스트 스크립트
│   ├── optimizers/      # Optuna 최적화
│   └── pipeline.py      # 통합 파이프라인 진입점
├── experiments/         # 실험, 분석, compliance 평가
├── fetchers/            # 데이터 수집 (KIS, Polygon, Polymarket)
├── polymarket/          # Polymarket 연동 모듈
├── data/                # 데이터 (gitignore)
│   ├── market/          # 시장 데이터
│   │   ├── cache/       # 종목별 일봉 캐시 (자동 갱신)
│   │   ├── ohlcv/       # 백테스트용 합산 분봉
│   │   ├── daily/       # 일봉 데이터
│   │   └── fx/          # 환율
│   ├── polymarket/      # Polymarket 확률 (연도별)
│   │   ├── 2024/
│   │   ├── 2025/
│   │   └── 2026/
│   ├── optuna/          # Optuna DB
│   ├── results/         # 출력 결과
│   │   ├── backtests/   # 백테스트 트레이드 로그
│   │   ├── optimization/# 최적화 결과
│   │   ├── analysis/    # 분석/compliance
│   │   ├── baselines/   # 기준 성과 JSON
│   │   └── events/      # 이벤트 로그
│   └── meta/            # 메타데이터 (ticker_mapping 등)
├── history/             # 거래내역
│   ├── 2024/            # 2024 거래내역 CSV
│   ├── 2025/            # 2025 거래내역
│   └── tools/           # PDF→CSV 변환 도구
├── docs/                # 문서 (아래 구조 참조)
│   ├── reports/         # 리포트
│   │   ├── backtest/    # 백테스트 결과
│   │   ├── optuna/      # Optuna 최적화
│   │   ├── stoploss/    # 손절 분석
│   │   └── baseline/    # 베이스라인 성능
│   ├── rules/           # 트레이딩 룰 v1~v5
│   ├── notes/           # 전략 아이디어, 메모
│   ├── architecture/    # 설계 문서
│   ├── templates/       # 문서 템플릿
│   ├── pdf/             # PDF 산출물
│   ├── charts/          # 차트/시각화
│   └── scripts/         # PDF 생성 스크립트
├── scripts/             # 유틸리티 스크립트
├── config.py            # 공통 설정
├── dashboard.py         # Streamlit 대시보드
├── app.py               # Legacy Streamlit app
└── run.py               # 실행 진입점 (대시보드)
```

## 실행 예시

### 백테스트
```bash
pyenv shell ptj_stock_lab && python simulation/backtests/backtest_v5.py
```

### Optuna 최적화
```bash
pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_v5_optuna.py
```

### 대시보드
```bash
pyenv shell ptj_stock_lab && streamlit run dashboard.py
```

## docs 폴더 규칙 (필수 준수)

**문서를 생성하거나 저장할 때 반드시 아래 규칙을 따른다. docs/ 루트에 파일을 직접 생성하지 않는다.**

### 파일 저장 위치 (강제)

| 파일 유형 | 저장 위치 | 절대 안 되는 곳 |
|----------|----------|---------------|
| 트레이딩 룰 (`.md`) | `docs/rules/` | docs/ 루트 |
| 백테스트 리포트 (`.md`) | `docs/reports/backtest/` | docs/ 루트 |
| Optuna 리포트 (`.md`) | `docs/reports/optuna/` | docs/ 루트 |
| 손절 리포트 (`.md`) | `docs/reports/stoploss/` | docs/ 루트 |
| 베이스라인 리포트 (`.md`) | `docs/reports/baseline/` | docs/ 루트 |
| 전략 메모/아이디어 (`.md`, `.csv`) | `docs/notes/` | docs/ 루트 |
| 설계/아키텍처 문서 (`.md`) | `docs/architecture/` | docs/ 루트 |
| PDF 산출물 (`.pdf`) | `docs/pdf/` | docs/ 루트, 다른 하위폴더 |
| 차트/이미지 (`.png`, `.html`) | `docs/charts/` | docs/ 루트 |
| PDF 생성 스크립트 (`.py`) | `docs/scripts/` | docs/ 루트 |
| 문서 템플릿 | `docs/templates/` | - |

### 파일 네이밍 규칙 (강제)

| 유형 | 패턴 | 예시 |
|------|------|------|
| 트레이딩 룰 | `trading_rules_v{N}.md` | `trading_rules_v6.md` |
| 백테스트 리포트 | `backtest_v{N}_report.md` | `backtest_v6_report.md` |
| Optuna 리포트 | `v{N}_optuna_report.md` | `v6_optuna_report.md` |
| 베이스라인 리포트 | `v{N}_baseline_report.md` | `v6_baseline_report.md` |
| 전략 노트 | `{주제}_{YYYY-MM-DD}.md` | `swing_strategy_2026-02-20.md` |
| 손절 리포트 | `stoploss_{주제}.md` | `stoploss_atr_comparison.md` |

### 새 문서 생성 시 체크리스트

1. **저장 위치 확인**: 위 테이블에서 해당 유형의 폴더 확인
2. **네이밍 확인**: 위 패턴에 맞는 파일명 사용
3. **템플릿 참조**: `docs/templates/`에서 해당 템플릿을 읽고 구조 따르기
4. **이전 버전 참조**: 버전이 있는 문서는 이전 버전을 먼저 읽기

### 커스텀 에이전트

- `docs-writer` 에이전트: 매매 전략 문서를 대화형으로 작성
- 사용법: Task 도구로 `docs-writer` 에이전트를 호출하거나, 직접 대화하면서 문서 작성

### 템플릿 (`docs/templates/`)

| 템플릿 | 용도 | 저장 위치 |
|--------|------|----------|
| `trading_rules.md` | 트레이딩 룰 새 버전 | `docs/rules/` |
| `backtest_report.md` | 백테스트 결과 리포트 | `docs/reports/backtest/` |
| `strategy_notes.md` | 전략 아이디어/메모 정리 | `docs/notes/` |

### 문서 작성 규칙

- 조건/규칙은 **테이블** 형식 우선
- 파라미터는 **구체적 수치** 필수 (애매한 표현 금지)
- 정정사항은 최종본에 반영 + 정정 이력 별도 기록
- 새 버전 작성 시 이전 버전의 `변경 요약` 테이블 필수 포함

## 전략 버전 히스토리
| 버전 | 파일 | 핵심 변경 |
|---|---|---|
| v1 | simulation/strategies/signals.py | 기본 5개 규칙 |
| v2 | simulation/strategies/signals_v2.py | 파라미터화, 임계값 조정 |
| v3 | simulation/strategies/signals_v3.py | Optuna 최적화 반영 |
| v4 | simulation/strategies/signals_v4.py | 스윙 이벤트, CB 감지 추가 |
| v5 | simulation/strategies/signals_v5.py | 횡보장 감지, 진입 시간 제한 |

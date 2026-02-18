# PTJ 매매 시그널 시스템 — 아키텍처 문서

## 1. 시스템 개요

PTJ 매매 시그널 시스템은 한국투자증권(KIS) OpenAPI를 통해 미국 주식 실시간 시세를 수집하고, 사전 정의된 매매 규칙에 따라 시그널을 생성하여 웹 대시보드로 제공하는 시스템입니다.

```
┌─────────────────────────────────────────────────────────────┐
│                        사용자 브라우저                        │
│                     http://localhost:3000                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  frontend (Nginx + React SPA)              :3000 → 80       │
│  ├── 정적 파일 서빙 (JS/CSS/HTML)                             │
│  └── /api/* → 리버스 프록시 ──────┐                           │
└───────────────────────────────────┼──────────────────────────┘
                                    │
                    ┌───────────────┘
                    ▼
┌──────────────────────────────────────────────────────────────┐
│  backend (FastAPI + Uvicorn)               :8000             │
│  ├── KIS API 시세 수집 (1초 간격)                              │
│  ├── 5개 매매 시그널 계산                                      │
│  ├── Redis 캐시 + Pub/Sub                                    │
│  └── SSE 스트리밍 → 프론트엔드                                 │
└──────┬────────────────────┬──────────────────────────────────┘
       │                    │
       ▼                    ▼
┌──────────────┐  ┌──────────────────────────────────────────┐
│  redis       │  │  postgres (PostgreSQL 16)    :5432       │
│  :6379       │  │  ├── review_sessions  (복기 세션)          │
│  ├── 시세 캐시 │  │  ├── review_snapshots (시점별 스냅샷)     │
│  └── pub/sub │  │  ├── daily_ohlcv      (일봉 캐시)         │
└──────────────┘  │  ├── signal_events    (시그널 이력)        │
                  │  └── trade_log        (매매 기록)          │
                  └──────────────────────┬───────────────────┘
                                         │
                    ┌────────────────────┘
                    ▼
┌──────────────────────────────────────────────────────────────┐
│  review (FastAPI + Uvicorn)                :8010 → 8001      │
│  ├── 복기 세션 CRUD                                           │
│  ├── yfinance 기반 날짜별 시그널 리플레이                       │
│  └── 기간 백테스트                                             │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. 컨테이너 구성

### 2.1 전체 목록

| 컨테이너 | 이미지 | 포트 (호스트:컨테이너) | 역할 | 헬스체크 |
|----------|--------|----------------------|------|---------|
| **frontend** | `ptj_stock-frontend` (Nginx Alpine) | `3000:80` | React SPA 서빙 + API 리버스 프록시 | HTTP 200 체크 |
| **backend** | `ptj_stock-backend` (Python 3.12) | `8000:8000` | 시세 수집 + 시그널 계산 + SSE 스트리밍 | `/api/health` 엔드포인트 |
| **review** | `ptj_stock-review` (Python 3.12) | `8010:8001` | 복기/백테스트 서비스 | `/docs` 엔드포인트 |
| **redis** | `redis:7-alpine` | `6379:6379` | 실시간 시세 캐시 + pub/sub | `redis-cli ping` → PONG |
| **postgres** | `postgres:16-alpine` | `5432:5432` | 영속 데이터 저장소 | `pg_isready -U ptj` |

### 2.2 컨테이너 의존 관계

```
redis (healthy) ──┐
                  ├──▶ backend ──▶ frontend
postgres (healthy)┤
                  └──▶ review
```

- `backend`는 redis와 postgres가 healthy 상태일 때만 시작
- `frontend`는 backend가 시작된 후 시작
- `review`는 postgres가 healthy 상태일 때만 시작

---

## 3. 컨테이너 상세

### 3.1 Frontend

| 항목 | 내용 |
|------|------|
| **기술 스택** | React 18 + TypeScript + Vite + TradingView Lightweight Charts |
| **운영 서버** | Nginx Alpine (gzip 압축, SPA 폴백, 정적 파일 1년 캐시) |
| **빌드** | Multi-stage Docker (node:20-alpine → nginx:alpine) |
| **리버스 프록시** | `/api/*` → `http://backend:8000` (SSE 지원: buffering off) |

**주요 기능:**
- 6개 페이지: 전체 요약, 시황, 쌍둥이, 조건부, 하락장, 전체 종목
- SSE(Server-Sent Events) 기반 실시간 시세 업데이트 (자동 재연결)
- Market Pulse 상태바: GLD/SPY/QQQ 시세 + KST/ET/PT 실시간 시계
- 접이식 왼쪽 사이드바 (매매 규칙 참조)
- 모바일 반응형 (480px/768px 이중 브레이크포인트)

### 3.2 Backend

| 항목 | 내용 |
|------|------|
| **기술 스택** | FastAPI + Uvicorn + httpx (비동기) |
| **데이터 소스** | 한국투자증권 KIS OpenAPI (해외주식 시세) |
| **인증** | `.env`의 `KIS_APP_KEY`, `KIS_APP_SECRET`, `KIS_ACCOUNT_NO` |

**핵심 모듈:**

| 모듈 | 경로 | 역할 |
|------|------|------|
| `kis_client.py` | `app/core/` | httpx 비동기 KIS API 클라이언트 (토큰 발급/갱신) |
| `fetcher.py` | `app/core/` | 전체 종목 시세 수집 (dict 반환, pandas 미사용) |
| `signals.py` | `app/core/` | 5개 시그널 함수 (파라미터화, config 미의존) |
| `price_service.py` | `app/services/` | 백그라운드 fetch loop + Redis 캐시 + SSE broadcast |
| `config.py` | `app/` | Pydantic BaseSettings (환경변수 + .env) |

**5개 매매 시그널:**

| 시그널 | 설명 |
|--------|------|
| `gold` | GLD 양전 시 매매금지 경고 |
| `twin_pairs` | 쌍둥이 ETF 페어 갭 기반 매수/매도 |
| `conditional` | ETHU+XXRP+SOLT 3종목 양전 → COIN 매수 |
| `stop_loss` | -3% 손절라인 도달 감지 |
| `bearish` | 하락장 방어주 (HIMZ, BRKU, BABX) 추천 |

### 3.3 Review

| 항목 | 내용 |
|------|------|
| **기술 스택** | FastAPI + Uvicorn + yfinance + SQLAlchemy |
| **데이터 소스** | yfinance (과거 주가 데이터) |
| **저장소** | PostgreSQL (세션, 스냅샷, 일봉 캐시) |

**주요 기능:**
- 복기 세션 생성/조회/삭제
- 특정 날짜의 시세를 재현하여 시그널 리플레이
- 기간 설정 백테스트 (전략 검증)

### 3.4 Redis

| 항목 | 내용 |
|------|------|
| **이미지** | `redis:7-alpine` |
| **용도** | 실시간 시세 캐시 + pub/sub 메시지 브로커 |
| **볼륨** | `redis-data` (영속화) |

**데이터 구조:**

| Key/Channel | 용도 |
|-------------|------|
| `ptj:latest` | 최신 시세 스냅샷 (JSON) |
| `prices:updates` | 시세 변경 pub/sub 채널 |

### 3.5 PostgreSQL

| 항목 | 내용 |
|------|------|
| **이미지** | `postgres:16-alpine` |
| **데이터베이스** | `ptj` (사용자: `ptj`) |
| **볼륨** | `postgres-data` (영속화) |
| **초기화** | `init.sql` 자동 실행 (컨테이너 최초 생성 시) |

**테이블 스키마:**

| 테이블 | 용도 | 주요 컬럼 |
|--------|------|----------|
| `review_sessions` | 복기 세션 | id, session_date, notes, status |
| `review_snapshots` | 시점별 스냅샷 | session_id, snapshot_time, tickers_json, signals_json |
| `daily_ohlcv` | 일봉 캐시 | symbol, trade_date, open/high/low/close, volume |
| `signal_events` | 시그널 이력 | event_time, signal_type, signal_data, is_actionable |
| `trade_log` | 매매 기록 | trade_time, symbol, action, price, quantity, amount_krw |

---

## 4. API 엔드포인트

### 4.1 Backend (:8000)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/api/health` | 서버 상태 확인 |
| `GET` | `/api/prices` | 전체 종목 스냅샷 |
| `GET` | `/api/prices/{symbol}` | 개별 종목 시세 |
| `GET` | `/api/signals` | 전체 시그널 |
| `GET` | `/api/signals/{type}` | 개별 시그널 (gold, twin_pairs, conditional, stop_loss, bearish) |
| `GET` | `/api/stream/prices` | SSE 실시간 스트리밍 |
| `GET` | `/api/config/parameters` | 매매 파라미터 조회 |
| `PUT` | `/api/config/parameters` | 매매 파라미터 수정 |
| `GET` | `/api/config/tickers` | 종목 레지스트리 조회 |

### 4.2 Review (:8010)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/api/sessions` | 복기 세션 목록 |
| `POST` | `/api/sessions` | 세션 생성 |
| `GET` | `/api/sessions/{id}` | 세션 조회 |
| `DELETE` | `/api/sessions/{id}` | 세션 삭제 |
| `POST` | `/api/replay/{date}` | 날짜별 리플레이 |
| `POST` | `/api/backtest` | 기간 백테스트 |

---

## 5. 데이터 흐름

### 5.1 실시간 시세 흐름

```
KIS OpenAPI ─── HTTP GET (1초 간격) ───▶ backend (fetcher.py)
                                              │
                                    시그널 계산 (signals.py)
                                              │
                                 ┌────────────┴────────────┐
                                 ▼                         ▼
                          Redis SET                  Redis PUBLISH
                        ptj:latest               prices:updates
                                                       │
                                                       ▼
                                                SSE broadcast
                                                       │
                                                       ▼
                                              frontend (React)
                                              useSSE() hook
```

### 5.2 복기/백테스트 흐름

```
사용자 요청 ──▶ review API
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
  yfinance (과거 데이터)   PostgreSQL (캐시)
        │                     │
        └──────────┬──────────┘
                   ▼
            시그널 재계산
                   │
                   ▼
            결과 반환 (JSON)
```

---

## 6. 헬스체크

### 6.1 수동 점검 명령

```bash
# 전체 컨테이너 상태
make ps

# 개별 서비스 확인
curl http://localhost:3000/                    # frontend → 200
curl http://localhost:8000/api/health           # backend  → {"status":"ok"}
curl http://localhost:8010/docs                 # review   → 200
docker exec ptj_stock-redis-1 redis-cli ping   # redis    → PONG
docker exec ptj_stock-postgres-1 pg_isready -U ptj  # postgres → accepting connections
```

### 6.2 Docker 내장 헬스체크

| 컨테이너 | 체크 방식 | 간격 | 타임아웃 | 재시도 |
|----------|----------|------|---------|--------|
| redis | `redis-cli ping` | 10초 | 5초 | 3회 |
| postgres | `pg_isready -U ptj` | 10초 | 5초 | 3회 |

### 6.3 로그 확인

```bash
make logs                                      # 전체 로그 (실시간)
docker compose logs backend --tail 20          # backend 최근 20줄
docker compose logs review --tail 20           # review 최근 20줄
docker compose logs frontend --tail 20         # frontend (Nginx 접근 로그)
```

---

## 7. 운영 명령어

### 7.1 Makefile 명령

| 명령 | 설명 |
|------|------|
| `make up` | 전체 시스템 기동 (백그라운드) |
| `make down` | 전체 시스템 종료 |
| `make logs` | 전체 로그 실시간 확인 |
| `make build` | 전체 이미지 빌드 |
| `make clean` | 볼륨 + 이미지 삭제 (초기화) |
| `make dev` | 개발 모드 기동 (hot reload) |
| `make ps` | 컨테이너 상태 확인 |
| `make redis-cli` | Redis CLI 접속 |
| `make db` | PostgreSQL psql 접속 |

### 7.2 개별 서비스 리빌드

```bash
# 프론트엔드만 리빌드 + 재시작
docker compose build frontend && docker compose up -d frontend

# 백엔드만 리빌드 + 재시작
docker compose build backend && docker compose up -d backend

# 리뷰 서비스만 리빌드 + 재시작
docker compose build review && docker compose up -d review
```

### 7.3 개발 모드

```bash
make dev
```

개발 모드에서는:
- backend: `--reload` 옵션으로 코드 변경 시 자동 재시작
- frontend: Vite dev server (HMR, 포트 5173 → 3000)
- review: `--reload` 옵션

---

## 8. 환경 변수

`.env` 파일에 설정 (`.env.example` 참조):

| 변수 | 설명 | 필수 |
|------|------|------|
| `KIS_APP_KEY` | 한국투자증권 앱 키 | O |
| `KIS_APP_SECRET` | 한국투자증권 앱 시크릿 | O |
| `KIS_ACCOUNT_NO` | 계좌번호 | O |
| `POSTGRES_PASSWORD` | PostgreSQL 비밀번호 (기본: ptj_dev_password) | |
| `REDIS_URL` | Redis 연결 URL (기본: redis://redis:6379/0) | |
| `DATABASE_URL` | PostgreSQL 연결 URL (자동 구성) | |

---

## 9. 볼륨

| 볼륨 | 마운트 대상 | 용도 |
|------|------------|------|
| `redis-data` | `/data` | Redis RDB/AOF 영속화 |
| `postgres-data` | `/var/lib/postgresql/data` | PostgreSQL 데이터 디렉토리 |

> `make clean` 실행 시 볼륨이 삭제되어 모든 데이터가 초기화됩니다.

---

## 10. 네트워크

모든 컨테이너는 Docker Compose 기본 네트워크(`ptj_stock_default`)에서 통신합니다.

| 출발 | 도착 | 프로토콜 | 포트 |
|------|------|---------|------|
| frontend (Nginx) | backend | HTTP | 8000 |
| backend | redis | TCP | 6379 |
| backend | postgres | TCP | 5432 |
| review | postgres | TCP | 5432 |
| 브라우저 | frontend | HTTP | 3000 |
| 브라우저 | backend (직접) | HTTP | 8000 |
| 브라우저 | review (직접) | HTTP | 8010 |

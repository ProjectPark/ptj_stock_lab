# product/ — 프로덕션 엔진 스테이징

`ptj_stock_lab`에서 검증된 시그널 엔진을 `ptj_stock` 프로덕션에 이식하기 위한 스테이징 폴더입니다.

## 구조

```
product/
├── README.md              # 이 파일 (전체 인덱스)
├── _template/             # 엔진 스캐폴딩 템플릿
├── engine_v4/             # (레거시) v4 엔진 패키징
└── {line}_{version}_{study}/  # 엔진별 격리 폴더
```

## 네이밍 규칙

`{line}_{version}_{study}` — 예: `line_a_v5_twin_pair`, `line_c_v1_d2s`

| 세그먼트 | 값 | 설명 |
|----------|-----|------|
| `line` | `line_a` ~ `line_d` | 4-Line 구조 대응 |
| `version` | `v1` ~ `v9` | 전략 버전 |
| `study` | snake_case | 시그널/기능명 |

## 엔진 목록

| 폴더 | 라인 | 버전 | 시그널 | Optuna Study | 상태 | 생성일 |
|------|------|------|--------|--------------|------|--------|
| `engine_v4` | line_a | v4 | CB / 횡보장 / 스윙 / 쌍둥이 / 조건부 / 매도 | phase1 (CB) + study2 (스윙) + study5 (CB·매도) | `promoted` | 2026-02-24 |

## 상태 값

| 상태 | 설명 |
|------|------|
| `draft` | 초안 생성됨 |
| `ready` | 검토 완료, 이식 가능 |
| `promoted` | ptj_stock에 복사 완료 |
| `deployed` | 서버 배포 완료 |

## 사용법

1. **engine-promoter** 에이전트로 엔진 생성 → `product/{engine_name}/` 폴더 생성
2. **execution-adapter** 에이전트로 실행 레이어 추가 → 같은 폴더에 `execution_layer.py` 추가
3. 검토 후 각 엔진 폴더의 `PROMOTION_GUIDE.md` 따라 `ptj_stock/`에 복사
4. 배포 완료 시 `metadata.json`의 status 갱신 + 이 테이블 업데이트

## 각 엔진 폴더 구조

```
{engine_name}/
├── signals.py             # 시그널 함수 (→ ptj_stock signals.py)
├── auto_trader.py         # 엔진 함수 (→ ptj_stock auto_trader.py)
├── execution_layer.py     # 초단위 실행 판단 (→ ptj_stock execution_layer.py)
├── signal_service.py      # 파라미터 주입 (→ ptj_stock signal_service.py)
├── config.py              # Settings 필드 (→ ptj_stock config.py)
├── PROMOTION_GUIDE.md     # 복사 위치/순서 가이드
└── metadata.json          # 엔진 메타정보
```

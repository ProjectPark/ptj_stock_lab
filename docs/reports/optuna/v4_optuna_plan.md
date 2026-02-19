# PTJ v4 Optuna 최적화 계획 (Win Rate / Return / MDD 동시 개선)

> 작성일: 2026-02-18  
> 기준 문서: `docs/v3_optuna_report.md`

## 1. 목표 정의 (v4)

단일 목표(수익률 최대화)만 쓰면 MDD/승률이 악화될 수 있으므로, v4는 아래 3개를 동시에 관리한다.

- 수익률: `total_return_pct` 최대화
- 리스크: `mdd` 최소화 (낙폭 제한)
- 품질: `win_rate` 하한 유지

권장 목표 기준:

- 목표 수익률: baseline 대비 `+5%p` 이상
- MDD 상한: `<= 8.0%`
- 승률 하한: `>= 40.0%`

## 2. 데이터/검증 구간

현재 데이터 상태:

- 1분봉: `data/backtest_1min_v2.parquet` (2025-01-03 ~ 2026-01-30)
- Polymarket: `polymarket/history` (2025-01-03 ~ 2026-02-17)

검증은 v3 방식과 동일하게 3단계로 분리한다.

1. In-sample 탐색
2. Walk-forward 검증
3. OOS 홀드아웃 최종 확인

권장 분할:

- Train: 2025-02-18 ~ 2025-11-30
- Validation: 2025-12-01 ~ 2026-01-15
- Holdout: 2026-01-16 ~ 2026-02-17

## 3. 목적함수(스코어) 설계

기본안:

```text
score =
  total_return_pct
  - w_mdd * max(0, mdd - mdd_target)
  - w_win * max(0, win_target - win_rate)
  - w_trade * max(0, min_trades - total_sells)
```

권장 초기값:

- `mdd_target = 8.0`
- `win_target = 40.0`
- `min_trades = 80`
- `w_mdd = 1.5`
- `w_win = 0.8`
- `w_trade = 0.05`

추가 하드 필터:

- `mdd > 12.0` 이면 trial 패널티 크게 부여
- `total_sells < 50` 이면 과소거래 패널티

## 4. 탐색 파라미터 범위 (v4 중심)

v3 대비 v4에서 반드시 포함할 핵심 축:

- 진입/쌍둥이: `V4_PAIR_GAP_ENTRY_THRESHOLD`, `V4_SPLIT_BUY_INTERVAL_MIN`
- 손절/보유: `STOP_LOSS_PCT`, `STOP_LOSS_BULLISH_PCT`, `MAX_HOLD_HOURS`, `TAKE_PROFIT_PCT`
- 횡보 필터: `V4_SIDEWAYS_MIN_SIGNALS`, `V4_SIDEWAYS_INDEX_THRESHOLD`
- CB/회피: `V4_CB_GLD_SPIKE_PCT`, `V4_CB_GLD_COOLDOWN_DAYS`, `V4_CB_BTC_CRASH_PCT`, `V4_CB_BTC_SURGE_PCT`
- v4 신규 리스크 축:
  - `V4_HIGH_VOL_STOP_LOSS_PCT`
  - `V4_PAIR_IMMEDIATE_SELL_PCT`
  - `V4_PAIR_FIXED_TP_PCT`
  - `V4_CRASH_BUY_THRESHOLD_PCT` (보수적으로 좁은 범위)
  - `V4_SWING_TRIGGER_PCT`, `V4_SWING_STAGE1/2_*` (스윙 활성 시)

## 5. 실행 단계 (v3 방식 준용)

### Phase 0: Baseline 고정

- baseline 1회 실행 후 JSON/리포트 저장
- 기준 지표 잠금

### Phase 1: Wide Search (탐색)

- Trial: 300~500
- Sampler: TPE
- 목적: 고성능 클러스터 발견

### Phase 2: Narrow Search (정밀)

- Trial: 200~300
- 범위: Phase 1 상위 20개 trial 주변으로 축소
- 목적: MDD/승률 제약을 만족하는 안정 조합 확보

### Phase 3: Robustness Check

- Walk-forward 2~3개 윈도우
- OOS holdout 1회
- Top 5 조합 재평가 후 최종 1개 선택

## 6. 선택 기준 (최종 배포 후보)

Top 후보 중 아래를 모두 만족하는 조합만 채택:

- Holdout 수익률이 baseline 이상
- Holdout MDD가 baseline 이하
- Win rate 38~45% 범위 유지 (과도한 저빈도 전략 제외)
- 파라미터 민감도 ±10%에서 성능 급붕괴 없음

## 7. 실행 커맨드 템플릿

```bash
# Stage 1: baseline
pyenv shell market
python optimize_v4_optuna.py --stage 1

# Phase 1: wide
python optimize_v4_optuna.py --stage 2 --n-trials 400 --n-jobs 8 --study-name ptj_v4_phase1 --db sqlite:///data/optuna_v4_phase1.db

# Phase 2: narrow (phase1 결과 반영 범위로 수정 후)
python optimize_v4_optuna.py --stage 2 --n-trials 250 --n-jobs 8 --study-name ptj_v4_phase2 --db sqlite:///data/optuna_v4_phase2.db
```

## 8. 바로 반영할 코드 변경 항목

`optimize_v4_optuna.py`에 다음을 추가하면 계획을 실행 가능한 상태로 전환할 수 있다.

1. 목적함수 모드 추가: `--objective return|balanced|risk_off`
2. balanced 모드에 MDD/승률 패널티 스코어 적용
3. v4 신규 파라미터(스윙/고변동성/고정익절/크래시) 탐색 옵션화
4. Walk-forward 옵션: `--train-end`, `--valid-start`, `--valid-end`
5. Top N 재평가 리포트 자동 생성

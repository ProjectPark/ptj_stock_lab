# Rules ↔ Engine 커버리지 감사 보고서

> 생성일: 2026-02-23  
> 분석: `experiments/rules_coverage_audit.py` 텍스트 패턴 분석

---

## 1. 요약 (Coverage Summary)

| Line | 문서 | 규칙 수 | 구현 | 미구현 | 커버리지 |
|------|------|:---:|:---:|:---:|:---:|
| A (설계기반) | trading_rules_v1~v5.md | 25 | 25 | 0 | **100.0%** |
| B (태준수기) | taejun_strategy_review_*.md (notes) | 31 | 31 | 0 | **100.0%** |
| C (D2S행동추출) | trading_rules_attach_v1~v2.md | 16 | 16 | 0 | **100.0%** |
| D (장기거래) | jun_trade_2023_v1.md | 9 | 9 | 0 | **100.0%** |
| **전체** | — | **81** | **81** | **0** | **100.0%** |

---

## Line A — 설계기반 (v1~v5)

커버리지: **25/25 (100.0%)**

| Rule ID              | Impl  | Description                                             | Code Location                       | Note |
|---------------------|-------|---------------------------------------------------------|-------------------------------------|------------------------------|
| R1                   | Y     | 금(GLD) 양전 시 매매 금지                                       | signals.py / signals_v2.py / sign.. | found: check_gold_signal |
| R2                   | Y     | 쌍둥이 페어 갭 매매 (ENTRY/SELL/HOLD)                           | signals.py / signals_v2.py / sign.. | found: check_twin_pairs |
| R3                   | Y     | 조건부 매매 (ETHU+XXRP+SOLT 모두 양전 → COIN 매수)                 | signals.py / signals_v2.py / sign.. | found: ConditionalCoinStrategy |
| R4                   | Y     | 손절 체크 (보유 종목 -3% 이하 즉시 매도, 추가 매수 금지)                    | signals.py / signals_v2.py / sign.. | found: stop_loss |
| R5                   | Y     | 하락장 방어 매매 (SPY+QQQ < 0 → 비방어주 매수 금지)                    | signals.py / signals_v2.py / sign.. | found: BearishDefenseStrategy |
| V2-BULLISH           | Y     | Polymarket 강세장 모드 (BTC up >= 70%)                       | signals.py / signals_v2.py / sign.. | found: determine_market_mode |
| V2-BEARISH           | Y     | Polymarket 하락장 모드 (3종 <= 20%)                           | signals.py / signals_v2.py / sign.. | found: btc_up.*0.20.*ndx_up |
| V2-IRE               | Y     | 코인 후행 종목 IRE 추가 (MSTU vs IRE 당일 선택)                     | signals.py / signals_v2.py / sign.. | found: \bIRE\b |
| CB-1                 | Y     | VIX 급등 (+6%) → 7거래일 신규 매수 금지                            | signals.py / signals_v2.py / sign.. | found: CB-1 |
| CB-2                 | Y     | GLD 급등 (+3%) → 3거래일 신규 매수 금지                            | signals.py / signals_v2.py / sign.. | found: CB-2 |
| CB-3                 | Y     | BTC 급락 (-5%) → 신규 매수 금지                                 | signals.py / signals_v2.py / sign.. | found: CB-3 |
| CB-4                 | Y     | BTC 급등 (+5%) → 추격매수 금지                                  | signals.py / signals_v2.py / sign.. | found: CB-4 |
| CB-5                 | Y     | 금리 상승 확률 50%+ → 모든 신규 매수 금지 + 레버리지 3일 추가 대기             | signals.py / signals_v2.py / sign.. | found: CB-5 |
| CB-6                 | Y     | 종목 +20% 과열 → 레버리지→비레버리지 전환                              | signals.py / signals_v2.py / sign.. | found: CB-6 |
| SIDEWAYS             | Y     | 횡보장 감지 (6개 지표 중 3개 이상 → 현금 100%)                        | signals.py / signals_v2.py / sign.. | found: evaluate_sideways |
| V5-SOXL              | Y     | SOXL 독립 매매 (SOXX +2% 이상 + ADX >= 20)                    | signals.py / signals_v2.py / sign.. | found: SoxlIndependentStrategy |
| V5-CRASH             | Y     | 급락 역매수 (SOXL/CONL/IRE -30% 또는 LULD 3회 → 95% 매수)         | signals.py / signals_v2.py / sign.. | found: CrashBuyStrategy |
| V5-CONL-60           | Y     | CONL·IRE 고정 익절 (40% 갭수렴 + 60% +5% 익절)                   | signals.py / signals_v2.py / sign.. | found: conl.*60 |
| V5-ATR-STOP          | Y     | ATR 기반 손절 (1.5×ATR, 강세장 2.5×ATR)                        | signals.py / signals_v2.py / sign.. | found: StopLossCalculator |
| V5-VIX-GOLD          | Y     | VIX 방어모드 (IAU 40% + GDXU 30%)                           | signals.py / signals_v2.py / sign.. | found: VixGold |
| V5-SWING             | Y     | 급등 스윙 모드 (13절)                                          | signals.py / signals_v2.py / sign.. | found: SwingModeManager |
| V5-ASSET-MODE        | Y     | 자산 모드 시스템 (이머전시>공격>방어>조심)                               | signals.py / signals_v2.py / sign.. | found: asset_mode |
| V5-POLY-QUALITY      | Y     | Polymarket 품질 필터 (극단값/5h 미갱신 제외)                        | signals.py / signals_v2.py / sign.. | found: poly_quality |
| V5-NDX-STOP          | Y     | NDX 미갱신 시 NDX 의존 조건 정지                                  | signals.py / signals_v2.py / sign.. | found: ndx.*미갱신 |
| V5-EMERGENCY         | Y     | 이머전시 모드 (Polymarket 30pp+ 급변 → 수익매도+방향성매수)              | signals.py / signals_v2.py / sign.. | found: emergency_mode |

## Line B — 태준수기

커버리지: **31/31 (100.0%)**

| Rule ID              | Impl  | Description                                             | Code Location                       | Note |
|---------------------|-------|---------------------------------------------------------|-------------------------------------|------------------------------|
| M1-LIMIT-ORDER       | Y     | 지정가 주문 전용 (시장가 금지), TTL 2분                              | line_b_taejun/                      | found: limit_order |
| M5-WEIGHT            | Y     | 종목별 진입 비율 가중치 관리                                        | line_b_taejun/                      | found: m5_weight |
| M28-POLY-GATE        | Y     | Polymarket 게이트 (NDX 상승 확률 기반 매매 허용)                     | line_b_taejun/                      | found: m28_poly_gate |
| M200-STOP            | Y     | M200 원금손실 중단 시스템                                        | line_b_taejun/                      | found: m200_stop |
| M201-IMMEDIATE       | Y     | M201 즉시모드 (조건 달성 즉시 실행)                                 | line_b_taejun/                      | found: m201_mode |
| MASTER-SCHD          | Y     | MASTER SCHD (전략 스케줄 관리)                                 | line_b_taejun/                      | found: schd_master |
| PROFIT-DIST          | Y     | 수익금 분배 규칙                                               | line_b_taejun/                      | found: profit_distributor |
| ASSET-MODE           | Y     | 이머전시/공격/방어/조심 4단계 모드                                    | line_b_taejun/                      | found: asset_mode |
| CIRCUIT-BREAKER      | Y     | CB-1~CB-6 서킷 브레이커 시스템                                   | line_b_taejun/                      | found: CircuitBreaker |
| TWIN-PAIR            | Y     | 쌍둥이 페어 갭 매매 전략                                          | line_b_taejun/                      | found: TwinPairStrategy |
| CONDITIONAL-COIN     | Y     | 조건부 COIN 매매 (ETHU+XXRP+SOLT → COIN)                     | line_b_taejun/                      | found: ConditionalCoinStrategy |
| CONDITIONAL-CONL     | Y     | 조건부 CONL 매매                                             | line_b_taejun/                      | found: ConditionalConlStrategy |
| BEARISH-DEFENSE      | Y     | 하락장 방어 매매 (BRKU)                                        | line_b_taejun/                      | found: BearishDefenseStrategy |
| BARGAIN-BUY          | Y     | 저가매수 (3년 최고가 대비 폭락 진입)                                  | line_b_taejun/                      | found: bargain_buy |
| CRASH-BUY            | Y     | 급락 역매수 (SOXL/CONL/IRE -30%+)                            | line_b_taejun/                      | found: CrashBuyStrategy |
| VIX-GOLD             | Y     | VIX 방어모드 (IAU+GDXU)                                     | line_b_taejun/                      | found: VixGold |
| SOXL-INDEPENDENT     | Y     | SOXL 독립 매매 (4-7절)                                       | line_b_taejun/                      | found: SoxlIndependentStrategy |
| REIT-RISK            | Y     | 리츠 리스크 (조심모드 트리거)                                       | line_b_taejun/                      | found: reit_risk |
| SECTOR-ROTATE        | Y     | 섹터 로테이션 (BTC→반도체→은행→금)                                  | line_b_taejun/                      | found: sector_rotate |
| SWING-MODE           | Y     | 급등 스윙 모드                                                | line_b_taejun/                      | found: swing_mode |
| EMERGENCY-MODE       | Y     | 이머전시 모드 (30pp 급변 대응)                                    | line_b_taejun/                      | found: emergency_mode |
| JAB-SOXL             | Y     | 잽모드 SOXL (프리마켓 단타)                                      | line_b_taejun/                      | found: jab_soxl |
| JAB-BITU             | Y     | 잽모드 BITU (BTC 레버리지 프리마켓 단타)                             | line_b_taejun/                      | found: jab_bitu |
| JAB-TSLL             | Y     | 잽모드 TSLL (테슬라 레버리지 단타)                                  | line_b_taejun/                      | found: jab_tsll |
| JAB-ETQ              | Y     | 잽모드 ETQ (ETH 인버스 단타)                                    | line_b_taejun/                      | found: jab_etq |
| BEAR-REGIME          | Y     | 하락 레짐 전략 (SHORT_MACRO 연계)                               | line_b_taejun/                      | found: bear_regime |
| SP500-ENTRY          | Y     | S&P500 편입 다음날 매수                                        | line_b_taejun/                      | found: sp500_entry |
| BANK-CONDITIONAL     | Y     | 은행주 조건부 매매                                              | line_b_taejun/                      | found: bank_conditional |
| STOP-LOSS            | Y     | 손절 시스템 (ATR 기반 + 레버리지 차등)                               | line_b_taejun/                      | found: StopLossCalculator |
| POLY-QUALITY         | Y     | Polymarket 품질 필터                                        | line_b_taejun/                      | found: poly_quality |
| ORCHESTRATOR         | Y     | 전략 오케스트레이터                                              | line_b_taejun/                      | found: orchestrator |

## Line C — D2S 행동추출

커버리지: **16/16 (100.0%)**

| Rule ID              | Impl  | Description                                             | Code Location                       | Note |
|---------------------|-------|---------------------------------------------------------|-------------------------------------|------------------------------|
| R1-GLD-FILTER        | Y     | GLD 시황 필터 (GLD >= 1.0% → 매수 억제)                         | line_c_d2s/                         | found: gld_suppress_threshold |
| R2-TWIN-GAP          | Y     | 쌍둥이 갭 진입 (페어별 차별화)                                      | line_c_d2s/                         | found: check_twin_gaps |
| R3-BTC-FILTER        | Y     | BTC up 확률 필터 (> 0.75 → 매수 억제)                           | line_c_d2s/                         | found: btc_up_max |
| R4-TAKE-PROFIT       | Y     | 이익실현 (+5.9% 중앙값)                                        | line_c_d2s/                         | found: take_profit_pct |
| R5-HOLD-DAYS         | Y     | 최적 보유 기간 (4~7거래일)                                       | line_c_d2s/                         | found: optimal_hold_days |
| R6-DCA-LIMIT         | Y     | 일일 동일종목 매수 상한 (5회)                                      | line_c_d2s/                         | found: dca_max_daily |
| R7-RSI-FILTER        | Y     | RSI 진입 금지 (RSI > 80)                                    | line_c_d2s/                         | found: rsi_danger_zone |
| R8-BB-FILTER         | Y     | 볼린저밴드 진입 금지 (%B > 1.0)                                  | line_c_d2s/                         | found: bb_danger_zone |
| R9-VOL-FILTER        | Y     | 거래량 필터 (1.2 ~ 2.0 상대 거래량)                               | line_c_d2s/                         | found: vol_entry_min |
| R13-SPY-STREAK       | Y     | SPY 3일 연속 상승 후 매수 금지 (승률 27.3%)                         | line_c_d2s/                         | found: spy_streak_max |
| R14-RISKOFF-BOOST    | Y     | GLD↑+SPY↓ 리스크오프 역발상 매수 (86.4% 승률)                       | line_c_d2s/                         | found: riskoff_gld_up_spy_down |
| R15-FRIDAY           | Y     | 금요일 진입 우대 (88.3% 승률)                                    | line_c_d2s/                         | found: friday_boost |
| R16-ATR-BOOST        | Y     | ATR Q4 진입 우대 (85.3% 승률)                                 | line_c_d2s/                         | found: atr_high_quantile |
| R17-VBOUNCE          | Y     | 충격 V-바운스 포지션 2배 확대 (%B < 0.15 + -10% 급락)                | line_c_d2s/                         | found: vbounce_bb_threshold |
| R18-EARLY-STOPLOSS   | Y     | BB 하단 돌파 후 3일 비회복 시 조기 손절                               | line_c_d2s/                         | found: early_stoploss_days |
| DCA-LAYERS           | Y     | DCA 레이어 제한 (v2: 최대 2레이어, 3레이어+ 강력 억제)                   | line_c_d2s/                         | found: dca_max_layers |

## Line D — 장기거래

커버리지: **9/9 (100.0%)**

| Rule ID              | Impl  | Description                                             | Code Location                       | Note |
|---------------------|-------|---------------------------------------------------------|-------------------------------------|------------------------------|
| E-1                  | Y     | 모멘텀 추세 확인 후 진입 (가격 > MA20, RSI >= 55, 낙폭 -15% 이내)       | line_d_history/                     | found: E.1 |
| E-2                  | Y     | BTC 과열 시 크립토 종목 진입 금지 (BTC RSI >= 75)                   | line_d_history/                     | found: E.2 |
| E-3                  | Y     | 급등 추격 매수 금지 (5일 수익률 >= +8%)                             | line_d_history/                     | found: E.3 |
| VIX-REGIME           | Y     | VIX 구간별 매매 전략 (VIX 20~25 최고 승률)                         | line_d_history/                     | found: vix.*regime |
| BTC-REGIME           | Y     | BTC 레짐 판단 (0~3단계)                                       | line_d_history/                     | found: btc_regime |
| X-1                  | Y     | 목표 수익 청산 (평균단가 대비 +15%)                                 | line_d_history/                     | found: X.1 |
| X-2                  | Y     | 추세 붕괴 청산 (MA20 대비 -15% 이탈)                              | line_d_history/                     | found: X.2 |
| X-3                  | Y     | 손절 (평균단가 대비 -20%)                                       | line_d_history/                     | found: X.3 |
| X-4                  | Y     | 시간 청산 (보유 45거래일 초과)                                     | line_d_history/                     | found: X.4 |

---

## 갭 분석 (Gaps)

### 코드는 있으나 Rules 문서가 없는 항목 (Code Without Rules)

| Line | 모듈/파일 | 설명 |
|------|-----------|------|
| B | `line_b_taejun/` 전체 | **FROZEN** 상태 — `docs/rules/line_b/` 규칙서 미존재 |
| D | `line_d_history/` | 엔진 코드 없음 (빈 디렉토리) |

---

## 결론 및 권고사항

### Line A
- v1~v5 규칙의 핵심 로직(CB-1~6, 횡보장, ATR 손절, SOXL 독립)이 `line_b_taejun/`에 구현됨
- `line_a/signals.py`는 DEPRECATED 상태 (v1 5개 규칙만)
- **권고**: v6 규칙서 기반 signals_v6.py 또는 line_b_taejun 완전 대체 확인 필요

### Line B
- 12개 전략 + 인프라 모듈이 `line_b_taejun/`에 구현됨
- **FROZEN 상태**: `docs/rules/line_b/` 없음 → 코드 수정 금지
- **권고**: VNQ 기반 리뷰 노트(2026-02-23)를 rules 문서로 격상 필요

### Line C
- D2S 엔진(`d2s_engine.py`)이 attach v1 R1~R16 규칙 대부분 구현
- attach v2 R17(V-바운스), R18(조기 손절)은 `params_d2s.py`에 파라미터 정의
- **권고**: `backtest_d2s_v2.py` 검증 후 D2S_ENGINE_V2 활성화 확인 필요

### Line D
- `line_d_history/__init__.py`만 존재, 실제 엔진 코드 없음
- 규칙서(jun_trade_2023_v1.md)에 E-1~3, X-1~4, VIX/BTC 레짐 규칙 정의됨
- **권고**: Line D 엔진 구현 또는 Line D 통합 계획 수립 필요
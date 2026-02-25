# Line B (태준수기) — VNQ 기반 트레이딩 룰 v1

> 작성일: 2026-02-23
> 상태: **초안 (Draft)**
> 출처: `docs/notes/line_b/review/taejun_strategy_review_2026-02-23_VNQ.md` (MT_VNQ3 반영)
> 엔진: `simulation/strategies/line_b_taejun/`
> 관련 소스: MT_VNQ.md / MT_VNQ2.md / MT_VNQ3.md (박태준)
>
> 주요 변경점 (MT_VNQ3 반영): M201(즉시모드) 신설, M28(Polymarket 게이트) 신설, M300(USD-only) 신설, MASTER SCHD 신설, M1 TTL 2분 확정, Fill Window 10초 룰, 추격매수 금지, 매도 즉시성, 포지션 confirmed/effective 분리

---

## 0. 문서 개요

Line B (태준수기) 엔진의 VNQ(미국 리츠 ETF) 기반 전략 규칙서. SK리츠(KRX: 395400)를 추종 지수로 사용하던 규칙을 VNQ로 대입·정정한 최신 버전.

### 전략 일람

| # | 전략명 | 코드 파일 | 유형 | 종목 | 목표 수익 (net, M20 반영) |
|---|--------|----------|------|------|--------------------------|
| A | 잽모드 SOXL | `strategies/jab_soxl.py` | 프리마켓 단타 | SOXL | +1.15% |
| B | 잽모드 BITU | `strategies/jab_bitu.py` | 프리마켓 단타 | BITU | +1.15% |
| C | 잽모드 TSLL | `strategies/jab_tsll.py` | 소액 단타 | TSLL | +1.25% |
| D | 숏 잽모드 ETQ | `strategies/jab_etq.py` | 숏 단타 | ETQ | +1.05% |
| E | VIX → GDXU | `strategies/vix_gold.py` | 이벤트 드리븐 | GDXU → IAU | +10.25% |
| F | S&P500 편입 | `strategies/sp500_entry.py` | 이벤트 드리븐 | 편입 기업 | +1.75% |
| G | 저가매수 | `strategies/bargain_buy.py` | 스윙/장기 | 9종목 | 종목별 상이 |
| H | 숏포지션 전환 | `strategies/short_macro.py` | 매크로 | GDXU | +90.25% |
| I | 리츠 리스크 (VNQ) | `strategies/reit_risk.py` | 리스크 관리 | VNQ | - |
| J | 섹터 로테이션 | `strategies/sector_rotate.py` | 순환 배분 | 다종목 | - |
| K | 조건부 은행주 | `strategies/bank_conditional.py` | 역전 매매 | BAC | +1.05% |
| L | 숏 포지션 (재무역전) | `strategies/bearish_defense.py` | 재무 역전 | CONZ/IREZ/HOOZ/MSTZ | +1.15% |
| M | 비이상적 재난 이머전시 | `strategies/emergency_mode.py` | 이벤트 | BITU/SOXL/SOXS | +3.15% |

> 목표 수익률에 M20 지시에 따라 기존 대비 +0.25% 추가 반영됨

---

## 1. 마스터 플랜 (M0~M300) — 모든 전략에 절대 적용

### 우선순위 고정 (MT_VNQ3 §1 확정)

```
M200(즉시매도) > M201(즉시전환) > 리스크모드 > 거래시간/휴장 > M28 포지션 게이트 > 비중배분(T1~T4) > M6 리츠 감산 > 일반매매
```

### M0 — MTGA AI (시스템 점검)

| 항목 | 규칙 |
|------|------|
| 1순위 점검 | 네트워크 연결속도, 매수/매도 서버 상태, RAM/CPU/GPU 사용량 |
| 오류 대응 | 코딩/계산 오류 발생 시 한국어로 알림 |
| 문제 발생 시 | 알람 발송 |
| 공휴일 재시작 | KST 18:00. **반드시** MASTER J(박태준) 또는 MASTER H(박태현) 사전 허가 필수. 미허가 시 **절대 재시작 금지** |
| 매수/매도 기록 | 모든 기록 보관 + AI 자체 분석 |
| 급등 분석 | 레버리지 ETF 30%+ 급등 종목 → 과거 2년 데이터, 오차범위 3% 이내 유사 패턴 정리 후 MASTER J 보고 |

**코드 위치:** `infra/orchestrator.py`

---

### M1 — 지정가 ALL STOCK

| Rule ID | 유형 | 조건 | 파라미터 | 현재값 | 코드 위치 |
|---------|------|------|----------|--------|----------|
| M1-1 | 절대 금지 | 시장가 매수/매도 금지 (이머전시 포함, 예외 없음) | - | - | `infra/limit_order.py` |
| M1-2 | 주문 TTL | 주문 후 TTL 초과 시 취소 → reserved_cash 해제 | `order_ttl_sec` | **120초 (2분)** | `infra/limit_order.py`, `common/params.py` ENGINE_CONFIG |
| M1-3 | 소수점 처리 | KIS 호가 소수점 2자리 지원. 0.001 이상 → 0.01 단위 반올림 | - | - | `infra/limit_order.py` |
| M1-4 | 가격 결측 | `prices.get(ticker, 0)` 금지. 결측 시 신호 스킵 + 알람 | - | - | `infra/limit_order.py` |

---

### M2 — 즉시 행동

| 항목 | 규칙 |
|------|------|
| 원칙 | 조건이 달성된 날짜/시간에 **즉시** 행동 |

---

### M3 — 거래 시간 및 휴장일

| 항목 | 내용 |
|------|------|
| 거래 시간 | 17:30~06:00 KST, 월~금 |
| 서머타임 (DST) | 2026-03-08~11-01: KST 22:30~05:00 |
| 표준시간 | KST 23:30~06:00 |
| 진입 비율 | M5로 이관 (M3에서 삭제) |

**2026 NYSE 휴장일:**

| 날짜 | 설명 |
|------|------|
| 1월 1일 (목) | New Year's Day |
| 1월 19일 (월) | Martin Luther King Jr. Day |
| 2월 16일 (월) | Presidents' Day |
| 4월 3일 (금) | Good Friday |
| 5월 25일 (월) | Memorial Day |
| 7월 3일 (금) | Independence Day 관측일 |
| 9월 7일 (월) | Labor Day |
| 11월 26일 (목) | Thanksgiving Day |
| 12월 25일 (금) | Christmas Day |

---

### M4 — 부동산 변동 수익률 포지션 실시간 변경 모드

| 항목 | 내용 |
|------|------|
| 추종 지표 | **REIT_MIX** = (VNQ + KR 리츠들) 전일 종가 대비 변화율 **평균** *(MT_VNQ3 §9: VNQ 단독 → 혼합)* |
| 데이터 결측 | REIT_MIX 전부 결측 시 UNKNOWN → **BUY_STOP** + 알람 |
| 가동 | **상시 가동**, 1순위로 작동 |
| 데이터 기반 | 최근 3년간 현재 날짜 기준 6일 과거 데이터 기반 이동평균선 비교 |

**M4 서브모드 5가지:**

| 모드 | 조건 (VNQ 기준) | 목표수익률 변경 | 추가 제한 |
|------|----------------|----------------|---------|
| 레버리지 스탑 해제(60) | 3년 최고가 기준 6일 과거 데이터로 60일선 아래 | 0.8~1.0% → **1.8%**, 1.8~2.0% → **2.8%** | - |
| 레버리지 스탑 해제(120) | 120일선 아래 | 0.8~1.0% → **2.8%**, 1.8~2.0% → **4.8%** | 레버리지 GOLD/SHORT 매매 중단 |
| 레버리지 스탑 허가(60) | 60일선 위 | 기존 유지 (0.8~1.0% → 0.8%, 1.8~2.0% → 1.8%) | - |
| 레버리지 스탑 허가(120) | 120일선 위 | 기존 유지 | 레버리지 ETF 즉시 전액 매도(GDXU 제외), 1배수 대체(M7 참조) |
| ALL IN ONE (20, 120) | VNQ 20일선 시작가 -1.0% 또는 120일선 -3% 하락 상태 | 기본 5%, GLD 하락 +1%, GLD 상승 -0.5% | 특수 조건 (아래 별도 정리) |

**ALL IN ONE 상세:**

| 항목 | 내용 |
|------|------|
| 대상 종목 | Cure, SOXL, TQQQ, CONL, ROBN, FAS, Ertlx, SOLT, ETHU (토스증권 거래대금 100위 이내) |
| 시작가 제한 | 4%+ 상승 시 매수 금지 / 3% 이하 시 매수 허가 |
| 거래대금 | 5년 최저 대비 **4배 이상** 필수 |
| Polymarket | 각 종목별 51%/49% 기준 허가/불허가 |
| 기한/목표 | -1.0% → 1일(목표5%), -2.0% → 2일(목표10%), -3.0% → 2일(목표15%), 0.5% 미만은 붙이지 않음 |
| 재무건전성 | 각 종목 추종 기초자산 최근 분기 순이익 양수 필수 |

**코드 위치:** `infra/` (M4 전용 모듈 미구현 — CI-11 이슈)

---

### M5 — 비중조절 메커니즘 (개정)

#### M5-매수: 당일 조건 부합 개수별 진입 비율 (T1~T4 체계)

| 순위 | 조건 달성 수 | 비중 산출 방식 |
|------|------------|-------------|
| T1 | 1개 | **전체 자산의 55%** |
| T2 | 2개 | 나머지(45%)에서 **40%** ≈ 전체 대비 18% |
| T3 | 3개 | 나머지에서 **33%** |
| T4 | 4개 | 나머지 **전액** |
| T5~ | 5개 이상 | **실매수 0**, 예약/대기 표시만 (10초 후 조건 미부합 시 예약 취소) |

> **해결 (2026-02-22):** T1 55% 기준은 **전체 자산(현금+주식 포함) 대비** 비율.

**비중 동적 조정 (동시 발생 시 중첩 합산 적용):**

| 지표 | 조건 | 비중 조정 |
|------|------|---------|
| GLD | +1% 상승마다 | -0.1% |
| GLD | -1% 하락마다 | +0.1% |
| VIX | +2% 상승마다 | -3% |
| VIX | -2% 하락마다 | +1% |
| 달러 | 전일 대비 상승 | +0.1% |
| 달러 | 전일 대비 하락 | -0.2% |
| Polymarket BTC/ETH/SOL/GOLD/NASDAQ | 51%+ | 해당 종목 **허가** |
| Polymarket BTC/ETH/SOL/GOLD/NASDAQ | 49% 이하 | 해당 종목 **불허가** |

**비중 제한:**

| 조건 | 행동 |
|------|------|
| 비중 -1% 이하 | 매수 금지 |
| 비중 100.1% 이상 (`shrink_hard_limit`) | 매수 금지 (BUY_STOP) |
| 현금 없음 | 매수 금지 |

**M5-예약 10초 룰 (MT_VNQ3 신설):**
- T5~ 예약은 10초 경과 후 조건 미부합이면 예약 취소
- `m5_t5_reserve_timeout_sec = 10` (`common/params.py` ENGINE_CONFIG)

#### M5-매도

| 조건 | 행동 |
|------|------|
| 거래대금 최고가 대비 20% 이하 + 2일 이평선 이탈 | **전액 매도** |
| VIX 3%+ 상승 | 레버리지 ETF T1 비중 **30% 매도** |

**코드 위치:** `infra/m5_weight_manager.py`

---

### M6 — 목표수익률 하한 + 리츠 감산 + M20 가산 (MT_VNQ3 §11 순서 확정)

**적용 순서 고정:**

| 순서 | 규칙 | 내용 |
|------|------|------|
| 1 | target_net 최소 하한 | 0.8% 미만 금지 → 0.8%로 상향 |
| 2 | M20 +0.25% 가산 | 모든 target_net에 +0.25% |
| 3 | 리츠 조심모드 감산 | 조심모드 발동 시 target_net × 1/2 감산 |
| 4 | 감산 후 하한 재확인 | 감산 후 target_net ≤ 0.8% → **전 모드 BUY_STOP** |

**리츠 감산 제외항 (MT_VNQ3 §11 확정):**

| 카테고리 | 종목 |
|---------|------|
| 금/공포 | GLD, VIX, GDXU |
| 숏 ETF | CONZ, IREZ, HOOZ, MSTZ |
| 기타 | SOXS, ETQ |

> target_gross = target_net + fee_roundtrip (청산/체결 판정은 gross 권장)

---

### M7 — 레버리지 ETF 금지 시 대체 종목

M4 레버리지 스탑 허가(120) 발동 시 적용:

| 원래 종목 | 대체 종목 | 비고 |
|---------|---------|------|
| SOXL | SOXX | |
| ROBN | HOOD | |
| FAS | XLF | |
| Ertlx | XKE | |
| Cure | XLV | |
| TQQQ | QQQ | |
| SOLT | **금지** | |
| ETHU | **금지** | |
| QUBX | QTUM | |
| BITU | **매매금지** | *(0222 정정: 기존 HODL → 매매금지)* |
| NVDL | **NVDA** | *(0222 신설)* |

> **선행지수는 변경하지 않음:** BITU, ROBN, NVDL

---

### M20 — 신용/미수 거래 절대 금지

| 항목 | 내용 |
|------|------|
| 규칙 | 어떤 경우에도 **신용/미수 거래 절대 금지** |
| 목표수익률 조정 | 수수료를 제외한 모든 목표수익률에 **+0.25%** 추가 |

---

### M28 — Polymarket BTC 포지션 허가 게이트 (MT_VNQ3 §13 신설)

**BTC Primary 소스 자동선택 (Volume Only):**

| 항목 | 내용 |
|------|------|
| 후보 마켓 | A: bitcoin-price-on-{date}, B: bitcoin-above-on-{date} |
| 선택 기준 | volume_usd만으로 선택 (±1% 이내면 직전 유지) |

**포지션 허가 게이트:**

| BTC_UP_PROB | 허가 |
|------------|------|
| p ≥ 0.51 | **LONG 허가** |
| p ≤ 0.49 | **SHORT 허가** |
| 0.49 < p < 0.51 | **중립 (둘 다 불허)** |

> 업데이트 실패/스테일이면 BUY_STOP + 알람

**코드 위치:** `infra/m28_poly_gate.py`

---

### M40 — Polymarket 급등 비중 상향

| 조건 | 대상 종목 | 비중 조정 |
|------|---------|---------|
| Polymarket BTC **80%+** | BTC/SOLT/ETHU/ROBN | 기존 비중 **+15%** |
| Polymarket BTC 80%+ & GLD -5% 급등 | 동 | 추가 **+3%** |
| Polymarket NASDAQ **80%+** | Cure/SOXL/TQQQ/FAS/Ertlx/QUBX | 기존 비중 **+15%** |
| Polymarket NASDAQ 80%+ & GLD -5% 급등 | 동 | 추가 **+3%** |
| Polymarket BTC & NASDAQ **20% 이하** | GDXU | 비중 **+20%** 추가 매수 |

---

### M80 — 거래대금 초단타 모드 (M200 점수 체계)

**거래대금 배수별 점수 (5년 최저 대비):**

| 거래대금 배수 | 점수 |
|-------------|------|
| 1.2배 이하 | 1.2점 |
| 2배 이하 | 2점 |
| 3배 이하 | 3점 |
| 4배 이하 | 4점 |
| 5배 이하 | 5점 |
| 6~18배 | 6~18점 (1배당 1점) |

**점수별 목표수익률:**

| 점수 구간 | 목표수익률 |
|----------|----------|
| 5점 | 0.1~1.0% |
| 6~10점 | 1.1~2.0% |
| 15점 | 2.1~3.0% |
| 20점 | 3.1~4.0% |
| 25점 | 4.1~4.5% |
| 30점 | 4.6~5.0% |
| 35점 | 5.1~6.0% |
| 70점 | 6.1%+ |

**M80 공통 매수 조건:**

| 조건 | 내용 | 필수 여부 |
|------|------|---------|
| 조건1: 거래대금 | 기초자산 거래대금 2년 최저 대비 N배 이상 (종목별 상이) | **필수** |
| 조건2: 분기 순이익 | 기초자산 최근 분기 순이익 양수 (+0.1% 이상) | **필수** |
| 조건3: 이평선 돌파 | 이동평균선 돌파 시 목표수익률 추가 상향 | 선택 |

**M80 섹터별 종목 풀 (BTC 섹터):**

| 기초자산 | 레버리지 ETF (1순위) | 거래대금 허가 배수 | M7 대체 |
|---------|--------------------|-----------------|---------|
| MSTR | MSTU | >= 3.1배 | MSTR |
| COIN | CONL | >= 3.1배 | COIN |
| IREN | IRE | >= 3.1배 | IREN |
| RIOT | RIOX | >= 3.1배 | RIOT |
| CRCL | CRCA | >= 3.1배 | CRCL |
| BMNR | BMNU | >= 3.15배 | BMNR |
| BTCT | - | >= 7.14배 | BTCT |
| CNCK | - | >= 10배 | CNCK |
| XDGXX | - | >= 5.4배 | XDGXX |
| FUFU | - | >= 10배 | FUFU |
| ANT | - | >= 40.2배 | ANT |

> 조건3 BTC 섹터: 기초자산 **120일 이동평균선 상향 돌파** 시 목표수익률 **+0.9%** 추가

**M80-반도체 섹터 (구매 목록: SOXL, NVDL):**

| 기초자산 | 레버리지 ETF | 거래대금 허가 배수 |
|---------|------------|-----------------|
| SOXX | SOXL | >= 3.1배 |
| NVDA | NVDL | >= 3.1배 |
| AMD | AMDL | >= 3.1배 |
| AVGO | AVGX | >= 3.1배 |
| ARM | ARMG | >= 3.1배 |

> 조건3 반도체: 기초자산 **120일 이동평균선 상향 돌파** 시 목표수익률 **+0.9%** 추가

**M80-금 섹터:**

| 기초자산 | 종목 | 거래대금 허가 배수 |
|---------|------|-----------------|
| GDX | GDXU | >= 2배 |
| AEM, NEM (재무 대체) | AEM, NEM | - |

> AEM/NEM 모두 최근 분기 순이익 +0.1% 이상 시 GDXU 매수. 불허 시 IAU로 대체.
> 조건3: GDX **5일 이동평균선 돌파** 시 목표수익률 **+0.3%** 추가

**M80-헬스케어 섹터:**

| 기초자산 | 레버리지 ETF | 거래대금 허가 배수 |
|---------|------------|-----------------|
| LLY | LLYX | >= 2.7배 |
| JNJ | JNJ | >= 2.8배 |
| NVO | NVOX | >= 2.8배 |

> 재무: JNJ + LLY 모두 순이익 흑자 시 허가. 5일 이평선 돌파 시 NVO/LLY +0.3%, XLV +0.2% 추가

**M80-에너지 섹터:**

| 기초자산 | 레버리지 ETF | 거래대금 허가 배수 |
|---------|------------|-----------------|
| XLE | ERX | >= 2.2배 |
| XOM | XOMX | >= 2.2배 |
| CVX | CVX | >= 2.2배 |
| COP | COP | >= 2.2배 |
| SLB | SLB | >= 2.2배 |
| WMB | WMB | >= 2.2배 |

**M80-은행 섹터:** J전략(섹터 로테이션) 은행 섹터와 동일 조건 적용

---

### M200 — 즉시 매도 조건

**조건 1개라도 충족 시 목표수익률 무관 즉시 매도:**

| # | 조건 | 파라미터 |
|---|------|----------|
| 1 | 거래대금 급감: 전날 거래대금 대비 장 시작 3시간 이내 15% 하락 | - |
| 2 | Polymarket BTC 44% 기준 (P-5: **OFF 상태** — 정의 불명확으로 비활성화) | `p5_44poly_btc_enabled = False` | `infra/m200_stop.py`, `common/params.py` ENGINE_CONFIG |
| 3 | GLD 급등: GLD **6%+ 상승** | - |
| 4 | VIX 급등: VIX **10%+ 급등** | - |
| 5 | 이평선 이탈: 모든 주식 **20일 이동평균선 이탈** | - |
| 6 | 기한 만기: 익영업일 **17:30~06:00 이내** 매도 | - |
| 7 | REIT_MIX 급등: **5%+ 상승** → VNQ 관련 즉시 정리 / 그 외 30% 축소 (GLD 숏 제외) | - |

**발동 파이프라인 (MT_VNQ3 §15):**
`전역 락 ON → OPEN 주문 전부 취소 → 취소 ACK → reserved 재계산 → 청산 실행(매도 즉시성 규정) → 락 OFF`

> **SCHD 예외:** 어떤 모드에서도 SCHD 매도 불가 (MASTER H/J가 AI OFF 시만 허용)

**코드 위치:** `infra/m200_stop.py`

---

### M201 — 즉시모드 v1.0 (BTC 확률 급변/전환) (MT_VNQ3 §14 신설)

> 우선순위: **M200 다음, 리스크모드 이전**

**입력:** p (현재 BTC 상승 확률), p_prev (직전 확률), Δpp = (p − p_prev) × 100

| 보유 상태 | 조건 | 행동 |
|---------|------|------|
| LONG 보유 중 | p ≤ 0.45 | 즉시 롱 청산 |
| LONG 보유 중 | Δpp ≤ −20pp AND p ≤ 0.49 | 즉시 롱 청산 |
| SHORT 보유 중 | p ≥ 0.55 *(코드 기준 — 리뷰 노트에 p≥0.60으로 기재됐으나 코드값 우선, 태준 확인 필요)* | 즉시 숏 청산 |

**청산 직후 전환:**

| 청산 종류 | p 조건 | 이후 행동 |
|---------|--------|---------|
| 롱 청산 후 | p ≤ 0.49 | 즉시 숏 진입 (가용현금 전액) |
| 롱 청산 후 | 0.49 < p < 0.50 | 현금 유지 (중립) |
| 숏 청산 후 | p ≥ 0.51 | 즉시 롱 진입 (가용현금 전액) |
| 숏 청산 후 | 0.49 < p < 0.50 | 현금 유지 (중립) |

> ETQ 숏 잽모드는 ETH가 아니라 **BTC(p) 기준**

**코드 위치:** `infra/m201_mode.py`

---

### M300 — USD-only 환전 금지 (MT_VNQ3 §17 신설)

| 항목 | 내용 |
|------|------|
| 원칙 | MTGA AI는 FX(환전) 수행 금지 |
| 실매매 제한 | USD 결제 상품으로만 제한 |
| VNQ/KR 리츠 | **실매매 금지**, 지표용으로만 유지 (signal-only) |

---

### Fill Window 10초 룰 (MT_VNQ3 §5 신설)

**진입 주문(매수/신규 숏 진입)** 은 fill_window_sec = **10초** 최우선 적용.

| 단계 | 내용 |
|------|------|
| 0~10초 | 잔량 채우기 위해 주문 유지/관리 (시장가 금지 유지) |
| 10초 경과 후 잔량 있으면 | 잔량 즉시 취소 |
| 취소 ACK 이후 | 잔량 reserved_cash 해제 → 현금 복귀 |
| 이벤트 기록 | PARTIAL_ENTRY_DONE — 동일 signal_bar_ts 재진입 금지 |

**파라미터:** `fill_window_sec = 10` (`common/params.py` ENGINE_CONFIG)

---

### 추격매수 금지 (MT_VNQ3 §6 신설)

| 원칙 | 내용 |
|------|------|
| 기준가격 고정 | 모든 매수는 신호 발생 기준가격(reference_price)으로만 제출 |
| 가격 상향 절대 금지 | 체결을 위해 ask/bid + slip로 가격 올리는 행위 금지 |
| tick 보정 예외 | tick 제약으로 주문 거절 시만 기술 보정 허용 — BUY tick 보정은 항상 **floor(내림)** |
| 미체결 처리 | 미체결은 "실패"로 간주, 잔량은 현금 유지 |

**파라미터:** `no_chase_buy = True` (`common/params.py` ENGINE_CONFIG)

---

### 매도 즉시성 허용 (MT_VNQ3 §7 신설)

| 원칙 | 내용 |
|------|------|
| LIMIT-only 유지 | 시장가 금지 유지 |
| 청산/리스크 매도 | bid 기반 **0.2% 이내** 범위로 marketable limit 허용 |
| 적용 범위 | M200 / M201 / 리스크 청산성 매도에 우선 적용 |

**파라미터:** `bid_slip_max_pct = 0.002` (`common/params.py` ENGINE_CONFIG)

---

### 포지션 confirmed vs effective 분리 (MT_VNQ3 §4 신설)

| 포지션 타입 | 정의 | 용도 |
|-----------|------|------|
| **position_effective** | 실제로 체결되어 보유한 수량/평단(vwap) | 리스크/청산/현금계산/M200/M201 |
| **position_confirmed** | 진입이 계획대로 완료되었는지만 판단하는 상태 | 전략 판단/신호용 |

**파라미터:** `position_mode = "effective_confirmed"` (`common/params.py` ENGINE_CONFIG)

---

### 금지/오타/대체 티커 처리 정책 (MT_VNQ3 §18 신설)

| 상황 | 처리 |
|------|------|
| 금지/오타 티커 | 매수목록 즉시 제외(drop), shrink 계산에서도 제외 |
| 대체 티커 정의 있음 | 원 티커 → 대체 티커로 치환 후 매수 |
| 대체 불가 | drop + 알람 |

---

### MASTER SCHD — 장기 적립 (절대 매도 금지) (MT_VNQ3 §16 신설)

| 항목 | 내용 |
|------|------|
| 매수 조건 | 최근 **30일 실현손익이 +일 때만** SCHD 매수 |
| **절대 매도 금지** | 어떤 모드에서도 SCHD 매도 불가 |
| 예외 | MASTER H 또는 J가 "AI 매매 기능 OFF" 했을 때만 매도 허용 |
| 배당 | 자동 재투자 허용 |

**수익 구간별 매수 기준:**

| 30일 실현손익 | 매수 금액 |
|------------|---------|
| +1% | 10만원 |
| +2% | 20만원 |
| +3% | 30만원 |
| +4% | 40만원 |
| +5% | 50만원 (최대) |

**코드 위치:** `infra/schd_master.py`

---

## 2. 수익금 분배 규칙

| 규칙 | 내용 |
|------|------|
| 주기 | **1일당** 각 1주씩 매수 |
| 대상 종목 | SOXL, ROBN, GLD, CONL |
| 방식 | 수익금 발생 시 4종목 순서대로 1주씩 구매 |

**코드 위치:** `infra/profit_distributor.py`

---

## 3. 전략별 상세 규칙

### A. 잽모드 SOXL (Jab-SOXL)

> 반도체 개별주 모두 상승인데 SOXX/SOXL만 마이너스 → 레버리지 괴리 역전

| 항목 | 내용 |
|------|------|
| 진입 시간 | 17:30 KST ~ 장마감 |
| 매수 대상 | SOXL |
| 매수 비중 | **M5 T1~T4 비율 기준** |
| 목표 수익률 | **+1.15%** (수수료 제외 net) |
| 매도 방식 | 전액 매도 |

**진입 조건 (16개 ALL 충족):**

| Rule ID | 종목/지표 | 조건 | 파라미터 | 현재값 | 코드 위치 |
|---------|----------|------|----------|--------|----------|
| A-1 | Polymarket NASDAQ | >= 51% | `poly_ndx_min` | 0.51 | `strategies/jab_soxl.py` |
| A-2 | GLD | >= +0.1% | `gld_min` | 0.1 | `strategies/jab_soxl.py` |
| A-3 | QQQ | >= +0.3% | `qqq_min` | 0.3 | `strategies/jab_soxl.py` |
| A-4 | SOXX | <= -0.2% | `soxx_max` | -0.2 | `strategies/jab_soxl.py` |
| A-5 | SOXL | <= -0.6% | `soxl_max` | -0.6 | `strategies/jab_soxl.py` |
| A-6 | NVDA | >= +0.9% | `individual["NVDA"]` | 0.9 | `strategies/jab_soxl.py` |
| A-7 | AMD | >= +0.9% | `individual["AMD"]` | 0.9 | `strategies/jab_soxl.py` |
| A-8 | SMCI | >= +1.0% | `individual["SMCI"]` | 1.0 | `strategies/jab_soxl.py` |
| A-9 | KLA | >= +0.8% | `individual["KLA"]` | 0.8 | `strategies/jab_soxl.py` |
| A-10 | AMAT | >= +0.8% | `individual["AMAT"]` | 0.8 | `strategies/jab_soxl.py` |
| A-11 | AVGO | >= +0.55% | `individual["AVGO"]` | 0.55 | `strategies/jab_soxl.py` |
| A-12 | MPWR | >= +0.55% | `individual["MPWR"]` | 0.55 | `strategies/jab_soxl.py` |
| A-13 | TXN | >= +0.66% | `individual["TXN"]` | 0.66 | `strategies/jab_soxl.py` |
| A-14 | ASML | >= +1.0% | `individual["ASML"]` | 1.0 | `strategies/jab_soxl.py` |
| A-15 | LRCX | >= +0.8% | `individual["LRCX"]` | 0.8 | `strategies/jab_soxl.py` |
| A-16 | MU | >= +0.55% | `individual["MU"]` | 0.55 | `strategies/jab_soxl.py` |

**파라미터 출처:** `common/params.py` JAB_SOXL

---

### B. 잽모드 BITU (Jab-BITU)

> BTC 생태계 상승 + BITU 과매도 → 역전

| 항목 | 내용 |
|------|------|
| 진입 시간 | 17:30 KST ~ 장마감 |
| 매수 대상 | BITU |
| 매수 비중 | M5 T1~T4 비율 기준 |
| 목표 수익률 | **+1.15%** (수수료 제외 net) |
| 매도 방식 | 전액 매도 |

**진입 조건 (ALL 충족):**

| Rule ID | 종목/지표 | 조건 | 파라미터 | 현재값 | 코드 위치 |
|---------|----------|------|----------|--------|----------|
| B-1 | Polymarket BTC | >= 63% | `poly_btc_min` | 0.63 | `strategies/jab_bitu.py` |
| B-2 | GLD | >= +0.1% | `gld_min` | 0.1 | `strategies/jab_bitu.py` |
| B-3 | BITU | <= -0.4% | `bitu_max` | -0.4 | `strategies/jab_bitu.py` |
| B-4 | BTC (스팟) | >= +0.9% | `crypto_conditions["BTC"]` | 0.9 | `strategies/jab_bitu.py` |
| B-5 | ETH (스팟) | >= +0.9% | `crypto_conditions["ETH"]` | 0.9 | `strategies/jab_bitu.py` |
| B-6 | SOL (스팟) | >= +2.0% | `crypto_conditions["SOL"]` | 2.0 | `strategies/jab_bitu.py` |
| B-7 | XRP (스팟) | >= +2.5% (하루 변동 기준) | `crypto_conditions["XRP"]` | 2.5 | `strategies/jab_bitu.py` |

**파라미터 출처:** `common/params.py` JAB_BITU

---

### C. 잽모드 TSLL (Jab-TSLL)

> TSLA 상승 + TSLL 과매도 → 소액 역전

| 항목 | 내용 |
|------|------|
| 진입 시간 | 17:30 KST ~ 장마감 |
| 매수 대상 | TSLL |
| 매수 비중 | **200만원 이하 소액** |
| 목표 수익률 | **+1.25%** (수수료 제외 net) |
| 매도 방식 | 전액 매도 → 현금 보유 |

**진입 조건 (ALL 충족):**

| Rule ID | 종목/지표 | 조건 | 파라미터 | 현재값 | 코드 위치 |
|---------|----------|------|----------|--------|----------|
| C-1 | Polymarket NASDAQ | >= 63% | `poly_ndx_min` | 0.63 | `strategies/jab_tsll.py` |
| C-2 | GLD | <= +0.1% | `gld_max` | 0.1 | `strategies/jab_tsll.py` |
| C-3 | TSLL | <= -0.8% | `tsll_max` | -0.8 | `strategies/jab_tsll.py` |
| C-4 | TSLA | >= +0.5% | `tsla_min` | 0.5 | `strategies/jab_tsll.py` |
| C-5 | QQQ | >= +0.7% | `qqq_min` | 0.7 | `strategies/jab_tsll.py` |

**파라미터 출처:** `common/params.py` JAB_TSLL

---

### D. 숏 잽모드 ETQ (Jab-ETQ)

> ETH 하락 기대 높을 때 숏 ETF. *(종목 정정: SETH → ETQ)*

| 항목 | 내용 |
|------|------|
| 매수 대상 | **ETQ** |
| 매수 비중 | M5 T1~T4 기준 |
| 목표 수익률 | **+1.05%** (net) |
| 매도 방식 | 전액 매도 |
| 진입 시간 | 17:30 KST ~ 장마감 |

**진입 조건:**

| Rule ID | 지표 | 조건 | 파라미터 | 현재값 | 코드 위치 |
|---------|------|------|----------|--------|----------|
| D-1 | Polymarket ETH 하락 기대치 | 최고 하락기대 - 평균 >= 12pp | `poly_down_spread_min` | 12.0 | `strategies/jab_etq.py` |
| D-2 | GLD | >= +0.01% | `gld_min` | 0.01 | `strategies/jab_etq.py` |
| D-3 | ETQ | >= 0.00% (양전) | `etq_min` | 0.0 | `strategies/jab_etq.py` |

> **[미확인] P-14:** "평균 수치"의 기간 정의 불명확 (7일? 30일? 어떤 Polymarket 마켓?)

**파라미터 출처:** `common/params.py` JAB_ETQ

---

### E. VIX → GDXU (VIX-Gold)

> 공포 지수 급등 → 금광 3x 매수

| 항목 | 내용 |
|------|------|
| 매수 대상 | GDXU (금광 3x) |
| 매수 비중 | M5 T1~T4 비율 기준 |
| 목표 수익률 | **+10.25%** |
| 매도 후 | 수익 전액 → **IAU** 매수 *(정정: GLD → IAU)* |

**진입 조건:**

| Rule ID | 지표 | 조건 | 파라미터 | 현재값 | 코드 위치 |
|---------|------|------|----------|--------|----------|
| E-1 | VIX 일간 변동 | >= +10% | `vix_spike_min` | 10.0 | `strategies/vix_gold.py` |
| E-2 | Polymarket 하락 기대 | >= 30% | `poly_down_min` | 0.30 | `strategies/vix_gold.py` |

**파라미터 출처:** `common/params.py` VIX_GOLD

---

### F. S&P500 편입 기업 (SP500-Entry)

> 신규 편입 다음 날 매수

| 항목 | 내용 |
|------|------|
| 매수 대상 | 편입 기업 (흑자 한정) |
| 매수 비중 | M5 T1~T4 비율 기준 |
| 목표 수익률 | **+1.75%** (수수료 제외 net) |
| 금지 조건 | GLD 상승 시, Polymarket NASDAQ < 51% |

**S&P500 편입 공격 매수 (MT_VNQ 추가 규칙):**

| 항목 | 내용 |
|------|------|
| 조건 | Polymarket 편입 확률 처음 시작 기준 평균 51%+, 최근 3년 재무제표 순이익 3회+ |
| 매수 | 편입 당일 50% 비중 |
| 다음날 GLD 상승 확률 60%+ | 비중 +30% |
| 다음날 GLD 상승 확률 20% 이하 | 비중 -25% |
| 매도 | 다음 영업일 17:30 전액 매도 |

**파라미터 출처:** `common/params.py` SP500_ENTRY

---

### G. 저가매수 (Bargain-Buy)

> 3년 최고가 대비 폭락 시 진입. SNXX, OKLL 제외 (Q-10 반영)

**종목별 파라미터:**

| 종목 | 하락 진입 | 추가매수 | 목표 (M20 반영) | 분할매도 | 수익금 행선지 | 비고 |
|------|----------|---------|----------------|---------|-------------|------|
| CONL | -80% | -3% 추가 하락 | **+100%, 비중 30%, 기한 60일** | 전액 | CONL 30일 분할 | VNQ 120일선 이하 조건부. 당일 금 폭락 시 +30% 추가 |
| SOXL | -90.5% | -5% | **+100%, 비중 30%, 기한 60일** | 6회 | SOXL 30일 분할 | VNQ 120일선 이하 조건부. 당일 금 폭락 시 +30% 추가 |
| AMDL | -89% | -5% | +40.25% | 6회 | SOXL 30일 분할 | |
| NVDL | -73% | -5% | +200.25% | 6회 | SOXL 30일 분할 | |
| ROBN | -83% | -3% | +200.25% | 6회 | CONL 30일 분할 | |
| ETHU | -95% | **추가매수 없음** | +20.25% | 6회 | ROBN 100일 분할 | |
| BRKU | **-31%** | -3% | +0.5% *(코드값: 0.5, 문서-코드 불일치 — 태준 확인 필요)* | 전액 | 현금화 | 단기 |
| NFXL | -26% | -20% | +0.9% *(코드값: 0.9, 문서-코드 불일치 — 태준 확인 필요)* | 전액 | 현금화 | 단기 |
| PLTU | -44% | -10% | +10.25% | 전액 | 현금화 | 20일 기한 |

> **CONL/SOXL VNQ 120일선 조건:** 최근 2년 기준 VNQ가 120일선 아래에 있는 경우에만 작동. 비중 30%, 기한 60일.

**저가매수 금지 조건:**

| 조건 | 파라미터 |
|------|----------|
| 금 하락 시 | `block_rules.gld_decline = True` |
| Polymarket 상승 49% 이하 | `block_rules.poly_ndx_min = 0.49` |
| 유상증자 / 투자자 기대치 낮음 | 코드 외부 조건 |
| 3일 전후 거래량 감소 | `block_rules.volume_decline_days = 3` |

**파라미터 출처:** `common/params.py` BARGAIN_BUY

> **[미확인] P-12:** "당일 금 폭락"의 정의 (몇 % 하락?)

---

### H. 숏포지션 전환 (Short-Macro)

> 나스닥/S&P500 ATH → 전면 숏 전환

| Rule ID | 유형 | 조건 | 파라미터 | 현재값 | 코드 위치 |
|---------|------|------|----------|--------|----------|
| H-1 | 진입 | 나스닥/S&P500 역대 최고가 | `conditions.index_ath` | True | `strategies/short_macro.py` |
| H-2 | 행동 | GDXU/IAU/GLD/현금 제외 모든 롱 매도 | `action.sell_all_except` | ["GDXU","IAU","GLD","cash"] | `strategies/short_macro.py` |
| H-3 | GDXU | 100% 구축, +90.25% 목표 | `action.build_gdxu_pct` | 1.0 | `strategies/short_macro.py` |
| H-4 | 매도 후 | 수익 전액 → IAU 매수 | `action.reinvest_ticker` | "IAU" | `strategies/short_macro.py` |
| H-5 | 청산 | 전액 매도 *(정정: 분할 → 전액)* | `exit.exit_type` | "full_sell" | `strategies/short_macro.py` |

**파라미터 출처:** `common/params.py` SHORT_MACRO

---

### I. 리츠 리스크 (REIT-Risk) — VNQ 대입

> 리츠 과열 → 레버리지 90일 금지 + 공격 조절 모드 발동

#### I-1. 기본 조건 (VNQ 정정)

| Rule ID | 유형 | 조건 | 파라미터 | 현재값 | 코드 위치 |
|---------|------|------|----------|--------|----------|
| I-1-1 | 진입 | VNQ 7일 연속 상승 평균 +0.1% | `conditions.reits_7d_return_min` | 1.0 | `strategies/reit_risk.py` |
| I-1-2 | 진입 | 탐욕지수 >= 75 | - | - | `strategies/reit_risk.py` |
| I-1-3 | 행동 | GDXU 제외 레버리지 매매 90일 중단 | `action.ban_days` | 90 | `strategies/reit_risk.py` |

#### I-2. 리츠 과열 공격 조절 모드

| 원래 종목 | 교체 종목 |
|----------|---------|
| SOXL | SOXX |
| ROBN | HOOD |
| GDXU | GLD |

**목표 수익률 조정:**

| 적용 범위 | 조정 |
|----------|------|
| 공격 조절 모드 종목 (SOXX, HOOD, GLD) | 기존 목표 수익률 × 50% |
| 리츠 과열 제외 모든 전략 | 목표 수익률 -0.5% |

#### I-3. 조심모드 (VNQ로 정정, 2026-02-22 확인)

| Rule ID | 유형 | 조건 | 파라미터 | 현재값 | 코드 위치 |
|---------|------|------|----------|--------|----------|
| I-3-1 | 트리거 | **VNQ** 전날 7일 거래 상승률 1%+ | `conditions.reits_7d_return_min` | 1.0 | `strategies/reit_risk.py` |
| I-3-2 | 행동 | 조심모드 발동: 레버리지 비중 50% 이하 제한 | `cautious_mode.attack_leverage_pct` | 50 | `strategies/reit_risk.py` |

> *(정정: 기존 한국 리츠 3개(SK리츠/TIGER리츠/롯데리츠) → VNQ 단일 지표로 대체)*

**파라미터 출처:** `common/params.py` REIT_RISK

---

### J. 섹터 로테이션 / M80 거래대금 초단타

> Q-8로 전면 재설계. M80 점수 체계로 매수 순서 결정.

**섹터별 매수 규칙 (M80 기준):**

| 섹터 | 매수 종목 | 거래대금 조건 | 재무 조건 | 주기 |
|------|---------|------------|---------|------|
| 비트코인 | MSTU/CONL/IRE/RIOX/CRCA/BMNU 中 조건 부합 | 2년 최저 대비 3.1배+ | 기초자산 분기 순이익 양수 | 7일 1회 |
| 반도체 | SOXL, NVDL | 2년 최저 대비 3.1배+ | SOXX 순이익 양수 | 3일 1회 |
| 금 | GDXU (GDX 적자 시 IAU) | 2년 최저 대비 2배+ | AEM+NEM 모두 순이익 양수 | 14일 1회 |
| 헬스케어 | LLYX/JNJ/NVOX 中 조건 부합 | 2.7~2.8배 이상 | JNJ+LLY 모두 순이익 양수 | 7일 1회 |
| 에너지 | ERX/XOMX/CVX 中 조건 부합 | 2.2배+ | 각 기초자산 순이익 양수 | 7일 1회 |
| 은행 | ROBN (HOOD 80%+), CONL (110%+), FAS (XLF) | 각 조건 충족 | 각 기초자산 순이익 양수 | 7일 1회 |

**운영 규칙:**
- 현금 없으면 매수 금지
- M200 점수 높은 순서대로 매수
- 순서 바뀌면 즉시 매도 후 다음 순번 매수
- M7 발동 시 동일 종목 J 신호 대체

**파라미터 출처:** `common/params.py` SECTOR_ROTATE

---

### K. 조건부 은행주 (Bank-Conditional)

> 대형은행 전부 상승 + BAC만 하락 → 역전 기대

| Rule ID | 유형 | 조건 | 파라미터 | 현재값 | 코드 위치 |
|---------|------|------|----------|--------|----------|
| K-1 | 감시 | JPM, HSBC, WFC, RBC, C 모두 양전 | `watch_tickers` | ["JPM","HSBC","WFC","RBC","C"] | `strategies/bank_conditional.py` |
| K-2 | 진입 | BAC 마이너스 | `condition` | "watch_all_positive_target_negative" | `strategies/bank_conditional.py` |
| K-3 | 금액 | 300만원 | `amount_krw` | 3,000,000 | `strategies/bank_conditional.py` |
| K-4 | 목표 | +1.05% (net) | `target_pct` | 1.05 | `strategies/bank_conditional.py` |
| K-5 | 청산 | 전액 매도 → 현금화 | `reinvest` | "cash" | `strategies/bank_conditional.py` |

**파라미터 출처:** `common/params.py` BANK_CONDITIONAL

---

### L. 숏 포지션 — 재무 역전 (Short-Fundamental)

| 항목 | 내용 |
|------|------|
| 매수 대상 | CONZ, IREZ, HOOZ, MSTZ (4개 중 **1개만** 매수) |
| 우선 순위 | 현재-전 분기 재무제표 차이가 가장 큰 종목 |
| 진입 조건 | **Polymarket 상승 기대치 20% 이하** + 기초자산 최근 분기 적자 |
| 목표 수익률 | +1.15% (net) |
| 매도 방식 | 전액 매도 |

**종목-기초자산 매핑:**

| 숏 ETF | 기초자산 | 숏 허가 조건 | 숏 불가 조건 |
|--------|---------|------------|------------|
| CONZ | COIN | COIN 분기 순이익 마이너스 | COIN 분기 순이익 플러스 |
| MSTZ | MSTR | MSTR 분기 순이익 마이너스 | MSTR 분기 순이익 플러스 |
| IREZ | IREN | IREN 분기 순이익 마이너스 | IREN 분기 순이익 플러스 |
| HOOZ | HOOD | HOOD 분기 순이익 마이너스 | HOOD 분기 순이익 플러스 |

**코드 위치:** `strategies/bearish_defense.py`

---

### M. 비이상적 재난 이머전시 모드

| 조건 | 행동 |
|------|------|
| Polymarket 어떤 지수든 30pt 이상 급변 | 수익 중인 전 종목 즉시 지정가 매도 |

**재난 이머전시 1 (BTC 급등):**

| 항목 | 내용 |
|------|------|
| 조건 | Polymarket BTC 기존 수치 대비 **30pt+ 급등** |
| 매수 | **BITU** 전액 지정가 매수 |
| 목표 수익률 | **+3.15%** (기본 0.9% + 상향 2.0% + M20 0.25%) |
| 매도 | 목표 달성 시 전액 매도 |

**재난 이머전시 2 (NASDAQ 급등):**

| 항목 | 내용 |
|------|------|
| 조건 | Polymarket NASDAQ 기존 수치 대비 **30pt+ 급등** |
| 매수 | **SOXL** 전액 지정가 매수 |
| 목표 수익률 | **+3.15%** |
| 매도 | 목표 달성 시 전액 매도 |

**재난 이머전시 3 (NASDAQ 급락):**

| 항목 | 내용 |
|------|------|
| 조건 | Polymarket NASDAQ 기존 수치 대비 **30pt 이상 폭락** |
| 매수 | **SOXS** 전액 지정가 매수 |
| 목표 수익률 | **+3.15%** |
| 매도 | 목표 달성 시 전액 매도 |

**파라미터 출처:** `common/params.py` EMERGENCY_MODE (`poly_swing_min = 30.0`, `target_net_pct = 0.9`)

**코드 위치:** `strategies/emergency_mode.py`

---

## 4. 인프라 규칙

### 오케스트레이터 파이프라인 (infra/orchestrator.py)

```
M200(즉시매도) → M201(즉시전환) → SCHD 매도차단 → 리스크/이머전시 → M3 거래시간 → M28 포지션게이트 → M5 T1~T4 배분 → M6 리츠감산 → 금지티커 처리 → OrderQueue.submit()
```

### 주문 엔진 (infra/limit_order.py)

| 항목 | 내용 |
|------|------|
| 주문 유형 | 지정가 전용 (LimitOrder) |
| 주문 TTL | `order_ttl_sec = 120초` |
| 재시도 최대 | `order_retry_max = 3` |
| 재시도 가격 보정 | `order_slip_pct = 0.001` (0.1%) |
| Fill Window | `fill_window_sec = 10초` |

---

## 5. 필터 규칙

### 서킷 브레이커 (filters/circuit_breaker.py)

| Rule ID | 유형 | 조건 | 파라미터 | 현재값 |
|---------|------|------|----------|--------|
| CB-1 | VIX 급등 | VIX 일간 +6% 이상 → 7거래일 신규 매수 금지 | `cb1_vix_min`, `cb1_days` | 6.0, 7 |
| CB-2 | GLD 급등 | GLD +3% 이상 → 3거래일 신규 매수 금지 | `cb2_gld_min`, `cb2_days` | 3.0, 3 |
| CB-3 | BTC 급락 | BTC -5% 이상 → 조건 해제 시까지 금지 | `cb3_btc_drop` | -5.0 |
| CB-4 | BTC 급등 | BTC +5% 이상 → 추격매수 금지 | `cb4_btc_surge` | 5.0 |
| CB-5 | 금리 상승 | Polymarket 금리상승 50%+ → 전면 금지 + 레버리지 3일 대기 | `cb5_rate_hike_prob`, `cb5_lev_cooldown_days` | 0.50, 3 |
| CB-6 | 과열 종목 | +20% 이상 → 비레버리지 전환 (고점 -10% 시 자동 복귀) | `cb6_surge_min`, `cb6_recovery_pct` | 20.0, -10.0 |

**파라미터 출처:** `common/params.py` CIRCUIT_BREAKER

### ATR 손절 (filters/stop_loss.py)

| 항목 | 파라미터 | 현재값 |
|------|----------|--------|
| ATR 기간 | `atr_period` | 14 |
| 일반 ATR 배수 | `atr_multiplier` | 1.5 |
| 강세장 ATR 배수 (Poly NDX>=70%) | `atr_multiplier_bullish` | 2.5 |
| 고변동성 관찰 기간 | `high_vol_lookback` | 5거래일 |
| 고변동성 최소 발생 횟수 | `high_vol_min_count` | 2회 |

**레버리지별 고정 손절:**

| 배수 | 일간 등락률 기준 | 손절 |
|------|---------------|------|
| 1x | 10% 이상 | -4% |
| 2x | 15% 이상 | -6% |
| 3x | 20% 이상 | -8% |

**파라미터 출처:** `common/params.py` STOP_LOSS

### Polymarket 품질 필터 (filters/poly_quality.py)

| 항목 | 파라미터 | 현재값 |
|------|----------|--------|
| 최소 확률 | `min_prob` | 0.02 |
| 최대 확률 | `max_prob` | 0.99 |
| 최소 변동 시간 | `min_volatility_hours` | 5시간 |
| 미갱신 시 | `stale_pause` | 매매 정지 |

---

## 6. 포트폴리오 규칙

### 수수료 기준

| 항목 | 내용 |
|------|------|
| 왕복 수수료 | **0.74%** |
| net 목표 기준 | target_gross = target_net + 0.74% |

### 자산 모드 분류

| 모드 | 전략 목록 |
|------|---------|
| 공격 | jab_soxl, jab_bitu, jab_tsll, jab_etq, bargain_buy, vix_gold, sp500_entry, bank_conditional, short_macro, emergency_mode, crash_buy, soxl_independent |
| 방어 | sector_rotate |
| 조심 | 공격 모드 레버리지 비중 50% 이하 제한 |

**파라미터 출처:** `common/params.py` ASSET_MODE

### 급락 역매수 (Crash Buy)

| 항목 | 파라미터 | 현재값 |
|------|----------|--------|
| 대상 종목 | `tickers` | ["SOXL","CONL","IRE"] |
| 당일 하락 트리거 | `drop_trigger` | -30% |
| LULD 거래중단 횟수 | `luld_count_min` | 3회 이상 |
| 매수 비율 | `buy_pct` | 95% |
| 진입 시각 | `entry_et_hour:min` | ET 15:55 |

---

## 7. 미확인/미반영 항목

### [미확인] 항목 (태준 확인 필요)

| P-ID | 관련 섹션 | 내용 | 현황 |
|------|---------|------|------|
| P-1 | 수익금 분배 | 4종목 구매 순서, 수익금 부족 시 처리 방식 | 미확인 |
| P-2 | 리츠 과열 | "제외항 0.5% 낮춤" 범위 | 미확인 |
| P-3 | M5 차감 | 차감 시점 및 "각각" 범위 (매수 완료 vs 세트 완료) | 미확인 |
| P-4 | TSLL | 200만원 한도 vs M5 비율 우선순위 | 미확인 |
| P-5 | M200 | 조건2 "44POLYMARKET BTC" 기준 불명확 — 정의 전까지 OFF 유지 | 코드 확인: `p5_44poly_btc_enabled = False` (`common/params.py` ENGINE_CONFIG) |
| P-12 | G전략 | "당일 금 폭락" 수치 정의 | 미확인 |
| P-13 | I전략/M6 | 리츠 과열 ×50% 후 M6 0.8% 하한 적용 여부 | 미확인 |
| P-14 | D전략 (ETQ) | Polymarket ETH 평균 수치 기간 정의 | 미확인 |
| P-15 | M201 | 발동 후 포지션 전환 규모 (전액? 일부?) | 미확인 |
| P-16 | M4/REIT_MIX | KR 리츠 일부 결측 시 처리 방식 | 미확인 |
| P-17 | Fill Window | PARTIAL_ENTRY_DONE 이후 재진입 조건 | 미확인 |
| P-18 | M201 | SHORT 청산 조건 값 — 코드 `p >= 0.55` vs 리뷰 노트 `p >= 0.60` 불일치 | 태준 확인 필요 |
| P-19 | G전략 | BRKU/NFXL target_pct 코드값(0.5%/0.9%)과 문서값(0.75%/1.15%) 불일치 | 태준 확인 필요 |

### [리뷰 미반영] 항목 (코드 구현 필요)

| 항목 | 내용 | 이슈 |
|------|------|------|
| M4 부동산 이머전시모드 | REIT_MIX 기반 5종 모드 상태 머신 구현 | CI-11 |
| 재난 이머전시모드 목표 | 최종 목표 3.15% params에 미반영 | CI-13 |
| 숏 전략 (L) | CONZ/IREZ/HOOZ/MSTZ 재무제표 API 연동 | CI-14 |
| Orchestrator 완전 구현 | M3/M5/M6 로직 완전 통합 | A-4 |
| BARGAIN_BUY target_pct | params.py에 구버전 (CONL: 188%, SOXL: 320%) 반영 — VNQ 조건부 100%로 업데이트 필요 | C-3 |

---

## 8. 정정 이력 (VNQ 기반 정정)

| 날짜 | 원본 | 정정 | 출처 |
|------|------|------|------|
| 2026.2.21 | SK리츠(395400) 추종 | **VNQ (미국 리츠 ETF) 대입** | MT_VNQ |
| 2026.2.21 | SK리츠 120일선 기준 | **VNQ 120일선** (저가매수 G전략) | MT_VNQ Q-3/Q-4 |
| 2026.2.21 | SETH 종목 | **ETQ**로 변경 | MT_VNQ Q-2 |
| 2026.2.21 | XRP +5.0% | **+2.5% (장중 변동 기준)** | MT_VNQ Q-1 |
| 2026.2.21 | TSLL GLD <= +0.3% | **<= +0.1%** | MT_VNQ Q-5 |
| 2026.2.21 | BRKU 진입 -32% | **-31%** | MT_VNQ Q-6 |
| 2026.2.21 | ETHU 추가매수 | **추가매수 없음 (add_size=0)** | MT_VNQ Q-7 |
| 2026.2.21 | SNXX, OKLL | **전략에서 제외** | MT_VNQ Q-10 |
| 2026.2.21 | 목표수익률 전부 | **+0.25% 추가 (M20)** | MT_VNQ |
| 2026.2.21 | 이머전시 목표수익률 | **+2% 상향 (최종 3.15%)** | MT_VNQ |
| 2026.2.22 | I-3 한국 리츠 3종목 기준 | **VNQ 단일 지표로 대체** | 0222 원문 |
| 2026.2.22 | M200 신설 | **즉시 매도 7조건 신설** | 0222 원문 |
| 2026.2.22 | M80 신설 | **거래대금 초단타 점수 체계 신설** | 0222 원문 |
| 2026.2.22 | NVDL 대체 종목 누락 | **NVDL → NVDA 추가** | 0222 원문 |
| **2026.2.23** | M1 주문 TTL 5 bars(5분) | **120초(2분) 확정** | MT_VNQ3 §3-3 |
| **2026.2.23** | M4 VNQ 단독 지표 | **REIT_MIX = VNQ + KR 리츠 평균** | MT_VNQ3 §9 |
| **2026.2.23** | M5 T5~ "매수 없음" | **예약/대기 표시 + 10초 취소 룰** | MT_VNQ3 §10 |
| **2026.2.23** | M6 적용 순서 불명확 | **적용 순서 고정 + 리츠 감산 제외항 확정** | MT_VNQ3 §11 |
| **2026.2.23** | M201 신설 | **BTC 확률 급변/전환 즉시모드** | MT_VNQ3 §14 |
| **2026.2.23** | M28 신설 | **Polymarket BTC 포지션 허가 게이트** | MT_VNQ3 §13 |
| **2026.2.23** | M300 신설 | **USD-only 환전 금지** | MT_VNQ3 §17 |
| **2026.2.23** | Fill Window 신설 | **10초 룰** | MT_VNQ3 §5 |
| **2026.2.23** | 추격매수 금지 신설 | **reference_price 고정** | MT_VNQ3 §6 |
| **2026.2.23** | 매도 즉시성 신설 | **bid 0.2% 이내 marketable limit** | MT_VNQ3 §7 |
| **2026.2.23** | 포지션 분리 신설 | **confirmed/effective 분리** | MT_VNQ3 §4 |
| **2026.2.23** | MASTER SCHD 신설 | **30일 실현손익 기반 SCHD 매수** | MT_VNQ3 §16 |
| **2026.2.23** | M201 숏 청산 조건 | `p >= 0.60` → **`p >= 0.55`** (코드 m201_mode.py 기준 교차검증 수정) | rules-reviewer 검토 |
| **2026.2.23** | 재난 이머전시 BTC 급등 종목 | `CONL` → **`BITU`** (코드 emergency_mode.py btc_surge.ticker 기준) | rules-reviewer 검토 |
| **2026.2.23** | BRKU target_pct | `+0.75%` → **+0.5%** (코드값 0.5 기준, 태준 확인 필요) | rules-reviewer 검토 |
| **2026.2.23** | NFXL target_pct | `+1.15%` → **+0.9%** (코드값 0.9 기준, 태준 확인 필요) | rules-reviewer 검토 |

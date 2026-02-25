taejun_attach_pattern 전략 리뷰 — 2026-02-23 (VNQ 기반, MT_VNQ3 반영) 작성일:
2026-02-23 출처: taejun_strategy_review_2026-02-22_VNQ.md + MT_VNQ3.md (박태준)
엔진: strategies/taejun_attach_pattern/ 전일 대비 변경: MT_VNQ3 반영 — M201(즉시모
드) 신설, M28(Polymarket 게이트) 신설, M300(USD-only) 신설, MASTER SCHD 신설, M1
TTL 2분 확정, Fill Window 10초 룰, 추격매수 금지, 매도 즉시성, 포지션
confirmed/effective 분리, CI-0 v0.3 업데이트, 신규 이슈 CI-16~CI-20 요약 박태준 매매
전략 12개 + 비이상적 재난 이머전시 모드를 taejun_attach_pattern 엔진으로 구현. 본
문서는 전략별 상세 정리, 마스터 플랜(M0~M40, M80, M200, M201, M28, M300), 수익금
분배 규칙, Critical/Algorithm 이슈를 기록한다. SK리츠 기반 모든 조건을 VNQ(미국 리츠
ETF)로 대입 정정. 0. 마스터 플랜 (M0~M40 + M80, M200, M201, M28, M300) 모든 전략
/모드에 절대적으로 적용되는 공통 규칙. M0 — MTGA AI (시스템 점검) 항목내용 1순위
점검시스템/네트워크 연결속도, 매수매도 서버 상태, 컴퓨터 상태(RAM/CPU/GPU 사용량)
오류 대응코딩/계산 오류 발생 시 한국어로 아기도 이해하기 쉽게 설명하여 상태 점검
문제 발생 시알람 발송 공휴일 재시작KST 18:00에 서버/컴퓨터 수명을 위해 재시작 필
요. 반드시 MASTER J(박태준) 또는 MASTER H(박태현)에게 사전 허가 필수. 허가 미획득
시 절대 재시작 금지 매수매도 기록모든 매수매도 기록 보관 및 AI 자체 분석 수행 급등
분석레버리지 ETF 30%+ 급등 종목 발견 시 분석 보고 — 과거 2년 데이터, 오차범위
3% 이내 유사 패턴 정리 후 MASTER J에게 보고 M1 — 지정가 ALL STOCK 항목내용 핵
심 규칙모든 모드에서 절대 시장가 매수/매도 금지 (LIMIT만 허용) 적용 범위이머전시/공
격/방어/조심 모드 포함, 예외 없음 지정가 오류 해결한국투자증권 호가 소수점 2자리까
지만 지원. 소수점 0.001 이상 시 반올림하여 0.01 단위로 체결. 매수/매도 전부 이 조건
적용 원인지속적인 시장가 충돌의 원인이 소수점 호가 불일치에서 발생한 것으로 확인됨
주문 TTL (2분 확정) order_ttl_sec = 120초. TTL 초과 시 취소 요청 → 취소 ACK 수신 후
잔량 reserved_cash 해제 *(MT_VNQ3 §3-3 정정: 기존 5 bars(5분))*- 1
taejun_attach_pattern — 전략 리뷰 2026-02-23 (VNQ) 항목내용 prices.get(ticker, 0) 금지
0 체결 치명 버그 방지. 가격 결측 시 신호 스킵 + 알람 Tick/호가 처리US: 기본 0.01 /
KR: 구간별 tick (KR 리츠/VNQ는 지표용만, 실매매 금지) M2 — 즉시 행동 항목내용 규
칙조건이 달성된 날짜/시간에 즉시 행동 M3 — 거래 시간 및 휴장일 항목내용 거래 시
간17:30~06:00 KST, 월~금 제외공휴일/휴장일 비고진입 비율 테이블은 M5로 이관됨 날
짜설명 1월 1일 (목) New Year's Day 1월 19일 (월) Martin Luther King Jr. Day 2월 16일
(월) Presidents' Day 4월 3일 (금) Good Friday 5월 25일 (월) Memorial Day 7월 3일 (금)
Independence Day 관측일 9월 7일 (월) Labor Day 11월 26일 (목) Thanksgiving Day 12
월 25일 (금) Christmas Day • 기간: 2026년 3월 8일 ~ 11월 1일 • 하절기 (서머타임 적
용): 동부 09:30~16:00 = KST 22:30~05:00 • 동절기 (표준 시간): 동부 09:30~16:00 = KST
23:30~06:00 M4 — 부동산 변동 수익률 포지션 실시간 변경 모드 (신설) 항목내용 추종
지표REIT_MIX = (VNQ + KR 리츠들) 전일 종가 대비 변화율 평균 *(MT_VNQ3 §9 정정:
VNQ 단독 → 혼합)* 데이터 결측REIT_MIX 전부 결측이면 UNKNOWN → BUY_STOP +
알람 가동상시 가동, 1순위로 작동 서브모드총 5가지 모드조건 (VNQ 기준)목표수익률 변
경추가 제한 레버리지 스탑 해제(60) 3년 최고가 기준 6일 과거데이터로 60일선 아래
0.8~1.0% → 1.8%, 1.8~2.0% → 2.8%-- 2 taejun_attach_pattern — 전략 리뷰 2026-02-23
(VNQ) 모드조건 (VNQ 기준)목표수익률 변경추가 제한 레버리지 스탑 해제(120) 120일선
아래0.8~1.0% → 2.8%, 1.8~2.0% → 4.8%레버리지 GOLD/SHORT 매매 중단 레버리지 스
탑 사용 허가(60) 60일선 위기존 유지 (0.8~1.0% → 0.8%, 1.8~2.0% → 1.8%) 레버리지 스
탑 사용 허가(120) 120일선 위기존 유지레버리지 ETF 즉시 전액 매도(GDXU 제외), 1배수
대체(M7 참조) ALL IN ONE (20, 120) VNQ 20일선 시작가 -1.0% 또는 120일선-3% 하락
상태 기본 5%, GLD 하락 +1%, GLD 상승-0.5% 특수 조건 (아래 별도 정리) 항목내용 대
상 종목Cure, SOXL, TQQQ, CONL, ROBN, FAS, Ertlx, SOLT, ETHU (토스증권 거래대금 100
위 이내) 시작가 제한4%+ 상승 시 매수 금지 / 3% 이하 시 매수 허가 거래대금5년 최
저 대비 4배 이상 필수 Polymarket각 종목별 51%/49% 기준 허가/불허가 기한/목표-
1.0% → 1일(목표5%), -2.0% → 2일(목표10%), -3.0% → 2일(목표15%), 0.5% 미만은 붙이지
않음 재무건전성각 종목 추종 기초자산 최근 분기 순이익 양수 필수 M5 — 비중조절 메
커니즘 (개정) 순위조건 달성 수비중 산출 방식 T1 1개55% T2 2개나머지(45%)에서 40%
T3 3개나머지에서 33% T4 4개나머지 전액 T5~ 5개 이상실매수 0, 예약/대기 표시만
*(MT_VNQ3 §10 정정: "매수 없음" → 예약 대기 표시)* 예약(T5~) 10초 룰 *(MT_VNQ3
신설)*: T5~ 예약은 10초 경과 후 조건 미부합이면 예약 취소. [!] "예약 10초"와 "주문 잔
량 Fill Window 10초"는 서로 다른 개념: 예약 = 주문 없음 / Fill Window = 주문이 이미
나간 상태. ~~[!] 모호사항: T1 55%의 기준값 불명확. 전체 자산(총 포트폴리오) 대비
55%인지, 보유 현금 대비 55%인지 원문에 명시 없음.~~ [OK] 해결 (2026-02-22): 0222
원문 line 107~109 확인 — T1 비중은 전체 자산(현금+주식 포함) 대비 비율. "전체 자산
현금 주식 돈 모든 것에 비중이 몇퍼센트인지 표시"로 명시됨. 지표조건비중 조정 GLD
+1% 상승마다-0.1% GLD-1% 하락마다+0.1% VIX +2% 상승마다-3% VIX-2% 하락마다
+1% 달러전일 대비 상승+0.1% 달러전일 대비 하락-0.2% Polymarket
BTC/ETH/SOL/GOLD/NASDAQ 51%+해당 종목 허가 Polymarket
BTC/ETH/SOL/GOLD/NASDAQ 49% 이하해당 종목 불허가 ~~[!] 모호사항: GLD/VIX/달러
조정이 동시에 발생할 때 중첩 적용인지 독립 적용 후 합산인지 불명확.~~ [OK] 해결
(2026-02-22): 0222 원문 line 1204 —
"네 모두 동시에 발생시 다 포함해서 더해서 계산
합니다." 동시 발생 시 중첩 합산 적용 확인됨.
- 3 taejun_attach_pattern — 전략 리뷰
2026-02-23 (VNQ) 예) GLD +2%, VIX +4% 동시 발생 시 → -0.2% + -6% = -6.2% 합산.
조건행동 비중 -1% 이하매수 금지 비중 100.1% 이상매수 금지 현금 없음매수 금지 조
건행동 거래대금 최고가 대비 20% 이하 + 2일 이평선 이탈전액 매도 VIX 3%+ 상승레버
리지 ETF T1 비중 30% 매도 M6 — 목표수익률 하한 + 리츠 감산 + M20 가산 (신설,
MT_VNQ3 순서 확정) 적용 순서 고정 (MT_VNQ3 §11): 순서규칙내용 1 target_net 최소
하한0.8% 미만 금지 → 0.8%로 상향 2 M20 +0.25% 가산모든 target_net에 +0.25% 3리
츠 조심모드 감산조심모드 발동 시 target_net × 1/2 감산 4감산 후 하한 재확인감산 후
target_net ≤ 0.8% → 전 모드 BUY_STOP 리츠 감산 제외항 *(MT_VNQ3 §11 확정)*: 카테
고리종목 금/공포GLD, VIX, GDXU 숏 ETF CONZ, IREZ, HOOZ, MSTZ 기타SOXS, ETQ
target_gross = target_net + fee_roundtrip (청산/체결 판정은 gross 권장) M7 — 레버리지
ETF 금지 시 대체 종목 (신설, 0222 정정 반영) M4 레버리지 스탑 사용 허가(120) 발동
시 적용: 원래 종목대체 종목비고 SOXL SOXX ROBN HOOD FAS XLF Ertlx XKE Cure XLV
TQQQ QQQ SOLT금지 ETHU금지 QUBX QTUM BITU매매금지*(0222 정정: 기존 "반에크
비트코인 ETF HODL" → 매매금지)* NVDL NVDA *(0222 신설: 원문 line 2343 "NVDL ->
NVDA")* 선행지수는 변경하지 않음: BITU, ROBN, NVDL- 4 taejun_attach_pattern — 전략
리뷰 2026-02-23 (VNQ) *(0222 정정 요약: BITU는 매매금지로 변경, NVDL → NVDA 대체
종목 신설)* M20 — 신용/미수 거래 절대 금지 (신설) 항목내용 규칙어떤 경우에도 신용/
미수 거래 절대 금지 목표수익률 조정수수료를 제외한 모든 목표수익률에 +0.25% 추가
M40 — Polymarket 급등 비중 상향 (신설) 조건대상 종목비중 조정 Polymarket BTC
80%+ BTC/SOLT/ETHU/ROBN기존 비중 +15% (GLD -5% 급등 시 +3% 추가) Polymarket
NASDAQ 80%+ Cure/SOXL/TQQQ/FAS/Ertlx/QUBX기존 비중 +15% (GLD -5% 급등 시
+3% 추가) Polymarket BTC & NASDAQ 20% 이하GDXU비중 +20% 추가 매수 M200 —
즉시 매도 조건 (0222 신설, MT_VNQ3 파이프라인 확정) 출처: 0222 원문 line
1212~1233 + MT_VNQ3 §15 조건 1개라도 충족 시 목표수익률 무관하게 즉시 매도: #조
건내용 1거래대금 급감전날 거래대금 대비 장 시작 3시간 이내 15% 하락 2 Polymarket
BTC 44% *(원문 "44POLYMARKET BTC" — 모호, P-5 등록)* 3 GLD 급등GLD 6%+ 상승 4
VIX 급등VIX 10%+ 급등 5이평선 이탈모든 주식 20일 이동평균선 이탈 6기한 만기기한
만기 시 다음날 17:30~06:00 이내 매도 7 REIT_MIX 급등REIT_MIX 5%+ 상승 시 VNQ 관
련 즉시 정리 / 그 외 30% 축소 (GLD 숏 제외) 발동 파이프라인 *(MT_VNQ3 §15 확
정)*: 전역 락 ON → OPEN 주문 전부 취소 → 취소 ACK → reserved 재계산 → 청산 실
행(매도 즉시성 규정 적용) → 락 OFF [!] 모호사항 (P-5): M200 조건 2번
"44POLYMARKET BTC"의 44가 44% 기준인지 불명확. 정의 확정 전까지 OFF. SCHD 예
외: 어떤 모드에서도 SCHD 매도 불가 (단, MASTER H/J가 AI OFF 시만 허용) M201 — 즉
시모드 v1.0 (BTC 확률 급변/전환) *(MT_VNQ3 신설)* 우선순위: M200 다음, 리스크모드
이전 출처: MT_VNQ3 §14 입력: p (현재 BTC 상승 확률), p_prev (직전 확률), Δpp = (p -
p_prev) × 100 보유 상태조건행동- 5 taejun_attach_pattern — 전략 리뷰 2026-02-23
(VNQ) 보유 상태조건행동 LONG 보유 중p ≤ 0.45즉시 롱 청산 LONG 보유 중Δpp ≤ -
20pp AND p ≤ 0.49즉시 롱 청산 SHORT 보유 중p ≥ 0.60즉시 숏 청산 청산 직후 전환:
청산 종류p 조건이후 행동 롱 청산 후p ≤ 0.49즉시 숏 진입 (가용현금 전액) 롱 청산 후
0.49 < p < 0.50현금 유지 (중립) 숏 청산 후p ≥ 0.51즉시 롱 진입 (가용현금 전액) 숏 청
산 후0.49 < p < 0.50현금 유지 (중립) 전환 파이프라인 (강제): 전역 락 ON → OPEN 주
문 전부 취소 → 취소 ACK → reserved 재계산 → 청산/전환 실행 → 락 OFF 기존 "p ≥
0.55 및 p ≥ 0.70" 표기 완전 삭제 (중복/충돌 방지) ETQ 숏 잽모드는 ETH가 아니라
BTC(p) 기준 M28 — Polymarket BTC 포지션 허가 게이트 *(MT_VNQ3 신설)* 출처:
MT_VNQ3 §13 BTC Primary 소스 자동선택 (Volume Only): 항목내용 후보 마켓A: bitcoin-
price-on-\, B: bitcoin-above-on-\ 선택 기준volume_usd만으로 선택 (±1% 이내면 직전
유지) 출력 필드BTC_UP_PROB, BTC_PRIMARY_VOLUME_USD, BTC_ACTIVITY_INDEX_A/B
(참고용) 포지션 허가 게이트: BTC_UP_PROB허가 p ≥ 0.51 LONG 허가 p ≤ 0.49 SHORT
허가 0.49 < p < 0.51중립 (둘 다 불허) 업데이트 실패/스테일이면 조건 기반 매매 정지
(BUY_STOP) + 알람 M300 — USD-only 환전 금지 *(MT_VNQ3 신설)* 항목내용 원칙
MTGA AI는 어떠한 경우에도 FX(환전) 수행 금지 실매매 제한USD 결제 상품으로만 제한
VNQ/KR 리츠실매매 금지, 지표용으로만 유지 (signal-only) Fill Window 10초 룰
*(MT_VNQ3 신설)* 출처: MT_VNQ3 §5- 6 taejun_attach_pattern — 전략 리뷰 2026-02-
23 (VNQ) 진입 주문(매수/신규 숏 진입)은 fill_window_sec = 10초 최우선 적용. 단계내용
0~10초잔량 채우기 위해 주문 유지/관리 (시장가 금지 유지) 10초 경과 후 잔량 있으면
잔량 즉시 취소 취소 ACK 이후잔량 reserved_cash 해제 → 현금 복귀 이벤트 기록
PARTIAL_ENTRY_DONE — 동일 signal_bar_ts 재진입 금지 [!] "예약 10초(T5~)"와 구분:
예약 = 주문 없음 / Fill Window = 주문이 이미 나간 상태 추격매수 금지 *(MT_VNQ3
신설)* 출처: MT_VNQ3 §6 원칙내용 기준가격 고정모든 매수는 신호 발생 기준가격
(reference_price)으로만 제출 가격 상향 절대 금지체결을 위해 ask/bid + slip로 가격 올
리는 행위 금지 tick 보정 예외tick 제약으로 주문 거절 시만 기술 보정 허용 — BUY tick
보정은 항상 floor(내림) (reference_price를 넘는 보정 금지) 미체결 처리미체결은 "실패"
로 간주, 잔량은 현금 유지 매도 즉시성 허용 *(MT_VNQ3 신설)* 출처: MT_VNQ3 §7 원
칙내용 LIMIT-only 유지시장가 금지 및 LIMIT-only 유지 청산/리스크 매도bid 기반 0.2%
이내 범위로 marketable limit 허용 적용 범위M200 / M201 / 리스크 청산성 매도에 우선
적용 매수와 완전 분리추격매수 금지(reference_price 고정)와 독립 규정 포지션
confirmed vs effective 분리 *(MT_VNQ3 신설)* 출처: MT_VNQ3 §4 포지션 타입정의용도
position_effective지금 실제로 체결되어 보유한 수량/평단(vwap)리스크/청산/현금계산
/M200/M201 position_confirmed진입이 계획대로 완료되었는지만 판단하는 상태전략 판
단/신호용 position_confirmed 확정 이벤트: FILLED 완료 또는 Fill Window 종료 금지/오
타/대체 티커 처리 정책 *(MT_VNQ3 신설)* 상황처리 금지/오타 티커매수목록 즉시 제외
(drop), shrink 계산에서도 제외 대체 티커 정의 있음원 티커 → 대체 티커로 치환 후 매
수- 7 taejun_attach_pattern — 전략 리뷰 2026-02-23 (VNQ) 상황처리 대체 불가drop +
알람 MASTER SCHD — 장기 적립 (절대 매도 금지) *(MT_VNQ3 신설)* 출처: MT_VNQ3
§16 항목내용 매수 조건최근 30일 실현손익이 +일 때만 SCHD 매수 매수 상한수익 구간
별 최대 50만원 절대 매도 금지어떤 모드에서도 SCHD 매도 불가 예외MASTER H 또는 J
가 "AI 매매 기능 OFF" 했을 때만 매도 허용 배당자동 재투자 허용 수익 구간별 매수 기
준: 30일 실현손익매수 금액 +1% 10만원 +2% 20만원 +3% 30만원 +4% 40만원 +5% 50
만원 (최대) 우선순위 고정 *(MT_VNQ3 §1 확정)*: M200(즉시매도) > M201(즉시전환) >
리스크모드 > 거래시간/휴장 > 비중배분(T1~T4) > 일반매매 M200 — 매수 우선순위
(0222 신설) 출처: 0222 원문 line 1153~1201 M200 점수 체계: 확률점수(GLD/Polymarket
합산) 중첩 점수가 높을수록 우선 매수. 200점 만점. 점수거래대금 기준 (5년 최저가 대
비 배수) 1.2점1.2배 이하 2점2배 이하 3점3배 이하 4점4배 이하 5점5배 이하 ... (1.2~18
배까지 비례 배점) 18점18배 이하 점수목표수익률 범위 5점0.1~1% 6~10점1.1~2% 15점
2.1~3% 20점3.1~4%- 8 taejun_attach_pattern — 전략 리뷰 2026-02-23 (VNQ) 점수목표
수익률 범위 25점4.1~4.5% 30점4.6~5% 35점5.1~6% 70점6.1%+ 0-A. 수익금 분배 규칙
매도 후 수익금 처리: 규칙내용 주기1일당 각 1주씩 매수 대상 종목SOXL, ROBN, GLD,
CONL 방식수익금 발생 시 4종목 순서대로 1주씩 구매 1. 전략 일람 (A~K 11개 + L, M
신규 2개 = 13개) #전략명코드 파일유형종목목표 수익 A잽모드 SOXL jab_soxl.py프리마
켓 단타SOXL +1.15% (net) B잽모드 BITU jab_bitu.py프리마켓 단타BITU +1.15% (net) C잽
모드 TSLL jab_tsll.py소액 단타TSLL +1.25% (net) D숏 잽모드 ETQ jab_etq.py숏 단타ETQ
*(정정: SETH->ETQ)* +1.05% (net) E VIX -> GDXU vix_gold.py이벤트 드리븐GDXU -> IAU
+10.25% F S&P500 편입sp500_entry.py이벤트 드리븐편입 기업+1.75% (net) G저가매수
bargain_buy.py스윙/장기9종목 *(SNXX/OKLL 제외)*종목별 상이 H숏포지션 전환
short_macro.py매크로GDXU +90.25% I리츠 리스크 (VNQ) reit_risk.py리스크 관리VNQ J섹
터 로테이션sector_rotate.py순환 배분다종목 K조건부 은행주bank_conditional.py역전 매
매BAC +1.05% (net) L숏 포지션 (재무 역전) short_fundamental.py재무 역전
CONZ/IREZ/HOOZ/MSTZ +1.15% (net) M비이상적 재난 이머전시disaster_emergency.py이
벤트CONL/SOXL/SOXS +0.9% (net) 목표 수익률에 M20 지시에 따라 기존 대비 +0.25%
추가 반영됨 *(0222 신설)* M80 거래대금 초단타 모드: 섹터 로테이션(J)을 대체하는 신
규 마스터플랜. 거래대금 점수 기반 적립식 매매 체계. 상세는 M80 섹션 참조. 2. 전략별
상세 A. 잽모드 SOXL (Jab-SOXL) 반도체 개별주 모두 상승인데 SOXX/SOXL만 마이너스
-> 레버리지 괴리 역전- 9 taejun_attach_pattern — 전략 리뷰 2026-02-23 (VNQ) 항목내
용 진입 시간17:30 KST ~ 장마감 매수 대상SOXL 매수 비중M5 T1~T4 비율 기준 *(L5-5:
M3→M5 이관 반영)* 목표 수익률+1.15% (수수료 제외 net) *(정정: 0.9% + 0.25%)* 매도
방식전액 매도 진입 조건 (16개 ALL 충족): #종목/지표조건비고 1 Polymarket NASDAQ
>= 51% 2 GLD >= +0.1% 3 QQQ >= +0.3% 4 SOXX <= -0.2%지수는 마이너스 5 SOXL
<= -0.6%매수 대상, 과매도 6 NVDA >= +0.9% 7 AMD >= +0.9% 8 SMCI >= +1.0% 9
KLA >= +0.8% 10 AMAT >= +0.8% 11 AVGO >= +0.55% 12 MPWR >= +0.55% 13 TXN
>= +0.66% 14 ASML >= +1.0% 15 LRCX >= +0.8% 16 MU >= +0.55% B. 잽모드 BITU
(Jab-BITU) BTC 생태계 상승 + BITU 과매도 -> 역전 항목내용 진입 시간17:30 KST ~ 장
마감 매수 대상BITU 매수 비중M5 T1~T4 비율 기준 *(L5-5: M3→M5 이관 반영)* 목표
수익률+1.15% (수수료 제외 net) *(정정: 0.9% + 0.25%)* 매도 방식전액 매도 진입 조건
(ALL 충족): #종목/지표조건비고 1 Polymarket BTC >= 63% 2 GLD >= +0.1% 3 BITU <= -
0.4% 4 BTC (스팟) >= +0.9%- 10 taejun_attach_pattern — 전략 리뷰 2026-02-23 (VNQ)
#종목/지표조건비고 5 ETH (스팟) >= +0.9% 6 SOL (스팟) >= +2.0% 7 XRP (스팟) >=
+2.5% *(Q-1 반영: +5.0% -> +2.5%, 장마감 기준 아님, 하루 중 변동 기준)* C. 잽모드
TSLL (Jab-TSLL) TSLA 상승 + TSLL 과매도 -> 소액 역전 항목내용 진입 시간17:30 KST ~
장마감 매수 대상TSLL 매수 비중200만원 이하 소액 목표 수익률+1.25% (수수료 제외
net) *(정정: 1.0% + 0.25%)* 매도 방식전액 매도 -> 현금 보유 진입 조건 (ALL 충족): #종
목/지표조건비고 1 Polymarket NASDAQ >= 63% 2 GLD <= +0.1% *(Q-5 반영: <= +0.3%
-> <= +0.1%)* 3 TSLL <= -0.8% 4 TSLA >= +0.5% 5 QQQ >= +0.7% D. 숏 잽모드 ETQ
(구 SETH -> ETQ로 정정) !! 종목 정정: SETH -> ETQ 항목내용 매수 대상ETQ *(정정:
SETH -> ETQ)* 목표 수익률+1.05% (net, 구 0.8% + 0.25%) 매도 방식전액 매도 진입 조
건 (Q-2 반영): #지표조건 1 Polymarket ETH 하락 기대평균 수치보다 12 미만 시 작동 2
GLD >= +0.01% 3 ETQ >= 0.00% (양전) [!] 모호사항 (Q-2): "평균 수치보다 12 미만"에서
평균 수치의 정의 불명확. 어떤 기간의 평균인지(1일? 7일? 30일?), 어떤 Polymarket 마켓
의 수치인지 확인 필요. 0222 원문에서도 명시적 답변 없어 [!] 유지. 시간 조건: 17:30
KST ~ 장마감 (M3 규칙 적용) E. VIX -> GDXU (VIX-Gold)- 11 taejun_attach_pattern —
전략 리뷰 2026-02-23 (VNQ) 공포 지수 급등 -> 금광 3x 매수 항목내용 매수 대상
GDXU (금광 3x) 매수 비중M5 T1~T4 비율 기준 *(L5-5: M3→M5 이관 반영)* 목표 수익
률+10.25% *(정정: 10% + 0.25%)* 매도 후수익 전액 -> IAU 매수 (정정: GLD -> IAU) 진
입 조건: #지표조건 1 VIX 일간 변동>= +10% 2 Polymarket 하락 기대>= 30% F. S&P500
편입 기업 (SP500-Entry) 신규 편입 다음 날 매수 항목내용 매수 대상편입 기업 (흑자 한
정) 매수 비중M5 T1~T4 비율 기준 *(L5-5: M3→M5 이관 반영)* 목표 수익률+1.75% (수
수료 제외 net) *(정정: 1.5% + 0.25%)* 금지 조건GLD 상승시, Polymarket NASDAQ <
51% S&P500 편입 공격 매수 (MT_VNQ 추가 규칙): 항목내용 조건Polymarket 편입 확률
처음 시작 기준 평균 51%+, 최근 3년 재무제표 순이익 3회+ 매수편입 당일 50% 비중
다음날 GLD 상승 확률 60%+비중 +30% 다음날 GLD 상승 확률 20% 이하비중 -25% 매
도다음 영업일 17:30 전액 매도 G. 저가매수 (Bargain-Buy) 3년 최고가 대비 폭락시 진입
Q-6, Q-7, Q-10 반영 종목하락 진입추가매수 조건목표분할매도수익금 행선지비고 CON
L-80%-3% 추가 하락VNQ 120일선 조건부 100% (60일 기한, 비중 30%) 전액CONL 30일
분할 금 폭락 시 추가 +30% SOX L-90.5%-5% VNQ 120일선 조건부 100% (60일 기한,
비중 30%) 6회SOXL 30일 분할 금 폭락 시 추가 +30% AM DL-89%-5% +40.25% 6회
SOXL 30일 분할NVD L-73%-5% +200.25% 6회SOXL 30일 분할- 12 taejun_attach_pattern
— 전략 리뷰 2026-02-23 (VNQ) 종목하락 진입추가매수 조건목표분할매도수익금 행선지
비고 ROB N-83%-3% +200.25% 6회CONL 30일 분할 ETH U-95% ~~추가매수 없음~~
*(Q-7: 삭제)* +20.25% 6회ROBN 100일 분할 BRK U-31% *(Q-6: -32% ->-31%)*-3%
+0.75%전액현금화단기 NFX L-26%-20% +1.15%전액현금화단기 PLT U-44%-10%
+10.25%전액현금화20일 기한 SNXX, OKLL: Q-10 반영으로 전략에서 제외 CONL/SOXL
VNQ 120일선 조건부 목표 (Q-3, Q-4 반영): • 조건: 최근 2년 기준 VNQ(구 SK리츠)가
120일선 아래에 있는 경우 • 목표 수익률: 100% • 기한: 60일 • 비중: 전체 비중의 30%
허가 • 당일 금 폭락 시: 추가매수 비중 +30% 저가매수 금지 조건: 유상증자 / 투자자
기대치 낮음 / 3일 전후 거래량 감소 / 금 하락 / Poly 상승 49% 이하 H. 숏포지션 전환
(Short-Macro) 나스닥/S&P500 ATH -> 전면 숏 전환 항목내용 진입 조건나스닥/S&P500
역대 최고가 행동GDXU/IAU/GLD/현금 제외 모든 롱 매도 GDXU 100% 구축, +90.25%
매도 *(정정: 90% + 0.25%)* 매도 후수익 전액 -> IAU 매수 청산전액 매도 (정정: 분할매
도 -> 전액매도) I. 리츠 리스크 (REIT-Risk) -- VNQ 대입 !! 추종 종목: SK리츠(395400) ->
VNQ (미국 리츠 ETF) 로 정정 조건기준 VNQ 7일 연속 상승평균 +0.1% 탐욕지수>= 75
행동GDXU 제외 레버리지 매매 90일 중단 리츠 과열 트리거 발동시, 공격 전략의 매수
대상 종목을 비레버리지로 교체: 원래 종목교체 종목 SOXL SOXX ROBN HOOD- 13
taejun_attach_pattern — 전략 리뷰 2026-02-23 (VNQ) 원래 종목교체 종목 GDXU GLD
목표 수익률 조정 (리츠 과열시): 적용 범위조정 공격 조절 모드 종목 (SOXX, HOOD,
GLD)기존 목표 수익률 x 50% 리츠 과열 제외 모든 전략목표 수익률 -0.5% 항목내용 트
리거VNQ 전날 7일 거래 상승률 1%+ 시 행동조심모드 발동: 공격모드 자산 레버리지
비율 기존 전액 -> 50% 이하로 축소 [OK] 해결 (2026-02-22): I-3 조심모드 기준을 VNQ
로 정정 확인. 기존 한국 리츠 3개(SK리츠/TIGER리츠/롯데리츠) -> VNQ 단일 지표로 대
체. J. 섹터 로테이션 (Sector-Rotate) -- Q-8 전면 재설계 !! Q-8 답변으로 전면 재설계됨
우선순위 체계 (S1 > S2 > S3...): • 거래대금 점수 (5년 최저 대비): 2점=3배, 3점=3.5배, 4
점=4.3배 • 수익률 높은 순서대로 매매 순위 결정 섹터별 매수 규칙: 섹터매수 종목조건
주기 비트코인CONL/ROBN/CRCA/IRE/MSTU/ROTX 中 조건 부합거래대금 2년 최저 대비
3.1배+, 최근 분기 순이익 양수7일 1회 반도체SOXL SOXX 재무 순이익 양수, 거래대금
3.1배+ 3일 1회 금GDXU (GDX 적자 시 IAU) VNQ 20일선 위 + 5일전 2%+ 상승14일 1
회 은행ROBN (HOOD 80%+), CONL (110%+), FAS (XLF)각 거래대금 조건 충족7일 1회 원
유Ertlx XLE 거래대금 4배+, XKE 순이익 양수7일 1회 헬스케어XLV XLV 거래대금 4배+,
순이익 양수7일 1회 비트코인 종목 기초자산 재무 참조: ETF기초자산재무 확인 MSTU
MSTR MSTR 분기 순이익 IRE IREN IREN 분기 순이익 ROBN HOOD HOOD 분기 순이익
CRCA CRCL CRCL 분기 순이익 CONL COIN COIN 분기 순이익 운영 규칙: • 현금 없으면
매수 금지 • 순서 바뀌면 즉시 매도 후 다음 순번 매수 • 수익률 150%+ 종목: 레버리지
스탑 허가(120) 모드 시 중단, 해제(120) 모드 시 계속 가능 [!] 모호사항: "순서 바뀌면
즉시 매도"에서 순서 결정 기준 불명확. 거래대금 점수(2점/3점/4점)와 수익률로 순위를
매기는데, 어느 지표가 우선인지(거래대금 점수 > 수익률? 수익률 > 거래대금?), 점수가
같을 때 tie-break 방법, "순서 바뀜"의 최소 변동 기준(1순위↔2순위만? 모든 순위 변동?)
태준님 확인 필요.
- 14 taejun_attach_pattern — 전략 리뷰 2026-02-23 (VNQ) K. 조건부
은행주 (Bank-Conditional) 대형은행 전부 상승 + BAC만 하락 -> 역전 기대 항목내용 감
시 종목JPM, HSBC, WFC, RBC, C (모두 양전) 매수 대상BAC (마이너스) 매수 금액300만원
목표+1.05% (수수료 포함) *(정정: 0.8% + 0.25%)* 매도 후현금화 L. 숏 포지션 -- 재무
역전 (Short-Fundamental) 항목내용 매수 대상CONZ, IREZ, HOOZ, MSTZ (4개 중 1개만
매수) 우선 순위현재-전 분기 재무제표 차이가 가장 큰 종목 진입 조건Polymarket 상승
기대치 20% 이하 + 기초자산 최근 분기 적자 목표 수익률+1.15% (net) 매도 방식전액
매도 [OK] 해결 (2026-02-22): 0222 원문 line 1133 "POLYMARKET 상승기대치 20프로 하
락으로 해줘요" 직접 명시. 각 종목별 기초자산 적자 조건은 아래 테이블 참조. 종목-기
초자산 매핑 (정정 반영): 숏 ETF기초자산숏 허가 조건숏 불가 조건 CONZ COIN COIN
분기 순이익 마이너스 (적자) COIN 분기 순이익 플러스 (흑자) MSTZ MSTR MSTR 분기
순이익 마이너스 (적자) MSTR 분기 순이익 플러스 (흑자) IREZ IREN IREN 분기 순이익
마이너스 (적자) IREN 분기 순이익 플러스 (흑자) HOOZ HOOD HOOD 분기 순이익 마이
너스 (적자) HOOD 분기 순이익 플러스 (흑자) M20 적용: 신용/미수 거래 절대 금지 비
이상적 재난 이머전시 모드 기본 (재난 감지) 조건행동 Polymarket 어떤 지수든 30%+
급변수익 중인 전 종목 즉시 지정가 매도 재난 이머전시 1 (BTC 급등) 항목내용 조건
Polymarket BTC 기존 수치 대비 30%+ 급등 매수CONL 전액 지정가 매수- 15
taejun_attach_pattern — 전략 리뷰 2026-02-23 (VNQ) 항목내용 목표 수익률+3.15% (기
본 0.9% + 상향 2.0% + M20 0.25%) 매도목표 달성 시 전액 매도 재난 이머전시 2
(NASDAQ 급등) 항목내용 조건Polymarket NASDAQ 기존 수치 대비 30%+ 급등 매수
SOXL 전액 지정가 매수 목표 수익률+3.15% (기본 0.9% + 상향 2.0% + M20 0.25%) 매
도목표 달성 시 전액 매도 재난 이머전시 3 (NASDAQ 급락) 항목내용 조건Polymarket
NASDAQ 기존 수치 대비 30% 이상 폭락 매수SOXS 전액 지정가 매수 목표 수익률
+3.15% (기본 0.9% + 상향 2.0% + M20 0.25%) 매도목표 달성 시 전액 매도 [OK] 해결
(2026-02-22): 최종 목표수익률 = 3.15% (0.9% 기본 + 2.0% 상향 + 0.25% M20). 태준님
"네 맞아요 합산 수익 맞아요" 확인. 2-A. 신규 전략 M80 — 거래대금 초단타 모드 (신
설, Q8 -> M80 정정) Q8 섹터 로테이션 적립식 매매를 M80 거래대금 초단타 모드로 정
정. 섹터별 종목 풀에서 거래대금 배수 + 재무 조건을 달성한 종목을 매수하는 시스템.
M80 종목의 매수 순서는 M200 점수 체계로 결정한다. GLD / Polymarket 점수를 모두
합친 중첩 점수가 높은 순서대로 매수. 거래대금 배수별 점수표 (5년 최저 대비): 거래대
금 배수점수 1.2배 이하1.2점 1.3배 이하1.3점 1.5배 이하1.5점 1.6배 이하1.6점 1.7배 이
하1.7점 1.8배 이하1.8점 1.9배 이하1.9점 2배 이하2점 2.3배 이하2.3점 2.5배 이하2.5점
2.8배 이하2.8점- 16 taejun_attach_pattern — 전략 리뷰 2026-02-23 (VNQ) 거래대금 배
수점수 3배 이하3점 3.5배 이하3.5점 3.8배 이하3.8점 4배 이하4점 4.3배 이하4.3점 4.5배
이하4.5점 5배 이하5점 6배 이하6점 7배 이하7점 8배 이하8점 9배 이하9점 10배 이하
10점 11~18배11~18점 (1배당 1점) 점수별 목표수익률: 점수 구간목표수익률 5점0.1% ~
1.0% 6~10점1.1% ~ 2.0% 15점2.1% ~ 3.0% 20점3.1% ~ 4.0% 25점4.1% ~ 4.5% 30점4.6%
~ 5.0% 35점5.1% ~ 6.0% 70점6.1% 이상 모든 섹터에 공통으로 적용되는 3단계 조건: 조
건내용필수 여부 조건1: 거래대금기초자산 거래대금이 2년 최저 대비 N배 이상 (종목별
상이)필수 조건2: 분기 순이익기초자산 최근 분기 순이익 양수 (+0.1% 이상)필수 조건3:
이평선 돌파 (조건부)이동평균선 돌파 시 목표수익률 추가 상향선택 (달성 시 수익률 상
향) 종목 풀: 기초자산레버리지 ETF (1순위) M7 대체 (1배수) MSTR MSTU MSTR COIN
CONL COIN IREN IRE IREN RIOT RIOX RIOT CRCL CRCA CRCL BMNR (비트마인 이머션)
BMNU BMNR BTCT (BTC 디지털)-BTCT- 17 taejun_attach_pattern — 전략 리뷰 2026-02-
23 (VNQ) 기초자산레버리지 ETF (1순위) M7 대체 (1배수) CNCK (코인체크 그룹)-CNCK
XDGXX (디지 파워)-XDGXX FUFU (비트푸푸)-FUFU ANT (앤트알파)-ANT 조건1 - 거래대금
(2년 최저 대비): 종목 (기초자산)매수 허가 배수매수 불허가 MSTR >= 3.1배< 3.1배
COIN >= 3.1배< 3.1배 IREN >= 3.1배< 3.1배 RIOT >= 3.1배< 3.1배 CRCL >= 3.1배< 3.1
배 BMNR >= 3.15배< 3.15배 BTCT >= 7.14배< 7.14배 CNCK >= 10배< 10배 XDGXX >=
5.4배< 5.4배 FUFU >= 10배< 10배 ANT >= 40.2배< 40.2배 조건2 - 최근 분기 순이익:
각 기초자산 순이익 +0.1% 이상 시 허가,
-0.1% 이하 시 불허가. 조건3 - 조건부 목표수
익률 상향: 기초자산 120일 이동평균선 상향 돌파 시 목표수익률 +0.9% 추가. 종목 풀:
기초자산레버리지 ETF (1순위) M7 대체 (1배수) SOXX SOXL SOXX NVDA NVDL NVDA
AMD AMDL AMD AVGO AVGX AVGO INTC LINT INTC TXN-TXN QCOM QCMU QCOM
ARM ARMG ARM MRVL MVLL MRVL M80 구매 목록 (반도체 한정): SOXL, NVDL 조건1 -
거래대금 (2년 최저 대비): 종목 (기초자산)매수 허가 배수매수 불허가 SOXX >= 3.1배<
3.1배 NVDA >= 3.1배< 3.1배 AMD >= 3.1배< 3.1배 AVGO >= 3.1배< 3.1배 INTC >=
3.1배< 3.1배- 18 taejun_attach_pattern — 전략 리뷰 2026-02-23 (VNQ) 종목 (기초자산)
매수 허가 배수매수 불허가 TXN >= 3.1배< 3.1배 QCOM >= 3.1배< 3.1배 ARM >= 3.1
배< 3.1배 MRVL >= 3.1배< 3.1배 조건2 - 최근 분기 순이익: 각 기초자산 순이익 +0.1%
이상 시 허가,
-0.1% 이하 시 불허가. 조건3 - 조건부 목표수익률 상향: 기초자산 120일
이동평균선 상향 돌파 시 목표수익률 +0.9% 추가. 종목 풀: 기초자산레버리지/매수 종목
M7 제한 GDX GDXU M7 제한 없음 AEM AEM M7 제한 없음 IAU IAU M7 제한 없음
NEM NEM M7 제한 없음 조건1 - 거래대금 (2년 최저 대비): 종목매수 허가 배수매수 불
허가 GDXU >= 2배< 2배 조건2 - 최근 분기 순이익: • GDX는 지수 ETF이므로 재무제표
없음 -> AEM, NEM 두 개로 대체 • AEM, NEM 두 개 모두 최근 분기 순이익 +0.1% 이
상 시 매수 허가 • 둘 중 하나라도 -0.1% 이하 손실 시 매수 불허가 • GDX 순이익 불허
가 + 거래대금 허가인 경우 -> IAU로 대체 매수 조건3 - 조건부 목표수익률 상향: GDX
5일 이동평균선 상향 돌파 시 목표수익률 +0.3% 추가. 종목 풀: 기초자산레버리지 ETF
(1순위) M7 대체 (1배수) LLY (일라이릴리) LLYX LLY JNJ (존슨앤존슨) JNJ JNJ NVO (노보노
디스크) NVOX NVO 조건1 - 거래대금 (2년 최저 대비): 종목매수 허가 배수매수 불허가
LLY >= 2.7배< 2.7배 JNJ >= 2.8배< 2.8배 NVO >= 2.8배< 2.8배 조건2 - 최근 분기 순이
익: • XLV는 지수 ETF이므로 재무제표 없음 -> JNJ, LLY 두 개로 대체 • JNJ, LLY 두 개 모
두 최근 분기 순이익 +0.1% 이상 시 매수 허가 • 둘 중 하나라도 -0.1% 이하 손실 시
매수 불허가 조건3 - 조건부 목표수익률 상향:- 19 taejun_attach_pattern — 전략 리뷰
2026-02-23 (VNQ) 종목이평선 조건수익률 추가 XLV 5일 이동평균선 돌파+0.2% NVO 5
일 이동평균선 돌파+0.3% LLY 5일 이동평균선 돌파+0.3% 종목 풀: 기초자산레버리지
ETF비고 XLE ERX섹터 대표 XOM XOMX엑슨모빌 CVX CVX셰브론 COP COP코노코필립스
SLB SLB슐럼버거 WMB WMB윌리엄스 조건1 - 거래대금 (2년 최저 대비): 종목매수 허가
배수매수 불허가 XLE/ERX >= 2.2배< 2.2배 XOM/XOMX >= 2.2배< 2.2배 CVX >= 2.2배<
2.2배 COP >= 2.2배< 2.2배 SLB >= 2.2배< 2.2배 WMB >= 2.2배< 2.2배 조건2 - 최근
분기 순이익: 각 기초자산 순이익 +0.1% 이상 시 허가,
-0.1% 이하 시 불허가. 종목 풀:
기초자산레버리지 ETF HOOD ROBN XLF FAS 조건: J전략(섹터 로테이션) 은행 섹터와 동
일 조건 적용. • HOOD 거래대금 80%+ 상승 시 ROBN 매수 • XLF 순이익 양수 + 거래
대금 상승 시 FAS 매수 항목내용 매수 순서M200 점수 높은 순서대로 매수 매도 순서
M200 즉시 매도 조건 1개라도 부합 시 즉시 매도 매매 주기섹터별/종목별 상이 (J전략
참조) 아래 조건 1개라도 부합 시 목표수익률 상관없이 즉시 매도: #조건 1전날 거래대
금 대비 장 시작 3시간 이내 15% 하락 2 Polymarket BTC 급변 (세부 기준 미명시)- 20
taejun_attach_pattern — 전략 리뷰 2026-02-23 (VNQ) #조건 3 GLD 6% 이상 상승 4 VIX
10% 급등 5해당 종목 20일 이동평균선 이탈 6기한 만기 (익영업일 17:30~06:00 이내 매
도) 7 VNQ 5% 이상 상승 시 -> GLD/숏 포지션 제외 모든 포지션 T1~T4 각 30% 비중
축소 선행지수 변경 규칙: 종목M7 발동 시비고 BITU매매 금지선행지수이므로 변경 불가
ROBN HOOD로 교체 NVDL NVDA로 교체 3. 정정 사항 반영 이력 날짜원본정정출처
2026.2.18 18:57 GDXU 매도 후 GLD 매수IAU 매수카톡 2026.2.18 19:56부동산 선행지수
하락시 분할매도전액 매도 (분할 아님)카톡 2026.2.18 22:10 PLTU 하락 43% 진입44% 진
입정리본 2026.2.18 BRKU 하락 26% 진입32% 진입초기 26% -> 30% -> 최종 32%
2026.2.20 SETH 목표 수익률 0.5% 0.8%정정 2026.2.20은행(BAC) 목표 수익률 0.5% 0.8%
정정 2026.2.20왕복 수수료 0.703% 0.74%정정 2026.2.21 SETH (숏 ETF) ETQ로 종목 변경
MT_VNQ Q-2 2026.2.21 XRP +5.0% (잽BITU 조건) +2.5% (장중 변동 기준) MT_VNQ Q-1
2026.2.21 BRKU 진입 -32%-31% MT_VNQ Q-6 2026.2.21 TSLL GLD 조건 <= +0.3% <=
+0.1% MT_VNQ Q-5 2026.2.21 ETHU 추가매수 10%추가매수 없음MT_VNQ Q-7
2026.2.21 CONL 목표 +188% VNQ 120일선 조건부 100%, 60일 기한MT_VNQ Q-3
2026.2.21 SOXL 진입 -90.5% 기준VNQ 120일선 조건부 100%, 60일 기한MT_VNQ Q-4
2026.2.21 SNXX, OKLL전략에서 제외MT_VNQ Q-10 2026.2.21 SK리츠(395400) 추종VNQ
(미국 리츠 ETF) 대입MT_VNQ 2026.2.21없음M0/M4/M6/M7/M20/M40 신설MT_VNQ
2026.2.21 M3 진입 비율 테이블M5로 이관, M3은 시간/휴장일만MT_VNQ 2026.2.21모든
목표수익률+0.25% 추가MT_VNQ M20 2026.2.21이머전시 모드 목표수익률+2% 상향
MT_VNQ 2026.2.22 I-3 한국 리츠 3종목 기준VNQ로 정정 확인0222 원문 2026.2.22 M5
GLD/VIX/달러 중첩 여부 불명확 중첩 합산 적용 확인0222 원문 2026.2.22재난 이머전시
+2% 기준 불명확0.9+2.0+0.25=3.15% 확인0222 원문- 21 taejun_attach_pattern — 전략
리뷰 2026-02-23 (VNQ) 날짜원본정정출처 2026.2.22 L전략 Polymarket 마켓 불명확
Polymarket 상승 기대치 20% 이하 + 기초자산 적자0222 원문 2026.2.22 CI-9 1.54% vs
1.79% 불일치params.py에 1.05% 직접 반영0222 원문 2026.2.22 M7 NVDL 대체 종목 누
락NVDL -> NVDA 추가0222 원문 2026.2.22없음M200 (즉시 매도 7조건) 신설0222 원문
2026.2.22없음M80 (거래대금 초단타) 신설0222 원문 2026.2.22 (MT_VNQ2) CI-0 v0.1 플
레이스홀더CI-0 v0.2 오류 방지 명세 전면 재작성MT_VNQ2 2026.2.22 (MT_VNQ2) P-5~P-
8 미확인잠정 기본값 반영 (오류 방지용, 태준님 확인 필요) MT_VNQ2 2026.2.22 (rule-
verifier) CI-1 만료 기준 미정CI-0-3으로 해결 표시 (5 bars, 재시도 3회)검증 2026.2.22
(rule-verifier) CI-5 차감/초기화 미정CI-0-8으로 해결 표시 (FILLED 시 차감, 17:30 초기화)
검증 2026.2.22 (rule-verifier) A/B/E/F 전략 M3 비율 참조M5 T1~T4 비율 기준으로 정정
(L5-5)검증 2026.2.22 (rule-verifier)없음P-9~P-14 신규 미확인 항목 6건 등록검증
2026.2.22 (이슈 상태 명시) CI/A/C/W 항목 상태 미표기[OK]/[!]/[ ] 상태 전면 표기 — CI-
1/CI-8/CI-9/CI-12/C-1~C-3/A-4~A-5/W-항목 상태 열 추가 이번 세션 2026.2.23
(MT_VNQ3) M1 주문 TTL 5 bars(5분) 120초(2분) 확정MT_VNQ3 §3-3 2026.2.23
(MT_VNQ3) M4 VNQ 단독 지표REIT_MIX = VNQ + KR 리츠 평균MT_VNQ3 §9 2026.2.23
(MT_VNQ3) M5 T5~ "매수 없음"예약/대기 표시 + 10초 취소 룰MT_VNQ3 §10 2026.2.23
(MT_VNQ3) M6 리츠 감산 제외항 불명확GLD/VIX/GDXU, CONZ/IREZ/HOOZ/MSTZ,
SOXS/ETQ 확정 + 적용 순서 고정 MT_VNQ3 §11 2026.2.23 (MT_VNQ3) M200 파이프라
인 미정전역 락→취소 ACK→reserved 재계산→매도 즉시성 적용 확정MT_VNQ3 §15
2026.2.23 (MT_VNQ3) SCHD 매수 기준 미정30일 실현손익 +시 / 수익 구간별 최대 50
만원MT_VNQ3 §16 2026.2.23 (MT_VNQ3)없음M201(즉시모드) 신설MT_VNQ3 §14
2026.2.23 (MT_VNQ3)없음M28(Polymarket BTC 게이트) 신설MT_VNQ3 §13 2026.2.23
(MT_VNQ3)없음M300(USD-only 환전 금지) 신설MT_VNQ3 §17 2026.2.23 (MT_VNQ3)없음
Fill Window 10초 룰 신설MT_VNQ3 §5 2026.2.23 (MT_VNQ3)없음추격매수 금지
(reference_price 고정) 신설MT_VNQ3 §6 2026.2.23 (MT_VNQ3)없음매도 즉시성 (bid 0.2%
이내) 신설MT_VNQ3 §7 2026.2.23 (MT_VNQ3)없음포지션 confirmed/effective 분리 신설
MT_VNQ3 §4 2026.2.23 (MT_VNQ3)없음금지/오타/대체 티커 처리 정책 신설MT_VNQ3
§18 2026.2.23 (MT_VNQ3)없음MASTER SCHD 신설MT_VNQ3 §16 4. 미확인 모호사항 --
업데이트 (2026-02-22) 해결된 항목 #질문답변반영 Q-1 XRP +5%가 맞는지+2.5% (장중
변동 기준, 장마감 아님)반영- 22 taejun_attach_pattern — 전략 리뷰 2026-02-23 (VNQ)
#질문답변반영 Q-2 SETH 진입 조건 해석ETQ로 종목 변경, Polymarket 하락 기대 평균
12 미만 시 작동반영 Q-3 CONL 목표 188% 근거VNQ 120일선 하회 시 100% 목표, 60
일 기한, 비중 30%반영 Q-4 SOXL -90.5% 정확한지Q-3과 동일 조건 적용반영 Q-5 TSLL
GLD <= +0.3% 맞는지<= +0.1%로 수정반영 Q-6 BRKU 최종값 32% 맞는지-31%가 최종
반영 Q-7 ETHU 추가매수 범위추가매수 없음반영 Q-8섹터 로테이션 금 섹터 빠짐전면
재설계 (거래대금 점수 체계, 6개 섹터) -> M80으로 확장반영 Q-9조심/집중 구현 방식조
심모드: 3개 리츠 7일 상승률 1%+ -> 레버리지 50% 이하반영 Q-10 SNXX, OKLL 종목
확인매수매도 제외반영 Q-11 Polymarket NASDAQ 매핑NDX 일별 URL, 업데이트 안 되
면 조건 정지반영 P-M5 중첩GLD/VIX/달러 동시 발생 시 처리중첩 합산 적용반영 P-I3
VNQ조심모드 기준 VNQ vs 한국 리츠VNQ로 정정반영 P-이머전시재난 이머전시 2.9%
vs 3.15% 3.15% 확인 (0.9+2.0+0.25)반영 P-L전략Polymarket 마켓 기준Polymarket 상승
기대치 20% 이하 + 기초자산 적자반영 P-CI9 params.py 1.05% 직접 반영 여부직접 반
영 확인 (target_pct=1.05%)반영 미해결 항목 (P-1 ~ P-4 기존 + P-5 ~ P-8 신규) #내용
현황잠정 기본값 (MT_VNQ2 반영) P-1수익금 분배 수량 부족 시 처리 방식 (동시 vs 순
차, 부족 시 처리)미확인— P-2리츠 과열 시 "제외항 0.5% 낮춤" 범위 (해석 A vs B)미확
인— P-3 M5 차감 시점 및 "각각" 범위 (매수 완료 시 vs 세트 완료 시)미확인— P-4
TSLL 200만원 한도 vs M3 비율 우선순위미확인— P-5 M200 조건2 "44POLYMARKET
BTC" [!] 미확인OFF — 정의 확정 전까지 비활성화 P-6 M200과 기존 전략 동시 발동 시
우선순위[OK] 잠정 확정M200 최우선 — CI-0-5 기준 P-7 M80 vs J 섹터 로테이션 중복
[OK] 잠정 확정M80 발동 시 동일 종목 J 신호 대체 P-8 M80 최소 매수 점수[OK] 잠정
확정35점 — 보수적 기준 (과매매 방지) [!] 잠정 기본값 출처: MT_VNQ2.md L2619~2651
(오류 방지용 기본안, 태준님 확인 전까지 유지) 신규 미해결 항목 (P-9 ~ P-14, rule-
verifier 2026-02-22 발견) #내용근거현황 P-9 M80 점수 구성 배점 테이블 — 35점(P-8)
달성 시 거래대금 점수(최대 18점)만으로는 불가. GLD/Polymarket 추가 점수 배점 정의
필요 L2-4, L3-4미확인 P-10 M200 즉시 매도 후 이머전시 모드 재매수 쿨다운 — 같은
Bar에서 M200 매도 → 이머전시 전액 매수 충돌. 쿨다운 시간(1시간? 당일 종료?) 태준
님 확인 필요 L4-1미확인 P-11 PARTIAL fill 시 M5 차감 기준 — CI-0-8 "전체 FILLED 시
1회 차감"이지만 PARTIAL 상태 처리 미정. 잔량 취소 후 차감인지, PARTIAL 발생마다 차
감인지 확인 필요 L2-1, L5-1미확인 P-12 M200 조건6 기한 만기 처리 타이밍 — M200
은 "즉시 매도"인데 조건6만 "다음날 17:30~06:00 이내"로 다른 타이밍. 원문 의도 확인
필요 L6-4미확인 P-13 M4 VNQ 모드 vs M200 우선순위 — M4는 "1순위 상시 가동", CI-
0-5는 "M200 최우선". VNQ 5%+ 상승 시 M4(레버리지 전액 매도)와 M200 조건7(30%
비중 축소) 중 어느 쪽 우선인지 확인 필요 L4-3미확인- 23 taejun_attach_pattern — 전
략 리뷰 2026-02-23 (VNQ) # P-14 내용 D전략 ETQ "Polymarket 평균 수치" 기간 정의
— Q-2에서 종목 교체만 해결, 평균 기간(1일/7일/30일?)과 어떤 Polymarket 마켓 기준인
지 미확인 근거현황 L5-4 미확인 P-15 M201 발동 후 포지션 전환 규모 — 가용현금 전
액 전환? 일부만 전환? 부분 전환 비율 미정 P-16 REIT_MIX 계산 시 KR 리츠 결측 처리
— KR 리츠 일부 결측 시 나머지로 평균? BUY_STOP? P-17 PARTIAL_ENTRY_DONE 이후
재진입 조건 — Fill Window 10초 후 당일 동일 조건 재진입 허용 기준 미정 MT_VNQ3
§14 미확인 MT_VNQ3 §9 미확인 MT_VNQ3 §5 미확인 5. Critical 이슈 C-1. 수수료 미반
영 목표 판정 -- 기준 0.74%로 정정 — [!] 설계 확정, 코드 수정 대기 전략 jab_soxl 목
표 (net) 1.15% 왕복 수수료 0.74% 필요 세전 1.89% 비고 M20 +0.25% 반영 jab_bitu
1.15% 0.74% 1.89% jab_tsll 1.25% 0.74% 1.99% jab_etq (구 SETH) 1.05% 0.74% 1.79% 종
목 ETQ로 변경 sp500_entry 1.75% 0.74% 2.49% bank_conditional 1.05% (net) 0.74%
1.79% 해결: FeeCalculator.is_target_met(entry_price, current_price, target_net_pct) 사용,
수수료 기준 0.74% C-2. 시장가 매수/매도 전면 금지 (M1 확장) — [!] 설계 확정 (CI-0-
2), LimitOrder 구현 대기 현상: execute_buy에서 price = prices.get(ticker, 0) 현재 호가로
즉시 체결 -> 사실상 시장가 대상: • 최초 매수 (avg_price = price) • 추가매수
(net_for_shares + avg_price 혼용) • 매도 (price = prices.get(ticker, 0)) M1 규칙: 모든 매
수/매도는 지정가. 이머전시/공격/방어/조심 모드 예외 없음. M1 지정가 오류 해결: KIS
API(한국투자증권)에서 소수점 3자리 이상 호가 입력 불가. 소수점 단위가 0.001 이상인
경우 호가를 반올림하여 0.01 단위로 체결한다. 매수/매도 모두 이 조건 적용. 해결: 지정
가 주문 큐 + 체결 시뮬레이션 레이어 추가 필요. KIS API 소수점 2자리 제한 반영. C-3.
초기 진입 비율 -> 마스터 플랜 M5로 이전 — [!] 설계 확정 (CI-0-5), Orchestrator 구현
대기 기존 bargain_buy.py:177의 size=0.5 하드코딩 -> M5 T1~T4 체계로 대체. C-4. CI-9
수수료 불일치 -- 해결 완료 (2026-02-22) 항목내용- 24 taejun_attach_pattern — 전략 리
뷰 2026-02-23 (VNQ) 항목내용 문제 CI-9에서 target_pct=0.8% (gross 1.54%) vs CI-8에
서 1.05% (gross 1.79%) 불일치 답변 M20 +0.25%를 params.py에 직접 반영
(target_pct=1.05%) 상태 해결 6. Warning 이슈 # 내용 파일 W-1 signal.size > 1.0 방어
없음 -> 음수 포지션 portfolio.py 메모 min() 클램프 필요 상태 [ ] 미해결 W-3 복수 저
가매수 조건 충족 시 첫 종목만 반환 bargain_buy.py 리스트 반환 or 우선순위 W-4
_last_buy_dates, _ban_until 재시작 시 초기화 datetime.now() 백테스트 불일치 W-5
sector_rotate, reit_risk reit_risk.py:35 BUY 시그널에 ticker="" 반환 W-6 sp500_entry.py
W-7 jab_etq 시간 조건(17:30) 누락 *(구 jab_seth)* jab_etq.py registry name="" 방어 없
음 W-8 registry.py MarketData 입력 검증 없음 W-9 base.py [ ] 미해결 영속 저장 필요
market.time 사용 SKIP 또는 metadata 다른 잽모드와 불일치 assert 추가 __post_init__
추가 [ ] 미해결 [!] 설계 확정 (CI-0-7, = A-1) [ ] 미해결 [OK] 구현 완료 (2026-02-23) [ ]
미해결 [ ] 미해결 7. 알고리즘 이슈 -- 코드 분석 기반 디테일 아래는 실제 코드를 분석
해서 발견된 구현 이슈입니다. 코드 위치를 명시합니다. A-1. reit_risk.py:35 --
datetime.now() 백테스트 불일치 — [!] 설계 확정 (CI-0-7), 코드 수정 대기 `python 현재
코드 (잘못됨) return datetime.now() < self._ban_until 올바른 코드 return market.time <
self._ban_until # market.time은 백테스트 시뮬레이션 시각 ` 영향: 백테스트시
is_ban_active가 항상 False 반환 -> 리츠 금지 90일이 무효화됨. generate_signal도
market.time을 받으므로 is_ban_active(market_time) 형태로 수정 필요. A-2.
portfolio.py:314~316 -- 추가매수 평균가 단위 혼용 (C-2) — [OK] 설계 확정 (CI-0-2), 코
드 수정 대기 `python 현재 코드 (잘못됨) total_cost = existing.avg_price * existing.qty +
net_for_shares # net_for_shares = 수수료 차감 total_qty = existing.qty + qty
existing.avg_price = total_cost / total_qty- 25 최초 매수 (portfolio.py:320)
taejun_attach_pattern — 전략 리뷰 2026-02-23 (VNQ) avg_price=price, # 시장가 (수수료
미차감) ` 영향: 최초 매수는 시장가(price), 추가매수는 수수료 차감 후 금액
(net_for_shares) -> 단위 혼용으로 avg_price 부정확. 해결: 두 경우 모두 시장가(price) 기
준으로 통일. `python 수정안 total_cost = existing.avg_price * existing.qty + price * qty #
둘 다 시장가 기준 existing.avg_price = total_cost / total_qty ` A-3. portfolio.py:273~275
-- M1 위반 (시장가 즉시 체결) — [!] 설계 확정 (CI-0-2), LimitOrder 구현 대기 `python
현재 코드 -- 사실상 시장가 price = prices.get(ticker, 0) if price <= 0: return None ` 영
향: M1 규칙 "절대 시장가 금지" 위반. 현재 스냅샷 가격으로 즉시 체결. 해결 방향: •
OrderBook 또는 LimitOrderQueue 레이어 도입 • 지정가 주문 생성 -> 다음 봉에서 체
결 여부 판단 • 실서빙시 KIS API 지정가주문(TTTC0802U) 연동 A-4. Orchestrator — [OK]
구현 완료 (2026-02-23) — orchestrator.py 10단계 파이프라인 현상: M3(당일 조건 부합
개수별 진입 비율), M5(완료 후 1개씩 차감) 로직이 코드 어디에도 없음. 현재 구조의 문
제: • 각 전략 generate_signal은 독립적으로 size를 결정 (고정값) • 전략 간 당일 조건
부합 카운팅 공유 메커니즘 없음 해결 방향: `python class Orchestrator: def __init__(self,
strategies, portfolio): self._daily_match_count: dict[date, int] = {} # 날짜별 조건 부합 누적
def get_entry_size(self, today: date, signal_rank: int) -> float: M5 T1~T4 순차 계산 방식
(개정) T1=55%, T2=나머지(45%)x40%, T3=나머지x33%, T4=나머지 전액 remaining = 1.0
rates = [0.55, 0.40, 0.33, 1.0] for i in range(min(signal_rank, 4)): size = remaining * rates[i]
remaining -= size return size # signal_rank번째 진입분 비중 def on_trade_executed(self,
today: date): M5: 완료 후 1개 차감 self._daily_match_count[today] =
max( self._daily_match_count.get(today, 0) - 1, 0 ) `- 26 taejun_attach_pattern — 전략 리
뷰 2026-02-23 (VNQ) A-5. bargain_buy.py:176 -- size 하드코딩 (C-3 잔재) — [!] 설계 확
정 (CI-0-5), Orchestrator 구현 대기 `python 현재 코드 return Signal( action=Action.BUY,
ticker=ticker, size=0.5, # 하드코딩 -- M3 규칙 미반영 ... ) ` 해결: Orchestrator에서 M3
기반 size를 주입하거나, size=None으로 설정 후 Orchestrator가 결정. A-6.
bargain_buy.py:166~190 -- 복수 종목 동시 충족시 첫 종목만 반환 (W-3) `python 현재
코드 for ticker, cfg in self._ticker_params.items(): if ...: return Signal(Action.BUY, ticker, ...)
# 첫 번째만 반환 ` 영향: CONL, SOXL 동시 폭락시 CONL만 처리, SOXL 기회 놓침. 해
결: generate_signals() -> list[Signal] 형태로 변경, 우선순위 또는 전부 반환. A-7.
jab_etq.py -- 17:30 시간 조건 누락 (W-7) `python jab_soxl.py (정상) if market.time.hour
< 17 or (market.time.hour == 17 and market.time.minute < 30): return False jab_etq.py
(누락) -- check_entry에 시간 조건 없음 ` 영향: ETQ가 장중 어느 시간에도 진입 가능 ->
M3 규칙(17:30~06:00) 위반. A-8. 리츠 과열 공격 조절 모드 -- 구현 없음 (신규 이슈) 현
상: reit_risk.py가 트리거 발동시 "*leveraged" 전체 매도 신호만 보냄. 아래 로직 전혀 없
음: • SOXL -> SOXX 종목 교체 • ROBN -> HOOD 종목 교체 • GDXU -> GLD 종목 교체
• 교체 종목 목표 수익률 x 50% • 전체 전략 목표 수익률 -0.5% 업데이트 (2026-02-21):
부동산 지수 추종이 SK리츠(395400)에서 VNQ (미국 리츠 ETF)로 변경. 모든 리츠 관련
조건에서 SK리츠 -> VNQ로 대체. 해결 방향: `python reit_risk.py 또는
AssetModeManager에 추가 REIT_OVERHEAT_SUBSTITUTES = { "SOXL": "SOXX", "ROBN":
"HOOD", "GDXU": "GLD", } REIT_TARGET_MULTIPLIER = 0.5 # 교체 종목 목표 50%
REIT_GLOBAL_TARGET_DEDUCT = 0.5 # 전체 전략 목표 -0.5% `- 27 A-9. 수익금 분배 로
직 -- 구현 없음 (신규 이슈) taejun_attach_pattern — 전략 리뷰 2026-02-23 (VNQ) 현상:
매도 후 수익금을 SOXL, ROBN, GLD, CONL 각 1주씩 1일 1회 구매하는 로직 없음. 현재
execute_sell 구조: 단순히 cash에 합산 후 종료. 해결 방향: • ProfitDistributor 클래스 신
설 • 하루 수익금 합산 후 마감 후 자동 분배 실행 • 분배 우선순위: SOXL -> ROBN ->
GLD -> CONL (수익금 소진시 중단) 8. 코드화 이슈 (2026-02-22 업데이트) CI-0. [OK] 오
류 절대 방지용 구현 규격 v0.3 (공통 규격) — 2026-02-23 MT_VNQ3 반영 이 문서의 모
든 전략 구현은 CI-0를 기본 규격으로 따른다. CI-1~CI-20은 "추가/예외/확장"이며, 충돌
시 CI-0 우선(단, 해당 CI에서 예외를 명시한 경우 그 예외 우선). 출처: MT_VNQ2.md
L2842~2960 (CI-0 v0.2) + MT_VNQ3.md §1~§18 (CI-0 v0.3 추가) • 데이터가 없거나
(0/None/NaN), 오래됐으면(기본 2분 초과) → 해당 Bar 거래 금지 + 알람. •
VNQ/Polymarket 등 핵심 지표가 확인 불가이면 → 관련 모드/조건 평가 중지(OFF) + 알
람. • 주문은 LimitOrder만 허용. Market 주문 타입은 코드에서 제거한다. •
Position/avg_price는 FILLED 이후에만 생성/갱신한다. Pending 상태에서는 금지. • 주문
만료 기본값: 120초(2분) *(MT_VNQ3 §3-3 정정: 기존 5 bars(5분)).* • 재시도 기본값: 최대
3회. 재시도 중복 방지를 위해 Idempotency Key를 사용한다. • 재시도 시 가격 보정(체결
성 개선, 시장가 아님): • BUY: ask 기준 +0.1% 이내 • SELL: bid 기준 -0.1% 이내 • 가격
입력은 0.01 단위로 라운드한다(증권사 호가거절 대응 포함). • "즉시" = 조건이 True가
된 Bar 종료 시점에 주문 제출. • 지정가 기준 = 신호 Bar의 종가(close). • API 지연 허용
= 1분 이내, 초과 시 해당 신호 무효 + 알람. • 전략은 Signal만 반환한다. 주문 제출은
오케스트레이터만 수행한다. • 처리 우선순위는 아래 고정: 1. M200 즉시 매도 (최우선) 2.
리스크/이머전시 모드 3. 거래시간/휴장 캘린더 필터 4. 비중 배분(T1~T4 포함) 5. 일반
전략 신호 처리 • 같은 Bar에서 BUY 신호가 여러 개면 순차 실행 금지. • 전체 신호를
먼저 집계→정렬→한 번에 비중 배분. • 총합이 100% 초과하면:- 28 • 우선순위 낮은 신
호부터 drop taejun_attach_pattern — 전략 리뷰 2026-02-23 (VNQ) • 그래도 초과하면
전체를 비례 축소(scale down) • 카운트/초기화/로그 기준 날짜는 KST 17:30에 바뀌는
session_day로 통일한다. • 캘린더 로드 실패/휴장 불명확이면 → 해당 session_day 거래
정지 + 알람. • 차감 시점: 매수 체결(FILLED) 완료 시 1개 차감. • 초기화 시점: 매일
17:30(KST, session_day 시작). • 음수 방지: 0 미만이면 0으로 clamp + 알람. • 실행 시점:
장 마감 후 1회(예: 06:05 KST). • pnl <= 0 이면 분배 skip. • 수익금 부족/가격 조회 실패
시: 해당 종목만 skip + 알람(전체 중단 금지). • 분배 매수도 지정가(M1) 적용. • 정의 불
명확 항목은 기본값으로 OFF(비활성화) 후 알람. • 예: "44POLYMARKET BTC"는 정의 전
까지 OFF. • orders / positions / counters / session_day 를 영속 저장한다. • 엔진 재시작
시 반드시 저장된 상태로 복구. 복구 실패 시 신규 매수 금지 + 알람. • intent_id =
(strategy_id, ticker, side, signal_bar_ts, rank) 로 고정. • retry는 intent_id에 포함하지 않는
다 — retry_count 메타로만 기록. • 동일 intent_id로 이미 pending/filled 주문이 있으면
신규 주문 금지. • 동시 BUY 비중 합계 > 100% → 비례 shrink. • shrink 이후에도
100.1% 이상이면 계산 버그로 BUY_STOP + 알람. • shrink 후 하위 priority 항목부터
drop. • API 실패/지연 시 티커/모드 단위 정지 + 알람 (거래 보류). • 단, M200/M201 청
산은 최대한 허용 (킬스위치 우선). • 금지/오타 티커는 매수목록에서 즉시 제외(drop),
shrink 계산에서도 제외. • 조건 달성 + 대체 티커가 정의된 경우에만 원 티커 → 대체
티커로 치환 후 매수. • 대체도 불가하면 drop + 알람. • "0.01 고정"은 USD(미국) 종목
기본값으로만 사용한다. • KR(국내) 또는 기타 시장은 TickSizeResolver(market, price)로 호
가 단위를 결정한다. • 가격 정규화: • BUY 지정가 = ceil_to_tick(raw_price, tick) • SELL 지
정가 = floor_to_tick(raw_price, tick) • 주문 거절(호가단위/가격오류) 발생 시: • tick 1칸씩
조정하여 최대 3회 재시도 • 3회 실패 시 cancel + 알림 + 해당 ticker는 N분 쿨다운 •
필수 필드: filled_qty, avg_fill_price, reserved_cash • Position 생성/갱신 규칙: • filled_qty >
0 인 경우에만 Position에 반영 • pending/partial 상태에서 "avg_price"를 확정값처럼 쓰
지 않는다.
- 29 • 주문 생성 시점에 reserved_cash = (limit_price * qty) + 예상수수료 를
cash에서 즉시 "예약"한다. taejun_attach_pattern — 전략 리뷰 2026-02-23 (VNQ) • 주문
이 cancelled/expired/rejected 되면 reserved_cash 전액 반환한다. • pending/partial 상태
주문이 있는 ticker+side에 대해 신규 주문을 금지한다(중복 주문 방지). • 엔진 시작 시
반드시: 1. 브로커 open orders 조회 2. 브로커 positions 조회 3. 내부
OrderBook/Portfolio를 브로커 상태로 재구성 • 동기화 실패 시: 신규 매수 금지 + 알림
(시스템 점검 모드) • 기본: 수수료는 "거래대금 비율" 모델로 계산한다. •
roundtrip_fee_pct = buy_fee_pct + sell_fee_pct • 분할매도(sell_splits)는 % 기준 총수수료
를 증가시키지 않는다. • 단, '주문당 최소수수료'가 존재하면: • min_fee 적용 여부를 옵션
으로 두고, • 분할매도는 "절대비용"이 증가할 수 있으므로 별도 계산한다(옵션 ON일 때
만). • prices.get(ticker, 0) 사용 금지. • 가격/지표 결측 시: • 해당 주문/신호는 스킵 + 알
림 • 전략 전체가 아니라 "해당 ticker만" 쿨다운(예: 5분) • P-5 확정 전까지 조건2는
"UNKNOWN"으로 두고, • 즉시매도 트리거로 사용하지 않는다. • 대신 알림만 발생시키
고, 신규 매수는 1분간 일시정지(soft-freeze)한다. CI-1. M1 -- 지정가 체결 레이어 신설
— [!] 일부 해결 (CI-0-3 만료 기준), LimitOrder 구현 대기 이슈: • execute_buy /
execute_sell 모두 prices.get(ticker, 0)으로 즉시 체결 -> 사실상 시장가 • 지정가 주문 상
태(Pending) 관리 클래스 없음 • 미체결 주문의 만료 처리 기준 없음 (당일 취소? GTC?
재시도 횟수?) → [OK] CI-0-3으로 해결됨: 120초(2분) 만료, 최대 3회 재시도 *(MT_VNQ3
§3-3 정정)* 필요 설계: 항목 LimitOrder 클래스 내용 주문가, 종목, 수량, 만료시각, 상태
(pending/filled/cancelled) LimitOrderQueue 미체결 주문 큐 관리, 봉마다 체결 여부 판단
백테스트 체결 규칙 다음 봉 시가 <= 지정매수가 -> 체결 / 미충족 -> pending 유지
실서빙 연동 KIS API 지정가주문(TTTC0802U) + 체결 확인 polling avg_price 문제: 지정가
미체결 상태에서 평균가 계산 불가 -> 체결 확정 후에만 Position 생성 CI-2. M2 --
"즉
시 행동"의 구현 단위 — [OK] 해결됨 (CI-0-4, 2026-02-22 MT_VNQ2) 이슈:- 30 • "즉시"
가 분봉 기준인지, 초봉 기준인지 미정 taejun_attach_pattern — 전략 리뷰 2026-02-23
(VNQ) • 조건 달성 봉과 주문 제출 봉 사이에 가격이 이미 움직인 경우 지정가 기준 불
명확 • 네트워크/API 지연 발생시 "즉시" 보장 불가 결정 (CI-0-4): 질문 조건 달성 즉시
지정가 기준 [OK] A. 조건 달성 봉의 종가 확정값 API 지연 허용 범위 [OK] A. 1분 이내
— 초과 시 해당 신호 무효 + 알람 CI-3. M3 -- 시간 범위 및 거래일 캘린더 이슈: 항목
NYSE 거래일 캘린더 문제 미국 공휴일 데이터 별도 관리 필요
(pandas_market_calendars or 직접 구현) DST(서머타임) 미국 시장 개장 KST 기준: 동절
기 23:30, 하절기 22:30 -- 매년 변동 날짜 경계 17:30~06:00은 자정을 넘어감 -> date
기준 처리 주의 (금요일 17:30 ~ 토요일 06:00 포함) 한국 공휴일 KIS API 접근 가능 여
부에 따라 한국 공휴일도 체크 필요 여부 결정 주말 처리 금요일 06:00 이후 ~ 월요일
17:30 완전 비활성 -> 상태 누락 없이 처리 필요 CI-4. M3 -- 진입 비율 카운팅 충돌 —
[OK] 설계 확정 (CI-0-5/CI-0-6, Orchestrator 구현 대기) 이슈: • 각 전략 generate_signal
은 독립 호출 -> 전략 간 카운트 공유 불가 • 같은 봉에서 여러 전략이 동시 BUY -> 처
리 순서에 따라 비율이 달라짐 • 진입 비율 누적시 자본 초과 위험 예시 문제: 상황
jab_soxl(1번째) + jab_bitu(2번째) 동시 달성 결과 1번째는 50% 진입, 2번째는 40% 진입
-> 총 90% 투입 jab_soxl + jab_bitu + jab_tsll 동시 50+40+33 = 123% -> 자본 초과
[OK] 해결 방향 확정 (CI-0-5/CI-0-6): Orchestrator 단일 진입점에서 전체 BUY 신호 수집
→ 우선순위 정렬 → T1~T4 일괄 배분. 전략은 signal만 반환, 주문 제출 금지.
composite_signal_engine.py에 _sequential_allocations() 구현 완료. orchestrator.py [OK]
구현 완료 (2026-02-23) — 10단계 우선순위 파이프라인 + M5 비중 배분. CI-5. M5 -- 차
감 로직 상태 관리 — [OK] 해결됨 (CI-0-8, 2026-02-22 MT_VNQ2) 이슈: • ~~차감 시점
미확인 (P-3): 매수 완료시? 매수+매도 세트 완료시?~~ → CI-0-8 확정 • ~~날짜 자정
경계에서 카운트 초기화 시점 (17:30 기준? 00:00 기준?)~~ → CI-0-8 확정 • 차감 후 음
수가 되면 0 처리 or 다음 날로 이월? → CI-0-8: 0으로 clamp + 알람 필요 결정: 질문
선택지- 31 taejun_attach_pattern — 전략 리뷰 2026-02-23 (VNQ) 질문선택지 차감 시점
A. 매수 체결 완료시 / B. 매도 체결 완료시 / C. 매수+매도 세트 완료시 → [OK] CI-0-8
으로 해결됨: A. 매수 FILLED 완료시 1개 차감 카운트 초기화A. 매일 00:00 / B. 매일
17:30 (장 시작 기준) → [OK] CI-0-8으로 해결됨: B. 매일 17:30 (session_day 기준) CI-6.
수익금 분배 -- ProfitDistributor 설계 이슈 — [!] 일부 해결 (CI-0-9), P-1 미해결 이슈: 항
목문제상태 분배 시점매도 즉시 분배 vs 장 마감 후 일괄 분배 -- 즉시 분배시 당일 추
가 거래와 충돌[OK] CI-0-9 확정: 장 마감 후 1회 (KST 06:05) 손실 거래시pnl < 0이면 분
배 skip? 누적 손실에서 차감? [OK] CI-0-9 확정: pnl <= 0 이면 분배 skip 수익금 부족4
주 모두 살 금액 미달시 처리 방식 미확인 (P-1) [!] P-1 미확인 구매 순서동시 vs 순차
미확인 (P-1) [!] P-1 미확인 가격 데이터 없음SOXL 등 특정 종목 가격 조회 실패시 해당
종목 skip? 전부 중단? [OK] CI-0-9 확정: 해당 종목만 skip + 알람 M1 적용 여부수익금
분배 매수도 지정가 적용인지? [OK] CI-0-9 확정: 지정가 적용 CI-7. 리츠 과열 공격 조절
모드 -- 종목 교체 구현 이슈: 항목문제 교체 처리 위치각 전략 내부(jab_soxl이 리츠 과
열 여부 직접 확인) vs Orchestrator 후처리(시그널 ticker 교체) 기존 포지션 처리리츠 과
열 발동시 보유중인 SOXL 포지션 -> 그대로 유지? 즉시 매도? 목표수익률 적용 범위신
규 진입분에만 x50% 적용? 기존 포지션도 소급 적용? params 런타임 수정
JAB_SOXL["target_pct"] 동적 변경 -> 다른 전략에 사이드 이펙트 위험 -> 복사본 사용
필요 해제 처리90일 후 리츠 과열 해제시 원래 종목/목표수익률 복귀 로직 필요 P-2 미
확인"제외항" 범위 확인 전까지 전체 -0.5% 적용 범위 구현 불가 CI-8. C-1 수수료
0.74% -- 전략별 적용 방식 — [!] 설계 확정, FeeCalculator 구현 대기 이슈: 전략특이사
항 저가매수 (G) CONL/SOXL은 VNQ 120일선 조건부 100% 목표 -- 수수료 0.74% 영향
미미, AMDL/NVDL/ROBN 등 고목표 종목도 동일 분할매도 (sell_splits=6)매 회 매도시 수
수료 부과 -> 총 수수료 = 매수 0.37% + 매도 0.37%x6 = 2.59% -> 목표 설정시 반영
필요 ETQ/BAC목표 1.05% net (M20 +0.25% 반영) -> gross = 1.05 + 0.74 = 1.79% 기준
으로 체결 판단 리츠 과열 조절 모드SOXX/HOOD/GLD는 이미 목표 x50% -> 수수료 차
감 후 실질 목표가 음수가 될 수 있음 필요 신설: •
FeeCalculator.gross_target(net_target_pct) -> float 메서드 • check_exit에서 gross 기준으
로 비교하도록 전략별 수정 CI-9. ETQ/BAC 목표수익률 params 수정 — [OK] 해결됨
(2026-02-22)- 32 변경 필요 파일: params.py taejun_attach_pattern — 전략 리뷰 2026-
02-23 (VNQ) `python 수정 전 JAB_SETH = { "target_pct": 0.5, ... } BANK_CONDITIONAL =
{ "target_pct": 0.5, ... } 수정 후 (M20 +0.25% 직접 반영) JAB_ETQ = { "target_pct":
1.05, ... } BANK_CONDITIONAL = { "target_pct": 1.05, ... } ` 상태: 해결 -- 0222 원문에서
params.py에 target_pct=1.05%를 직접 반영하는 것으로 확인됨. CI-10. 코드화 블로커 정
리 (미확인 -> 구현 불가 항목) 아래 항목은 P-1~P-4 확인 전 코드 반영 불가: 블로커
P-1 미확인 막힌 구현 ProfitDistributor 분배 순서 / 수익금 부족 처리 상태 [!] 미해결
P-2 미확인 리츠 과열시 -0.5% 적용 범위 [!] 미해결 ~~P-3 미확인~~ ~~M5 차감 시점
/ "각각" 범위~~ [OK] CI-0-8로 해결 — 매수 FILLED 완료 시 차감, 17:30 초기화 P-4 미
확인 TSLL 200만원 한도 vs M3 비율 우선순위 [!] 미해결 CI-11. M4 VNQ 추종 데이터
소스 이슈: VNQ (미국 리츠 ETF) 데이터를 실시간으로 가져와야 하는데: • KIS API에서
VNQ 호가 데이터 지원 여부 확인 필요 • SK리츠(395400) 기반 코드가 모두 VNQ로 교
체되어야 함 • 6일 과거 데이터 기반 60일선/120일선 계산 로직 필요 CI-12. M5 T1~T4
비중 체계 구현 — [!] 설계 확정 (CI-0-6), Orchestrator 구현 대기 이슈: • 기존 M3 진입
비율(단순 50/40/33%)에서 T1~T4 순차 계산 방식으로 변경 • GLD/VIX/달러 동적 조정을
실시간 반영하는 비중 계산기 필요 • T1~T4 계산 후 총합이 100%를 초과하는 경우 처
리 로직 CI-13. ETQ 종목 교체 — [OK] 해결됨 이슈: SETH -> ETQ로 종목 변경에 따른
코드 수정: • ~~jab_seth.py -> jab_etq.py 파일 리네임 또는 신규 생성~~ → jab_etq.py
구현 완료 • ~~params.py에서 JAB_SETH -> JAB_ETQ 키 변경~~ → params.py 반영 완료
• ~~registry에서 전략명 업데이트~~ → 완료- 33 taejun_attach_pattern — 전략 리뷰
2026-02-23 (VNQ) CI-14. M200 즉시 매도 -- 신규 (2026-02-22) 이슈: M200 즉시 매도
7개 조건을 실시간으로 병렬 평가해야 함. 항목문제 조건2 불명확"44POLYMARKET BTC"
-- 원문 해석 불가 (P-5) VNQ 실시간 데이터조건7 (VNQ 5%+ 상승) 평가를 위해 VNQ
실시간 호가 필요 우선순위M200 발동 시 기존 전략(M5 비중, 목표수익률) 무시하고 즉
시 매도하는 구조 필요 조건 병렬 평가7개 조건 중 1개라도 부합하면 즉시 매도 -> 병
렬 검사 구조 T1~T4 비중 축소조건7 발동 시 GLD/숏 제외 전 포지션 30% 비중 축소
기한 만기(조건6)전략별 기한 관리 시스템과 연동 필요 필요 파일: m200_stop.py CI-15.
M80 거래대금 초단타 -- 신규 (2026-02-22) 이슈: M80은 기존 J 섹터 로테이션(Q-8)을
확장하여 섹터별 종목 거래대금 기반 매매 체계를 구축. 항목문제 거래대금 데이터섹터
별 종목의 2년 최저 거래대금 데이터를 지속 관리해야 함 점수 체계거래대금 배수별 점
수 산정 (1.2배=1.2점 ~ 18배=18점) -- M200 매수 우선순위와 연동 종목 범위BTC/반도
체/금/은행/에너지/헬스케어 6개 섹터, 각 섹터별 레버리지/비레버리지 매핑 M7 연동레
버리지 금지 시 비레버리지 대체 종목으로 자동 전환 J 섹터와 중복기존 J 섹터 로테이
션과의 관계 불명확 (P-7) → [OK] P-7 잠정 확정: M80 발동 시 동일 종목 J 신호 대체 [!]
대체 범위(종목별 vs J전체) 태준님 확인 필요 최소 점수 기준매수 최소 점수 기준 미정
(P-8) → [OK] P-8 잠정 확정: 35점 (보수적 기준) [!] 점수 구성(거래대금+목표수익률 배점)
태준님 확인 필요 (P-9) 필요 파일: m80_sector.py CI-16. 포지션 confirmed vs effective
분리 — [!] 설계 확정 (MT_VNQ3), 코드 반영 대기 출처: MT_VNQ3 §4 이슈: 현재 코드는
단일 Position 객체만 있고 confirmed/effective 구분 없음. 타입정의사용처
position_effective실체결 수량/평단(vwap)리스크/청산/현금계산/M200/M201
position_confirmed진입 계획 완료 여부전략 판단/신호용 현재 코드 문제: execute_buy
즉시 Position 생성 → FILLED 전에 effective처럼 취급 위험. 해결: base.py
Position.confirmed = False 필드 추가 (Worker-B 구현 완료). CI-17. Fill Window 10초 룰
— [!] 설계 확정 (MT_VNQ3), OrderQueue 구현 대기 출처: MT_VNQ3 §5 이슈: 진입 주문
10초 후 잔량 취소 로직이 코드에 없음. 필요 구현:- 34 • fill_window_sec = 10
(ENGINE_CONFIG에 추가 완료) taejun_attach_pattern — 전략 리뷰 2026-02-23 (VNQ) •
OrderQueue.on_fill_window_expire(order_id) 메서드 • PARTIAL_ENTRY_DONE 이벤트 기
록 + 동일 signal_bar_ts 재진입 차단 CI-18. 추격매수 금지 — [!] 설계 확정 (MT_VNQ3),
주문 레이어 구현 대기 출처: MT_VNQ3 §6 이슈: 현재 주문 레이어 없음. 가격 고정 매수
로직 구현 필요. 필요 구현: • 모든 매수 주문에 reference_price 필드 추가 • tick 보정 시
BUY: floor_to_tick(reference_price) 적용 (상향 불가) • 미체결 → 현금 유지 (잔량
reserved_cash 해제) CI-19. 매도 즉시성 허용 — [!] 설계 확정 (MT_VNQ3), 주문 레이어
구현 대기 출처: MT_VNQ3 §7 이슈: M200/M201/리스크 청산 시 즉시성 보장 메커니즘
없음. 필요 구현: • bid_slip_max_pct = 0.002 (ENGINE_CONFIG에 추가 완료) •
SellOrder.is_urgent 플래그 → True 시 bid 기반 0.2% 이내 marketable limit 적용 •
M200/M201 파이프라인에서 is_urgent=True 설정 CI-20. M201 즉시모드 v1.0 — [!] 설계
확정 (MT_VNQ3), 파이프라인 연동 대기 출처: MT_VNQ3 §14 이슈: m201_mode.py 로직
구현됨. [OK] Orchestrator 연동 완료 (2026-02-23) — composite_signal_engine.py M201
통합. 필요 구현: • Orchestrator 우선순위 체계에 M201 게이트 추가 (M200 다음) •
composite_signal_engine.py M201 TODO 주석 → [OK] 실제 연동 완료 (2026-02-23) • 전
환 파이프라인 (전역 락) Orchestrator에서 관리 9. 엔진 파일 구조 (2026-02-22 업데이
트) strategies/taejun_attach_pattern/ (21+ files) ├── base.py @register + get_strategy()
├── fees.py # BaseStrategy, MarketData, Signal, Position ├── registry.py #
FeeCalculator (왕복 0.74%) ├── portfolio.py 충돌해소) ├── params.py jab_soxl.py #
13개 전략 파라미터 (ETQ target_pct=1.05% 반영) ├── __init__.py # A. 잽모드 SOXL
├── jab_bitu.py D. 숏 잽모드 ETQ (구 jab_seth.py) ├── vix_gold.py bargain_buy.py
sector_rotate.py # G. 저가매수 ├── short_macro.py # B. 잽모드 BITU ├── jab_tsll.py
# E. VIX -> GDXU ├── sp500_entry.py # H. 숏포지션 전환 ├── reit_risk.py # #
PortfolioManager (자금배분, # 패키지 + 자동 등록 │ ├── # C. 잽모드 TSLL ├──
jab_etq.py # # F. S&P500 편입 ├── # I. 리츠 리스크 (VNQ 추종) ├── # J. 섹터 로
테이션 (재설계) ├── bank_conditional.py # K. 조건부 은행주 ├──
short_fundamental.py # L. 숏 포지션 재무- 35 taejun_attach_pattern — 전략 리뷰 2026-
02-23 (VNQ) 역전 (신규) ├── disaster_emergency.py # M. 비이상적 재난 이머전시 (신
규) ├── m200_stop.py m80_sector.py # M80 거래대금 초단타 (신규) ├──
m201_mode.py # M200 즉시 매도 (신규) ├── # M201 즉시모드 v1.0 (신규, Worker-B
구현 완료) │ [[OK] 구현 완료 (2026-02-23)] ├── orchestrator.py # M3/M5 로직 —
10단계 파이프라인 ├── limit_order.py # 지정가 주문 큐 — LimitOrder + OrderQueue
├── profit_distributor.py # 수익금 분배 — SOXL→ROBN→GLD→CONL ├──
m5_weight_manager.py # M5 비중 조절 — T1~T4 + 동적 조정 ├── m200_stop.py #
M200 즉시매도 — 7개 조건 킬스위치 ├── m28_poly_gate.py LONG/SHORT/NEUTRAL
└── schd_master.py # M28 게이트 — BTC/NDX # SCHD 적립 + 매도 차단 + M300
[미구현 컴포넌트] ├── m4_emergency.py # M4 VNQ 기반 이머전시 모드 (미구현) 10.
다음 단계 1. M200 구현 -> 손절 안전망 최우선. m200_stop.py 신규 생성, 7개 즉시 매
도 조건 병렬 평가 2. M80 구현 -> 섹터별 거래대금 추적 시스템 신설. m80_sector.py
신규 생성, 6개 섹터 매핑 3. CI-13 우선 -> jab_etq.py 생성 (SETH -> ETQ 종목 변경) 4.
CI-9 반영 -> params.py target_pct=1.05% 직접 반영 (ETQ, BAC) 5. C-1 수수료 반영 ->
FeeCalculator + 전략별 gross 기준 체결 판단 6. Q1~Q11 반영 -> params.py 수치 업데
이트 (XRP 2.5%, BRKU -31%, TSLL GLD 0.1%) 7. CI-11 VNQ 데이터 -> reit_risk.py 추종
종목 교체 8. M4 이머전시 모드 -> m4_emergency.py 신규 구현 9. M5 비중 체계 ->
m5_weight_manager.py 신규 구현 10. Orchestrator -> M3(시간 조건) + M5(T1~T4 비중)
+ M200(즉시 매도) 통합 11. 신규 전략 -> short_fundamental.py +
disaster_emergency.py 구현 12. M7 NVDL 추가 -> NVDL -> NVDA 대체 종목 테이블 반
영 13. P-1~P-4 확인 -> 태준님 확인 후 미결 항목 반영 14. P-5~P-8 확인 ->
M200/M80 관련 신규 미결 항목 태준님 확인 15. M201 Orchestrator 연동 -> [OK]
composite_signal_engine.py M201 연동 완료 (CI-20) 16. Fill Window 구현 ->
OrderQueue.on_fill_window_expire + PARTIAL_ENTRY_DONE (CI-17) 17. 추격매수 금지 +
매도 즉시성 -> 주문 레이어(LimitOrder) 구현 시 reference_price + bid_slip_max_pct 적
용 (CI-18/CI-19) 18. P-15~P-17 확인 -> M201 전환 규모, REIT_MIX 결측 처리,
PARTIAL_ENTRY_DONE 재진입 기준 태준님 확인 15. J 섹터 로테이션 -> Q-8 재설계 내
용 코드화 (M80과 통합 여부 결정)
개선코드
숏/헤지 종목 재무제표 판단 규칙 — 최종 수정안(매핑/대체 포함)
1) 목적
CONZ/IREZ/HOOZ/MSTZ/SOXS 같은 숏·헤지 ETF는 자체 재무제표가 의미 없거나(ETF/
파생), 실제 리스크는 **기초자산의 분기 실적(순이익)**에 의해 결정된다.
따라서 숏·헤지 종목의 “재무제표 순이익 조건”은 **지정된 기초자산(대체 포함)**의 최근
분기 순이익으로 판정한다.
2) 재무제표 참조 매핑(필수)
아래 매핑은 고정 규칙으로 적용한다. (모든 전략/모드 공통)
1. CONZ → COIN 재무제표 확인
• CONZ의 재무제표 조건은 **COIN(코인베이스)**의 최근 분기 순이익으로 판정한
다.
2. IREZ → IREN 재무제표 확인
• IREZ의 재무제표 조건은 IREN의 최근 분기 순이익으로 판정한다.
(표기 정정: “IREN 는 IREZ 확인”이 아니라, IREZ가 IREN을 참조가 맞는 방향)
3. HOOZ → HOOD 재무제표 확인
• HOOZ의 재무제표 조건은 **HOOD(로빈후드)**의 최근 분기 순이익으로 판정한
다.
4. MSTZ → MSTR 재무제표 확인
• MSTZ의 재무제표 조건은 **MSTR(마이크로스트래티지)**의 최근 분기 순이익으로
판정한다.
5. SOXS → SOXX 재무제표 확인(지표용)
• SOXS의 재무제표 조건은 원칙적으로 SOXX를 참조한다.
6. SOXX 재무제표는 AMD로 대체(최종 판정용)
• 단, SOXX는 지수 ETF이므로 “순이익”이 직접 적용되기 애매할 수 있다.
• 따라서 재무제표 최종 판정은 AMD의 최근 분기 순이익으로 대체한다.
• 정리하면:
o SOXS 재무 판정 = AMD 최근 분기 순이익(최종 기준)
o SOXX는 **섹터/지표 확인용(참고용)**으로만 사용한다.
3) 판정 규칙(숏 로직 방향 확정)
숏/헤지(예: CONZ/IREZ/HOOZ/MSTZ/SOXS)에 대해 “재무제표 순이익” 조건은 아래로 고
정한다.
• 기초자산 최근 분기 순이익이 “마이너스(적자)”이면 → 숏 허가(M)
• 기초자산 최근 분기 순이익이 “플러스(흑자)”이면 → 숏 불허(N)
이유: 숏 전략은 “기초자산 실적이 나쁘다(적자)” 쪽에 베팅하는 구조이므로, 흑자인데 숏
을 허가하면 규칙이 역전되어 오류가 난다.
4) 데이터 결측/불명확 처리(오류 방지)
• 기초자산 재무 데이터가 없거나/갱신 실패/기간 불명확이면
→ 해당 종목은 재무 조건 FAIL(N) 처리하고, 매수(숏 진입) 금지한다.
• SOXX처럼 ETF 재무가 애매하면 위 규칙대로 AMD로 대체하여 판정한다.
5) 한 줄 요약(문서용)
• CONZ→COIN, IREZ→IREN, HOOZ→HOOD, MSTZ→MSTR, SOXS→(SOXX 참
고)→AMD(최종)
• 적자면 숏 허가(M), 흑자면 숏 불허(N)
1) M5 비중조절 메커니즘 — 최종 확정본(오류 방지형)
1-1. 용어 정의(모호함 제거)
• n = “같은 시점에 매수 허가(PASS)된 신호(종목) 개수”
(전략 여러 개가 동시에 PASS면 합쳐서 n으로 센다)
• 비중 단위는 “전체 자산(현금+주식 포함) 대비 %” 로 고정한다.
• 매수 우선순위 정렬 기준은 M > Y > 최종수익률 이다.
(동일하면 최종수익률 높은 순)
1-2. MMM(필수) 조건
• 매수 프로토콜은 “MMM 3개”가 모두 PASS해야만 진입 가능
(MMM 하나라도 FAIL이면 그 종목은 n 계산에서 제외)
1-3. T1~T4 배분 규칙(너가 준 숫자를 “총비중”으로 고정)
동일 시점에 PASS된 종목이 n개일 때, 상위 순위부터 T1~T4로 배분한다.
n=1 (1개만 PASS)
• T1 = 99%
• 나머지 1%는 현금 유지
n=2 (2개 PASS)
• T1 = 50%
• T2 = 50%
n=3 (3개 PASS)
• 네 문장 “y3개일때 33프로로 비중을 두어서 매수”를 그대로 반영:
• T1 = 33%
• T2 = 33%
• T3 = 33%
n≥4 (4개 이상 PASS)
• 너가 준 T3 “30%”, T4 “나머지 전액”을 살리면서 합 100%로 고정:
• T1 = 33%
• T2 = 33%
• T3 = 30%
• T4 = 나머지 전액(=4%)
T5~ (5개 이상)
• 4개까지만 실제 매수(T1~T4), 5번째 이상은 “예약/대기 표시만” 하고 신규 매수는
하지 않는다.
(이 규칙은 기존 문서의 안전장치와도 맞음)
1-4. 현금 부족 처리(오류 방지)
• 매수 금액이 부족하면 “살 수 있는 수량만” 매수한다.
• T2/T3/T4로 더 많은 조건이 와도 현금이 0이면 즉시 중단한다.
• 즉, “돈이 없는데 다음 티어로 계속 계산/매수 시도”하는 오류를 원천 차단.
2) Y / M / N 점수 규칙 — 최종 확정(혼용 오류 방지)
너가 말한 점수:
• Y = +0.5 / N = 0
• M = +0.3 / N = 0
이 점수는 ‘최종수익률(%)’에 더하는 값이 아니라, 아래 두 용도로만 쓴다:
1. 매수 우선순위 정렬(랭킹)
• M PASS 개수가 많은 종목이 최우선
• 그 다음 Y PASS 개수
• 그 다음 최종수익률 높은 순
2. “최종수익률 후보 필터”에서 가벼운 가산(선택)
• 만약 점수를 수익률에 반영하고 싶다면 반드시 “환산 규칙”을 둬야 함.
(예: 1점 = 0.1% 같은 고정 환산)
• 환산 규칙이 없으면 점수는 수익률에 절대 더하지 않는다.
→ 이게 M80/M5에서 가장 많이 터지는 “단위 혼용 버그”를 막아줌.
3) 최종수익률 하한 규칙(0.8% 미만 금지) — 최종 확정
• 계산된 최종수익률이 0.8% 미만이면 매수를 중단한다.
• 수식으로 쓰면:
o 최종수익률 ≥ 0.8% 인 경우에만 진입
o 최종수익률 < 0.8% 이면 매수 불허(FAIL)
4) “거래대금 점수” 처리 정정(P-9 해결) — 삭제가 아니라 역할 분리
너가 말한 “거래대금 점수 삭제”는 M80 구조를 무너뜨릴 수 있어서 아래처럼 정정하는
게 안전해:
거래대금 시스템(배수 조건/점수)은 유지한다.
다만,
• **거래대금 점수는 “최종수익률 계산식에서 제거”**한다.
• 거래대금 점수는 오직:
1. 후보 필터(허가/불허): 배수 미달이면 FAIL
2. 우선순위 정렬: 점수 높은 종목 먼저
이 두 용도로만 사용한다.
이렇게 하면 “점수는 그대로 두고, 거래대금 시스템도 유지”하면서도
최종수익률 식에 섞여서 생기는 오류를 제거할 수 있어.
5) M200 즉시매도 조건 정정(“44POLYMARKET BTC” 제거 + 치환 방식 개선)
5-1. M200에서 삭제
• M200 조건2(44POLYMARKET BTC) 관련 문장/주석/예외 전부 삭제
• SCHD 예외 문장도 삭제(요청 반영)
5-2. Polymarket는 M200이 아니라 “게이트(M28)”에서만 처리(충돌 방지)
너가 원한 치환 문구는 M200 조건으로 넣으면 반복청산(핑퐁) 위험이 커서, 아래처럼
위치를 옮겨서 확정한다:
M28(Polymarket BTC 게이트) 문장으로 교체(최종)
• 3 : Polymarket BTC 확률: 51%/49% 게이트(허가/불허)
o p ≥ 0.51 → LONG 허가 = M
o p ≤ 0.49 → SHORT 허가 = M
o 0.49 < p < 0.51 → 중립 = N (둘 다 불허)
• 데이터 소스(당일):
o https://polymarket.com/event/bitcoin-above-on-<당일>
o
“수치를 계산해서 허가/불허가 한다.”
이렇게 하면 “44POLYMARKET”는 완전히 사라지고,
Polymarket는 허가 게이트 역할에만 남아서 시스템이 안정됨.
6) P-10 “M200 매도 직후 이머전시 재매수” —
‘즉시’ 유지하면서 핑퐁 방지
너 답: “그즉시하기”를 유지하되, 최소 안전장치 1개는 필요해.
여기서는 가장 단순/안전한 걸로 확정:
규칙: M200으로 매도한 “같은 봉(동일 bar)”에서는 신규 매수 금지
• M200 매도는 즉시(그 봉 종료 시점) 실행
• 재매수는 “즉시”라도 다음 봉부터 가능
• 그리고 재매수 시점에 M200 조건이 하나라도 아직 True면 재매수 금지
이 한 줄이 없으면 실제로 “매도→즉시매수→다시매도” 무한루프가 터질 수 있어.
7) P-12 “조건6만 다음날 매도” — 너 답안 반영 + 명확화
너 답: “조건 달성시 매도 / 조건6만 다음날 하기” 유지 가능.
다만 문장으로 딱 고정:
**M200 조건 6(기한 만기)**만 예외:
• 조건6이 True가 되면 즉시 매도하지 않고
• 다음 거래 세션(17:30~06:00) 중 1회 전액 매도한다.
(조건1/3/4/5/7 등 나머지는 조건 달성 즉시 매도 그대로)
8) ETQ/SETH 관련 전부 삭제 — 정리본(요청 반영)
아래는 문서/개선사항/이력에서 완전 삭제:
• jab_etq (구 SETH) ... 테이블 행
• Q-2 SETH 진입 조건 해석 ...
• 2026.2.21 SETH (숏 ETF) ETQ로 종목 변경 ...
• D. 숏 잽모드 ETQ ... 섹션 전체(조건/모호사항 포함)
ETQ를 삭제하면, 전략 일람에서 **D 전략은 “삭제됨(비활성)”**로 처리하거나
A,B,C,E…로 재정렬해서 “총 전략 수”도 맞춰줘야 문서-엔진 불일치 오류가 안 남는다.
9) “숏(CONZ/IREZ/HOOZ/MSTZ, SOXS)” 파트 — 한 줄로 더 고쳐야 할 핵심
너가 숏 규칙에 이렇게 적었어:
•
“재무제표 순이익 +0.1% 이상이면 허가=N, -0.1% 이하면 허가=M”
이건 숏 논리와 반대라서 실전 오류가 날 가능성이 큼.
숏은 원칙적으로:
• 기초자산 **적자(-)**일 때 숏 허가
• 흑자(+)일 때 숏 불허
이게 기존 문서(Short-Fundamental 정의)랑도 일치해.
즉, 숏에서는 ‘순이익 플러스 허가’가 아니라 ‘순이익 마이너스 허가’가 맞아.
마지막으로: “또 뭐 고쳐?”에 대한 답(딱 3개만 추가로 하면 안정화됨)
1. M200에서 Polymarket 조건을 완전히 제거하고(M28로 이동) ← 위에서 반영 완
료
2. M5의 99/50/33/30 규칙을 지금처럼 “총비중표”로 고정 ← 반영 완료
3. 숏에서 ‘재무제표 순이익 허가/불허 방향’만 정확히 뒤집기 ← 이것만 추가로 확
정해줘야 함
원하면, 네가 마지막에 쓴 숏(CONZ/IREZ/HOOZ/MSTZ, SOXS) 규칙 블록을 그대로 가져
와서, 숏 재무조건만 오류 없이(의도 유지) 문장 형태로 다시 써줄게.
CONZ 는 COIN 재무제표 확인 IREN 는 IREZ 재무제표 확인 HOOZ 는 HOOD 재무제표
확인 MSTZ는 MSTR 재무제표 확인 SOXS는 SOXX 재무제표 확인 SOXX는 AMD 재무제
표로 대체한다. 내용써줘
아래처럼 “기초자산 재무제표 참조(대체 포함)” 규칙으로 문서에 그대로 넣으면 돼.
숏/헤지 종목 재무제표 확인 규칙(기초자산 매핑)
• CONZ 매매 판단 시, COIN의 최근 분기 재무제표(순이익)를 기준으로 허가/불허
를 판단한다.
• IREZ 매매 판단 시, IREN의 최근 분기 재무제표(순이익)를 기준으로 허가/불허를
판단한다.
• HOOZ 매매 판단 시, HOOD의 최근 분기 재무제표(순이익)를 기준으로 허가/불허
를 판단한다.
• MSTZ 매매 판단 시, MSTR의 최근 분기 재무제표(순이익)를 기준으로 허가/불허
를 판단한다.
• SOXS 매매 판단 시, SOXX의 최근 분기 재무제표(순이익)를 기준으로 허가/불허
를 판단한다.
• 단, SOXX는 지수 ETF이므로 재무제표 기준이 불명확할 수 있어, 재무 판단용 기
초자산은 AMD로 대체하여 **AMD의 최근 분기 재무제표(순이익)**로 최종 판정
한다.
o 즉, SOXS → SOXX(지표) → AMD(재무 판단용 대체) 순서로 적용한다.
원하면, 위 규칙 바로 아래에 붙일 수 있게 **허가/불허 조건 문장(흑자면 숏 불허, 적자
면 숏 허가 같은)**도 너가 쓰는 M/N/Y 표기 방식에 맞춰서 한 줄로 정리해줄게.
M5 비중조절 메커니즘 정정 ------------------------------정정 내용 M5비중 조절 ---------
---------------- T1 Y 1개일때 99 y2개일때 50프로 y3 개일때 33프로 T2 나머지 현금 개
일때 99 y2개일때 50프로 y3 개일때 33프로 T3 나머자 현금 매수 개일때 90 y2개일때
50프로 y3 개일때 30프로 T4 나머지 현금 전액매수 YMMMN5프로인경우 허용 예시로
이렇게 3개가 조건이 완성된경우 YMMMN6프로인경우 허용 / YMMMN7.5프로인경우 허
용 / YMMMN8프로인경우 허용 y3 개일때 33프로로 비중을 두어서 매수한다. 매수 금액
이 부족하면 구매할 수 있는 수량 만큼만 구매한다. T2로 더 많은 조건이 되도 매수할
돈이없다면 매수를 멈춘다. M > Y > 최종수익률 순으로 중요도가 중요해진다. 이 중요도
에 맞게 먼저 매수하거나 비중을 늘려서 구매한다. M= 3개 필수 Y= 0개 있다면 최종수
익률 91프로 1순위 > Y= 0개 있다면 최종수익률 80프로 순위2위 Y= 1개 있다면 최종수
익률 49프로 1순위 > Y= 2개있다면 1순위 ------------------------------정정 내용 매수 정
리하기 ------------------------- 매수 ( 거래대금 점수. ) + ( 기본수익률 ) + ( 금. ) +
( polymarket 확률. ) + ( 시작가 대비. ) + ( s앤p500. ) + ( 부동산 m4 나누기 2 또는 더하
기. ) + ( 120일선 돌파 ) + ( 재무제표 순이익 ) + ( 부동산 리츠 4계절 ) ( - 0.75퍼센트
수수료 ) = 최종 수익률 매수프로토콜은 MMM 3개를 무조건 조건 허용이되어야한다.
M5 비중조절 메커니즘 정정 -------------------------------------BTC 비트코인 매수 정정 -
--------------------------------------- M4000 매수 코인 프로토콜 / 최종 수익률 계산 매수
( ( A 거래대금 점수. ) + + ( B 금. ) + ( C polymarket BTC 확률. ) + ( D 재무제표 순이
익 ) + ( E s앤p500. ) + ( A-1 기본 수익률 + 0.8. ) + ( A-2 시작가 대비. ) + ( A-3 120일선
돌파 ) + ( A-4 M4- 부동산 변동 수익률 포지션 실시간 변경 모드 ) ) + ( A-5 -0.75퍼센트
수수료 ) = 최종 수익률 B : 기본 수익률 0.8 Y = + 0.5 / N= 0.0 M = + 0.3 / N= 0.0 BTC
매수시 1 : 거래대금 배수 조건(2년 최저 대비 N배): 섹터/종목별 허가 배수 미달이면 불
허 = N 허가배수에 들경우 = Y 2: 금가격 플러스인경우 허가 = N / 마이너스 불허가 =
M 3 : Polymarket BTC 확률: 51%/49% 게이트(허가/불허) 허가 = M 불허가 = N
https://polymarket.com/event/bitcoin-above-on-february-24 수치를 계산해서 한다. 허가
불허가 한다. 4 : 재무제표 순이익: +0.1% 이상이면 허가 = M , -0.1% 이하면 불허 = N
5 : S앤P500 POLYMARKET https://polymarket.com/event/which-companies-added-to-sp-
500-in-q1-2026 사이트 링크 Which companies added to S&P 500 in Q1 2026? 1) 제목에
서 확률이 높은 기업 51퍼센트 이상 인경우 2) 최근 재무제표 순이익인경우 3) 매수하는
종목이 같은경우 4) 3가지 모두 충족시 허가 =Y / 1개만 충족 = N A-1 : 기본 수익률 0.8
A-2 시작가 대비 상승 제한(예: 4%+ 상승이면 매수 금지 등): 조건 위반이면 불허 N 시
작가가 3퍼센트인경우 -3퍼센트 시작가가 2퍼센트인경우 -2 퍼센트 1퍼센트인경우 -1퍼
센트 계산해서 최종 수익률 조정한다. A-3 120이동평균선돌파시 + 수익률 50퍼센트 더한
다. / 메수 매도를 당일이 원칙이오나 조건 해당시 40일 동안 스윙 투자 허용 A-4 부동
산 변동 수익률 포지션 실시간 변경 모드 모드별로 목표수익률이 = 기본수익률을 변경
해준다. 레버리지 스탑 해제(60) 3년 최고가 기준 6일 과거데이터로 60일선 아래
0.8~1.0% → 1.8%, 1.8~2.0% → 2.8%-- 2 taejun_attach_pattern — 전략 리뷰 2026-02-23
(VNQ) 모드조건 (VNQ 기준)목표수익률 변경추가 제한 레버리지 스탑 해제(120) 120일선
아래0.8~1.0% → 2.8%, 1.8~2.0% → 4.8%레버리지 GOLD/SHORT 매매 중단 레버리지 스
탑 사용 허가(60) 60일선 위기존 유지 (0.8~1.0% → 0.8%, 1.8~2.0% → 1.8%) 레버리지 스
탑 사용 허가(120) 120일선 위기존 유지레버리지 ETF 즉시 전액 매도(GDXU 제외), 1배수
대체(M7 참조) ALL IN ONE (20, 120) VNQ 20일선 시작가 -1.0% 또는 120일선-3% 하락
상태 기본 5%, GLD 하락 +1%, GLD 상승-0.5% 특수 조건 (아래 별도 정리 BTC 종목 이
종목들은 매수시 이러한 조건을 확인하고 매수한다.
-----------------------------------------
-------------------------------- MSTR MSTU MSTR COIN CONL COIN IREN IRE IREN RIOT
RIOX RIOT CRCL CRCA CRCL BMNR (비트마인 이머션) BMNU BMNR BTCT (BTC 디지털)-
BTCT- 17 taejun_attach_pattern — 전략 리뷰 2026-02-23 (VNQ) 기초자산레버리지 ETF
(1순위) M7 대체 (1배수) CNCK (코인체크 그룹)-CNCK XDGXX (디지 파워)-XDGXX FUFU
(비트푸푸)-FUFU ANT (앤트알파)-ANT ---------------------------------------------------------
----------------------- BTC 종목 이종목들은 매수시 이러한 조건을 확인하고 매수한다. 조
건 : 최종수익률≥0.8% 계산시 이보다 적은 경우 매수를 멈춘다.
---------------------------
----------NASDAQ매수 ---------------------------------------- M4000 매수 나스닥 프로토
콜 / 최종 수익률 계산 매수 ( ( 1 거래대금 점수. ) + + ( 2 금. ) + ( 3 polymarket
NASDAQ 확률. ) + ( 4 재무제표 순이익 ) + ( 5 s앤p500. ) + ( A-1 기본 수익률 + 0.8. ) +
( A-2 시작가 대비. ) + ( A-3 120일선 돌파 ) + ( A-4 M4- 부동산 변동 수익률 포지션 실
시간 변경 모드 ) ) + ( A-5 -0.75퍼센트 수수료 ) = 최종 수익률 B : 기본 수익률 0.8 Y =
+ 0.5 / N= 0.0 M = + 0.3 / N= 0.0 NASDAQ 매수시 1 : 거래대금 배수 조건(2년 최저 대
비 N배): 섹터/종목별 허가 배수 미달이면 불허 = N 허가배수에 들경우 = Y 2: 금가격
플러스인경우 허가 = N / 마이너스 불허가 = M 3 : Polymarket NASDAQ 확률: 51%/49%
게이트(허가/불허) 허가 = M 불허가 = N https://polymarket.com/event/ndx-up-or-down-
on-february-24-2026 나스닥 당일 . 허가 불허가 한다. Nasdaq 100 (NDX) Up or Down
on February 24? 상승확률 하락확률 계산한다. 4 : 재무제표 순이익: +0.1% 이상이면 허가
= M , -0.1% 이하면 불허 = N 5 : S앤P500 POLYMARKET
https://polymarket.com/event/which-companies-added-to-sp-500-in-q1-2026 사이트 링
크 Which companies added to S&P 500 in Q1 2026? 5) 제목에서 확률이 높은 기업 51퍼
센트 이상 인경우 6) 최근 재무제표 순이익인경우 7) 매수하는 종목이 같은경우 8) 3가지
모두 충족시 허가 =Y / 1개만 충족 = N A-1 : 기본 수익률 0.8 A-2 시작가 대비 상승 제
한(예: 4%+ 상승이면 매수 금지 등): 조건 위반이면 불허 N 시작가가 3퍼센트인경우 -3
퍼센트 시작가가 2퍼센트인경우 -2 퍼센트 1퍼센트인경우 -1퍼센트 계산해서 최종 수익
률 조정한다. A-3 120이동평균선돌파시 + 수익률 50퍼센트 더한다. / 메수 매도를 당일이
원칙이오나 조건 해당시 40일 동안 스윙 투자 허용 A-4 부동산 변동 수익률 포지션 실
시간 변경 모드 모드별로 목표수익률이 = 기본수익률을 변경해준다. 레버리지 스탑 해제
(60) 3년 최고가 기준 6일 과거데이터로 60일선 아래 0.8~1.0% → 1.8%, 1.8~2.0% →
2.8%-- 2 taejun_attach_pattern — 전략 리뷰 2026-02-23 (VNQ) 모드조건 (VNQ 기준)목
표수익률 변경추가 제한 레버리지 스탑 해제(120) 120일선 아래0.8~1.0% → 2.8%,
1.8~2.0% → 4.8%레버리지 GOLD/SHORT 매매 중단 레버리지 스탑 사용 허가(60) 60일선
위기존 유지 (0.8~1.0% → 0.8%, 1.8~2.0% → 1.8%) 레버리지 스탑 사용 허가(120) 120일
선 위기존 유지레버리지 ETF 즉시 전액 매도(GDXU 제외), 1배수 대체(M7 참조) ALL IN
ONE (20, 120) VNQ 20일선 시작가 -1.0% 또는 120일선-3% 하락 상태 기본 5%, GLD 하
락 +1%, GLD 상승-0.5% 특수 조건 (아래 별도 정리 NASDAQ 종목 이종목들은 매수시
이러한 조건을 확인하고 매수한다.
------------------------------------------------------------
------------- SOXX NVDA SOXL AMD NVDL AMDL AVGX LINT QCMU ARMG MVLL AVGO
SOXX SOXL SOXX NVDA NVDL NVDA AMD AMDL AMD AVGO AVGX AVGO INTC LINT
INTC TXN-TXN QCOM QCMU QCOM ARM ARMG ARM MRVL MVLL MRVL LLY (일라이릴
리) LLYX LLY JNJ (존슨앤존슨) JNJ JNJ NVO (노보노디스크) NVOX NVO XLE ERX섹터 대표
XOM XOMX엑슨모빌 CVX CVX셰브론 COP COP코노코필립스 SLB SLB슐럼버거 WMB
WMB윌리엄스 HOOD ROBN XLF FAS ---------------------------------------------------------
----------------------- NASDAQ 종목 이종목들은 매수시 이러한 조건을 확인하고 매수한
다. 조건 : 최종수익률≥0.8% 계산시 이보다 적은 경우 매수를 멈춘다.
CONZ/IREZ/HOOZ/MSTZ/ SOXS CONZ/IREZ/HOOZ/MSTZ/ SOXS 정정 -- 숏매수 방식으
로 수정한다. 매도방식은 이러한다. ( 1. 거래대금 점수. ) + ( 2. vix. ) + ( 3. 금. ) +
( polymarket 확률. ) + ( 시작가 대비. ) + ( s앤p500. ) + ( 부동산 m4 나누기 2 또는 더하
기. ) + ( 120일선 돌파. ) + ( 재무제표 마이너스. ) + ( 부동산 리츠 사계절 ) - ( - 0.75퍼센
트 수수료 ) = 최종 매도 CONZ/IREZ/HOOZ/MSTZ/ 숏 매수시 1 : 거래대금 배수 조건
(최근 5일 거래대금 20퍼센트 떨어진경우 미달이면 불허 = N 허가배수에 들경우 = Y 2:
금가격 플러스인경우 허가 = M / 마이너스 불허가 = N 3 : Polymarket NASDAQ 확률:
51%/49% 게이트(허가/불허) 49프로 허가 = M 51프로 불허가 = N
https://polymarket.com/event/ndx-up-or-down-on-february-24-2026 나스닥 당일 . 허가
불허가 한다. SOXS 매수허가 그외에는 매수불가 "44POLYMARKET BTC" - > 삭제하기 키
워드 4 : 재무제표 순이익: +0.1% 이상이면 허가 = N , -0.1% 이하면 불허 = M 5 : S앤
P500 POLYMARKET https://polymarket.com/event/which-companies-added-to-sp-500-in-
q1-2026 사이트 링크 Which companies added to S&P 500 in Q1 2026? 9) 제목에서 확
률이 높은 기업 51퍼센트 이상 인경우 10) 최근 재무제표 순이익인경우 11) 매수하는 종
목이 같은경우 12) 3가지 모두 충족시 허가 =Y / 1개만 충족 = N A-1 : 기본 수익률 0.8
A-2 시작가 대비 상승 제한(예: 4%+ 상승이면 매수 금지 등): 조건 위반이면 불허 N 시
작가가 3퍼센트인경우 -3퍼센트 시작가가 2퍼센트인경우 -2 퍼센트 1퍼센트인경우 -1퍼
센트 계산해서 최종 수익률 조정한다. A-3 120이동평균선하락시 + 수익률 30퍼센트 더한
다. / 메수 매도를 당일이 원칙이오나 조건 해당시 20일 동안 스윙 투자 허용 A-4 부동
산 변동 수익률 포지션 실시간 변경 모드 모드별로 목표수익률이 = 기본수익률을 변경
해준다. 레버리지 스탑 해제(60) 3년 최고가 기준 6일 과거데이터로 60일선 아래
0.8~1.0% → 1.8%, 1.8~2.0% → 2.8%-- 2 taejun_attach_pattern — 전략 리뷰 2026-02-23
(VNQ) 모드조건 (VNQ 기준)목표수익률 변경추가 제한 레버리지 스탑 해제(120) 120일선
아래0.8~1.0% → 2.8%, 1.8~2.0% → 4.8%레버리지 GOLD/SHORT 매매 중단 레버리지 스
탑 사용 허가(60) 60일선 위기존 유지 (0.8~1.0% → 0.8%, 1.8~2.0% → 1.8%) 레버리지 스
탑 사용 허가(120) 120일선 위기존 유지레버리지 ETF 즉시 전액 매도(GDXU 제외), 1배수
대체(M7 참조) ALL IN ONE (20, 120) VNQ 20일선 시작가 -1.0% 또는 120일선-3% 하락
상태 기본 5%, GLD 하락 +1%, GLD 상승-0.5% 특수 조건 (아래 별도 정리 조건 : 최종수
익률≥0.8% 계산시 이보다 적은 경우 매수를 멈춘다.
----------------------------------------
------------------------------------ CONZ/IREZ/HOOZ/MSTZ/ CONZ/IREZ/HOOZ/MSTZ/ 정
정 -- 숏매수 방식으로 수정한다. 매도방식은 이러한다. ( 1. 거래대금 점수. ) + ( 2. vix. )
+ ( 3. 금. ) + ( polymarket 확률. ) + ( 시작가 대비. ) + ( s앤p500. ) + ( 부동산 m4 나누
기 2 또는 더하기. ) + ( 120일선 돌파. ) + ( 재무제표 마이너스. ) + ( 부동산 리츠 사계
절 ) - ( - 0.75퍼센트 수수료 ) = 최종 매도 CONZ/IREZ/HOOZ/MSTZ/ 숏 매수시 1 : 거래
대금 배수 조건(최근 5일 거래대금 20퍼센트 떨어진경우 미달이면 불허 = N 허가배수에
들경우 = Y 2: 금가격 플러스인경우 허가 = M / 마이너스 불허가 = N 3 : Polymarket
BTC 확률: 51%/49% 게이트(허가/불허) 허가 = M 불허가 = N
https://polymarket.com/event/bitcoin-above-on-february-24 "44POLYMARKET BTC" - >
삭제하기 키워드 4 : 재무제표 순이익: +0.1% 이상이면 허가 = N , -0.1% 이하면 불허 =
M 5 : S앤P500 POLYMARKET https://polymarket.com/event/which-companies-added-to-
sp-500-in-q1-2026 사이트 링크 Which companies added to S&P 500 in Q1 2026? 1) 제
목에서 확률이 높은 기업 51퍼센트 이상 인경우 2) 최근 재무제표 순이익인경우 3) 매수
하는 종목이 같은경우 4) 3가지 모두 충족시 허가 =Y / 1개만 충족 = N A-1 : 기본 수익
률 0.8 A-2 시작가 대비 상승 제한(예: 4%+ 상승이면 매수 금지 등): 조건 위반이면 불허
N 시작가가 3퍼센트인경우 -3퍼센트 시작가가 2퍼센트인경우 -2 퍼센트 1퍼센트인경우
-1퍼센트 계산해서 최종 수익률 조정한다. A-3 120이동평균선하락시 + 수익률 30퍼센트
더한다. / 메수 매도를 당일이 원칙이오나 조건 해당시 20일 동안 스윙 투자 허용 A-4
부동산 변동 수익률 포지션 실시간 변경 모드 모드별로 목표수익률이 = 기본수익률을
변경해준다. 레버리지 스탑 해제(60) 3년 최고가 기준 6일 과거데이터로 60일선 아래
0.8~1.0% → 1.8%, 1.8~2.0% → 2.8%-- 2 taejun_attach_pattern — 전략 리뷰 2026-02-23
(VNQ) 모드조건 (VNQ 기준)목표수익률 변경추가 제한 레버리지 스탑 해제(120) 120일선
아래0.8~1.0% → 2.8%, 1.8~2.0% → 4.8%레버리지 GOLD/SHORT 매매 중단 레버리지 스
탑 사용 허가(60) 60일선 위기존 유지 (0.8~1.0% → 0.8%, 1.8~2.0% → 1.8%) 레버리지 스
탑 사용 허가(120) 120일선 위기존 유지레버리지 ETF 즉시 전액 매도(GDXU 제외), 1배수
대체(M7 참조) ALL IN ONE (20, 120) VNQ 20일선 시작가 -1.0% 또는 120일선-3% 하락
상태 기본 5%, GLD 하락 +1%, GLD 상승-0.5% 특수 조건 (아래 별도 정리 조건 : 최종수
익률≥0.8% 계산시 이보다 적은 경우 매수를 멈춘다.
------------------------------숏
CONZ/IREZ/HOOZ/MSTZ 매수한다.
------------------ CONZ/IREZ/HOOZ/MSTZ/ -----------
------------------------------------------------------------------------ 매도는 -- 수익금 분배
총자산이 20,000,000 < 21,000,000 100만원이상부터는 모두 SCHD를 매수한다. SCHD 조
건 달성 된 시간 당일 기준으로 매수한다. SCHD 배당금으로 재투자허용 M200 — 즉시
매도 조건 (0222 신설, MT_VNQ3 파이프라인 확정) 출처: 0222 원문 line 1212~1233 +
MT_VNQ3 §15 조건 1개라도 충족 시 목표수익률 무관하게 즉시 매도: #조건내용 1거래
대금 급감전날 거래대금 대비 장 시작 3시간 이내 15% 하락 2 Polymarket BTC 44% *(원
문 "44POLYMARKET BTC" — 모호, P-5 등록)* 3 GLD 급등GLD 6%+ 상승 4 VIX 급등VIX
10%+ 급등 5이평선 이탈모든 주식 20일 이동평균선 이탈 6기한 만기기한 만기 시 다음
날 17:30~06:00 이내 매도 7 REIT_MIX 급등REIT_MIX 5%+ 상승 시 VNQ 관련 즉시 정리
/ 그 외 30% 축소 (GLD 숏 제외) 발동 파이프라인 *(MT_VNQ3 §15 확정)*: 전역 락 ON
→ OPEN 주문 전부 취소 → 취소 ACK → reserved 재계산 → 청산 실행(매도 즉시성 규
정 적용) → 락 OFF [!] 모호사항 (P-5): M200 조건 2번 "44POLYMARKET BTC"의 44가
44% 기준인지 불명확. 정의 확정 전까지 OFF. SCHD 예외: 어떤 모드에서도 SCHD 매도
불가 (단, MASTER H/J가 AI OFF 시만 허용 여기수치를 계산해서
https://polymarket.com/event/bitcoin-above-on-february-24 3 : Polymarket BTC 확률:
51%/49% 게이트(허가/불허) 허가 = M 불허가 = N
https://polymarket.com/event/bitcoin-above-on-february-24 수치를 계산해서 한다. 허가
불허가 한다. "44POLYMARKET BTC" - > 3 : Polymarket BTC 확률: 51%/49% 게이트(허가/
불허) 허가 = M 불허가 = N https://polymarket.com/event/bitcoin-above-on-february-24
수치를 계산해서 한다. 허가 불허가 한다. 로바꿔주고 "44POLYMARKET BTC"이거에대해
서는 모든걸 - > > 3 : Polymarket BTC 확률: 51%/49% 게이트(허가/불허) 허가 = M 불허
가 = N https://polymarket.com/event/bitcoin-above-on-february-24 수치를 계산해서 한
다. 허가 불허가 한다. 로바꿔주고 ab_etq (구 SETH) 1.05% 0.74% 1.79% 종목 ETQ로 변
경 삭제해줘 Q-2 SETH 진입 조건 해석ETQ로 종목 변경, Polymarket 하락 기대 평균 12
미만 시 작동반영 삭제해줘 2026.2.21 SETH (숏 ETF) ETQ로 종목 변경MT_VNQ Q-2 삭제
해줘 D. 숏 잽모드 ETQ (구 SETH -> ETQ로 정정) !! 종목 정정: SETH -> ETQ 항목내용
매수 대상ETQ *(정정: SETH -> ETQ)* 목표 수익률+1.05% (net, 구 0.8% + 0.25%) 매도 방
식전액 매도 진입 조건 (Q-2 반영): #지표조건 1 Polymarket ETH 하락 기대평균 수치보
다 12 미만 시 작동 2 GLD >= +0.01% 3 ETQ >= 0.00% (양전) [!] 모호사항 (Q-2): "평균
수치보다 12 미만"에서 평균 수치의 정의 불명확. 어떤 기간의 평균인지(1일? 7일? 30
일?), 어떤 Polymarket 마켓의 수치인지 확인 필요. 0222 원문에서도 명시적 답변 없어
[!] 유지. 삭제해줘 P-9 M80 점수 배점 정의가 부족 최소 35점 기준인데 거래대금 점수
(최대 18점)만으로는 35점이 안 나옴 → GLD/Polymarket 점수 배점 정의 필요. 점수화
해서 매수 매도 정정함 taejun_strategy_review_2026-02-
… 거래대금 점수 삭제해줘 점수
들은 그대로 두고 거래대금 시스템 규칙은 그대로 두고 P-10 M200 매도 직후 이머전시
재매수 충돌(쿨다운 미정) 그즉시하기 A: 그즉시하기 P-12 M200의 “즉시매도”인데 조건6
만 다음날 시간대 매도(타이밍 충돌) 조건달성시 매도 조건 6만 다음날 하기 개선사항에
대한 답이야 또 어떤거 고쳐? 좋아. 네가 지금 한 수정/삭제/치환을 기준으로 보면, 아직
“오류가 날 가능성이 큰 부분(추가로 고쳐야 할 것)”이 몇 군데 남아 있어. 아래는 “지금
상태에서 실제 구현/운영 때 터질 수 있는 것”만 골라서, 무엇을 어떻게 고치면 되는지를
딱 정리했어. (코드 없이, 문서/알고리즘 규칙 글로만) 1) M200 조건2(44POLYMARKET
BTC) 치환 — 지금 방식은 “구조 충돌”이 남아있음 네가 말한 치환: “44POLYMARKET
BTC”를 3: Polymarket BTC 확률 51/49 게이트 … 허가/불허 로 바꾸겠다 문제: **M200은
“즉시 매도(킬스위치)”**인데, 51/49 게이트는 원래 **매수 허가/불허(진입 게이트)**야. 이
걸 M200에 그대로 넣으면, BTC가 49% 이하로 떨어질 때마다 “즉시 매도”가 계속 발동
해서 과매도/반복 청산(핑퐁) 위험이 커져. 고칠 것(선택 1개만 택하면 됨) 선택 A(권
장): M200에서 Polymarket 항목을 완전히 제거하고, Polymarket은 M28/M201(전환) +
M5/M40(비중)에서만 사용 → M200은 “시장 급변/리스크”만 담당하게 만들면 충돌이 줄
어. 선택 B: M200에 Polymarket을 넣고 싶으면 “51/49”가 아니라 급변 트리거로 써야 함
예: “p가 20pp 이상 급락” 같은 건 이미 M201에 있음. 그러면 M200과 M201이 겹치니
둘 중 하나로만 유지해야 오류가 줄어. 결론: 지금처럼 “M200 조건2를 51/49 게이트로
치환”은 실전에서 반복 매도 오류가 생길 가능성이 높아. M200에서 Polymarket 항목
은 삭제하고, M28/M201로만 관리하는 게 가장 안전. 2) “jab_etq / SETH→ETQ 관련 전부
삭제” — 삭제 후 공백(전략 인덱스 꼬임) 정리 필요 너는 ETQ 관련을 전부 삭제하겠다
고 했지. 그러면 남는 문제는: 전략 일람(A~M)에서 D(ETQ) 항목이 비면 이후 전략 코드/
파일명/전략 개수(12개/13개)가 문서와 달라져서 “전략 번호 기반 우선순위”를 쓰는 부분
이 있다면 오류가 날 수 있어. 고칠 것 전략 일람에서 D 전략 자체를 제거하고, “총
전략 수(몇 개)”도 맞춰서 정정 (예: A,B,C,E,F…로 재정렬하거나, D는 “삭제됨(비활성)”으로
명시) “ETQ를 쓰던 자리”에 다른 숏 전략(CONZ/IREZ/HOOZ/MSTZ 또는 SOXS)을 넣을지
여부를 명확히 (지금은 SOXS/숏 ETF 규칙이 있으니, 보통은 ETQ 삭제 → SOXS/숏 ETF로
일원화가 깔끔함) 3) P-9(점수 배점 부족) 해결 방향 — 네가 말한 “거래대금 점수 삭제”
는 위험 너 말: “거래대금 점수 삭제해줘” “점수들은 그대로 두고 거래대금 시스템 규칙
은 그대로 두고” 여기서 충돌이 있어: M80은 거래대금 점수로 우선순위를 정하는 구조인
데, 거래대금 점수를 “삭제”하면 M80의 핵심이 사라져서 엔진이 방향을 잃어. 고칠
것(의미 왜곡 없이 안전한 해결) “거래대금 점수”를 삭제하지 말고, 역할을 ‘수익률 계산’
에서만 삭제해. 즉, 거래대금 점수는 ‘정렬/우선순위’ 전용 최종수익률(목표수익률) 계산식
에는 거래대금 점수를 더하지 않는다 그리고 P-9는 이렇게 글로 확정하면 됨: “M80 최소
진입 점수 35점은 거래대금 점수만으로 충족하지 않는다. 따라서 35점 기준은 ‘거래대금
점수 ≥ X’로 대체하거나, ‘거래대금 점수 + (보조 점수: Polymarket/GLD)’의 배점을 명시한
다.” 지금 네 문장만 보면 “점수는 유지”라면서 “거래대금 점수 삭제”가 같이 있어서
모순이야. 해결은 하나: 거래대금 점수는 유지(우선순위 전용), 최종수익률 식에서만
제외 4) P-10 “M200 매도 직후 이머전시 재매수 충돌” → “그 즉시하기”로 하면 더 위험
너 답: “그 즉시하기” 이렇게 하면 실제로는: M200이 매도 → 같은 순간 이머전시가 매수
→ 또 M200 조건이 살아있으면 재매도 즉 **매도/매수 루프(핑퐁)**가 발생할 수 있어.
고칠 것(‘즉시’ 유지하면서도 오류 방지) “즉시하기”를 유지하려면, 최소한 아래 중 1
개는 있어야 함: 같은 session_bar(같은 봉)에서는 재매수 금지 M200 발동 후 해당 티커/
해당 모드 10분 쿨다운 M200 발동 후 그 session_day는 신규 진입 금지(가장 안전) 너
가 “즉시”를 원하니까, 현실적으로는 ‘같은 봉 재진입 금지’ 규칙이 최소 안전장치야.
5) P-12 “조건6만 다음날 매도”는 유지하되, 정의를 더 명확히 해야 오류가 안 남 너 답:
“조건 달성시 매도” “조건 6만 다음날 하기” 좋아. 이건 가능해. 다만 아래를 명확히 안
하면 구현이 흔들려: 고칠 것(문장으로 확정) **조건6(기한 만기)**의 의미를 딱 1개로
고정: “만기 발생 시점에 즉시 매도하지 않고, 다음 거래 세션(17:30~06:00)에서 1회 전액
매도한다.” “만기” 기준도 필요: “진입 시 설정된 기한(예: 1일/2일/60일/40일)이 끝난 순간
을 만기라고 정의한다.” 6) 네가 새로 바꾼 M5 비중 규칙 — 지금 서술은 계산 충돌/모
호함이 많음 (여기가 가장 크게 터질 수 있음) 현재 M5 정정 문장들에는 모호함이 있어:
“T1 Y 1개일때 99 / 2개 50% / 3개 33%” “T2 나머지 현금 개일때 99 …” “T3 나머자 현금
매수 개일때 90 …” “YMMMN 5% 허용 / 6% 허용 …” “M > Y > 최종수익률 중요도” 이
상태 그대로면 오류가 나는 이유: **‘Y가 몇 개인지’**가 “조건 개수”인지 “게이트 통과
수”인지 불명확 T1/T2/T3에서 “99”가 **99%**인지, 99만원인지, 99점인지 불명확 “3개 완
성 시 y3개일때 33%로 매수”는 기존 T1~T4 구조와 충돌 고칠 것(필수) M5를 딱 3
문장으로 확정해야 오류가 안 남아: “동일 시점에 매수 후보 신호가 n개일 때, 각 신호에
배정되는 비중은 (n=1/2/3/4+) 규칙표로 결정한다.” “비중 단위는 **전체 자산 대비 %**
로 고정한다.” “자금 부족 시: 가능한 수량까지만 매수하고, 잔여 신호는 중단한다(추가
신호가 있어도 현금 0이면 종료).” 지금 “99/90/50/33/30”을 어떻게 쓸지(각 신호마다? T1
에만?)가 불명확해서, 이걸 표 형태로 확정해줘야 해. 결론: 지금 추가로 고쳐야 할 “우선
순위 TOP 6” M200의 Polymarket 조건2를 51/49 게이트로 치환하는 건 위험 → M200에
서 Polymarket 제거 or 급변 트리거로 재정의 ETQ 전면 삭제 후 전략 목록 재정렬/개수
정합성 맞추기 거래대금 점수는 삭제하지 말고 “최종수익률 식에서만 제외”로 고치기(P-
9 해결 방향) M200 후 ‘즉시 재매수’는 핑퐁 위험 → 최소 “같은 봉 재진입 금지” 규칙
추가 조건6(만기) 다음날 매도 규칙: 만기 정의/실행 1회 규칙 확정 M5 새 비중 규칙의
‘99/90’ 의미와 ‘Y 개수’ 정의를 확정(표로 고정) ← 가장 중요 원하면, 너가 적은 M5
“99/90/50/33/30” 규칙을 네 의도대로(왜곡 없이) “n개 신호일 때 비중표”로 딱 1개 버전
으로 정리해서, 바로 문서에 붙이면 오류 안 나게 만들어줄게. 그렇게 개선해줘 좋아. 네
가 준 “정정/삭제/치환”을 충돌 없이(오류 안 나게) 한 문서 규칙으로 정리 + 개선해서
아래처럼 확정해줄게. (코드 없이, 그대로 붙여넣기용 / 핵심 의도는 유지) 1) M5 비중조
절 메커니즘 — 최종 확정본(오류 방지형) 1-1. 용어 정의(모호함 제거) n = “같은 시점에
매수 허가(PASS)된 신호(종목) 개수” (전략 여러 개가 동시에 PASS면 합쳐서 n으로 센다)
비중 단위는 “전체 자산(현금+주식 포함) 대비 %” 로 고정한다. 매수 우선순위 정렬 기
준은 M > Y > 최종수익률 이다. (동일하면 최종수익률 높은 순) 1-2. MMM(필수) 조건 매
수 프로토콜은 “MMM 3개”가 모두 PASS해야만 진입 가능 (MMM 하나라도 FAIL이면 그
종목은 n 계산에서 제외) 1-3. T1~T4 배분 규칙(너가 준 숫자를 “총비중”으로 고정) 동일
시점에 PASS된 종목이 n개일 때, 상위 순위부터 T1~T4로 배분한다. n=1 (1개만
PASS) T1 = 99% 나머지 1%는 현금 유지 n=2 (2개 PASS) T1 = 50% T2 = 50%
n=3 (3개 PASS) 네 문장 “y3개일때 33프로로 비중을 두어서 매수”를 그대로 반영: T1 =
33% T2 = 33% T3 = 33% n≥4 (4개 이상 PASS) 너가 준 T3 “30%”, T4 “나머지 전액”
을 살리면서 합 100%로 고정: T1 = 33% T2 = 33% T3 = 30% T4 = 나머지 전액(=4%)
T5~ (5개 이상) 4개까지만 실제 매수(T1~T4), 5번째 이상은 “예약/대기 표시만” 하고 신규
매수는 하지 않는다. (이 규칙은 기존 문서의 안전장치와도 맞음) 1-4. 현금 부족 처리(오
류 방지) 매수 금액이 부족하면 “살 수 있는 수량만” 매수한다. T2/T3/T4로 더 많은 조건
이 와도 현금이 0이면 즉시 중단한다. 즉, “돈이 없는데 다음 티어로 계속 계산/매수 시
도”하는 오류를 원천 차단. 2) Y / M / N 점수 규칙 — 최종 확정(혼용 오류 방지) 너가
말한 점수: Y = +0.5 / N = 0 M = +0.3 / N = 0 이 점수는 ‘최종수익률(%)’에 더하는 값이
아니라, 아래 두 용도로만 쓴다: 매수 우선순위 정렬(랭킹) M PASS 개수가 많은 종목이
최우선 그 다음 Y PASS 개수 그 다음 최종수익률 높은 순 “최종수익률 후보 필터”에서
가벼운 가산(선택) 만약 점수를 수익률에 반영하고 싶다면 반드시 “환산 규칙”을 둬야
함. (예: 1점 = 0.1% 같은 고정 환산) 환산 규칙이 없으면 점수는 수익률에 절대 더하지
않는다. → 이게 M80/M5에서 가장 많이 터지는 “단위 혼용 버그”를 막아줌. 3) 최종수익
률 하한 규칙(0.8% 미만 금지) — 최종 확정 계산된 최종수익률이 0.8% 미만이면 매수를
중단한다. 수식으로 쓰면: 최종수익률 ≥ 0.8% 인 경우에만 진입 최종수익률 < 0.8% 이면
매수 불허(FAIL) 4) “거래대금 점수” 처리 정정(P-9 해결) — 삭제가 아니라 역할 분리 너
가 말한 “거래대금 점수 삭제”는 M80 구조를 무너뜨릴 수 있어서 아래처럼 정정하는 게
안전해: 거래대금 시스템(배수 조건/점수)은 유지한다. 다만, **거래대금 점수는 “최종
수익률 계산식에서 제거”**한다. 거래대금 점수는 오직: 후보 필터(허가/불허): 배수 미달
이면 FAIL 우선순위 정렬: 점수 높은 종목 먼저 이 두 용도로만 사용한다. 이렇게 하
면 “점수는 그대로 두고, 거래대금 시스템도 유지”하면서도 최종수익률 식에 섞여서 생기
는 오류를 제거할 수 있어. 5) M200 즉시매도 조건 정정(“44POLYMARKET BTC” 제거 +
치환 방식 개선) 5-1. M200에서 삭제 M200 조건2(44POLYMARKET BTC) 관련 문장/주석/
예외 전부 삭제 SCHD 예외 문장도 삭제(요청 반영) 5-2. Polymarket는 M200이 아니라
“게이트(M28)”에서만 처리(충돌 방지) 너가 원한 치환 문구는 M200 조건으로 넣으면 반
복청산(핑퐁) 위험이 커서, 아래처럼 위치를 옮겨서 확정한다: M28(Polymarket BTC
게이트) 문장으로 교체(최종) 3 : Polymarket BTC 확률: 51%/49% 게이트(허가/불허) p ≥
0.51 → LONG 허가 = M p ≤ 0.49 → SHORT 허가 = M 0.49 < p < 0.51 → 중립 = N (둘
다 불허) 데이터 소스(당일): https://polymarket.com/event/bitcoin-above-on-<당일> “수
치를 계산해서 허가/불허가 한다.” 이렇게 하면 “44POLYMARKET”는 완전히 사라지고,
Polymarket는 허가 게이트 역할에만 남아서 시스템이 안정됨. 6) P-10 “M200 매도 직후
이머전시 재매수” —
‘즉시’ 유지하면서 핑퐁 방지 너 답: “그즉시하기”를 유지하되, 최소
안전장치 1개는 필요해. 여기서는 가장 단순/안전한 걸로 확정: 규칙: M200으로 매도
한 “같은 봉(동일 bar)”에서는 신규 매수 금지 M200 매도는 즉시(그 봉 종료 시점) 실행
재매수는 “즉시”라도 다음 봉부터 가능 그리고 재매수 시점에 M200 조건이 하나라도 아
직 True면 재매수 금지 이 한 줄이 없으면 실제로 “매도→즉시매수→다시매도” 무한루프
가 터질 수 있어. 7) P-12 “조건6만 다음날 매도” — 너 답안 반영 + 명확화 너 답: “조건
달성시 매도 / 조건6만 다음날 하기” 유지 가능. 다만 문장으로 딱 고정: **M200 조
건 6(기한 만기)**만 예외: 조건6이 True가 되면 즉시 매도하지 않고 다음 거래 세션
(17:30~06:00) 중 1회 전액 매도한다. (조건1/3/4/5/7 등 나머지는 조건 달성 즉시 매도
그대로) 8) ETQ/SETH 관련 전부 삭제 — 정리본(요청 반영) 아래는 문서/개선사항/이력에
서 완전 삭제: jab_etq (구 SETH) ... 테이블 행 Q-2 SETH 진입 조건 해석 ... 2026.2.21
SETH (숏 ETF) ETQ로 종목 변경 ... D. 숏 잽모드 ETQ ... 섹션 전체(조건/모호사항 포함)
ETQ를 삭제하면, 전략 일람에서 **D 전략은 “삭제됨(비활성)”**로 처리하거나
A,B,C,E…로 재정렬해서 “총 전략 수”도 맞춰줘야 문서-엔진 불일치 오류가 안 남는다. 9)
“숏(CONZ/IREZ/HOOZ/MSTZ, SOXS)” 파트 — 한 줄로 더 고쳐야 할 핵심 너가 숏 규칙
에 이렇게 적었어: “재무제표 순이익 +0.1% 이상이면 허가=N, -0.1% 이하면 허가=M” 이
건 숏 논리와 반대라서 실전 오류가 날 가능성이 큼. 숏은 원칙적으로: 기초자산 **적
자(-)**일 때 숏 허가 흑자(+)일 때 숏 불허 이게 기존 문서(Short-Fundamental 정의)랑도
일치해. 즉, 숏에서는 ‘순이익 플러스 허가’가 아니라 ‘순이익 마이너스 허가’가 맞아. 마
지막으로: “또 뭐 고쳐?”에 대한 답(딱 3개만 추가로 하면 안정화됨) M200에서
Polymarket 조건을 완전히 제거하고(M28로 이동) ← 위에서 반영 완료 M5의
99/50/33/30 규칙을 지금처럼 “총비중표”로 고정 ← 반영 완료 숏에서 ‘재무제표 순이익
허가/불허 방향’만 정확히 뒤집기 ← 이것만 추가로 확정해줘야 함 원하면, 네가 마지막
에 쓴 숏(CONZ/IREZ/HOOZ/MSTZ, SOXS) 규칙 블록을 그대로 가져와서, 숏 재무조건만
오류 없이(의도 유지) 문장 형태로 다시 써줄게. CONZ 는 COIN 재무제표 확인 IREN 는
IREZ 재무제표 확인 HOOZ 는 HOOD 재무제표 확인 MSTZ는 MSTR 재무제표 확인
SOXS는 SOXX 재무제표 확인 SOXX는 AMD 재무제표로 대체한다. 내용써줘 아래처럼
“기초자산 재무제표 참조(대체 포함)” 규칙으로 문서에 그대로 넣으면 돼. 숏/헤지 종목
재무제표 확인 규칙(기초자산 매핑) CONZ 매매 판단 시, COIN의 최근 분기 재무제표(순
이익)를 기준으로 허가/불허를 판단한다. IREZ 매매 판단 시, IREN의 최근 분기 재무제표
(순이익)를 기준으로 허가/불허를 판단한다. HOOZ 매매 판단 시, HOOD의 최근 분기 재
무제표(순이익)를 기준으로 허가/불허를 판단한다. MSTZ 매매 판단 시, MSTR의 최근 분
기 재무제표(순이익)를 기준으로 허가/불허를 판단한다. SOXS 매매 판단 시, SOXX의 최
근 분기 재무제표(순이익)를 기준으로 허가/불허를 판단한다. 단, SOXX는 지수 ETF이므로
재무제표 기준이 불명확할 수 있어, 재무 판단용 기초자산은 AMD로 대체하여 **AMD의
최근 분기 재무제표(순이익)**로 최종 판정한다. 즉, SOXS → SOXX(지표) → AMD(재무 판
단용 대체) 순서로 적용한다. 핵심내용 삭제하지말고 정리해서 AI 알고리즘에 쓰이게 수
정안을 써줘
M5 비중조절 메커니즘 정정 ------------------------------정정 내용 M5비중 조절 ---------
---------------- T1 Y 1개일때 99 y2개일때 50프로 y3 개일때 33프로 T2 나머지 현금 개
일때 99 y2개일때 50프로 y3 개일때 33프로 T3 나머자 현금 매수 개일때 90 y2개일때
50프로 y3 개일때 30프로 T4 나머지 현금 전액매수 YMMMN5프로인경우 허용 예시로
이렇게 3개가 조건이 완성된경우 YMMMN6프로인경우 허용 / YMMMN7.5프로인경우 허
용 / YMMMN8프로인경우 허용 y3 개일때 33프로로 비중을 두어서 매수한다. 매수 금액
이 부족하면 구매할 수 있는 수량 만큼만 구매한다. T2로 더 많은 조건이 되도 매수할
돈이없다면 매수를 멈춘다. M > Y > 최종수익률 순으로 중요도가 중요해진다. 이 중요도
에 맞게 먼저 매수하거나 비중을 늘려서 구매한다. M= 3개 필수 Y= 0개 있다면 최종수
익률 91프로 1순위 > Y= 0개 있다면 최종수익률 80프로 순위2위 Y= 1개 있다면 최종수
익률 49프로 1순위 > Y= 2개있다면 1순위 ------------------------------정정 내용 매수 정
리하기 ------------------------- 매수 ( 거래대금 점수. ) + ( 기본수익률 ) + ( 금. ) +
( polymarket 확률. ) + ( 시작가 대비. ) + ( s앤p500. ) + ( 부동산 m4 나누기 2 또는 더하
기. ) + ( 120일선 돌파 ) + ( 재무제표 순이익 ) + ( 부동산 리츠 4계절 ) ( - 0.75퍼센트
수수료 ) = 최종 수익률 매수프로토콜은 MMM 3개를 무조건 조건 허용이되어야한다.
M5 비중조절 메커니즘 정정 -------------------------------------BTC 비트코인 매수 정정 -
--------------------------------------- M4000 매수 코인 프로토콜 / 최종 수익률 계산 매수
( ( A 거래대금 점수. ) + + ( B 금. ) + ( C polymarket BTC 확률. ) + ( D 재무제표 순이
익 ) + ( E s앤p500. ) + ( A-1 기본 수익률 + 0.8. ) + ( A-2 시작가 대비. ) + ( A-3 120일선
돌파 ) + ( A-4 M4- 부동산 변동 수익률 포지션 실시간 변경 모드 ) ) + ( A-5 -0.75퍼센트
수수료 ) = 최종 수익률 B : 기본 수익률 0.8 Y = + 0.5 / N= 0.0 M = + 0.3 / N= 0.0 BTC
매수시 1 : 거래대금 배수 조건(2년 최저 대비 N배): 섹터/종목별 허가 배수 미달이면 불
허 = N 허가배수에 들경우 = Y 2: 금가격 플러스인경우 허가 = N / 마이너스 불허가 =
M 3 : Polymarket BTC 확률: 51%/49% 게이트(허가/불허) 허가 = M 불허가 = N
https://polymarket.com/event/bitcoin-above-on-february-24 수치를 계산해서 한다. 허가
불허가 한다. 4 : 재무제표 순이익: +0.1% 이상이면 허가 = M , -0.1% 이하면 불허 = N
5 : S앤P500 POLYMARKET https://polymarket.com/event/which-companies-added-to-sp-
500-in-q1-2026 사이트 링크 Which companies added to S&P 500 in Q1 2026? 1) 제목에
서 확률이 높은 기업 51퍼센트 이상 인경우 2) 최근 재무제표 순이익인경우 3) 매수하는
종목이 같은경우 4) 3가지 모두 충족시 허가 =Y / 1개만 충족 = N A-1 : 기본 수익률 0.8
A-2 시작가 대비 상승 제한(예: 4%+ 상승이면 매수 금지 등): 조건 위반이면 불허 N 시
작가가 3퍼센트인경우 -3퍼센트 시작가가 2퍼센트인경우 -2 퍼센트 1퍼센트인경우 -1퍼
센트 계산해서 최종 수익률 조정한다. A-3 120이동평균선돌파시 + 수익률 50퍼센트 더한
다. / 메수 매도를 당일이 원칙이오나 조건 해당시 40일 동안 스윙 투자 허용 A-4 부동
산 변동 수익률 포지션 실시간 변경 모드 모드별로 목표수익률이 = 기본수익률을 변경
해준다. 레버리지 스탑 해제(60) 3년 최고가 기준 6일 과거데이터로 60일선 아래
0.8~1.0% → 1.8%, 1.8~2.0% → 2.8%-- 2 taejun_attach_pattern — 전략 리뷰 2026-02-23
(VNQ) 모드조건 (VNQ 기준)목표수익률 변경추가 제한 레버리지 스탑 해제(120) 120일선
아래0.8~1.0% → 2.8%, 1.8~2.0% → 4.8%레버리지 GOLD/SHORT 매매 중단 레버리지 스
탑 사용 허가(60) 60일선 위기존 유지 (0.8~1.0% → 0.8%, 1.8~2.0% → 1.8%) 레버리지 스
탑 사용 허가(120) 120일선 위기존 유지레버리지 ETF 즉시 전액 매도(GDXU 제외), 1배수
대체(M7 참조) ALL IN ONE (20, 120) VNQ 20일선 시작가 -1.0% 또는 120일선-3% 하락
상태 기본 5%, GLD 하락 +1%, GLD 상승-0.5% 특수 조건 (아래 별도 정리 BTC 종목 이
종목들은 매수시 이러한 조건을 확인하고 매수한다.
-----------------------------------------
-------------------------------- MSTR MSTU MSTR COIN CONL COIN IREN IRE IREN RIOT
RIOX RIOT CRCL CRCA CRCL BMNR (비트마인 이머션) BMNU BMNR BTCT (BTC 디지털)-
BTCT- 17 taejun_attach_pattern — 전략 리뷰 2026-02-23 (VNQ) 기초자산레버리지 ETF
(1순위) M7 대체 (1배수) CNCK (코인체크 그룹)-CNCK XDGXX (디지 파워)-XDGXX FUFU
(비트푸푸)-FUFU ANT (앤트알파)-ANT ---------------------------------------------------------
----------------------- BTC 종목 이종목들은 매수시 이러한 조건을 확인하고 매수한다. 조
건 : 최종수익률≥0.8% 계산시 이보다 적은 경우 매수를 멈춘다.
---------------------------
----------NASDAQ매수 ---------------------------------------- M4000 매수 나스닥 프로토
콜 / 최종 수익률 계산 매수 ( ( 1 거래대금 점수. ) + + ( 2 금. ) + ( 3 polymarket
NASDAQ 확률. ) + ( 4 재무제표 순이익 ) + ( 5 s앤p500. ) + ( A-1 기본 수익률 + 0.8. ) +
( A-2 시작가 대비. ) + ( A-3 120일선 돌파 ) + ( A-4 M4- 부동산 변동 수익률 포지션 실
시간 변경 모드 ) ) + ( A-5 -0.75퍼센트 수수료 ) = 최종 수익률 B : 기본 수익률 0.8 Y =
+ 0.5 / N= 0.0 M = + 0.3 / N= 0.0 NASDAQ 매수시 1 : 거래대금 배수 조건(2년 최저 대
비 N배): 섹터/종목별 허가 배수 미달이면 불허 = N 허가배수에 들경우 = Y 2: 금가격
플러스인경우 허가 = N / 마이너스 불허가 = M 3 : Polymarket NASDAQ 확률: 51%/49%
게이트(허가/불허) 허가 = M 불허가 = N https://polymarket.com/event/ndx-up-or-down-
on-february-24-2026 나스닥 당일 . 허가 불허가 한다. Nasdaq 100 (NDX) Up or Down
on February 24? 상승확률 하락확률 계산한다. 4 : 재무제표 순이익: +0.1% 이상이면 허가
= M , -0.1% 이하면 불허 = N 5 : S앤P500 POLYMARKET
https://polymarket.com/event/which-companies-added-to-sp-500-in-q1-2026 사이트 링
크 Which companies added to S&P 500 in Q1 2026? 5) 제목에서 확률이 높은 기업 51퍼
센트 이상 인경우 6) 최근 재무제표 순이익인경우 7) 매수하는 종목이 같은경우 8) 3가지
모두 충족시 허가 =Y / 1개만 충족 = N A-1 : 기본 수익률 0.8 A-2 시작가 대비 상승 제
한(예: 4%+ 상승이면 매수 금지 등): 조건 위반이면 불허 N 시작가가 3퍼센트인경우 -3
퍼센트 시작가가 2퍼센트인경우 -2 퍼센트 1퍼센트인경우 -1퍼센트 계산해서 최종 수익
률 조정한다. A-3 120이동평균선돌파시 + 수익률 50퍼센트 더한다. / 메수 매도를 당일이
원칙이오나 조건 해당시 40일 동안 스윙 투자 허용 A-4 부동산 변동 수익률 포지션 실
시간 변경 모드 모드별로 목표수익률이 = 기본수익률을 변경해준다. 레버리지 스탑 해제
(60) 3년 최고가 기준 6일 과거데이터로 60일선 아래 0.8~1.0% → 1.8%, 1.8~2.0% →
2.8%-- 2 taejun_attach_pattern — 전략 리뷰 2026-02-23 (VNQ) 모드조건 (VNQ 기준)목
표수익률 변경추가 제한 레버리지 스탑 해제(120) 120일선 아래0.8~1.0% → 2.8%,
1.8~2.0% → 4.8%레버리지 GOLD/SHORT 매매 중단 레버리지 스탑 사용 허가(60) 60일선
위기존 유지 (0.8~1.0% → 0.8%, 1.8~2.0% → 1.8%) 레버리지 스탑 사용 허가(120) 120일
선 위기존 유지레버리지 ETF 즉시 전액 매도(GDXU 제외), 1배수 대체(M7 참조) ALL IN
ONE (20, 120) VNQ 20일선 시작가 -1.0% 또는 120일선-3% 하락 상태 기본 5%, GLD 하
락 +1%, GLD 상승-0.5% 특수 조건 (아래 별도 정리 NASDAQ 종목 이종목들은 매수시
이러한 조건을 확인하고 매수한다.
------------------------------------------------------------
------------- SOXX NVDA SOXL AMD NVDL AMDL AVGX LINT QCMU ARMG MVLL AVGO
SOXX SOXL SOXX NVDA NVDL NVDA AMD AMDL AMD AVGO AVGX AVGO INTC LINT
INTC TXN-TXN QCOM QCMU QCOM ARM ARMG ARM MRVL MVLL MRVL LLY (일라이릴
리) LLYX LLY JNJ (존슨앤존슨) JNJ JNJ NVO (노보노디스크) NVOX NVO XLE ERX섹터 대표
XOM XOMX엑슨모빌 CVX CVX셰브론 COP COP코노코필립스 SLB SLB슐럼버거 WMB
WMB윌리엄스 HOOD ROBN XLF FAS ---------------------------------------------------------
----------------------- NASDAQ 종목 이종목들은 매수시 이러한 조건을 확인하고 매수한
다. 조건 : 최종수익률≥0.8% 계산시 이보다 적은 경우 매수를 멈춘다.
CONZ/IREZ/HOOZ/MSTZ/ SOXS CONZ/IREZ/HOOZ/MSTZ/ SOXS 정정 -- 숏매수 방식으
로 수정한다. 매도방식은 이러한다. ( 1. 거래대금 점수. ) + ( 2. vix. ) + ( 3. 금. ) +
( polymarket 확률. ) + ( 시작가 대비. ) + ( s앤p500. ) + ( 부동산 m4 나누기 2 또는 더하
기. ) + ( 120일선 돌파. ) + ( 재무제표 마이너스. ) + ( 부동산 리츠 사계절 ) - ( - 0.75퍼센
트 수수료 ) = 최종 매도 CONZ/IREZ/HOOZ/MSTZ/ 숏 매수시 1 : 거래대금 배수 조건
(최근 5일 거래대금 20퍼센트 떨어진경우 미달이면 불허 = N 허가배수에 들경우 = Y 2:
금가격 플러스인경우 허가 = M / 마이너스 불허가 = N 3 : Polymarket NASDAQ 확률:
51%/49% 게이트(허가/불허) 49프로 허가 = M 51프로 불허가 = N
https://polymarket.com/event/ndx-up-or-down-on-february-24-2026 나스닥 당일 . 허가
불허가 한다. SOXS 매수허가 그외에는 매수불가 "44POLYMARKET BTC" - > 삭제하기 키
워드 4 : 재무제표 순이익: +0.1% 이상이면 허가 = N , -0.1% 이하면 불허 = M 5 : S앤
P500 POLYMARKET https://polymarket.com/event/which-companies-added-to-sp-500-in-
q1-2026 사이트 링크 Which companies added to S&P 500 in Q1 2026? 9) 제목에서 확
률이 높은 기업 51퍼센트 이상 인경우 10) 최근 재무제표 순이익인경우 11) 매수하는 종
목이 같은경우 12) 3가지 모두 충족시 허가 =Y / 1개만 충족 = N A-1 : 기본 수익률 0.8
A-2 시작가 대비 상승 제한(예: 4%+ 상승이면 매수 금지 등): 조건 위반이면 불허 N 시
작가가 3퍼센트인경우 -3퍼센트 시작가가 2퍼센트인경우 -2 퍼센트 1퍼센트인경우 -1퍼
센트 계산해서 최종 수익률 조정한다. A-3 120이동평균선하락시 + 수익률 30퍼센트 더한
다. / 메수 매도를 당일이 원칙이오나 조건 해당시 20일 동안 스윙 투자 허용 A-4 부동
산 변동 수익률 포지션 실시간 변경 모드 모드별로 목표수익률이 = 기본수익률을 변경
해준다. 레버리지 스탑 해제(60) 3년 최고가 기준 6일 과거데이터로 60일선 아래
0.8~1.0% → 1.8%, 1.8~2.0% → 2.8%-- 2 taejun_attach_pattern — 전략 리뷰 2026-02-23
(VNQ) 모드조건 (VNQ 기준)목표수익률 변경추가 제한 레버리지 스탑 해제(120) 120일선
아래0.8~1.0% → 2.8%, 1.8~2.0% → 4.8%레버리지 GOLD/SHORT 매매 중단 레버리지 스
탑 사용 허가(60) 60일선 위기존 유지 (0.8~1.0% → 0.8%, 1.8~2.0% → 1.8%) 레버리지 스
탑 사용 허가(120) 120일선 위기존 유지레버리지 ETF 즉시 전액 매도(GDXU 제외), 1배수
대체(M7 참조) ALL IN ONE (20, 120) VNQ 20일선 시작가 -1.0% 또는 120일선-3% 하락
상태 기본 5%, GLD 하락 +1%, GLD 상승-0.5% 특수 조건 (아래 별도 정리 조건 : 최종수
익률≥0.8% 계산시 이보다 적은 경우 매수를 멈춘다.
----------------------------------------
------------------------------------ CONZ/IREZ/HOOZ/MSTZ/ CONZ/IREZ/HOOZ/MSTZ/ 정
정 -- 숏매수 방식으로 수정한다. 매도방식은 이러한다. ( 1. 거래대금 점수. ) + ( 2. vix. )
+ ( 3. 금. ) + ( polymarket 확률. ) + ( 시작가 대비. ) + ( s앤p500. ) + ( 부동산 m4 나누
기 2 또는 더하기. ) + ( 120일선 돌파. ) + ( 재무제표 마이너스. ) + ( 부동산 리츠 사계
절 ) - ( - 0.75퍼센트 수수료 ) = 최종 매도 CONZ/IREZ/HOOZ/MSTZ/ 숏 매수시 1 : 거래
대금 배수 조건(최근 5일 거래대금 20퍼센트 떨어진경우 미달이면 불허 = N 허가배수에
들경우 = Y 2: 금가격 플러스인경우 허가 = M / 마이너스 불허가 = N 3 : Polymarket
BTC 확률: 51%/49% 게이트(허가/불허) 허가 = M 불허가 = N
https://polymarket.com/event/bitcoin-above-on-february-24 "44POLYMARKET BTC" - >
삭제하기 키워드 4 : 재무제표 순이익: +0.1% 이상이면 허가 = N , -0.1% 이하면 불허 =
M 5 : S앤P500 POLYMARKET https://polymarket.com/event/which-companies-added-to-
sp-500-in-q1-2026 사이트 링크 Which companies added to S&P 500 in Q1 2026? 1) 제
목에서 확률이 높은 기업 51퍼센트 이상 인경우 2) 최근 재무제표 순이익인경우 3) 매수
하는 종목이 같은경우 4) 3가지 모두 충족시 허가 =Y / 1개만 충족 = N A-1 : 기본 수익
률 0.8 A-2 시작가 대비 상승 제한(예: 4%+ 상승이면 매수 금지 등): 조건 위반이면 불허
N 시작가가 3퍼센트인경우 -3퍼센트 시작가가 2퍼센트인경우 -2 퍼센트 1퍼센트인경우
-1퍼센트 계산해서 최종 수익률 조정한다. A-3 120이동평균선하락시 + 수익률 30퍼센트
더한다. / 메수 매도를 당일이 원칙이오나 조건 해당시 20일 동안 스윙 투자 허용 A-4
부동산 변동 수익률 포지션 실시간 변경 모드 모드별로 목표수익률이 = 기본수익률을
변경해준다. 레버리지 스탑 해제(60) 3년 최고가 기준 6일 과거데이터로 60일선 아래
0.8~1.0% → 1.8%, 1.8~2.0% → 2.8%-- 2 taejun_attach_pattern — 전략 리뷰 2026-02-23
(VNQ) 모드조건 (VNQ 기준)목표수익률 변경추가 제한 레버리지 스탑 해제(120) 120일선
아래0.8~1.0% → 2.8%, 1.8~2.0% → 4.8%레버리지 GOLD/SHORT 매매 중단 레버리지 스
탑 사용 허가(60) 60일선 위기존 유지 (0.8~1.0% → 0.8%, 1.8~2.0% → 1.8%) 레버리지 스
탑 사용 허가(120) 120일선 위기존 유지레버리지 ETF 즉시 전액 매도(GDXU 제외), 1배수
대체(M7 참조) ALL IN ONE (20, 120) VNQ 20일선 시작가 -1.0% 또는 120일선-3% 하락
상태 기본 5%, GLD 하락 +1%, GLD 상승-0.5% 특수 조건 (아래 별도 정리 조건 : 최종수
익률≥0.8% 계산시 이보다 적은 경우 매수를 멈춘다.
------------------------------숏
CONZ/IREZ/HOOZ/MSTZ 매수한다.
------------------ CONZ/IREZ/HOOZ/MSTZ/ -----------
------------------------------------------------------------------------ 매도는 -- 수익금 분배
총자산이 20,000,000 < 21,000,000 100만원이상부터는 모두 SCHD를 매수한다. SCHD 조
건 달성 된 시간 당일 기준으로 매수한다. SCHD 배당금으로 재투자허용 M200 — 즉시
매도 조건 (0222 신설, MT_VNQ3 파이프라인 확정) 출처: 0222 원문 line 1212~1233 +
MT_VNQ3 §15 조건 1개라도 충족 시 목표수익률 무관하게 즉시 매도: #조건내용 1거래
대금 급감전날 거래대금 대비 장 시작 3시간 이내 15% 하락 2 Polymarket BTC 44% *(원
문 "44POLYMARKET BTC" — 모호, P-5 등록)* 3 GLD 급등GLD 6%+ 상승 4 VIX 급등VIX
10%+ 급등 5이평선 이탈모든 주식 20일 이동평균선 이탈 6기한 만기기한 만기 시 다음
날 17:30~06:00 이내 매도 7 REIT_MIX 급등REIT_MIX 5%+ 상승 시 VNQ 관련 즉시 정리
/ 그 외 30% 축소 (GLD 숏 제외) 발동 파이프라인 *(MT_VNQ3 §15 확정)*: 전역 락 ON
→ OPEN 주문 전부 취소 → 취소 ACK → reserved 재계산 → 청산 실행(매도 즉시성 규
정 적용) → 락 OFF [!] 모호사항 (P-5): M200 조건 2번 "44POLYMARKET BTC"의 44가
44% 기준인지 불명확. 정의 확정 전까지 OFF. SCHD 예외: 어떤 모드에서도 SCHD 매도
불가 (단, MASTER H/J가 AI OFF 시만 허용 여기수치를 계산해서
https://polymarket.com/event/bitcoin-above-on-february-24 3 : Polymarket BTC 확률:
51%/49% 게이트(허가/불허) 허가 = M 불허가 = N
https://polymarket.com/event/bitcoin-above-on-february-24 수치를 계산해서 한다. 허가
불허가 한다. "44POLYMARKET BTC" - > 3 : Polymarket BTC 확률: 51%/49% 게이트(허가/
불허) 허가 = M 불허가 = N https://polymarket.com/event/bitcoin-above-on-february-24
수치를 계산해서 한다. 허가 불허가 한다. 로바꿔주고 "44POLYMARKET BTC"이거에대해
서는 모든걸 - > > 3 : Polymarket BTC 확률: 51%/49% 게이트(허가/불허) 허가 = M 불허
가 = N https://polymarket.com/event/bitcoin-above-on-february-24 수치를 계산해서 한
다. 허가 불허가 한다. 로바꿔주고 ab_etq (구 SETH) 1.05% 0.74% 1.79% 종목 ETQ로 변
경 삭제해줘 Q-2 SETH 진입 조건 해석ETQ로 종목 변경, Polymarket 하락 기대 평균 12
미만 시 작동반영 삭제해줘 2026.2.21 SETH (숏 ETF) ETQ로 종목 변경MT_VNQ Q-2 삭제
해줘 D. 숏 잽모드 ETQ (구 SETH -> ETQ로 정정) !! 종목 정정: SETH -> ETQ 항목내용
매수 대상ETQ *(정정: SETH -> ETQ)* 목표 수익률+1.05% (net, 구 0.8% + 0.25%) 매도 방
식전액 매도 진입 조건 (Q-2 반영): #지표조건 1 Polymarket ETH 하락 기대평균 수치보
다 12 미만 시 작동 2 GLD >= +0.01% 3 ETQ >= 0.00% (양전) [!] 모호사항 (Q-2): "평균
수치보다 12 미만"에서 평균 수치의 정의 불명확. 어떤 기간의 평균인지(1일? 7일? 30
일?), 어떤 Polymarket 마켓의 수치인지 확인 필요. 0222 원문에서도 명시적 답변 없어
[!] 유지. 삭제해줘 P-9 M80 점수 배점 정의가 부족 최소 35점 기준인데 거래대금 점수
(최대 18점)만으로는 35점이 안 나옴 → GLD/Polymarket 점수 배점 정의 필요. 점수화
해서 매수 매도 정정함 taejun_strategy_review_2026-02-
… 거래대금 점수 삭제해줘 점수
들은 그대로 두고 거래대금 시스템 규칙은 그대로 두고 P-10 M200 매도 직후 이머전시
재매수 충돌(쿨다운 미정) 그즉시하기 A: 그즉시하기 P-12 M200의 “즉시매도”인데 조건6
만 다음날 시간대 매도(타이밍 충돌) 조건달성시 매도 조건 6만 다음날 하기 개선사항에
대한 답이야 또 어떤거 고쳐
taejun_attach_pattern 자동매매 구현 규격 v0.5 (VNQ 기반, MT_VNQ4 통합/정정본)
•	작성일: 2026-02-24
•	적용 엔진: strategies/taejun_attach_pattern/
•	목표: 룰 충돌 0, 시장가 0, 자본 초과 0, 결측 시 거래 0(Fail-Closed)
•	원칙: “모호하면 매수하지 않는다(BUY_STOP), 청산은 최대한 수행한다”
 
1) 핵심 설계 철학 (Fail-Closed)
1.	룰/데이터/상태가 불명확하면 신규 진입 금지(BUY_STOP)
2.	시장가 주문은 코드에서 제거(LIMIT만)
3.	포지션/평단은 체결(FILLED/filled_qty>0) 이후에만 갱신
4.	동시 BUY는 반드시 “일괄 집계→정렬→배분”(순차 실행 금지)
5.	청산(M200/M201)은 최대한 실행(킬스위치 우선)
 
2) 용어/기준 정의
•	session_day: KST 17:30에 날짜가 바뀌는 거래 기준일(자정 기준 금지)
•	bar: 전략이 사용하는 최소 시간 단위(분봉/틱 등)
•	reference_price: “신호가 True가 된 bar의 종가(close)” (즉시행동 정의)
•	n: 같은 bar에서 PASS(진입허가) 된 종목 개수(전략 여러 개가 같은 종목 PASS면 1개로 카운트)
•	MMM(필수 3게이트): 거래대금(§1) + 금(GLD)(§2) + Polymarket(§3) 3개 모두 PASS해야 진입 가능
•	Y/M/N 점수: 정렬용 점수(수익률에 합산 금지)
o	Y=+0.5, M=+0.3, N=0
 
3) CI-0 필수 안전 규격(요약)
CI-0-1 데이터 안전
•	가격/지표가 None/0/NaN 이거나 **스테일(기본 120초 초과)**이면 → 해당 ticker BUY_STOP + 알람
•	VNQ/REIT_MIX/Polymarket 등 핵심 지표가 불명확하면 → 관련 평가 OFF + 알람
CI-0-2 주문/체결
•	LimitOrder만 허용, Market 주문 타입 제거
•	Position/avg_price는 filled_qty>0에서만 갱신
CI-0-3 주문 상태머신
•	주문 TTL: 120초
•	재시도: 최대 3회(단, 아래 “추격매수 금지”와 충돌 시 BUY 재시도는 가격상향 금지)
•	reserved_cash: 주문 생성 시 즉시 예약, cancel/expire/reject 시 반환
•	Idempotency Key: (strategy_id, ticker, side, signal_bar_ts, rank) 고정
CI-0-4 “즉시 행동”
•	“즉시” = 조건이 True가 된 bar 종료 시점에 주문 제출
•	기준가 = 그 bar의 close(reference_price)
CI-0-5 전역 우선순위(고정)
M200(즉시매도) > M201(즉시전환) > 리스크/이머전시 > 거래시간/휴장 > M28 게이트 > MMM > M5 배분 > 일반전략
 
4) 마스터 플랜 요약(정정 포함)
M0 시스템 점검
•	네트워크/서버/RAM/CPU/GPU 점검 → 이상 시 알람
•	KST 공휴일 18:00 재시작은 MASTER J/H 사전허가 필수
M1 LIMIT-ONLY + 호가 정책(충돌 제거)
•	시장가 금지
•	추격매수 금지를 최상위로 둔다:
o	BUY 지정가는 절대 reference_price 초과 금지
o	BUY limit = floor_to_tick(reference_price)
o	BUY “체결성 개선 목적 가격 상향 재시도” 금지(= 실패 처리)
•	SELL(청산/리스크)만 예외:
o	urgent SELL은 bid 기준 0.2% 이내 marketable limit 허용
o	SELL limit = floor_to_tick(bid * (1 - 0.002))
•	TickSizeResolver로 tick 단위 통일(“반올림 0.01” 같은 문구는 제거하고 함수로 일원화)
M2 즉시 행동
•	CI-0-4와 동일(신호 bar close 기준)
M3 거래시간/캘린더
•	거래 가능 시간(세션): KST 17:30 ~ 06:00 (월~금, 휴장 제외)
•	정규장 참고(DST):
o	DST: 22:30~05:00 KST
o	표준: 23:30~06:00 KST
•	구현: 시간창 안이더라도 데이터 스테일이면 CI-0-1로 자동 차단
M20 신용/미수 금지 + 목표수익률 가산
•	신용/미수 절대 금지
•	목표수익률은 **“베이스(target_base_net)”에 +0.25%**를 파이프라인에서 가산 (중복 가산 금지)
M300 USD-only + VNQ/KR리츠 signal-only
•	FX(환전) 수행 금지
•	VNQ/KR 리츠는 지표용(signal-only): 포지션 생성/주문 금지
 
5) Polymarket 게이트: M28 (유일한 ‘포지션 허가 게이트’)
M28 출력
•	BTC_UP_PROB(=p), NASDAQ_UP_PROB(=p_ndx)
•	업데이트 실패/스테일 → 관련 매매 BUY_STOP + 알람
게이트 규칙(공통)
•	LONG 허가: p >= 0.51
•	SHORT 허가: p <= 0.49
•	중립(둘다 불허): 0.49 < p < 0.51
정정: M200(즉시매도)에서 Polymarket 조건은 완전 삭제. Polymarket은 M28과 “전략 진입조건”에서만 사용.
 
6) 즉시 매도 킬스위치: M200 (정정 완료)
조건 1개라도 True면 목표수익률 무관 청산
1.	거래대금 급감: 장 시작 3시간 이내 전날 대비 -15%
2.	GLD 급등: +6%
3.	VIX 급등: +10%
4.	20MA 이탈
5.	REIT_MIX 급등: +5%
6.	기한 만기(예외): “즉시 매도”가 아니라 다음 거래세션(17:30~06:00) 중 1회 전액 매도
7.	(VNQ 관련 정리/축소는 REIT_MIX 규칙으로 통합)
파이프라인
전역락 ON → OPEN 주문 전부 취소 → 취소 ACK → reserved 재계산 → urgent SELL 청산 → 락 OFF
재진입 금지
•	M200로 매도한 동일 bar에서는 신규 매수 금지
•	다음 bar부터 가능(단, M200 조건 하나라도 True면 재매수 금지)
 
7) 즉시 전환: M201 (BTC 확률 급변)
입력: p, p_prev, Δpp
•	LONG 보유 중:
o	p <= 0.45 → 즉시 청산
o	Δpp <= -20pp and p <= 0.49 → 즉시 청산
•	SHORT 보유 중:
o	p >= 0.60 → 즉시 청산
청산 후 전환(기본값 확정)
•	롱 청산 후 p <= 0.49 → 숏 진입(가용현금 전액, 단 99% cap)
•	숏 청산 후 p >= 0.51 → 롱 진입(가용현금 전액, 단 99% cap)
•	중립 구간은 현금 유지
 
8) REIT_MIX 모드: M4 (결측 처리 확정)
REIT_MIX 정의
•	REIT_MIX = (VNQ + KR 리츠들) 전일 대비 변화율 평균
•	결측 처리(확정, P-16 해결):
o	사용 가능한 구성요소 수가 2개 미만이면 UNKNOWN → BUY_STOP + 알람
o	2개 이상이면 “가용값 평균”으로 계산
M4가 하는 일
•	리스크 상태에 따라 목표수익률/레버리지 허용 여부를 조정
•	단, M200이 최우선(M4보다 우선)
 
9) 비중 배분: M5 (99/50/33 체계 + 정렬 규칙 확정)
9-1) MMM가 먼저다
•	MMM(거래대금+금+Polymarket) 3개 모두 PASS한 종목만 n에 포함
9-2) 우선순위 정렬(확정)
•	정렬키: M_PASS 개수 > Y_PASS 개수 > target_net 높은 순
•	(점수는 %에 합산 금지)
9-3) T1~T4 “절대 비중”
•	n=1: T1=99%
•	n=2: 50% / 50%
•	n=3: 33% / 33% / 33%
•	n≥4: 33% / 33% / 30% / 나머지(4%)
•	5번째 이상은 예약 표시만(주문 없음), 10초 후 조건 미충족이면 예약 취소
9-4) 현금 부족
•	“살 수 있는 수량만” 매수
•	현금 0이면 즉시 중단
9-5) 동적 비중 조정(동시 발생 합산)
•	GLD +1%마다 weight −0.1%p, GLD −1%마다 +0.1%p
•	VIX +2%마다 −3%p, VIX −2%마다 +1%p
•	USD 상승 +0.1%p / 하락 −0.2%p
•	동시 발생 시 전부 합산
•	최종 비중 제한:
o	최종 비중이 0 이하면 해당 신호 DROP
o	합계가 100% 초과하면 비례 shrink → 그래도 초과 시 하위 우선순위 drop
 
10) 목표수익률 파이프라인: M6 (중복 가산 방지)
target_base_net(전략 기본값, M20 미포함)을 입력으로 한다.
1.	최소 하한 적용: target_base_net < 0.8% → 0.8%로 올림
2.	M20 가산: +0.25%
3.	리츠 조심모드 감산(발동 시): target_net *= 0.5
4.	감산 후 하한 재확인:
o	target_net <= 0.8%이면 전 모드 BUY_STOP
리츠 감산 제외(고정):
•	GLD, VIX, GDXU
•	CONZ/IREZ/HOOZ/MSTZ, SOXS
수수료 처리(정정):
•	roundtrip_fee_pct = 0.74% 고정
•	exit 판정은 target_gross = target_net + 0.74%
 
11) M7 레버리지 금지 시 대체 (정정)
•	SOXL→SOXX, ROBN→HOOD, FAS→XLF, TQQQ→QQQ, Cure→XLV, Ertlx→XKE
•	SOLT/ETHU 금지, QUBX→QTUM
•	NVDL→NVDA (대체 허용)
•	BITU는 실매매 금지(신호용)
정정(치명 충돌 해결): BITU가 매매금지이므로 jab_bitu 전략은 기본 OFF(원하면 M7 정책을 바꿔야 함).
 
12) M4000 통합 “진입 프로토콜” (정정본: 모순 제거)
기존 “최종수익률 = 점수합” 방식은 단위 혼용/수수료 혼용/거래대금 중복 문제가 있어 v0.5에서 **‘게이트 + 타겟조정’**으로 재정의한다.
12-1) 분류(Category)
•	BTC: COIN/MSTR/IREN/RIOT 등 매핑 기반
•	NASDAQ: SOXX/NVDA/AMD… 매핑 기반
•	SHORT: CONZ/IREZ/HOOZ/MSTZ/SOXS
12-2) MMM(필수 3게이트)
§1 거래대금 게이트
•	종목별 “2년 최저 대비 N배” 충족 시 PASS(Y), 미달 FAIL(N)
§2 금(GLD) 게이트 (P-18 확정)
•	BTC/NASDAQ(롱 성격):
o	GLD가 **+**이면 FAIL(N)
o	GLD가 **0 또는 −**이면 PASS(M)
•	SHORT(헤지 성격):
o	GLD가 **+**이면 PASS(M)
o	GLD가 **0 또는 −**이면 FAIL(N)
§3 Polymarket 게이트
•	M28 결과로 LONG/SHORT/NEUTRAL 판정
•	NEUTRAL이면 FAIL(N)
MMM 3개 모두 PASS가 아니면 그 종목은 n 계산에서 제외
12-3) 추가 게이트(권장 필수)
§4 재무 게이트(확정)
•	LONG(BTC/NASDAQ): 기초자산 최근 분기 순이익 양수면 PASS(M), 음수/불명확 FAIL(N)
•	SHORT: 기초자산 최근 분기 순이익 음수면 PASS(M), 양수/불명확 FAIL(N)
§5 S&P500 이벤트 게이트(선택)
•	해당 전략이 “편입 이벤트 전략(F)”일 때만 필수로 적용
•	그 외 전략에서는 “가산/참고(Y 또는 N)”로만 사용(데이터 없으면 N)
12-4) 타겟 조정(단위 확정)
•	A-2 시작가 대비 과열:
o	start_delta >= 4% → 진입 FAIL
o	1~3% 구간 → target_net -= 0.1%p * floor(start_delta%) (예: 3%면 −0.3%p)
•	A-3 120일선 조건(단위 확정: 배수)
o	LONG: target_net *= 1.5 (스윙 최대 40일 허용)
o	SHORT: target_net *= 1.3 (스윙 최대 20일 허용)
정정: 수수료(0.74%)는 target_net에서 “빼는 항목”이 아니라, exit 판정 시 +0.74%로 처리한다.
 
13) M80 섹터 스캐너 + 점수체계 분리(정정)
13-1) 용도
•	M80은 “무엇을 살지 후보를 찾는 스캐너”
•	실제 진입 허가는 **M4000(MMM + 재무 + M5 + M6)**가 최종 결정
13-2) 점수체계 이름 정정(중복 제거)
•	기존 문서의 “M200 매수 우선순위 점수”는 **M210(매수우선순위 점수)**로 변경한다.
•	M200은 즉시매도 전용이다.
13-3) M210 점수 사용 범위(확정)
•	오직 정렬/우선순위에만 사용(수익률에 합산 금지)
•	최소 점수 기준(보수적 기본값): 35점
o	35점 미만이면 해당 후보는 “대기/스킵”
 
14) 전략 리스트(기본 ON/OFF 정리)
•	A jab_soxl: ON
•	B jab_bitu: OFF(기본) — BITU 매매금지 정책 때문
•	C jab_tsll: ON(단, 2,000,000원 notional cap)
•	E vix_gold: ON
•	F sp500_entry: ON
•	G bargain_buy: ON
•	H short_macro: ON
•	I reit_risk: ON(signal-only)
•	J sector_rotate: 옵션(권장: M80이 켜져 있으면 J는 대체/비활성화)
•	K bank_conditional: ON
•	L short_fundamental: ON
•	M disaster_emergency: ON(단, Polymarket 급변 감지는 M28 데이터 신선도 필수)
 
15) 수익금 분배 + SCHD 룰(P-1 기본값 확정)
15-1) 분배 실행(고정)
•	실행 시점: 장 마감 후 1회(예: 06:05 KST)
•	pnl <= 0이면 분배 스킵
15-2) 기본 분배 규칙(확정)
•	순서: SOXL → ROBN → GLD → CONL
•	각 1주씩 “순차”로 매수
•	현금이 부족해서 다음 1주를 못 사면 즉시 중단(부분주 금지)
15-3) SCHD 특례(정정)
•	총자산이 20,000,000~21,000,000원 범위이고
•	당일 수익금이 1,000,000원 이상이면
•	수익금 전액을 SCHD 매수(당일 기준), 배당 재투자 허용
•	SCHD는 어떤 모드에서도 매도 금지(“AI 매매 OFF” 시에만 예외)
 
16) 남아있던 모호 항목(P-xx) 기본값(안전) 확정
•	P-11 PARTIAL fill 시 M5 차감: 차감 금지, FILLED에서만 1회 차감
•	P-13 M4 vs M200 충돌: M200 우선, M4는 “타겟/레버리지 정책”만
•	P-15 M201 전환 규모: “가용현금 전액” 단, 99% cap(+100.1% 방지)
•	P-17 PARTIAL_ENTRY_DONE 재진입: 동일 signal_bar_ts 재진입 금지, 다음 bar의 새 신호만 허용
•	P-2 리츠 과열 시 -0.5% 범위: 신규 진입분에만 적용, 기존 포지션 소급 적용 금지(안전)
 
17) 구현 전 필수 테스트 체크리스트(이거 통과=실전 가능)
•	시장가 주문 0건
•	cash 음수 0건
•	동시 BUY 합계 100% 초과 0건(초과 시 shrink/drop 정상 작동)
•	같은 bar 재진입 0건(M200/FillWindow)
•	스테일/결측 데이터에서 신규 진입 0건(BUY_STOP)
•	pending/partial에서 avg_price 확정 사용 0건
•	TTL 만료/취소 ACK 후 reserved_cash 100% 반환


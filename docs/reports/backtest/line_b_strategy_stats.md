# Line B 전략별 시그널 통계 리포트

- **평가 기간**: 2025-01-03 ~ 2026-01-30 (269 거래일)
- **평가 전략 수**: 19 / 19
- **데이터 소스**: `data/market/ohlcv/backtest_1min_v2.parquet` (일봉 변환)
- **Polymarket**: `data/polymarket/` JSON (btc_up, ndx_up)
- **생성일**: 2026-02-23 20:56

## 전략 분류

- **잽모드 단타 (Jab)**: jab_soxl, jab_bitu, jab_tsll, jab_etq
- **이벤트 드리븐**: vix_gold, emergency_mode, crash_buy
- **매크로/방어**: short_macro, reit_risk, bear_regime_long, bearish_defense
- **조건부 매매**: bank_conditional, sp500_entry, conditional_coin, conditional_conl, twin_pair
- **장기/로테이션**: bargain_buy, sector_rotate, soxl_independent

## 전략별 시그널 통계

| 전략 | BUY | SELL | HOLD | SKIP | ERROR | 진입률 | 신호/일 | 비고 |
|------|----:|-----:|-----:|-----:|------:|-------:|--------:|------|
| twin_pair | 158 | 101 | 0 | 10 | 0 | 58.7% | 0.963 |  |
| conditional_coin | 46 | 0 | 0 | 223 | 0 | 17.1% | 0.171 |  |
| conditional_conl | 41 | 0 | 0 | 228 | 0 | 15.2% | 0.152 |  |
| bank_conditional | 0 | 0 | 0 | 269 | 0 | 0.0% | 0.000 |  |
| bargain_buy | 0 | 0 | 0 | 269 | 0 | 0.0% | 0.000 |  |
| bear_regime_long | 0 | 0 | 0 | 269 | 0 | 0.0% | 0.000 |  |
| bearish_defense | 0 | 0 | 0 | 269 | 0 | 0.0% | 0.000 |  |
| crash_buy | 0 | 0 | 0 | 269 | 0 | 0.0% | 0.000 |  |
| emergency_mode | 0 | 0 | 0 | 269 | 0 | 0.0% | 0.000 |  |
| jab_bitu | 0 | 0 | 0 | 269 | 0 | 0.0% | 0.000 |  |
| jab_etq | 0 | 0 | 0 | 269 | 0 | 0.0% | 0.000 |  |
| jab_soxl | 0 | 0 | 0 | 269 | 0 | 0.0% | 0.000 |  |
| jab_tsll | 0 | 0 | 0 | 269 | 0 | 0.0% | 0.000 |  |
| reit_risk | 0 | 0 | 0 | 269 | 0 | 0.0% | 0.000 |  |
| sector_rotate | 0 | 0 | 0 | 269 | 0 | 0.0% | 0.000 |  |
| short_macro | 0 | 0 | 0 | 269 | 0 | 0.0% | 0.000 |  |
| soxl_independent | 0 | 0 | 0 | 269 | 0 | 0.0% | 0.000 |  |
| sp500_entry | 0 | 0 | 0 | 269 | 0 | 0.0% | 0.000 |  |
| vix_gold | 0 | 0 | 0 | 269 | 0 | 0.0% | 0.000 |  |

## 활동도 순위 (BUY 시그널 기준)

1. **twin_pair** — BUY 158회 (58.7%일)
2. **conditional_coin** — BUY 46회 (17.1%일)
3. **conditional_conl** — BUY 41회 (15.2%일)
4. **bank_conditional** — BUY 0회 (0.0%일)
5. **bargain_buy** — BUY 0회 (0.0%일)
6. **bear_regime_long** — BUY 0회 (0.0%일)
7. **bearish_defense** — BUY 0회 (0.0%일)
8. **crash_buy** — BUY 0회 (0.0%일)
9. **emergency_mode** — BUY 0회 (0.0%일)
10. **jab_bitu** — BUY 0회 (0.0%일)

## 진입 조건 통과율 (check_entry)

| 전략 | 검사일 | 통과 | 통과율 |
|------|-------:|-----:|-------:|
| twin_pair | 269 | 158 | 58.7% |
| conditional_coin | 269 | 46 | 17.1% |
| conditional_conl | 269 | 41 | 15.2% |
| bank_conditional | 269 | 0 | 0.0% |
| bargain_buy | 269 | 0 | 0.0% |
| bear_regime_long | 269 | 0 | 0.0% |
| bearish_defense | 269 | 0 | 0.0% |
| crash_buy | 269 | 0 | 0.0% |
| emergency_mode | 269 | 0 | 0.0% |
| jab_bitu | 269 | 0 | 0.0% |
| jab_etq | 269 | 0 | 0.0% |
| jab_soxl | 269 | 0 | 0.0% |
| jab_tsll | 269 | 0 | 0.0% |
| reit_risk | 269 | 0 | 0.0% |
| sector_rotate | 269 | 0 | 0.0% |
| short_macro | 269 | 0 | 0.0% |
| soxl_independent | 269 | 0 | 0.0% |
| sp500_entry | 269 | 0 | 0.0% |
| vix_gold | 269 | 0 | 0.0% |

## 비활성 전략 (BUY/SELL 시그널 0회) — 원인 분석

| 전략 | 미활성 원인 | 필요 데이터 | 백테스트 데이터 보유 여부 |
|------|-----------|-----------|:---:|
| jab_soxl | 개별 반도체 11종목(NVDA,AMD,SMCI...) 변동률 ALL 충족 필요 | SOXX, 개별 반도체 11종목 | 미보유 |
| jab_bitu | 크립토 스팟 변동률(BTC,ETH,SOL,XRP) 필요 | crypto 스팟 데이터 | 미보유 |
| jab_tsll | TSLA, TSLL 변동률 필요 | TSLA, TSLL | 미보유 |
| jab_etq | ETQ 변동률 + Polymarket 하락기대 스프레드 필요 | ETQ 가격 | 미보유 |
| vix_gold | VIX 일간 +10% 급등 감지 필요 | VIX 가격 | 미보유 |
| crash_buy | -30% 급락 + 15:55 ET 시간 조건 | 시간=17:35 KST (미매칭) | 조건부 |
| emergency_mode | poly_prev(이전 확률) 필요 | Polymarket 이전 시점 확률 | 미제공 |
| short_macro | 나스닥/S&P500 ATH(history._macro) 필요 | history 메타데이터 | 미보유 |
| reit_risk | VNQ+KR리츠 7일 수익률(history) 필요 | history 메타데이터 | 미보유 |
| bear_regime_long | btc_up < 0.43 + btc_monthly_dip > 0.30 필요 | btc_monthly_dip | 미보유 |
| bargain_buy | 3년 최고가 대비 하락률(history.high_3y) 필요 | history 메타데이터 | 미보유 |
| sector_rotate | 1Y 저가 대비 상승률(history.low_1y) 필요 | history 메타데이터 | 미보유 |
| soxl_independent | SOXX +2% 조건 필요 | SOXX 가격 | 미보유 |
| sp500_entry | GLD 상승시 매수 금지 (gld_block_positive=True) → 전 기간 GLD > 0 | Polymarket ndx_up | 조건부 |
| bank_conditional | JPM/HSBC/WFC/RBC/C 모두 양전 + BAC 음전 필요 | 은행 5종목 가격 | 미보유 |
| bearish_defense | mode="bearish" 외부 입력 필요 (기본 "normal") | 외부 모드 판정 | 해당없음 |

## 데이터 제약 사항

1. **시장 데이터 한계**: backtest_1min_v2.parquet에 포함된 16개 종목만 평가 가능
   - 포함: BABX, BITU, BRKU, COIN, CONL, ETHU, GLD, IRE, MSTU, NVDL, QQQ, ROBN, SOLT, SOXL, SPY, XXRP
   - 미포함: SOXX, TSLA, TSLL, VIX, GDXU, IAU, BAC, JPM, HSBC, WFC, RBC, C, ETQ, SOXS, BITI 등
2. **Polymarket 데이터**: btc_up, ndx_up만 추출 (btc_monthly_dip 등 미포함)
3. **history 데이터 없음**: 3년 최고가, 1Y 저가, VNQ 120일선 등 미제공 → bargain_buy, sector_rotate 등 제한
4. **crypto 스팟 데이터 없음**: jab_bitu의 BTC/ETH/SOL/XRP 스팟 변동률 미제공
5. **OHLCV 히스토리 없음**: ADX/EMA 기반 필터(conditional_conl, soxl_independent) 제한적
6. **evaluate()-only 전략**: conditional_coin, conditional_conl, twin_pair, bearish_defense는 evaluate() 직접 호출로 평가

## 해석 및 참고사항

- **잽모드 전략 (jab_soxl/bitu/tsll/etq)**: 프리마켓 17:30 KST 이후, Polymarket + 개별종목 조건 ALL 충족 필요.
  실제 데이터에서는 개별 반도체 11종목(NVDA, AMD 등)이나 크립토 스팟이 없어 조건 미충족이 대부분.
- **bargain_buy**: 3년 최고가 대비 하락률 데이터(history) 필요. 미제공시 항상 SKIP.
- **sector_rotate**: 1Y 저가 대비 상승률(history) 필요. 미제공시 항상 SKIP.
- **vix_gold**: VIX 변동률 데이터 필요. backtest 데이터에 VIX 미포함 → SKIP.
- **short_macro**: 나스닥/S&P500 ATH 데이터(history._macro) 필요. 미제공 → SKIP.
- **crash_buy**: -30% 급락 + 15:55 ET 시간 필요. 시간 조건이 17:35 KST로 설정되어 미매칭.
- **emergency_mode**: poly_prev(이전 확률) 필요. 미제공 → SKIP.
- **bear_regime_long**: btc_up < 0.43 + btc_monthly_dip > 0.30 필요. dip 미제공.
- **twin_pair**: 갭 분석으로 실제 ENTRY/SELL 빈도 측정 가능.
- **conditional_coin/conl**: 트리거 종목(ETHU/XXRP/SOLT) 변동률 기반 직접 평가 가능.

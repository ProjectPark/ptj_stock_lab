# Line B 데이터 확대 리포트
생성일: 2026-02-23 23:25

## 현재 데이터 현황

### 1분봉 OHLCV (backtest_1min_v2.parquet)
- 종목 수: 16개
- 날짜 범위: 2025-01-03 ~ 2026-01-30
- 포함 종목: BABX, BITU, BRKU, COIN, CONL, ETHU, GLD, IRE, MSTU, NVDL, QQQ, ROBN, SOLT, SOXL, SPY, XXRP
- SOXX 별도 파일: 있음

### 일봉 3년 (strategy_daily_3y.parquet)
- 종목 수: 28개
- 날짜 범위: 2023-02-21 ~ 2026-02-18
- 포함 종목: AMAT, AMD, ASML, AVGO, BAC, C, GDXU, HSBC, IAU, JPM, KLA, LRCX, MPWR, MU, NFXL, NVDA, OKLL, PLTU, RBC, SETH, SMCI, SNXX, SOXL, SOXX, TSLA, TSLL, TXN, WFC

## Group A: 1분봉 OHLCV 추가 수집 결과

| 종목 | 필요 전략 | 상태 | 비고 |
|------|----------|------|------|
| SOXX | jab_soxl, sector_rotate | 있음 (1분봉) |  |
| TSLA | jab_tsll | 수집완료 (110,364 rows) |  |
| TSLL | jab_tsll | 수집완료 (110,357 rows) |  |
| UVXY | vix_gold (VIX 대리) | 수집완료 (110,001 rows) |  |
| IAU | vix_gold | 수집완료 (109,802 rows) |  |
| GDXU | vix_gold, short_macro | 수집완료 (87,773 rows) |  |
| ETQ | jab_etq | 수집완료 (18,514 rows) | ETP — Alpaca 미지원, 대안: yfinance 확인 필요 |
| BAC | bank_conditional | 수집완료 (109,924 rows) |  |
| JPM | bank_conditional | 수집완료 (109,936 rows) |  |
| HSBC | bank_conditional | 수집완료 (105,555 rows) | NYSE ADR — Alpaca 지원 가능 |
| WFC | bank_conditional | 수집완료 (109,873 rows) |  |
| C | bank_conditional | 수집완료 (109,909 rows) |  |
| RBC | sector_rotate | 수집완료 (39,589 rows) | Alpaca 지원 가능 |
| VNQ | reit_risk | 수집완료 (109,745 rows) |  |
| SOXS | bear_regime | 수집완료 (110,320 rows) |  |
| BITI | bear_regime | 수집완료 (99,529 rows) |  |
| CONZ | bear_regime (인버스) | 미수집 | Alpaca 미지원 (non-US ETP/역방향) |
| IREZ | bear_regime (인버스) | 미수집 | Alpaca 미지원 (non-US ETP/역방향) |
| HOOZ | bear_regime (인버스) | 미수집 | Alpaca 미지원 (non-US ETP/역방향) |
| MSTZ | bear_regime (인버스) | 미수집 | Alpaca 미지원 (non-US ETP/역방향) |

## Group B: 히스토리 메타데이터

- 파일: `data/meta/history_metadata.json`
- 상태: 수집완료

| 용도 | 대상 종목 | 데이터 항목 |
|------|----------|------------|
| bargain_buy 진입 기준 | CONL, SOXL, AMDL, NVDL, ROBN, ETHU, BRKU | 3년 최고가 (high_3y) |
| sector_rotate 활성화 | BITU, SOXX, ROBN, GLD | 1년 저가 (low_1y) |
| short_macro 발동 | QQQ, SPY | ATH 근사값 (ath_approx) |
| reit_risk 감지 | VNQ | 120일 이동평균 (ma120) |

### 계산된 메타데이터 요약

| 종목 | high_3y | low_1y | ath_approx | 최근 종가 |
|------|---------|--------|------------|----------|
| AMDL | 25.51 | 2.9 | 25.51 | 12.38 |
| BITU | 49.3192 | 10.99 | 49.3192 | 12.15 |
| BRKU | 31.6767 | 21.8176 | 31.6767 | 23.958 |
| CONL | 85.17 | 5.15 | 85.17 | 7.4 |
| ETHU | 315.3901 | 19.3414 | 315.3901 | 21.18 |
| GLD | 495.9 | 263.27 | 495.9 | 468.62 |
| NVDL | 112.99 | 27.66 | 112.99 | 88.48 |
| QQQ | 634.952 | 414.5779 | 634.952 | 608.81 |
| ROBN | 115.9466 | 7.8523 | 115.9466 | 22.03 |
| SOXL | 70.47 | 8.2216 | 70.47 | 67.11 |
| SOXX | 361.13 | 153.9716 | 361.13 | 359.43 |
| SPY | 695.49 | 492.1936 | 695.49 | 689.43 |
| VNQ | 95.49 | 77.5585 | 95.49 | 94.88 |

## Group C: 크립토 스팟 데이터

- 파일: `data/market/daily/crypto_daily.parquet`
- 상태: unknown
- 필요 전략: jab_bitu (BTC, ETH, SOL, XRP 당일 변동률 조건)

| 심볼 | 설명 | 소스 |
|------|------|------|
| BTC-USD | Bitcoin | yfinance |
| ETH-USD | Ethereum | yfinance |
| SOL-USD | Solana | yfinance |
| XRP-USD | Ripple | yfinance |

## Group D: Polymarket 확장

| 지표 | 필요 전략 | 현재 상태 | 비고 |
|------|----------|----------|------|
| btc_monthly_dip | bear_regime_long | poly_config.py에 btc_monthly 존재 | Dip 컴포넌트 파싱 추가 필요 |

## 남은 작업 (수동/설정 필요)

### 즉시 가능
1. `python experiments/expand_line_b_data.py --fetch-1min` — Alpaca로 Group A 수집
2. `python experiments/expand_line_b_data.py --fetch-meta` — yfinance로 Group B 메타데이터
3. `python experiments/expand_line_b_data.py --fetch-crypto` — yfinance로 Group C 크립토

### Alpaca 미지원 (대안 필요)
- **ETQ**: GraniteShares ETP — Alpha Vantage, 직접 API 또는 수동 수집 필요
- **HSBC/RBC**: Alpaca 지원 여부 실행 시 자동 확인
- **CONZ/IREZ/HOOZ/MSTZ**: 역방향 인버스 ETP — 대부분 비US 상장으로 미지원

### Polygon API (미설정)
- `.env`에 `POLYGON_API_KEY` 없음
- VIX 지수 직접 조회 시 필요 (현재 UVXY로 대체 가능)

### Polymarket btc_monthly_dip
- `polymarket/poly_config.py`의 `btc_monthly` 지표 활용 가능
- `poly_history.py`에서 Dip 마켓 컴포넌트만 추출하면 됨
- 별도 API 키 불필요

## 권장 실행 순서

```bash
# 1. 전체 dry-run으로 계획 확인
pyenv shell ptj_stock_lab && python experiments/expand_line_b_data.py --dry-run

# 2. 메타데이터 먼저 수집 (API 부하 없음)
pyenv shell ptj_stock_lab && python experiments/expand_line_b_data.py --fetch-meta

# 3. 크립토 일봉 수집
pyenv shell ptj_stock_lab && python experiments/expand_line_b_data.py --fetch-crypto

# 4. Alpaca 1분봉 수집 (시간 소요: 약 15~30분)
pyenv shell ptj_stock_lab && python experiments/expand_line_b_data.py --fetch-1min
```
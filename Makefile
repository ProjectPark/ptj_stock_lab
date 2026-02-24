# ── 서버 설정 ──
REMOTE_HOST  := gigaflops-node54
REMOTE_DIR   := /mnt/giga/project/ptj_stock_lab
REMOTE       := $(REMOTE_HOST):$(REMOTE_DIR)

# ── 공통 제외 패턴 ──
RSYNC_EXCLUDES := \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude '.env' \
  --exclude '.venv' \
  --exclude '*.log' \
  --exclude '.vscode' \
  --exclude '.idea'

# ── 코드 동기화 (로컬→서버) ──
sync-code:
	rsync -avz --delete \
	  $(RSYNC_EXCLUDES) \
	  --exclude 'data/' \
	  --exclude 'charts/' \
	  --exclude 'history/' \
	  ./ $(REMOTE)/

# ── 데이터: 로컬→서버 (market 데이터 push) ──
sync-data-push:
	rsync -avz $(RSYNC_EXCLUDES) \
	  ./data/market/ $(REMOTE)/data/market/

# ── 데이터: 서버→로컬 (결과 pull) ──
sync-data-pull:
	rsync -avz $(RSYNC_EXCLUDES) \
	  $(REMOTE)/data/results/ ./data/results/

# ── 전체 동기화 (코드 + 데이터 양방향) ──
sync-all: sync-code sync-data-push sync-data-pull

# ── 서버→로컬 전체 pull (초기 세팅/복구용) ──
sync-pull-all:
	rsync -avz $(RSYNC_EXCLUDES) \
	  $(REMOTE)/ ./

# ── Dry-run (실제 전송 없이 미리보기) ──
sync-dry:
	rsync -avzn --delete \
	  $(RSYNC_EXCLUDES) \
	  --exclude 'data/' \
	  --exclude 'charts/' \
	  --exclude 'history/' \
	  ./ $(REMOTE)/

# ── 서버에 프로젝트 폴더 초기 생성 ──
init-remote:
	ssh $(REMOTE_HOST) "mkdir -p $(REMOTE_DIR)"
	$(MAKE) sync-code

# ── mutagen 세션 관리 ──
mutagen-start:
	mutagen sync create \
	  --name=ptj-stock-lab \
	  --ignore-vcs \
	  --ignore=data/ \
	  --ignore=charts/ \
	  --ignore=history/ \
	  --ignore=__pycache__ \
	  --ignore=.DS_Store \
	  --ignore=*.pyc \
	  --ignore=*.log \
	  --ignore=.env \
	  --sync-mode=two-way-resolved \
	  ./ $(REMOTE)/

mutagen-stop:
	mutagen sync terminate ptj-stock-lab

mutagen-status:
	mutagen sync list

mutagen-pause:
	mutagen sync pause ptj-stock-lab

mutagen-resume:
	mutagen sync resume ptj-stock-lab

# ── 도움말 ──
help:
	@echo ""
	@echo "=== ptj_stock_lab 동기화 명령어 ==="
	@echo ""
	@echo "  make sync-code        코드 → 서버 (data 제외)"
	@echo "  make sync-data-push   data/market → 서버"
	@echo "  make sync-data-pull   서버 data/results → 로컬"
	@echo "  make sync-all         코드 + 데이터 전체 동기화"
	@echo "  make sync-pull-all    서버 전체 → 로컬 (복구용)"
	@echo "  make sync-dry         코드 동기화 미리보기"
	@echo "  make init-remote      서버에 폴더 생성 + 초기 push"
	@echo ""
	@echo "  make mutagen-start    실시간 동기화 시작"
	@echo "  make mutagen-stop     실시간 동기화 종료"
	@echo "  make mutagen-status   동기화 상태 확인"
	@echo "  make mutagen-pause    일시정지"
	@echo "  make mutagen-resume   재개"
	@echo ""

.PHONY: sync-code sync-data-push sync-data-pull sync-all sync-pull-all sync-dry \
        init-remote mutagen-start mutagen-stop mutagen-status mutagen-pause mutagen-resume help

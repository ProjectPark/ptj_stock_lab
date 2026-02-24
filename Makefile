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

# ══════════════════════════════════════════
# 기본 동기화
# ══════════════════════════════════════════

# ── 코드 동기화 (로컬→서버) ──
sync-code:
	rsync -avz --delete \
	  $(RSYNC_EXCLUDES) \
	  --exclude 'data/' \
	  --exclude 'charts/' \
	  --exclude 'history/' \
	  ./ $(REMOTE)/

# ── 데이터: 로컬→서버 (market 데이터 push, 초기 1회) ──
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

# ══════════════════════════════════════════
# 보안 워크플로우 (코드 흔적 안 남기기)
# ══════════════════════════════════════════

# ── 특정 파일/폴더만 push ──
# Usage: make push F=experiments/backtest_evaluation.py
push:
	@if [ -z "$(F)" ]; then echo "Usage: make push F=path/to/file"; exit 1; fi
	rsync -avzR $(RSYNC_EXCLUDES) ./$(F) $(REMOTE)/

# ── 서버에서 명령 실행 ──
# Usage: make remote-exec CMD="python simulation/backtests/backtest_v5.py"
remote-exec:
	@if [ -z "$(CMD)" ]; then echo "Usage: make remote-exec CMD=\"python script.py\""; exit 1; fi
	ssh $(REMOTE_HOST) "cd $(REMOTE_DIR) && $(CMD)"

# ── 서버 코드 삭제 (data/ 만 유지) ──
clean-remote-code:
	ssh $(REMOTE_HOST) "cd $(REMOTE_DIR) && find . -maxdepth 1 ! -name '.' ! -name 'data' -exec rm -rf {} +"

# ── 서버 결과 삭제 ──
clean-remote-results:
	ssh $(REMOTE_HOST) "rm -rf $(REMOTE_DIR)/data/results/*"

# ── 서버 전체 정리 (data/market 만 유지) ──
clean-remote:
	$(MAKE) clean-remote-code
	$(MAKE) clean-remote-results

# ── 원스텝: push → run → pull → clean ──
# Usage: make run-remote SCRIPT=simulation/backtests/backtest_v5.py
run-remote:
	@if [ -z "$(SCRIPT)" ]; then echo "Usage: make run-remote SCRIPT=path/to/script.py"; exit 1; fi
	@echo ">>> [1/4] 코드 push..."
	@$(MAKE) sync-code --no-print-directory
	@echo ">>> [2/4] 서버 실행: $(SCRIPT)"
	@ssh $(REMOTE_HOST) "cd $(REMOTE_DIR) && python $(SCRIPT)"
	@echo ">>> [3/4] 결과 pull..."
	@$(MAKE) sync-data-pull --no-print-directory
	@echo ">>> [4/4] 서버 코드 정리..."
	@$(MAKE) clean-remote-code --no-print-directory
	@echo ">>> 완료. 서버에 data/ 만 남아있습니다."

# ══════════════════════════════════════════
# mutagen 세션 관리
# ══════════════════════════════════════════

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

# ══════════════════════════════════════════
# 도움말
# ══════════════════════════════════════════

help:
	@echo ""
	@echo "=== ptj_stock_lab 동기화 명령어 ==="
	@echo ""
	@echo "  [기본 동기화]"
	@echo "  make sync-code          코드 → 서버 (data 제외)"
	@echo "  make sync-data-push     data/market → 서버 (초기 1회)"
	@echo "  make sync-data-pull     서버 data/results → 로컬"
	@echo "  make sync-all           코드 + 데이터 전체 동기화"
	@echo "  make sync-pull-all      서버 전체 → 로컬 (복구용)"
	@echo "  make sync-dry           코드 동기화 미리보기"
	@echo "  make init-remote        서버에 폴더 생성 + 초기 push"
	@echo ""
	@echo "  [보안 워크플로우]"
	@echo "  make push F=path        특정 파일만 서버에 push"
	@echo "  make remote-exec CMD=.. 서버에서 명령 실행"
	@echo "  make clean-remote-code  서버 코드 삭제 (data/ 유지)"
	@echo "  make clean-remote-results 서버 결과 삭제"
	@echo "  make clean-remote       서버 전체 정리 (data/market 유지)"
	@echo "  make run-remote SCRIPT=.. push→실행→pull→코드삭제 원스텝"
	@echo ""
	@echo "  [실시간 동기화]"
	@echo "  make mutagen-start      실시간 동기화 시작"
	@echo "  make mutagen-stop       실시간 동기화 종료"
	@echo "  make mutagen-status     동기화 상태 확인"
	@echo "  make mutagen-pause      일시정지"
	@echo "  make mutagen-resume     재개"
	@echo ""

.PHONY: sync-code sync-data-push sync-data-pull sync-all sync-pull-all sync-dry \
        init-remote push remote-exec clean-remote-code clean-remote-results clean-remote \
        run-remote mutagen-start mutagen-stop mutagen-status mutagen-pause mutagen-resume help

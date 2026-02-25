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
# SLURM Profile 기반 보안 실행 (Job별 격리)
# ══════════════════════════════════════════
#
# 구조: runs/<profile>-<timestamp>/ 에 코드를 격리하여 동시 실행 안전
#
# Usage:
#   make slurm-full PROFILE=backtest_v5              # 원스텝 (push→submit→wait→collect)
#   make slurm-run  PROFILE=optimizer_v5             # 비동기 (push→submit)
#   make slurm-run  PROFILE=backtest_v5 ARGS="--dry" # 스크립트 인자 전달
#   make slurm-watch   PROFILE=optimizer_v5          # 완료 대기
#   make slurm-collect PROFILE=optimizer_v5          # 결과 수집 + run dir 삭제
#   make slurm-recover PROFILE=optimizer_v5          # 네트워크 끊김 후 복구
#   make slurm-status                                # 내 전체 Job 목록
#   make slurm-log PROFILE=backtest_v5               # 실행 중 로그 tail
#   make slurm-clean                                 # 완료된 모든 run dir 정리

PARTITION     ?= titanx
ACCOUNT       ?= default
SBATCH_MEM    ?= 120G
ARGS          ?=
SLURM_POLL    ?= 30

# ── Docker 이미지 설정 ──
DOCKER_IMAGE  ?= ptj_stock_lab
DOCKER_TAG    ?= latest
SLURM_IMG_DIR  = $(REMOTE_DIR)/slurm/images
SLURM_IMG_PATH = $(SLURM_IMG_DIR)/$(DOCKER_IMAGE).sqsh

# ── slurm-image-build: 로컬에서 linux/amd64 이미지 빌드 (Docker Desktop 필요) ──
slurm-image-build:
	@echo ">>> [slurm-image-build] Building $(DOCKER_IMAGE):$(DOCKER_TAG) for linux/amd64 ..."
	docker buildx build --platform linux/amd64 -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo ">>> [slurm-image-build] 완료."

# ── slurm-image-push: 로컬 이미지 → 서버 전송 → enroot import (Docker Desktop 필요) ──
slurm-image-push:
	@echo ">>> [slurm-image-push] Saving image to /tmp/$(DOCKER_IMAGE).tar.gz ..."
	docker save $(DOCKER_IMAGE):$(DOCKER_TAG) | gzip > /tmp/$(DOCKER_IMAGE).tar.gz
	@echo ">>> [slurm-image-push] Uploading to server ..."
	ssh $(REMOTE_HOST) "mkdir -p $(SLURM_IMG_DIR)"
	rsync -avz --progress /tmp/$(DOCKER_IMAGE).tar.gz $(REMOTE)/slurm/images/$(DOCKER_IMAGE).tar.gz
	@echo ">>> [slurm-image-push] Importing as enroot .sqsh on server ..."
	ssh $(REMOTE_HOST) "enroot import --output $(SLURM_IMG_PATH) file:///$(SLURM_IMG_DIR)/$(DOCKER_IMAGE).tar.gz && rm $(SLURM_IMG_DIR)/$(DOCKER_IMAGE).tar.gz"
	@echo ">>> [slurm-image-push] 완료: $(SLURM_IMG_PATH)"

# ── slurm-image-setup: build + push 원스텝 (Docker Desktop 필요) ──
slurm-image-setup: slurm-image-build slurm-image-push
	@echo ">>> [slurm-image-setup] Docker 이미지 준비 완료."

# ── slurm-image-refresh: 서버에서 직접 이미지 재빌드 (Docker Desktop 불필요) ──
# enroot pull + pip install + export (패키지 업데이트 시 사용)
slurm-image-refresh:
	@echo ">>> [slurm-image-refresh] Pulling python:3.10-slim via enroot ..."
	ssh $(REMOTE_HOST) "cd ~ && enroot import docker://python:3.10-slim"
	@echo ">>> [slurm-image-refresh] Creating container and installing packages ..."
	ssh $(REMOTE_HOST) "enroot delete ptj_stock_lab 2>/dev/null || true && enroot create --name ptj_stock_lab ~/python+3.10-slim.sqsh && enroot start --rw --root ptj_stock_lab -- pip install --no-cache-dir numpy pandas pyarrow optuna scipy matplotlib tqdm PyYAML python-dateutil pytz"
	@echo ">>> [slurm-image-refresh] Exporting to $(SLURM_IMG_PATH) ..."
	ssh $(REMOTE_HOST) "mkdir -p $(SLURM_IMG_DIR) && enroot export --output $(SLURM_IMG_PATH) ptj_stock_lab"
	@echo ">>> [slurm-image-refresh] 완료: $(SLURM_IMG_PATH)"

# ── slurm-setup: 서버 venv 생성 (초기 1회) ──
slurm-setup:
	@echo ">>> [slurm-setup] 서버에 venv 생성 중..."
	rsync -avzR $(RSYNC_EXCLUDES) ./slurm/setup_env.sh $(REMOTE)/
	ssh $(REMOTE_HOST) "bash $(REMOTE_DIR)/slurm/setup_env.sh"
	@echo ">>> [slurm-setup] 완료."

# ── slurm-push: Profile 기반 코드를 격리된 run dir에 push ──
slurm-push:
	@if [ -z "$(PROFILE)" ]; then echo "Usage: make slurm-push PROFILE=backtest_v5"; exit 1; fi
	@if [ ! -f slurm/profiles/$(PROFILE).conf ]; then echo "Error: slurm/profiles/$(PROFILE).conf not found"; exit 1; fi
	@echo ">>> [slurm-push] Profile: $(PROFILE)"
	@. slurm/profiles/$(PROFILE).conf && \
	  TIMESTAMP=$$(date +%Y%m%d-%H%M%S) && \
	  RUN_DIR=$(REMOTE_DIR)/runs/$(PROFILE)-$$TIMESTAMP && \
	  echo ">>> [slurm-push] Run dir: $$RUN_DIR" && \
	  ssh $(REMOTE_HOST) "mkdir -p $$RUN_DIR $(REMOTE_DIR)/slurm/logs $(REMOTE_DIR)/slurm/jobs" && \
	  for f in $$CODE_FILES; do \
	    rsync -avzR $(RSYNC_EXCLUDES) ./$$f $(REMOTE_HOST):$$RUN_DIR/; \
	  done && \
	  ssh $(REMOTE_HOST) "ln -sfn $(REMOTE_DIR)/data $$RUN_DIR/data" && \
	  echo "$$RUN_DIR" > slurm/jobs/$(PROFILE).rundir
	@echo ">>> [slurm-push] 완료. 격리된 run dir에 코드 push됨."

# ── slurm-submit: sbatch 생성 + 제출 (run dir 기반) ──
slurm-submit:
	@if [ -z "$(PROFILE)" ]; then echo "Usage: make slurm-submit PROFILE=backtest_v5"; exit 1; fi
	@if [ ! -f slurm/profiles/$(PROFILE).conf ]; then echo "Error: slurm/profiles/$(PROFILE).conf not found"; exit 1; fi
	@if [ ! -f slurm/jobs/$(PROFILE).rundir ]; then echo "Error: slurm/jobs/$(PROFILE).rundir not found. slurm-push 먼저 실행하세요."; exit 1; fi
	@echo ">>> [slurm-submit] Profile: $(PROFILE)"
	@. slurm/profiles/$(PROFILE).conf && \
	  SBATCH_MEM=$${SBATCH_MEM:-$(SBATCH_MEM)} && \
	  RUN_DIR=$$(cat slurm/jobs/$(PROFILE).rundir) && \
	  _ARGS="$(ARGS)" && \
	  EFFECTIVE_ARGS="$${_ARGS:-$$SCRIPT_ARGS}" && \
	  sed \
	    -e 's|{{PROFILE}}|$(PROFILE)|g' \
	    -e "s|{{PARTITION}}|$(PARTITION)|g" \
	    -e "s|{{ACCOUNT}}|$(ACCOUNT)|g" \
	    -e "s|{{SBATCH_CPUS}}|$$SBATCH_CPUS|g" \
	    -e "s|{{SBATCH_MEM}}|$$SBATCH_MEM|g" \
	    -e "s|{{SBATCH_TIME}}|$$SBATCH_TIME|g" \
	    -e "s|{{REMOTE_DIR}}|$(REMOTE_DIR)|g" \
	    -e "s|{{RUN_DIR}}|$$RUN_DIR|g" \
	    -e "s|{{SCRIPT}}|$$SCRIPT|g" \
	    -e "s|{{SCRIPT_ARGS}}|$$EFFECTIVE_ARGS|g" \
	    slurm/templates/default.sbatch > slurm/jobs/$(PROFILE).sbatch
	@echo ">>> [slurm-submit] sbatch 파일 생성: slurm/jobs/$(PROFILE).sbatch"
	@rsync -avz slurm/jobs/$(PROFILE).sbatch $(REMOTE)/slurm/jobs/
	@JOB_ID=$$(ssh $(REMOTE_HOST) "cd $(REMOTE_DIR) && sbatch slurm/jobs/$(PROFILE).sbatch" | grep -oE '[0-9]+') && \
	  echo "$$JOB_ID" > slurm/jobs/$(PROFILE).jobid && \
	  echo ">>> [slurm-submit] Job 제출 완료: $$JOB_ID"

# ── slurm-watch: Job 완료 대기 (폴링, 네트워크 끊김 내성) ──
slurm-watch:
	@if [ -z "$(PROFILE)" ]; then echo "Usage: make slurm-watch PROFILE=backtest_v5"; exit 1; fi
	@if [ ! -f slurm/jobs/$(PROFILE).jobid ]; then echo "Error: slurm/jobs/$(PROFILE).jobid not found. submit 먼저 실행하세요."; exit 1; fi
	@JOB_ID=$$(cat slurm/jobs/$(PROFILE).jobid) && \
	  echo ">>> [slurm-watch] Job $$JOB_ID 모니터링 시작 ($(SLURM_POLL)초 간격)..." && \
	  while true; do \
	    STATE=$$(ssh $(REMOTE_HOST) "squeue -j $$JOB_ID -h -o '%T'" 2>/dev/null) || { \
	      echo ">>> [slurm-watch] 네트워크 오류. $(SLURM_POLL)초 후 재시도..."; \
	      sleep $(SLURM_POLL); continue; \
	    }; \
	    if [ -z "$$STATE" ]; then \
	      echo ">>> [slurm-watch] Job $$JOB_ID 완료."; break; \
	    fi; \
	    echo "  [$$(date +%H:%M:%S)] Job $$JOB_ID: $$STATE"; \
	    sleep $(SLURM_POLL); \
	  done

# ── slurm-log: 실행 중 로그 tail ──
slurm-log:
	@if [ -z "$(PROFILE)" ]; then echo "Usage: make slurm-log PROFILE=backtest_v5"; exit 1; fi
	@if [ ! -f slurm/jobs/$(PROFILE).jobid ]; then echo "Error: jobid not found"; exit 1; fi
	@JOB_ID=$$(cat slurm/jobs/$(PROFILE).jobid) && \
	  echo ">>> [slurm-log] Job $$JOB_ID stdout (Ctrl+C to stop):" && \
	  ssh $(REMOTE_HOST) "tail -f $(REMOTE_DIR)/slurm/logs/$$JOB_ID.out" || true

# ── slurm-collect: 결과 pull + run dir만 삭제 (다른 Job 영향 없음) ──
slurm-collect:
	@if [ -z "$(PROFILE)" ]; then echo "Usage: make slurm-collect PROFILE=backtest_v5"; exit 1; fi
	@if [ ! -f slurm/profiles/$(PROFILE).conf ]; then echo "Error: profile not found"; exit 1; fi
	@echo ">>> [slurm-collect] 결과 pull 중..."
	@. slurm/profiles/$(PROFILE).conf && \
	  for d in $$RESULTS_PULL; do \
	    rsync -avz $(RSYNC_EXCLUDES) $(REMOTE)/$$d ./$$d; \
	  done
	@if [ -f slurm/jobs/$(PROFILE).jobid ]; then \
	  JOB_ID=$$(cat slurm/jobs/$(PROFILE).jobid) && \
	  echo ">>> [slurm-collect] 로그 pull 중..." && \
	  rsync -avz $(REMOTE)/slurm/logs/$$JOB_ID.out ./slurm/logs/ 2>/dev/null || true && \
	  rsync -avz $(REMOTE)/slurm/logs/$$JOB_ID.err ./slurm/logs/ 2>/dev/null || true; \
	fi
	@if [ -f slurm/jobs/$(PROFILE).rundir ]; then \
	  RUN_DIR=$$(cat slurm/jobs/$(PROFILE).rundir) && \
	  echo ">>> [slurm-collect] run dir 삭제: $$RUN_DIR" && \
	  ssh $(REMOTE_HOST) "rm -rf $$RUN_DIR" && \
	  rm -f slurm/jobs/$(PROFILE).rundir; \
	fi
	@echo ">>> [slurm-collect] 완료. 해당 run dir만 삭제됨 (다른 Job 영향 없음)."

# ── slurm-status: 현재 내 Job 목록 ──
slurm-status:
	@echo ">>> [slurm-status] 서버 SLURM Job 목록:"
	@ssh $(REMOTE_HOST) "squeue -u $$(whoami) -o '%.10i %.15j %.8T %.10M %.6D %R'" 2>/dev/null || \
	  echo ">>> 서버 연결 실패. 네트워크를 확인하세요."
	@echo ""
	@echo ">>> [slurm-status] 로컬 저장된 Job:"
	@for f in slurm/jobs/*.jobid; do \
	  if [ -f "$$f" ]; then \
	    PROF=$$(basename "$$f" .jobid); \
	    RUNDIR=""; \
	    if [ -f "slurm/jobs/$$PROF.rundir" ]; then RUNDIR=" -> $$(cat slurm/jobs/$$PROF.rundir)"; fi; \
	    echo "  $$PROF: $$(cat $$f)$$RUNDIR"; \
	  fi; \
	done 2>/dev/null || echo "  (없음)"

# ── slurm-recover: 네트워크 끊김 후 복구 (상태확인 → 완료시 collect) ──
slurm-recover:
	@if [ -z "$(PROFILE)" ]; then echo "Usage: make slurm-recover PROFILE=backtest_v5"; exit 1; fi
	@if [ ! -f slurm/jobs/$(PROFILE).jobid ]; then echo "Error: jobid not found. 이전에 submit한 적이 없습니다."; exit 1; fi
	@JOB_ID=$$(cat slurm/jobs/$(PROFILE).jobid) && \
	  echo ">>> [slurm-recover] Job $$JOB_ID 상태 확인..." && \
	  STATE=$$(ssh $(REMOTE_HOST) "squeue -j $$JOB_ID -h -o '%T'" 2>/dev/null) && \
	  if [ -z "$$STATE" ]; then \
	    echo ">>> [slurm-recover] Job $$JOB_ID 완료됨. 결과 수집 시작..." && \
	    $(MAKE) slurm-collect PROFILE=$(PROFILE) --no-print-directory; \
	  else \
	    echo ">>> [slurm-recover] Job $$JOB_ID 아직 실행 중: $$STATE" && \
	    echo ">>> slurm-watch로 대기하거나 나중에 다시 recover하세요."; \
	  fi

# ── slurm-run: push + submit (비동기) ──
slurm-run:
	@if [ -z "$(PROFILE)" ]; then echo "Usage: make slurm-run PROFILE=backtest_v5"; exit 1; fi
	@$(MAKE) slurm-push PROFILE=$(PROFILE) --no-print-directory
	@$(MAKE) slurm-submit PROFILE=$(PROFILE) PARTITION=$(PARTITION) ARGS="$(ARGS)" --no-print-directory
	@echo ">>> [slurm-run] 비동기 제출 완료. slurm-watch 또는 slurm-recover로 모니터링하세요."

# ── slurm-full: push + submit + wait + collect (동기, 원스텝) ──
slurm-full:
	@if [ -z "$(PROFILE)" ]; then echo "Usage: make slurm-full PROFILE=backtest_v5"; exit 1; fi
	@echo "══════════════════════════════════════════"
	@echo " SLURM Full Run: $(PROFILE)"
	@echo "══════════════════════════════════════════"
	@$(MAKE) slurm-push PROFILE=$(PROFILE) --no-print-directory
	@$(MAKE) slurm-submit PROFILE=$(PROFILE) PARTITION=$(PARTITION) ARGS="$(ARGS)" --no-print-directory
	@$(MAKE) slurm-watch PROFILE=$(PROFILE) --no-print-directory
	@$(MAKE) slurm-collect PROFILE=$(PROFILE) --no-print-directory
	@echo "══════════════════════════════════════════"
	@echo " 완료. 결과가 로컬에 저장되었습니다."
	@echo "══════════════════════════════════════════"

# ── slurm-clean: 완료된 모든 run dir 정리 ──
slurm-clean:
	@echo ">>> [slurm-clean] 서버 runs/ 디렉토리 정리 중..."
	@ssh $(REMOTE_HOST) "if ls -d $(REMOTE_DIR)/runs/*/ 2>/dev/null; then rm -rf $(REMOTE_DIR)/runs/*/; echo '>>> 삭제 완료.'; else echo '>>> runs/ 비어있음.'; fi"
	@rm -f slurm/jobs/*.rundir
	@echo ">>> [slurm-clean] 완료."

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
	@echo "  [SLURM Profile 실행 — Job별 격리 (runs/)]"
	@echo "  make slurm-setup                    서버 venv 생성 (초기 1회)"
	@echo "  make slurm-full    PROFILE=name     원스텝: push→submit→wait→collect"
	@echo "  make slurm-run     PROFILE=name     비동기: push→submit"
	@echo "  make slurm-push    PROFILE=name     격리 run dir에 코드 push"
	@echo "  make slurm-submit  PROFILE=name     sbatch 생성+제출"
	@echo "  make slurm-watch   PROFILE=name     완료 대기 (폴링)"
	@echo "  make slurm-log     PROFILE=name     실행 로그 tail"
	@echo "  make slurm-collect PROFILE=name     결과 pull + run dir 삭제"
	@echo "  make slurm-recover PROFILE=name     네트워크 끊김 후 복구"
	@echo "  make slurm-status                   내 전체 Job 목록"
	@echo "  make slurm-clean                    완료된 모든 run dir 정리"
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
        run-remote \
        slurm-image-build slurm-image-push slurm-image-setup slurm-image-refresh \
        slurm-setup slurm-push slurm-submit slurm-watch slurm-log slurm-collect \
        slurm-status slurm-recover slurm-run slurm-full slurm-clean \
        mutagen-start mutagen-stop mutagen-status mutagen-pause mutagen-resume help

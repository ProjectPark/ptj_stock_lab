FROM python:3.10-slim

# 비대화형 모드 + locale
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace

# 패키지 설치 (백테스트/최적화 전용 — streamlit/alpaca 등 제외)
COPY requirements-slurm.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements-slurm.txt \
 && rm -f requirements-slurm.txt

# 소스 코드는 런타임에 마운트 — 이미지에 복사하지 않음

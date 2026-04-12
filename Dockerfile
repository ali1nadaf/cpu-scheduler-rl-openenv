# ── CPU Scheduler RL Environment — Docker image ──────────────────────────────
FROM python:3.10-slim

LABEL maintainer="Mohtra AI"
LABEL description="OpenEnv CPU Scheduler RL Environment"
LABEL version="1.0"

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy project files
COPY cpu_env.py    .
COPY inference.py  .
COPY openenv.yaml  .
COPY README.md     .

# default: run full benchmark
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "inference.py"]
CMD ["--mode", "benchmark", "--episodes", "20"]

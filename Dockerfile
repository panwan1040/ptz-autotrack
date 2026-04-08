FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt README.md ./
COPY app ./app
COPY configs ./configs
COPY tests ./tests

RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -e .

EXPOSE 8080
CMD ["ptz-autotrack", "run"]

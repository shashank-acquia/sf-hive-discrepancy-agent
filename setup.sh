# ===================================================================================
# Dockerfile for CDP AI TOOLS (Corrected Version)
# ===================================================================================
FROM python:3.10-slim-bullseye

LABEL author="CDP AI Team"
LABEL description="Flask application for CDP AI Tools including discrepancy checks and an AI helpdesk."

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from tools.utils import download_nltk_data; download_nltk_data()"

COPY . .

EXPOSE 8081

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8081/ || exit 1

CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8081", "app:app"]
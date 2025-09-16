# ===================================================================================
# Dockerfile for CDP AI TOOLS (Final Production-Ready Version)
# ===================================================================================
FROM python:3.10-slim-bullseye


LABEL author="CDP AI Team"
LABEL description="Flask application for CDP AI Tools including discrepancy checks and an AI helpdesk."

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV SENTENCE_TRANSFORMERS_HOME=/root/.cache/torch/sentence_transformers

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY certs/*.pem /usr/local/share/ca-certificates/
RUN update-ca-certificates

ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY ./all-MiniLM-L6-v2 /app/models/all-MiniLM-L6-v2

COPY . .

RUN python -c "from tools.utils import download_nltk_data; download_nltk_data()"

EXPOSE 8081

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8081/ || exit 1

CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8081", "app:app"]
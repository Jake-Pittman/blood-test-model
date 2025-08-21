
FROM python:3.11-slim

# System deps (LightGBM + downloads)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates curl unzip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY recommender.py api.py ./
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Start via entrypoint (fetch artifacts, then launch uvicorn)
CMD ["/app/entrypoint.sh"]

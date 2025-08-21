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

# Normalize potential Windows/mac CRLFs and ensure executable
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod 755 /app/entrypoint.sh

# IMPORTANT: run via shell to avoid any shebang/format issues
CMD ["sh", "/app/entrypoint.sh"]

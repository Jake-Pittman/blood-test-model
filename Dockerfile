FROM python:3.11-slim

# System deps needed by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the app code
COPY recommender.py api.py ./

# Environment knobs (override at run/deploy time as needed)
# - MODEL_DIR: path inside the container to the FinalModel artifacts
# - FOODS_PATH: path to the foods parquet (or CSV)
ENV MODEL_DIR=/app/models/FinalModel
ENV FOODS_PATH=/app/data/processed/foods_dictionary_plus_enriched.parquet
ENV CORS_ALLOW_ORIGINS=*

EXPOSE 8080
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]

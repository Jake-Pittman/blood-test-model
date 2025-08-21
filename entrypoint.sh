
set -euo pipefail

echo "[entrypoint] preparing model and foods..."
mkdir -p /app/models /app/data/processed

# Download & unzip model ZIP to /app/models
if [ -n "${MODEL_ZIP_URL:-}" ]; then
  echo "[entrypoint] downloading model zip..."
  curl -fsSL ${AUTH_HEADER:+-H "$AUTH_HEADER"} "$MODEL_ZIP_URL" -o /tmp/FinalModel.zip
  unzip -o /tmp/FinalModel.zip -d /app/models

  # Normalize common layouts so we end up with /app/models/FinalModel/meta.json
  if [ ! -f /app/models/FinalModel/meta.json ]; then
    if [ -f /app/models/FinalModel/FinalModel/meta.json ]; then
      mv /app/models/FinalModel/FinalModel/* /app/models/FinalModel/
      rmdir /app/models/FinalModel/FinalModel || true
    elif [ -f /app/models/meta.json ]; then
      mkdir -p /app/models/FinalModel
      mv /app/models/*.joblib /app/models/meta.json /app/models/FinalModel/ 2>/dev/null || true
    fi
  fi
  echo "[entrypoint] model layout:" && ls -R /app/models | sed 's/^/  /'
fi

# Download foods file
if [ -n "${FOODS_URL:-}" ]; then
  echo "[entrypoint] downloading foods file..."
  curl -fsSL ${AUTH_HEADER:+-H "$AUTH_HEADER"} "$FOODS_URL" \
    -o /app/data/processed/foods_dictionary_plus_enriched.parquet
  echo "[entrypoint] foods present:" && ls -l /app/data/processed
fi

echo "[entrypoint] starting API..."
exec uvicorn api:app --host 0.0.0.0 --port 8080 --workers 1

#!/bin/sh
set -euo pipefail

echo "[entrypoint] preparing model and foods..."
mkdir -p /app/models /app/data/processed

# ---------- Resolve URLs (prefer *_B64 if provided) ----------
resolve_url() {
  b64_name="$1"; url_name="$2"
  val_b64="$(printenv "$b64_name" 2>/dev/null || true)"
  val_url="$(printenv "$url_name" 2>/dev/null || true)"
  if [ -n "$val_b64" ] && [ -z "$val_url" ]; then
    decoded="$(printf '%s' "$val_b64" | base64 -d 2>/dev/null || printf '%s' "$val_b64" | base64 -Di 2>/dev/null || true)"
    printf '%s' "$decoded"
  else
    printf '%s' "$val_url"
  fi
}

# Show which envs are present (without printing secrets)
echo "[entrypoint] env present: MODEL_ZIP_B64=$( [ -n "${MODEL_ZIP_B64:-}" ] && echo set || echo empty ), MODEL_ZIP_URL=$( [ -n "${MODEL_ZIP_URL:-}" ] && echo set || echo empty )"
echo "[entrypoint] env present: FOODS_URL_B64=$( [ -n "${FOODS_URL_B64:-}" ] && echo set || echo empty ), FOODS_URL=$( [ -n "${FOODS_URL:-}" ] && echo set || echo empty )"

MODEL_URL="$(resolve_url MODEL_ZIP_B64 MODEL_ZIP_URL)"
FOODS_URL_RESOLVED="$(resolve_url FOODS_URL_B64 FOODS_URL)"

# ---------- Validate + log URL health ----------
validate_url() {
  name="$1"; val="$2"
  if [ -z "$val" ]; then
    echo "[entrypoint] ERROR: $name is empty"; return 1
  fi
  # single line only (no control chars/newlines)
  if printf '%s' "$val" | grep -q '[[:cntrl:]]'; then
    echo "[entrypoint] ERROR: $name contains control chars (likely a newline)"; return 1
  fi
  case "$val" in
    http://*|https://*) : ;;
    *) echo "[entrypoint] ERROR: $name does not start with http(s)"; return 1 ;;
  esac
  if printf '%s' "$val" | grep -q '"'; then
    echo "[entrypoint] ERROR: $name contains quotes"; return 1
  fi
  echo "[entrypoint] $name OK (len=$(printf '%s' "$val" | wc -c | tr -d ' '))"
  return 0
}

# ---------- Model ZIP (REQUIRED) ----------
if ! validate_url "MODEL_URL" "$MODEL_URL"; then
  echo "[entrypoint] FATAL: No valid model URL provided via MODEL_ZIP_B64 or MODEL_ZIP_URL."
  exit 1
fi

echo "[entrypoint] HEAD check model..."
curl -I -g --max-time 15 "$MODEL_URL" | sed 's/^/[curl HEAD] /' || true

echo "[entrypoint] downloading model zip..."
curl -fsSLg "$MODEL_URL" -o /tmp/FinalModel.zip
unzip -o /tmp/FinalModel.zip -d /app/models

# Normalize to /app/models/FinalModel/meta.json
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

# Double-check meta.json before starting API
if [ ! -f /app/models/FinalModel/meta.json ]; then
  echo "[entrypoint] FATAL: meta.json still not found at /app/models/FinalModel/meta.json"
  exit 1
fi

# ---------- Foods file (optional but recommended) ----------
if [ -n "$FOODS_URL_RESOLVED" ]; then
  if validate_url "FOODS_URL" "$FOODS_URL_RESOLVED"; then
    echo "[entrypoint] HEAD check foods..."
    curl -I -g --max-time 15 "$FOODS_URL_RESOLVED" | sed 's/^/[curl HEAD] /' || true

    echo "[entrypoint] downloading foods file..."
    curl -fsSLg "$FOODS_URL_RESOLVED" -o /app/data/processed/foods_dictionary_plus_enriched.parquet
    echo "[entrypoint] foods present:" && ls -l /app/data/processed
  else
    echo "[entrypoint] WARN: FOODS_URL invalid; skipping foods download"
  fi
else
  echo "[entrypoint] WARN: FOODS_URL not provided; API will start without foods DB"
fi

echo "[entrypoint] starting API..."
exec uvicorn api:app --host 0.0.0.0 --port 8080 --workers 1

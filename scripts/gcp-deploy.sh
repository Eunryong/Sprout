#!/usr/bin/env bash
# Deploy Sprout to Cloud Run (idempotent).
set -euo pipefail

PROJECT=ninth-goal-499012-m4
REGION=us-central1
SERVICE=sprout
DOMAIN=sprout.eunryong.com
IMAGE=us-central1-docker.pkg.dev/ninth-goal-499012-m4/sprout/sprout:latest

# Always run from the repo root.
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# (1) Enable required APIs.
gcloud services enable \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  secretmanager.googleapis.com \
  --project="$PROJECT"

# (2) Create Artifact Registry repo (skip if exists).
if gcloud artifacts repositories describe sprout \
  --location="$REGION" --project="$PROJECT" >/dev/null 2>&1; then
  echo "Artifact Registry repo sprout exists, skipping."
else
  gcloud artifacts repositories create sprout \
    --repository-format=docker \
    --location="$REGION" \
    --project="$PROJECT"
fi

# (3) Ensure secrets exist; never overwrite, never echo values.
ensure_secret() {
  local name="$1"
  if gcloud secrets describe "$name" \
    --project="$PROJECT" >/dev/null 2>&1; then
    echo "Secret $name exists, skipping."
  else
    printf 'Enter value for %s: ' "$name"
    read -rs value
    echo
    printf '%s' "$value" | gcloud secrets create "$name" \
      --data-file=- \
      --project="$PROJECT"
    unset value
  fi
}
ensure_secret MUSE_API_KEY
ensure_secret MUSE_BASE_URL

# (4) Grant the runtime identity access to only the required secrets.
EXISTING_SERVICE="$(gcloud run services list \
  --project="$PROJECT" --region="$REGION" \
  --filter="metadata.name=$SERVICE" --format='value(metadata.name)')"
RUNTIME_SERVICE_ACCOUNT=""
if [[ -n "$EXISTING_SERVICE" ]]; then
  RUNTIME_SERVICE_ACCOUNT="$(gcloud run services describe "$SERVICE" \
    --project="$PROJECT" --region="$REGION" \
    --format='value(spec.template.spec.serviceAccountName)')"
fi
if [[ -z "$RUNTIME_SERVICE_ACCOUNT" ]]; then
  PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" \
    --format='value(projectNumber)')"
  RUNTIME_SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
fi
for secret in MUSE_API_KEY MUSE_BASE_URL; do
  gcloud secrets add-iam-policy-binding "$secret" \
    --project="$PROJECT" \
    --member="serviceAccount:$RUNTIME_SERVICE_ACCOUNT" \
    --role=roles/secretmanager.secretAccessor \
    --condition=None \
    --quiet
done

# (5) Build and push the image.
gcloud builds submit \
  --tag "$IMAGE" \
  --project="$PROJECT" .

# (6) Deploy to Cloud Run using the identity granted access above.
# Shared in-memory state and local saves require a single instance.
gcloud run deploy "$SERVICE" \
  --image="$IMAGE" \
  --project="$PROJECT" \
  --region="$REGION" \
  --service-account="$RUNTIME_SERVICE_ACCOUNT" \
  --platform=managed \
  --allow-unauthenticated \
  --port=8080 \
  --memory=2Gi \
  --cpu=1 \
  --min-instances=1 \
  --max-instances=1 \
  --concurrency=10 \
  --timeout=300 \
  --set-env-vars=GROWING_LLM=live,GROWING_LANG=en \
  --set-secrets=MUSE_API_KEY=MUSE_API_KEY:latest,MUSE_BASE_URL=MUSE_BASE_URL:latest

# (7) Reuse only a matching domain mapping; create it only when absent.
DOMAIN_ERROR_FILE="$(mktemp)"
trap 'rm -f "$DOMAIN_ERROR_FILE"' EXIT
if MAPPED_SERVICE="$(gcloud beta run domain-mappings describe \
  --domain="$DOMAIN" --region="$REGION" --project="$PROJECT" \
  --format='value(spec.routeName)' --quiet 2>"$DOMAIN_ERROR_FILE")"; then
  if [[ "$MAPPED_SERVICE" != "$SERVICE" ]]; then
    printf 'Domain %s maps to %s, expected %s.\n' "$DOMAIN" "$MAPPED_SERVICE" "$SERVICE" >&2
    exit 1
  fi
  echo "Domain mapping $DOMAIN already points to $SERVICE, skipping creation."
else
  DOMAIN_DESCRIBE_STATUS=$?
  DOMAIN_ERROR="$(cat "$DOMAIN_ERROR_FILE")"
  case "$DOMAIN_ERROR" in
    *NOT_FOUND*|*"HTTPError 404"*|*"HTTP 404"*|*"(404)"*)
      gcloud beta run domain-mappings create \
        --service="$SERVICE" --domain="$DOMAIN" \
        --region="$REGION" --project="$PROJECT" --quiet
      ;;
    *)
      printf '%s\n' "$DOMAIN_ERROR" >&2
      exit "$DOMAIN_DESCRIBE_STATUS"
      ;;
  esac
fi
rm -f "$DOMAIN_ERROR_FILE"
trap - EXIT
gcloud beta run domain-mappings describe \
  --domain="$DOMAIN" --region="$REGION" --project="$PROJECT" \
  --format='yaml(spec.routeName,status.conditions,status.resourceRecords)' --quiet

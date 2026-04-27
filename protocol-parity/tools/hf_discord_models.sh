#!/bin/sh
# hf_discord_models.sh — Search Hugging Face for chat/text-generation models
## Modernized POSIX shell version (no Bashisms)
set -eu

usage() {
  cat <<'EOF'
Usage: hf_discord_models.sh [OPTIONS]

Search Hugging Face for text-generation models suitable for Discord bot usage.

Options:
  --query QUERY     Search query (default: "chat")
  --limit N         Max results (default: 10)
  --pipeline TAG    Pipeline tag filter (default: text-generation)
  --library LIB     Library filter (e.g., transformers, gguf, mlx)
  --sort FIELD      Sort by: downloads, likes, trending (default: downloads)
  --compact         One-line-per-model output (NDJSON)
  --ids-only        Print only model IDs (one per line)
  --help            Show this help

Examples:
  # Top 10 chat models by downloads
  hf_discord_models.sh

  # Search for small GGUF models
  hf_discord_models.sh --query "small chat" --library gguf --limit 5

  # Get model IDs for ABI connector config
  hf_discord_models.sh --ids-only --limit 20

  # Pipe to jq for custom filtering
  hf_discord_models.sh --limit 50 --compact | jq 'select(.likes > 100)'

  # Find MLX models for Apple Silicon
  hf_discord_models.sh --library mlx --query "instruct"
EOF
  exit 0
}

# Defaults
QUERY="chat"
LIMIT=10
PIPELINE="text-generation"
LIBRARY=""
SORT="downloads"
COMPACT=0
IDS_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --help) usage ;;
    --query) QUERY="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --pipeline) PIPELINE="$2"; shift 2 ;;
    --library) LIBRARY="$2"; shift 2 ;;
    --sort) SORT="$2"; shift 2 ;;
    --compact) COMPACT=1; shift ;;
    --ids-only) IDS_ONLY=1; shift ;;
    *) echo "Unknown option: $1" >&2; usage ;;
  esac
done

# Build URL
URL="https://huggingface.co/api/models?pipeline_tag=${PIPELINE}&sort=${SORT}&direction=-1&limit=${LIMIT}&search=${QUERY}"
if [ -n "$LIBRARY" ]; then
  URL="$URL&library=${LIBRARY}"
fi

# Fetch
RESPONSE="$(curl -fsSL "$URL" || true)" || {
  echo "Error: Failed to fetch from Hugging Face API" >&2; exit 1;
}

# Output
if [ "$IDS_ONLY" -eq 1 ]; then
  echo "$RESPONSE" | jq -r '.[].id'
elif [ "$COMPACT" -eq 1 ]; then
  echo "$RESPONSE" | jq -c '.[] | {id, downloads, likes, library: .library_name, pipeline: .pipeline_tag}'
else
  echo "$RESPONSE" | jq '[.[] | {
    id,
    downloads,
    likes,
    library: .library_name,
    pipeline: .pipeline_tag,
    tags: [.tags[] | select(. == "chat" or . == "conversational" or . == "instruct" or . == "gguf" or . == "mlx" or . == "quantized")]
  }]]'
fi

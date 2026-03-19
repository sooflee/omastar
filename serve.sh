#!/usr/bin/env bash
# Live-reload dev server for the frontend.
# Watches frontend/ and output/ for changes, serves from project root.
# Usage: ./serve.sh [port]

PORT="${1:-8080}"

exec npx --yes live-server \
  --port="$PORT" \
  --open="frontend/index.html" \
  --watch="frontend,output" \
  --no-css-inject

#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

if [[ ! -x .venv/bin/uvicorn || ! -d frontend/node_modules ]]; then
  echo "Dependencies are missing. Run: make bootstrap"
  exit 1
fi

.venv/bin/uvicorn kinetrace.api:app --reload --host 127.0.0.1 --port 8000 &
api_pid=$!
npm --prefix frontend run dev &
web_pid=$!

cleanup() {
  kill "$api_pid" "$web_pid" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

# macOS still ships an older Bash without `wait -n`. Polling also lets us stop
# both processes as soon as either development server exits.
while kill -0 "$api_pid" 2>/dev/null && kill -0 "$web_pid" 2>/dev/null; do
  sleep 1
done

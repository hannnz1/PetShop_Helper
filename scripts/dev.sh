#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONUTF8=1

if [[ ! -f .env ]]; then
  echo "Missing .env. Run scripts/configure.ps1 on Windows or copy .env.example to .env, then set CHAT_MODEL and CHAT_API_KEY." >&2
  exit 1
fi

uv sync --locked
make mcp-up
exec uv run --locked uvicorn app.main:app --host 127.0.0.1 --port 8000

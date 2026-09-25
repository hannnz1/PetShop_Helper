#!/usr/bin/env bash
set -euo pipefail
uv run --locked python scripts/demo_agent.py "$@"

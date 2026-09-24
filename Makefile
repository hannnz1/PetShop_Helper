.PHONY: dev test eval

ifeq ($(OS),Windows_NT)
dev:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/dev.ps1
else
dev:
	./scripts/dev.sh
endif

test:
	uv run --locked pytest -v

eval:
	uv run --locked python scripts/eval_extract.py

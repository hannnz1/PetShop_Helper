.PHONY: dev test eval eval-agent seed seed-conv kb-preview kb-build kb-vectorize eval-retrieval eval-mining

ifeq ($(OS),Windows_NT)
dev:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/dev.ps1
else
dev:
	./scripts/dev.sh
endif

ifeq ($(OS),Windows_NT)
kb-preview:
	.\.venv\Scripts\python.exe -X utf8 -m scripts.preview_kb

kb-build:
	.\.venv\Scripts\python.exe -X utf8 -m scripts.build_kb

kb-vectorize:
	.\.venv\Scripts\python.exe -X utf8 -m scripts.vectorize_kb

eval-retrieval:
	.\.venv\Scripts\python.exe -X utf8 -m scripts.eval_retrieval

eval-mining:
	.\.venv\Scripts\python.exe -X utf8 -m scripts.eval_mining
else
kb-preview:
	uv run --locked python -m scripts.preview_kb

kb-build:
	uv run --locked python -m scripts.build_kb

kb-vectorize:
	uv run --locked python -m scripts.vectorize_kb

eval-retrieval:
	uv run --locked python -m scripts.eval_retrieval

eval-mining:
	uv run --locked python -m scripts.eval_mining
endif

test:
	uv run --locked pytest -v

eval:
	uv run --locked python scripts/eval_extract.py

eval-agent:
	uv run --locked python scripts/eval_agent.py

ifeq ($(OS),Windows_NT)
seed:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/seed.ps1

seed-conv:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/seed-conv.ps1
else
seed:
	docker compose exec -T mysql sh -c 'mysql -uroot -proot --default-character-set=utf8mb4 mewhelp < /docker-entrypoint-initdb.d/ch02-seed.sql'

seed-conv:
	docker compose exec -T mysql sh -c 'mysql -uroot -proot --default-character-set=utf8mb4 mewhelp < /docker-entrypoint-initdb.d/ch03-seed.sql'
endif

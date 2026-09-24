.PHONY: dev test eval seed

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

ifeq ($(OS),Windows_NT)
seed:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/seed.ps1
else
seed:
	docker compose exec -T mysql sh -c 'mysql -uroot -proot --default-character-set=utf8mb4 mewhelp < /docker-entrypoint-initdb.d/ch02-seed.sql'
endif

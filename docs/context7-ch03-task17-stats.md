# Ch03 Task17 repository read queries: Context7 check

- Resolved `SQLAlchemy` to `/websites/sqlalchemy_en_20` (official SQLAlchemy 2.0 documentation, high reputation) on 2026-09-25.
- Queried aggregate `select(func.count()).select_from(...)`, grouped counts, `order_by`/`limit`, and result extraction with `scalars()` / `scalar_one()`.
- Official source: <https://docs.sqlalchemy.org/en/20/tutorial/data_select.html> and <https://docs.sqlalchemy.org/en/20/core/functions.html> document `select(func.count()).select_from(table)`; <https://docs.sqlalchemy.org/en/20/changelog/migration_20.html> documents `Session.execute(select(...)).scalars()` and explicit `limit(1)`.
- Applied these SQLAlchemy 2.0 patterns in `app/db/repository.py` for `knowledge_stats`, `list_recent_chunks`, `list_chunk_pairs`, and `staging_stats`.
- TDD red: all 4 offline SQL tests failed with missing repository attributes. Green: all 4 passed using actual SQLite queries through a small async-session adapter. MySQL integration remains pending while Docker's Linux engine is unavailable.
# API validation follow-up (2026-09-25)

Context7 `/pydantic/pydantic` official fields docs confirm optional numeric fields may use `Field(default=None, ge=..., le=...)`; the KB search request now rejects invalid Top-K and cosine score bounds before embedding.

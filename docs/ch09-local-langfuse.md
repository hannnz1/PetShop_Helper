# Chapter 9: local Langfuse

This deployment is based on the [official Langfuse v4.46.0 Compose file](https://github.com/langfuse/langfuse/blob/v4.46.0/docker-compose.yml). The web, worker, PostgreSQL, ClickHouse, Redis, and MinIO services belong to the separate `petshop-langfuse` Compose project. Its five persistent volumes are named under that project. All published ports bind to `127.0.0.1`; no trace is sent to Langfuse Cloud. The upstream `langfuse-web` and `langfuse-worker` image tags are fixed at `4.46.0`.

The Langfuse PostgreSQL container uses port 5432 internally and binds to host `127.0.0.1:15432`, leaving an existing host service on 5432 untouched. The startup script checks for unrelated listeners on its published host ports, accepts ports already owned by this Compose project on a repeated Start, and ignores comments when checking the local environment file for example credentials.

## Initialize local credentials

1. Copy `infra/langfuse/.env.example` to `infra/langfuse/.env.langfuse`. The latter is Git ignored. Replace every `CHANGE_ME` value with a different long random value. In PowerShell, generate one with `[Convert]::ToHexString([Security.Cryptography.RandomNumberGenerator]::GetBytes(32))`; `ENCRYPTION_KEY` must be exactly 64 hexadecimal characters. Never commit or paste the real file into an issue or log.
2. Inspect the values and run `docker compose --env-file infra/langfuse/.env.langfuse -f infra/langfuse/compose.yaml config --quiet` from the repository root. The script also validates secrets, Docker resources (at least 18 GiB memory and four CPUs assigned), Docker daemon availability, and published port conflicts with unrelated services before startup.
3. Run `./scripts/ch09/langfuse_local.ps1 -Action Start`. Open [http://127.0.0.1:3000](http://127.0.0.1:3000), register the first local account, create a local project, and copy its API public/secret keys. Do not use the optional `LANGFUSE_INIT_*` headless values unless you deliberately set strong credentials in the ignored file.
4. Put the project keys in the application's ignored root `.env` as `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`; set `LANGFUSE_BASE_URL=http://127.0.0.1:3000` and `LANGFUSE_ENABLED=true` only when tracing is desired. `LANGFUSE_ENABLED=false` needs no project keys or running Langfuse. Set `OBSERVABILITY_ADMIN_TOKEN` to a separate random secret to allow the later read-only usage API.

`./scripts/ch09/langfuse_local.ps1 -Action Status` shows the isolated services. A repeated `-Action Start` is safe for an already-running project. `./scripts/ch09/langfuse_local.ps1 -Action Stop` runs `docker compose down` and retains every named volume. Do not use `down -v` or delete existing MySQL, Milvus, or MCP containers. The local start, fake-model trace ingestion, and stopped-service Graph/SSE acceptance results are recorded in [the acceptance report](ch09-observability-acceptance.md).

The official v4.46.0 source uses `clickhouse/clickhouse-server:25.12`, `redis:7`, `postgres:17`, and an unversioned Chainguard MinIO image. The Langfuse web/worker tags are pinned here. Verify the dependency image tags again before an integration startup, since upstream dependency tags can change independently.

## Model usage and tool audit migration

Apply `sql/ch09-observability.sql` once to the **application** MySQL schema after the Chapter 8 schema. It adds nullable `tool_audit_logs.turn_id` and an index on `(conversation_id, turn_id)`, then creates `model_usage_events` keyed by the LangChain model run ID. Existing tool rows keep a null turn ID; new Graph tool rows carry the same turn ID as the local trace and usage event. The SQL file is a one-time migration, so do not apply it twice. For a fresh schema, apply Chapter 2–8 DDL first and this file last. The isolated pytest fixture does this only in `mewhelp_test`.

`CHAT_STREAM_USAGE=false` remains the default. Set it to `true` only when the selected OpenAI-compatible upstream supports streamed usage metadata. A provider that omits metadata produces `unavailable` usage rows with null token counts. The usage table stores IDs, intent, model, counts, status and time; it has no prompt, reply, tool argument or credential columns.

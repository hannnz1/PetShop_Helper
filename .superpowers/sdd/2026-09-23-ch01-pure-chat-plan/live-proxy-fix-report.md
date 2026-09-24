# Ch01 live proxy fix report

The three live validation CLIs now create clients through `scripts.eval_extract.live_client`, which sets `trust_env=False` so HTTPX does not route local service calls through inherited proxy variables.

A regression test runs each CLI entry point with HTTP_PROXY, HTTPS_PROXY, and ALL_PROXY set to an invalid proxy and confirms the constructed localhost client disables environment settings.

Validation: `uv run --with pytest pytest tests/test_demo_eval.py -q` — 17 passed. No live model calls were made.

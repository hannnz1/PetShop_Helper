# Task 2 implementation evidence

- Added chat and extraction request schemas with whitespace-only rejection and the agreed length limits.
- Added the five-value `RequestType` enum and `AfterSalesTicket`, with nullable/default-null `order_id` and a required nonblank expected solution.
- Added focused tests for required fields, whitespace, length boundaries, nullable order IDs, and enum validation.
- TDD red: before adding the schema modules, `pytest tests/test_schemas.py -q` failed during collection with `ModuleNotFoundError: No module named 'app.schemas'`.
- TDD green: `.venv/Scripts/python.exe -m pytest tests/test_schemas.py -q` passed (8 tests).

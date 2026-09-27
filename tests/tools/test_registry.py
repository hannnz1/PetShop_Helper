"""Chapter-eight builtin registry contracts."""

import logging

from app.tools import registry


def test_builtin_tools_register_without_legacy_logistics():
    assert {spec.name for spec in registry.builtin_specs()} == {
        "query_order", "query_product", "query_faq", "create_ticket", "submit_refund",
    }
    assert "query_logistics" not in {tool.name for tool in registry.get_chat_tools("business")}


def test_every_builtin_has_description_and_json_schema():
    for spec in registry.builtin_specs():
        assert spec.name and spec.description
        assert isinstance(spec.json_schema, dict)
        assert isinstance(spec.json_schema.get("properties"), dict)


def test_write_permission_and_injected_argument_are_local_rules():
    specs = {spec.name: spec for spec in registry.builtin_specs()}
    assert specs["create_ticket"].permission == "write"
    assert all(spec.permission == "read" for name, spec in specs.items()
               if name != "create_ticket")
    assert specs["create_ticket"].inject_conversation
    assert "conversation_id" not in specs["create_ticket"].json_schema["properties"]


def test_duplicate_builtin_registration_keeps_first(caplog):
    original = registry.builtin_specs()[0]
    duplicate = registry.ToolSpec(
        name=original.name, description="duplicate", json_schema={"type": "object", "properties": {}},
        tool=original.tool, permission="read", source="builtin",
    )
    with caplog.at_level(logging.WARNING):
        registry.register(duplicate)
    assert registry.get_builtin_spec(original.name) is original
    assert any("重名" in record.message for record in caplog.records)


def test_app_startup_scans_builtin_registry(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from app.config import Settings
    from app import main

    calls = []
    original = registry.scan_builtin

    def tracked_scan():
        calls.append("scan")
        original()

    monkeypatch.setattr(registry, "scan_builtin", tracked_scan)
    original_budget_check = main.validate_startup_budget

    def tracked_budget_check(settings):
        calls.append("budget")
        original_budget_check(settings)

    monkeypatch.setattr(main, "validate_startup_budget", tracked_budget_check)

    class InertGraphRuntime:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    async def no_recovery(_graph):
        pass

    monkeypatch.setattr(main, "GraphRuntime", InertGraphRuntime)
    monkeypatch.setattr(main, "schedule_recovery", no_recovery)
    settings = Settings(
        _env_file=None, chat_model="test", chat_base_url="http://127.0.0.1:9/v1",
        chat_api_key="test", graph_checkpoint_path=str(tmp_path / "graph.sqlite"),
        token_budget=32768,
    )

    class UnusedModel:
        pass

    with TestClient(main.create_app(settings=settings, model=UnusedModel())):
        pass
    assert calls[0] == "scan"

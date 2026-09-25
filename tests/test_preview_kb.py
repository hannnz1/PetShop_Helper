"""Offline preview must reuse the ingestion chunk builder."""

from scripts import preview_kb


def test_preview_reports_all_sources_without_database(capsys):
    preview_kb.main()
    output = capsys.readouterr().out
    assert "product-faq.md" in output
    assert "returns-policy.md" in output
    assert "after-sales-manual.md" in output
    assert "运费怎么算" in output
    assert "key-clause" not in output

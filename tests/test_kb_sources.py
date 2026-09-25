"""One authoritative manifest for source ingestion and previews."""

import pytest

from app.kb import sources
from scripts import build_kb, preview_kb


def test_scripts_share_manifest():
    assert build_kb.KB_DIR == sources.KB_DIR
    assert build_kb.DOCS is sources.SOURCE_TYPES
    assert preview_kb.DOCS is sources.SOURCE_TYPES
    assert set(sources.SOURCE_TYPES.values()) <= sources.CONTENT_TYPES


def test_resolve_source_rejects_unknown_or_traversal():
    with pytest.raises(ValueError):
        sources.resolve_source("../product-faq.md")
    with pytest.raises(ValueError):
        sources.resolve_source("unexpected.md")
    path, kind = sources.resolve_source("product-faq.md")
    assert path == sources.KB_DIR / "product-faq.md"
    assert kind == "faq"

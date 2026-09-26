"""Authoritative local knowledge-source manifest and type vocabulary."""

from pathlib import Path


KB_DIR = Path(__file__).resolve().parents[2] / "data" / "kb"
SOURCE_TYPES = {
    "product-faq.md": "faq",
    "returns-policy.md": "policy",
    "after-sales-manual.md": "manual",
    "product-specs.md": "spec",
}
CONTENT_TYPES = frozenset({"faq", "policy", "manual", "spec", "mined"})


def resolve_source(filename: str) -> tuple[Path, str]:
    """Resolve only a listed source file; never accept a caller path."""
    try:
        kind = SOURCE_TYPES[filename]
    except KeyError as exc:
        raise ValueError("unknown source file") from exc
    return KB_DIR / filename, kind

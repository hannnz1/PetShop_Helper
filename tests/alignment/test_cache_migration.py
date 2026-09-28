import hashlib
import json
from types import SimpleNamespace

from scripts.ch04_cache_migration import compatible_retrieval, migrate_legacy


SOURCE = b'''K = 10
def _read_cache(): pass
def _append_cache(): pass
def _cache_key(): pass
async def _retrieve_all(): pass
async def _generation(): return 1
'''


def test_compatibility_rejects_retrieval_but_accepts_generation_changes():
    assert compatible_retrieval(SOURCE, SOURCE.replace(b'return 1', b'return 2'))
    assert not compatible_retrieval(SOURCE, SOURCE.replace(b'K = 10', b'K = 20'))


def test_exact_legacy_fingerprint_preserves_authority_and_config(tmp_path, monkeypatch):
    from scripts import ch04_cache_migration as migration
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts/eval_ch04.py').write_bytes(SOURCE.replace(b'return 1', b'return 2'))
    monkeypatch.setattr(migration.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=SOURCE))
    old_scope = [1, ['knowledge-v1'], [SOURCE.hex(), 'dependency'], 'model-v1']
    digest = hashlib.sha256(json.dumps(old_scope, ensure_ascii=False, default=str).encode()).hexdigest()[:20]
    legacy = tmp_path / f'retrieval-{digest}.jsonl'
    legacy.write_bytes(b'{"key":"cached","value":[]}\n')
    changed = [1, ['knowledge-v2'], old_scope[2], 'model-v1']
    target = tmp_path / 'retrieval-new.jsonl'
    assert not migrate_legacy(tmp_path, target, changed)
    assert not migrate_legacy(tmp_path, target, [*old_scope[:3], 'model-v2'])
    assert migrate_legacy(tmp_path, target, old_scope)
    assert target.read_bytes() == legacy.read_bytes()
    assert legacy.exists()

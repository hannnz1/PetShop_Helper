"""Read fixed, version-bound artifacts without recomputing model metrics."""
import asyncio
import hashlib
import json
from pathlib import Path
from app.core.taxonomy import terminology_table
from app.topics.client import health_metadata
from scripts.ch10.inference_lib import bundle_metadata
from scripts.ch10.corpus_lib import desensitize

ROOT = Path(__file__).resolve().parents[2] / 'data/ch10'
ARTIFACTS = (
    ('golden', '黄金预标闸', 'corpus/golden_report.json', 'python -m scripts.ch10.build_corpus'),
    ('corpus', '语料血缘', 'corpus/corpus_report.json', 'python -m scripts.ch10.build_corpus'),
    ('dataset', '数据划分', 'dataset/manifest.json', 'python -m scripts.ch10.build_dataset'),
    ('model', '训练模型', 'model/train_report.json', 'python -m scripts.ch10.train'),
    ('threshold', '阈值重演', 'threshold_replay.json', 'python -m scripts.ch10.scan_threshold_replay'),
    ('evaluation', '质量红线', 'evaluation.json', 'python -m scripts.ch10.evaluate'),
    ('onnx', 'ONNX 一致性', 'onnx/export_report.json', 'python -m scripts.ch10.export_onnx'),
    ('service', '服务实测', 'service_smoke.json', 'python -m scripts.ch10.smoke_service'),
    ('pool', '分类入库', 'pool_categories.json', 'python -m scripts.ch10.classify_pool'),
)


def _masked(value):
    if isinstance(value, str): return desensitize(value)
    if isinstance(value, list): return [_masked(item) for item in value]
    if isinstance(value, dict): return {key: _masked(item) for key, item in value.items()}
    return value


def read_artifact(path: Path, expected: dict) -> dict:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict): raise ValueError('report must be object')
    except FileNotFoundError:
        return {'status': 'missing', 'data': None, 'reason': '报告尚未生成'}
    except (OSError, ValueError):
        return {'status': 'error', 'data': None, 'reason': '报告无法解析'}
    status = data.get('status', 'pending_verification')
    if not isinstance(status, str): status = 'error'
    reason = data.get('reason', '')
    if status not in ('failed', 'error') and not status.startswith('pending'):
        meta = {**data, **(data.get('metadata') if isinstance(data.get('metadata'), dict) else {})}
        mismatch = [key for key, value in expected.items() if value is not None and meta.get(key) != value]
        if mismatch:
            status, reason = 'stale', '版本不一致或缺少版本字段：' + ', '.join(mismatch)
        elif any(value is None for value in expected.values()):
            status, reason = 'pending_version', '当前版本产物缺失，无法核验历史报告'
    return {'status': status, 'data': _masked(data), 'reason': desensitize(str(reason))}


def _hash(path: Path):
    try: return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError: return None


def current_versions() -> dict:
    from scripts.ch10.train import checkpoint_hash
    result = {'taxonomy_hash': hashlib.sha256(terminology_table().encode('utf-8')).hexdigest(),
              'test_hash': _hash(ROOT/'dataset/test.jsonl'), 'val_hash': _hash(ROOT/'dataset/val.jsonl'),
              'model_version': None, 'checkpoint_hash': checkpoint_hash(ROOT/'model')}
    try:
        threshold = json.loads((ROOT/'onnx/threshold.json').read_text(encoding='utf-8'))['threshold']
        result.update(bundle_metadata(ROOT/'onnx', threshold))
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return result


def artifact_item(identity: str, versions: dict | None = None) -> dict:
    versions = versions or current_versions()
    key, label, path, command = next(item for item in ARTIFACTS if item[0] == identity)
    expected = {'taxonomy_hash': versions['taxonomy_hash']}
    if key in ('threshold', 'evaluation', 'onnx', 'service', 'pool'):
        expected['model_version'] = versions['model_version']
    if key in ('evaluation', 'onnx', 'service'):
        expected['test_hash'] = versions['test_hash']
    if key == 'threshold': expected['val_hash'] = versions['val_hash']
    if key == 'golden':
        from scripts.ch10.prelabel import prelabel_fingerprint
        expected['prelabel_version'] = prelabel_fingerprint()
        expected['golden_hash'] = _hash(Path(__file__).resolve().parents[2]/'scripts/ch10/golden_samples.jsonl')
    if key == 'corpus': expected['clean_hash'] = _hash(ROOT/'corpus/corpus_clean.jsonl')
    if key in ('model', 'onnx'):
        from scripts.ch10.train import checkpoint_hash
        expected['checkpoint_hash'] = (versions['checkpoint_hash'] if 'checkpoint_hash' in versions
                                       else checkpoint_hash(ROOT/'model'))
    if key == 'model':
        expected['dataset_manifest_hash'] = _hash(ROOT/'dataset/manifest.json')
    if key == 'dataset':
        from scripts.ch10.build_dataset import _hash_json
        split_hashes = {}
        try:
            for name in ('train', 'val', 'test'):
                rows = [json.loads(line) for line in (ROOT/f'dataset/{name}.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
                split_hashes[name] = _hash_json(rows)
            expected['split_hashes'] = split_hashes
        except (OSError, ValueError): expected['split_hashes'] = None
    result = read_artifact(ROOT/path, expected)
    if key in ('threshold', 'evaluation', 'service', 'pool') and result['status'] in ('passed', 'done', 'empty'):
        export = artifact_item('onnx', versions)
        if export['status'] != 'passed':
            result.update(status='stale' if export['status']=='stale' else 'pending_version',
                          reason='依赖的ONNX导出与当前训练版本未通过核验')
    return {'id': key, 'title': label, 'command': command, **result}


async def acceptance_overview() -> dict:
    versions = current_versions()
    items = [artifact_item(key, versions) for key, *_ in ARTIFACTS]
    try:
        meta = await asyncio.wait_for(health_metadata(), timeout=2)
        health = {'status': 'ready' if meta['model_version'] == versions['model_version'] else 'stale', **meta}
    except Exception as exc:
        health = {'status': 'unavailable', 'reason': type(exc).__name__}
    return {'versions': versions, 'health': health, 'items': items}

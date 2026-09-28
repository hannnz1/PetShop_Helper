import hashlib
import json
import pytest
from fastapi.testclient import TestClient
from app.core.taxonomy import TOPIC_NAMES, terminology_table


def metadata():
    return {'model_version': 'a'*64, 'taxonomy_hash': hashlib.sha256(terminology_table().encode()).hexdigest(),
            'threshold': .5}


def test_health_metadata_has_no_local_path():
    from scripts.ch10.serve import create_app
    class Runtime:
        metadata = metadata()
        def classify(self, texts):
            return [{'labels': [TOPIC_NAMES[0]], 'scores': dict.fromkeys(TOPIC_NAMES, .6)} for _ in texts]
    with TestClient(create_app(Runtime())) as client:
        health = client.get('/healthz')
        assert health.status_code == 200 and health.json()['ready'] is True
        assert health.json()['model_version'] == metadata()['model_version']
        assert 'model_dir' not in health.json()
        result = client.post('/classify', json={'texts': ['问题']}).json()
        assert result['model_version'] == health.json()['model_version']


def test_model_version_changes_with_bundle(tmp_path):
    from scripts.ch10.inference_lib import bundle_metadata
    for name in ('model.onnx', 'tokenizer.json', 'threshold.json'):
        (tmp_path/name).write_bytes(b'first')
    first = bundle_metadata(tmp_path, .5)
    (tmp_path/'model.onnx').write_bytes(b'second')
    assert bundle_metadata(tmp_path, .5)['model_version'] != first['model_version']


@pytest.mark.parametrize('change', ['version', 'length', 'nan', 'labels', 'scores'])
def test_classifier_rejects_invalid_response(change):
    from app.topics.client import validate_response
    payload = {**metadata(), 'results': [{'labels': [TOPIC_NAMES[0]], 'scores': dict.fromkeys(TOPIC_NAMES, .6)}]}
    if change == 'version': payload['model_version'] = 'b'*64
    if change == 'length': payload['results'] = []
    if change == 'nan': payload['results'][0]['scores'][TOPIC_NAMES[0]] = float('nan')
    if change == 'labels': payload['results'][0]['labels'] = ['非法']
    if change == 'scores': payload['results'][0]['scores'] = {}
    with pytest.raises(ValueError):
        validate_response(payload, 1, metadata())


def test_classifier_url_must_stay_local():
    from app.config import Settings
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Settings(_env_file=None, classifier_base_url='https://example.com',
                 chat_model='test', chat_api_key='test', chat_base_url='http://localhost')


def test_stale_export_pass_cannot_start_new_model(tmp_path):
    from scripts.ch10.inference_lib import load_runtime
    (tmp_path/'model.onnx').write_bytes(b'new model')
    (tmp_path/'tokenizer.json').write_text('{}')
    (tmp_path/'threshold.json').write_text(json.dumps({**metadata(), 'label_order':list(TOPIC_NAMES)}))
    (tmp_path/'export_report.json').write_text(json.dumps({'status':'passed', 'metadata':metadata()}))
    with pytest.raises(ValueError, match='version'):
        load_runtime(tmp_path)

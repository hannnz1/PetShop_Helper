import json
import pytest
import httpx
from fastapi import FastAPI
from types import SimpleNamespace
from pydantic import SecretStr


def test_old_passed_artifact_is_stale_for_new_model(tmp_path):
    from app.topics.acceptance import read_artifact
    path = tmp_path/'report.json'
    assert read_artifact(path, {})['status'] == 'missing'
    path.write_text('{oops', encoding='utf-8')
    assert read_artifact(path, {})['status'] == 'error'
    path.write_text(json.dumps({'status': 'passed', 'metadata': {'model_version': 'old', 'test_hash': 'x'}}), encoding='utf-8')
    assert read_artifact(path, {'model_version': 'new'})['status'] == 'stale'
    assert read_artifact(path, {'model_version': 'old', 'test_hash': 'new'})['status'] == 'stale'
    path.write_text(json.dumps({'status': 'pending_compute'}), encoding='utf-8')
    assert read_artifact(path, {'model_version': 'new'})['status'] == 'pending_compute'


@pytest.mark.asyncio
async def test_nine_reports_survive_unhealthy_service(tmp_path, monkeypatch):
    from app.topics import acceptance
    monkeypatch.setattr(acceptance, 'ROOT', tmp_path)
    async def unavailable(): raise ConnectionError('private local path')
    monkeypatch.setattr(acceptance, 'health_metadata', unavailable)
    folder = tmp_path/'onnx'
    folder.mkdir()
    (folder/'model.onnx').write_bytes(b'not proof')
    (folder/'export_report.json').write_text(json.dumps({'status':'failed', 'reason':'mismatch'}))
    report = await acceptance.acceptance_overview()
    assert len(report['items']) == 9
    assert report['health']['status'] == 'unavailable'
    assert next(row for row in report['items'] if row['id']=='onnx')['status'] == 'failed'
    assert 'private local path' not in str(report)


@pytest.mark.asyncio
async def test_acceptance_requires_observability_token(tmp_path, monkeypatch):
    from app.api.acceptance import router
    from app.topics import acceptance
    monkeypatch.setattr(acceptance, 'ROOT', tmp_path)
    app = FastAPI()
    app.state.settings = SimpleNamespace(observability_admin_token=SecretStr('secret'))
    app.include_router(router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
        for endpoint in ('data', 'evaluation', 'errors'):
            assert (await client.get('/api/acceptance/'+endpoint)).status_code == 401
            response = await client.get('/api/acceptance/'+endpoint, headers={'Authorization':'Bearer secret'})
            assert response.status_code == 200
        assert (await client.get('/api/acceptance/arbitrary')).status_code == 404


def test_corpus_producer_emits_version_bound_summary(tmp_path):
    from scripts.ch10.build_corpus import run_synthetic
    run_synthetic(tmp_path)
    report = json.loads((tmp_path/'corpus_report.json').read_text(encoding='utf-8'))
    assert report['status'] == 'pending_data'
    assert len(report['metadata']['clean_hash']) == 64
    assert report['counts']['labeled'] > 0


@pytest.mark.asyncio
async def test_service_smoke_records_missing_model_without_inference(tmp_path, monkeypatch):
    from scripts.ch10 import smoke_service
    async def forbidden(): raise AssertionError('missing bundle must not call service')
    monkeypatch.setattr(smoke_service, 'health_metadata', forbidden)
    report = await smoke_service.run(tmp_path/'missing', tmp_path/'test.jsonl', tmp_path/'smoke.json')
    assert report['status'] == 'pending_compute'


def test_golden_report_binds_current_prelabel_version(tmp_path, monkeypatch):
    from app.topics import acceptance
    from app.config import get_settings
    from scripts.ch10.prelabel import prelabel_fingerprint
    config = get_settings().model_copy(update={'chat_model':'model-one'})
    import scripts.ch10.prelabel as prelabel
    monkeypatch.setattr(prelabel, 'get_settings', lambda: config)
    monkeypatch.setattr(acceptance, 'ROOT', tmp_path)
    folder=tmp_path/'corpus'
    folder.mkdir()
    golden = acceptance._hash(acceptance.Path(__file__).resolve().parents[2]/'scripts/ch10/golden_samples.jsonl')
    report={'status':'passed', 'metadata':{'taxonomy_hash':acceptance.current_versions()['taxonomy_hash'],
        'golden_hash':golden, 'prelabel_version':prelabel_fingerprint()}}
    (folder/'golden_report.json').write_text(json.dumps(report),encoding='utf-8')
    assert acceptance.artifact_item('golden')['status']=='passed'
    config.chat_model='model-two'
    assert acceptance.artifact_item('golden')['status']=='stale'


def test_retraining_invalidates_old_export_and_downstream_reports(tmp_path, monkeypatch):
    from app.topics import acceptance
    from scripts.ch10.train import checkpoint_hash
    monkeypatch.setattr(acceptance, 'ROOT', tmp_path)
    model = tmp_path/'model'
    model.mkdir()
    (model/'model.safetensors').write_bytes(b'checkpoint-one')
    versions = {'taxonomy_hash':'taxonomy', 'model_version':'onnx-one',
                'test_hash':'test', 'val_hash':'val'}
    meta = {**versions, 'checkpoint_hash':checkpoint_hash(model)}
    for key in ('onnx', 'threshold', 'evaluation', 'service', 'pool'):
        path = tmp_path/next(item[2] for item in acceptance.ARTIFACTS if item[0]==key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'status':'passed', 'metadata':meta}), encoding='utf-8')
        assert acceptance.artifact_item(key, versions)['status']=='passed'
    (model/'model.safetensors').write_bytes(b'checkpoint-two')
    for key in ('onnx', 'threshold', 'evaluation', 'service', 'pool'):
        assert acceptance.artifact_item(key, versions)['status']=='stale', key


def test_export_report_binds_source_checkpoint(tmp_path, monkeypatch):
    from scripts.ch10 import export_onnx
    from scripts.ch10.train import checkpoint_hash
    from scripts.ch10 import inference_lib
    model, output = tmp_path/'model', tmp_path/'onnx'
    model.mkdir(); output.mkdir()
    (model/'model.safetensors').write_bytes(b'test-checkpoint')
    (output/'threshold.json').write_text('{"threshold": 0.5}')
    test = tmp_path/'test.jsonl'
    test.write_text('test')
    monkeypatch.setattr(export_onnx, '_export_and_verify', lambda *args: export_onnx.ExportReport('passed'))
    monkeypatch.setattr(inference_lib, 'bundle_metadata', lambda *args: {'model_version':'onnx-test'})
    result = export_onnx.export_and_verify(model, test, output)
    assert result.metadata['checkpoint_hash']==checkpoint_hash(model)

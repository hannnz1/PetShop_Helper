from types import SimpleNamespace

import pytest

from scripts.ch10 import text_jobs


class Model:
    def __init__(self, output=None, fail=False):
        self.output, self.fail, self.calls = output, fail, 0
    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        return self
    async def ainvoke(self, prompt):
        self.calls += 1
        if self.fail:
            raise ConnectionError('down')
        return self.schema(text=self.output)


@pytest.mark.asyncio
async def test_cleaning_failure_preserves_review_lineage(monkeypatch):
    monkeypatch.setattr(text_jobs, 'get_settings', lambda: SimpleNamespace(chat_base_url='http://127.0.0.1:9', structured_output_method='json_mode'))
    source = {'id': 7, 'review_id': 'pool-7', 'text': '手机13812345678想退款', 'origin': 'pool'}
    result = (await text_jobs.clean_rows([source], Model(fail=True)))[0]
    assert result['text'] == '手机[手机号]想退款'
    assert result['review_id'] == source['review_id']
    assert result['clean_status'] == 'failed'
    assert '13812345678' not in str(result)


@pytest.mark.asyncio
async def test_cleaning_rejects_changed_models(monkeypatch):
    monkeypatch.setattr(text_jobs, 'get_settings', lambda: SimpleNamespace(chat_base_url='http://127.0.0.1:9', structured_output_method='json_mode'))
    result = await text_jobs.clean_one('MH-CAM1咋退货', Model('MH-CAD1怎么退货'))
    assert result['text'] == 'MH-CAM1咋退货'
    assert result['status'] == 'rejected'


@pytest.mark.asyncio
async def test_no_external_permission_means_no_cleaning_call(monkeypatch):
    monkeypatch.setattr(text_jobs, 'get_settings', lambda: SimpleNamespace(chat_base_url='https://external.example/v1'))
    model = Model('退款')
    rows = await text_jobs.clean_rows([{'id': 1, 'text': '退欵'}], model)
    assert model.calls == 0
    assert rows[0]['clean_status'] == 'pending_upstream'


def test_deduplication_preserves_all_source_ids():
    rows = text_jobs.prepare_rows([{'id': 1, 'text': '退款'}, {'id': 2, 'text': '退款'}])
    assert len(rows) == 1
    assert set(rows[0]['source_review_ids']) == {'pool-1', 'pool-2'}

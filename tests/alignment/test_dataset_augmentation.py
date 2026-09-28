import pytest
import json

from app.core.taxonomy import TOPIC_NAMES
from scripts.ch10.build_dataset import build_dataset, fingerprint
from scripts.ch10.text_jobs import prepare_augmentation


@pytest.mark.asyncio
async def test_cli_augmentation_preserves_holdout(tmp_path):
    rows = [{'text': f'测试宠物问题{i}', 'labels': [TOPIC_NAMES[0]], 'reviewed': True,
             'review_id': f'pool-{i}', 'origin': 'pool'} for i in range(20)]
    original = build_dataset(rows)
    inputs = []
    class Model:
        def with_structured_output(self, schema, **kwargs):
            self.schema = schema
            return self
        async def ainvoke(self, prompt):
            text = prompt.to_messages()[-1].content
            inputs.append(text)
            return self.schema(text=text + '请问')
    cache = tmp_path / 'cache.json'
    mapping = await prepare_augmentation(list(original.train), Model(), cache)
    again = await prepare_augmentation(list(original.train), Model(), cache)
    assert mapping == again
    assert len(inputs) == len(original.train)
    assert set(inputs) == {row['text'] for row in original.train}
    augmented = build_dataset(rows, augment_fn=lambda row: mapping[fingerprint(row['text'])].get('text'))
    assert original.val == augmented.val and original.test == augmented.test
    assert original.manifest['split_hashes']['val'] == augmented.manifest['split_hashes']['val']
    assert original.manifest['split_hashes']['test'] == augmented.manifest['split_hashes']['test']
    assert all(row.get('parent_review_id') for row in augmented.train if row['origin']=='augmented')
    assert augmented.manifest['status'] == 'pending_review'
    assert all(row['reviewed'] is False for row in augmented.train if row['origin']=='augmented')


def test_failed_augmentation_cli_preserves_holdout_and_reports_pending(tmp_path, monkeypatch):
    from scripts.ch10 import build_dataset as module
    import app.core.llm
    rows = [{'text': f'离线问题{i}', 'labels': [TOPIC_NAMES[0]], 'reviewed': True,
             'review_id': f'pool-{i}'} for i in range(20)]
    source = tmp_path / 'source.jsonl'
    source.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows), encoding='utf-8')
    class Failed:
        def with_structured_output(self, *args, **kwargs):
            return self
        async def ainvoke(self, *args):
            raise RuntimeError('quota')
    monkeypatch.setattr(app.core.llm, 'get_chat_model', lambda **kwargs: Failed())
    before, after = tmp_path/'before', tmp_path/'after'
    module.main(['--source', str(source), '--out', str(before)])
    module.main(['--source', str(source), '--out', str(after), '--augment',
                 '--augmentation-cache', str(tmp_path/'cache.json')])
    for name in ('train', 'val', 'test'):
        assert (before/f'{name}.jsonl').read_bytes() == (after/f'{name}.jsonl').read_bytes()
    manifest = json.loads((after/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['augmentation']['status'] == 'pending_upstream'
    assert manifest['augmentation']['counts']['failed'] == 16


def test_human_augmented_decisions_preserve_holdout():
    from scripts.ch10.build_dataset import review_augmentation
    rows=[{'text':f'问题{i}', 'labels':[TOPIC_NAMES[0]], 'reviewed':True, 'review_id':str(i)} for i in range(20)]
    build=build_dataset(rows, augment_fn=lambda row: row['text']+'请帮忙')
    decisions=[{'review_id':row['review_id'],'approved':i%2==0} for i,row in enumerate(build.train) if row.get('origin')=='augmented']
    reviewed=review_augmentation(build,decisions)
    assert reviewed.val == build.val and reviewed.test == build.test
    assert all(row['reviewed'] for row in reviewed.train)
    assert reviewed.manifest['augmentation']['semantic_review']=='complete'

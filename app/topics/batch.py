"""Version-locked local classification; validate the whole batch before writing."""
from uuid import uuid4
from app.core.taxonomy import TOPIC_NAMES
from app.db.topics import list_unclassified, write_classifications, record_run
from app.topics.client import health_metadata, classify_batch, validate_response


async def classify_pool(*, limit=500, min_batch=10, force=False, reclassify=False) -> dict:
    if type(min_batch) is not int or min_batch < 1:
        raise ValueError('min_batch must be positive')
    rows = await list_unclassified(limit, reclassify=reclassify)
    report = {'run_id': uuid4().hex, 'status': 'empty', 'pending_count': len(rows),
              'written': 0, 'failed_count': 0, 'question_count': 0,
              'class_counts': dict.fromkeys(TOPIC_NAMES, 0), 'reclassify': reclassify}
    if not rows:
        await record_run(report)
        return report
    if len(rows) < min_batch and not force:
        report['status'] = 'below_batch'
        await record_run(report)
        return report
    report['status'] = 'running'
    await record_run(report)
    try:
        meta = await health_metadata()
        report.update(meta)
        predictions = []
        for start in range(0, len(rows), 100):
            chunk = rows[start:start+100]
            response = await classify_batch([row['text'] for row in chunk], meta)
            validate_response(response, len(chunk), meta)
            predictions.extend(response['results'])
        prepared = [{**row, 'labels': result['labels']} for row, result in zip(rows, predictions, strict=True)]
        report.update(await write_classifications(prepared, meta, report['run_id'], reclassify))
        report['counts_scope'] = 'written_in_this_run'
        if not report['written']:
            report['attempted_model_version'] = report.pop('model_version')
    except Exception as exc:
        report.update(status='failed', failed_count=len(rows), reason=type(exc).__name__)
        await record_run(report)
    return report

"""Topic result authority. Lock source rows to serialize concurrent writers."""
import hashlib
from datetime import datetime, timezone
from sqlalchemy import select, func
import app.db.base as db
from app.db.models import LowConfidenceQuestion, TopicClassification, TopicClassificationRun, CanonicalOccurrence, CanonicalQuestion
from app.core.taxonomy import TOPIC_NAMES
from scripts.ch10.corpus_lib import desensitize
from app.topics.client import validate_metadata


async def list_unclassified(limit: int, *, reclassify: bool = False) -> list[dict]:
    if not 1 <= limit <= 10000:
        raise ValueError('limit must be 1..10000')
    stmt = select(LowConfidenceQuestion).order_by(LowConfidenceQuestion.id).limit(limit)
    if not reclassify:
        stmt = stmt.where(~select(TopicClassification.id).where(
            TopicClassification.question_id == LowConfidenceQuestion.id).exists())
    async with db.async_session() as session:
        return [{'id': row.id, 'text': desensitize(row.raw_question), 'source': row.source}
                for row in (await session.scalars(stmt)).all()]


async def write_classifications(rows: list[dict], meta: dict, run_id: str, reclassify: bool = False) -> dict:
    validate_metadata(meta)
    ids = [row['id'] for row in rows]
    if len(ids) != len(set(ids)) or any(type(value) is not int or value <= 0 for value in ids):
        raise ValueError('invalid or duplicate question IDs')
    if not run_id or len(run_id) > 64:
        raise ValueError('invalid run ID')
    for row in rows:
        labels = row.get('labels')
        if not labels or not isinstance(labels, list) or any(label not in TOPIC_NAMES for label in labels):
            raise ValueError('invalid topic labels')
        if desensitize(row['text']) != row['text']:
            raise ValueError('unmasked classification input')
    written, skipped, previous, skipped_versions = 0, 0, set(), set()
    counts = dict.fromkeys(TOPIC_NAMES, 0)
    async with db.async_session.begin() as session:
        sources = (await session.scalars(select(LowConfidenceQuestion).where(
            LowConfidenceQuestion.id.in_(ids)).order_by(LowConfidenceQuestion.id).with_for_update())).all()
        source_map = {row.id: row for row in sources}
        if set(source_map) != set(ids):
            raise ValueError('classification source no longer exists')
        # Current locking read sees prior concurrent commit even under MySQL REPEATABLE READ.
        existing = {row.question_id: row for row in (await session.scalars(select(TopicClassification).where(
            TopicClassification.question_id.in_(ids)).with_for_update())).all()}
        for row in rows:
            if desensitize(source_map[row['id']].raw_question) != row['text']:
                raise ValueError('classification source changed')
            old = existing.get(row['id'])
            if old is not None and not reclassify:
                skipped += 1
                skipped_versions.add(old.model_version)
                continue
            values = {**meta, 'labels': list(dict.fromkeys(row['labels'])),
                      'input_hash': hashlib.sha256(row['text'].encode('utf-8')).hexdigest(),
                      'run_id': run_id, 'classified_at': datetime.now(timezone.utc).replace(tzinfo=None)}
            if old is None:
                session.add(TopicClassification(question_id=row['id'], **values))
            else:
                previous.add(old.model_version)
                for key, value in values.items():
                    setattr(old, key, value)
            written += 1
            for label in values['labels']:
                counts[label] += 1
        status = 'empty' if not written else 'done'
        result = {'written': written, 'skipped': skipped, 'previous_versions': sorted(previous),
                  'skipped_versions': sorted(skipped_versions), 'class_counts': counts,
                  'question_count': written, 'status': status}
        # Audit commit shares the classification transaction, so neither can survive alone.
        run = await session.scalar(select(TopicClassificationRun).where(TopicClassificationRun.run_id == run_id).with_for_update())
        if run is None:
            run = TopicClassificationRun(run_id=run_id, pending_count=len(rows), success_count=0, failed_count=0, status='running')
            session.add(run)
        run.model_version = meta['model_version'] if written else None
        run.success_count = written
        run.failed_count = 0
        run.status = status
        run.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
    return result


async def record_run(report: dict, report_path: str | None = None):
    async with db.async_session.begin() as session:
        run = await session.get(TopicClassificationRun, report['run_id'])
        if run is None:
            run = TopicClassificationRun(run_id=report['run_id'])
            session.add(run)
        for key, value in {'model_version':report.get('model_version'), 'pending_count':report['pending_count'],
                           'success_count':report.get('written',0), 'failed_count':report.get('failed_count',0),
                           'status':report['status'], 'report_path':report_path,
                           'finished_at':None if report['status']=='running' else datetime.now(timezone.utc).replace(tzinfo=None)}.items():
            setattr(run, key, value)


async def topic_distribution() -> dict:
    async with db.async_session() as session:
        # Aggregate in SQL; no raw question text is loaded for the chart.
        counts = {name: int(await session.scalar(select(func.count()).select_from(TopicClassification).where(
            func.JSON_CONTAINS(TopicClassification.labels, func.JSON_QUOTE(name)) == 1))) for name in TOPIC_NAMES}
        classified = int(await session.scalar(select(func.count()).select_from(TopicClassification)))
        total = int(await session.scalar(select(func.count()).select_from(LowConfidenceQuestion)))
        versions = (await session.execute(select(TopicClassification.model_version, func.count()).group_by(
            TopicClassification.model_version))).all()
    return {'question_count': classified, 'unclassified_count': total-classified, 'class_counts': counts,
            'versions': [{'model_version': version, 'question_count': count} for version, count in versions]}


async def topic_questions(label: str, offset: int, limit: int) -> dict:
    if label not in TOPIC_NAMES or offset < 0 or not 1 <= limit <= 100:
        raise ValueError('invalid topic pagination')
    condition = func.JSON_CONTAINS(TopicClassification.labels, func.JSON_QUOTE(label)) == 1
    async with db.async_session() as session:
        total = int(await session.scalar(select(func.count()).select_from(TopicClassification).where(condition)))
        stmt = (select(TopicClassification, LowConfidenceQuestion, CanonicalQuestion)
                .join(LowConfidenceQuestion, TopicClassification.question_id == LowConfidenceQuestion.id)
                .outerjoin(CanonicalOccurrence, CanonicalOccurrence.raw_question_id == LowConfidenceQuestion.id)
                .outerjoin(CanonicalQuestion, CanonicalQuestion.id == CanonicalOccurrence.canonical_id)
                .where(condition).order_by(TopicClassification.classified_at.desc(), TopicClassification.id.desc())
                .offset(offset).limit(limit))
        items = [{'id': raw.id, 'text': desensitize(raw.raw_question), 'source': raw.source,
                  'labels': classified.labels, 'model_version': classified.model_version,
                  'classified_at': classified.classified_at.isoformat(),
                  'review': {'id': canonical.id, 'status': canonical.status} if canonical else None}
                 for classified, raw, canonical in (await session.execute(stmt)).all()]
    return {'label': label, 'total': total, 'offset': offset, 'limit': limit, 'items': items}

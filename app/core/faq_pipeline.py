"""Shared FAQ retrieval; tool and Graph paths select explicit evidence gates."""

import hashlib
from pathlib import Path

from app.config import get_settings
from app.core import query_understanding, retrieval, selfcheck
from app.core.evidence_confidence import (
    SIGNAL_VERSION, compute_evidence_confidence, load_calibration, retrieval_fingerprint,
)


CALIBRATION_DATASET = Path(__file__).resolve().parents[2] / 'tests/data/eval_ch04.jsonl'


async def prepare_retrieval(keyword: str) -> tuple[str, str]:
    understood = await query_understanding.understand(keyword)
    standard, expanded = understood['standard'], understood['expanded']
    return standard, standard + (' ' + ' '.join(expanded) if expanded else '')


def active_calibration(settings) -> dict:
    try:
        dataset_hash = hashlib.sha256(CALIBRATION_DATASET.read_bytes()).hexdigest()
    except OSError:
        return {'status': 'pending_calibration', 'threshold': None, 'in_use': False}
    return load_calibration(Path(settings.evidence_calibration_path), {
        'dataset_hash': dataset_hash, 'strategy': 'hybrid_rerank',
        'signal_version': SIGNAL_VERSION, 'retrieval_config_hash': retrieval_fingerprint(settings),
    })


async def run_faq_pipeline(keyword: str, category: str | None = None, *,
                           gate: str = 'top1', settings=None) -> dict:
    if gate not in {'top1', 'calibrated'}:
        raise ValueError('unsupported FAQ gate')
    settings = settings or get_settings()
    if not settings.milvus_uri.startswith(('http://', 'https://')):
        hits = await retrieval.search_knowledge(keyword)
        if not hits:
            return {'hits': [], 'message': f'未找到与「{keyword}」相关的常见问题'}
        return {'hits': [{'question': hit['question'], 'answer': hit['answer']} for hit in hits]}

    standard, search_query = await prepare_retrieval(keyword)
    calibration = active_calibration(settings) if gate == 'calibrated' else None
    trace = {}

    def passed(hits):
        confidence = compute_evidence_confidence(hits)
        trace['confidence'] = confidence
        if calibration is not None:
            trace['calibration'] = calibration
        if not hits:
            return False
        if calibration and calibration['status'] == 'ready':
            return confidence['score'] >= calibration['threshold']
        return confidence['signals']['top1_score'] >= settings.rerank_min_score

    hits = await retrieval.search_knowledge(
        standard, strategy='hybrid_rerank', category=category, bm25_query=search_query,
    )
    accepted = passed(hits)
    if not accepted and category:
        # Keep standard question/model constraints; only remove the semantic category.
        hits = await retrieval.search_knowledge(
            standard, strategy='hybrid_rerank', category=None, bm25_query=search_query,
        )
        trace['category_retry'] = True
        accepted = passed(hits)

    def result(payload):
        # Preserve the existing tool response schema; Graph receives diagnostics.
        return {**payload, 'trace': trace} if gate == 'calibrated' else payload

    if not accepted:
        return result({'sufficient': False, 'source': 'retrieval_low_conf',
                       'reason': f"检索证据不足(top={trace['confidence']['signals']['top1_score']:.3f})",
                       'citations': []})
    answer_hits = hits[:3]
    check = await selfcheck.check_sufficient(
        standard, [f"{hit['question']} {hit['answer']}" for hit in answer_hits],
    )
    if not check['useful']:
        return result({'sufficient': False, 'source': 'self_check',
                       'reason': check['reason'], 'citations': []})
    citations = [
        {'n': index, 'id': hit['id'], 'section_path': hit['section_path'],
         'question': hit['question'], 'answer': hit['answer'], 'content_type': hit['content_type']}
        for index, hit in enumerate(retrieval.arrange_head_tail(answer_hits), 1)
    ]
    evidence = '\n'.join(f"[{item['n']}] {item['question']}: {item['answer']}" for item in citations)
    return result({'sufficient': True, 'evidence': evidence, 'citations': citations})

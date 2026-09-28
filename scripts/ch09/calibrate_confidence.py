"""Replay a fixed score set offline; retrieval is opt-in via --live."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from app.core.evidence_confidence import (
    SIGNAL_VERSION, calibrate, compute_evidence_confidence, retrieval_fingerprint,
)


async def live_scores(samples):
    from app.core.retrieval import search_knowledge
    from app.core.faq_pipeline import prepare_retrieval
    rows = []
    for sample in samples:
        standard, lexical = await prepare_retrieval(sample['query'])
        hits = await search_knowledge(standard, strategy='hybrid_rerank', category=None, bm25_query=lexical)
        rows.append({'id': sample['id'], 'bucket': sample['bucket'], **compute_evidence_confidence(hits)})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--scores-file', type=Path)
    source.add_argument('--live', action='store_true', help='uses query rewrite model, embeddings and reranker')
    parser.add_argument('--dataset', type=Path, help='original labeled dataset; required to bind offline scores')
    parser.add_argument('--retrieval-config-hash', default='offline-unbound')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    dataset = args.dataset or (Path('tests/data/eval_ch04.jsonl') if args.live else None)
    source_path = dataset if args.live else args.scores_file
    content = source_path.read_bytes()
    rows = [json.loads(line) for line in content.decode('utf-8-sig').splitlines() if line.strip()]
    config_hash = args.retrieval_config_hash
    if args.live:
        from app.config import get_settings
        config_hash = retrieval_fingerprint(get_settings())
        rows = asyncio.run(live_scores(rows))
    elif dataset is not None:
        samples = [json.loads(line) for line in dataset.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        labels = {sample['id']: sample['bucket'] for sample in samples}
        if len(rows) != len(labels) or {row.get('id') for row in rows} != set(labels):
            raise ValueError('bound scores must contain each dataset ID exactly once')
        if any(row['bucket'] != labels[row['id']] for row in rows):
            raise ValueError('score labels do not match dataset')
    report = calibrate(rows)
    report['meta'] = {'dataset_hash': hashlib.sha256(dataset.read_bytes()).hexdigest() if dataset else None,
                      'scores_hash': hashlib.sha256(content).hexdigest() if not args.live else None,
                      'strategy': 'hybrid_rerank', 'signal_version': SIGNAL_VERSION,
                      'retrieval_config_hash': config_hash, 'source': str(source_path)}
    report['scores'] = rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'recommended': report['recommended'], 'in_use': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

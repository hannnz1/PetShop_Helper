"""Compare actual local HTTP predictions with the verified bundle on held-out texts."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from app.topics.client import health_metadata, classify_batch
from scripts.ch10.inference_lib import load_runtime
from scripts.ch10.corpus_lib import desensitize


async def run(model: Path, test: Path, output: Path) -> dict:
    report = {'status': 'pending_compute', 'reason': 'verified model bundle missing'}
    if all((model/name).exists() for name in ('model.onnx', 'threshold.json', 'tokenizer.json', 'export_report.json')):
        try:
            runtime = load_runtime(model)
            if not test.exists():
                report = {'status': 'pending_data', 'reason': 'held-out test missing'}
            else:
                rows = [json.loads(line) for line in test.read_text(encoding='utf-8').splitlines() if line.strip()]
                if not rows: raise ValueError('empty test set')
                meta = await health_metadata()
                if meta != runtime.metadata: raise ValueError('service version differs from bundle')
                mismatches = 0
                for start in range(0, len(rows), 100):
                    texts = [desensitize(row['text']) for row in rows[start:start+100]]
                    remote = (await classify_batch(texts, meta))['results']
                    local = runtime.classify(texts)
                    mismatches += sum(set(a['labels']) != set(b['labels']) for a, b in zip(remote, local, strict=True))
                report = {'status': 'passed' if mismatches == 0 else 'failed', 'checked': len(rows),
                          'mismatches': mismatches, 'metadata': {**meta, 'test_hash': hashlib.sha256(test.read_bytes()).hexdigest()}}
        except Exception as exc:
            report = {'status': 'failed', 'reason': type(exc).__name__}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('data/ch10/onnx'))
    parser.add_argument('--test', type=Path, default=Path('data/ch10/dataset/test.jsonl'))
    parser.add_argument('--out', type=Path, default=Path('data/ch10/service_smoke.json'))
    args = parser.parse_args()
    report = asyncio.run(run(args.model, args.test, args.out))
    print('status='+report['status'])
    if report['status'] != 'passed': raise SystemExit(1)

if __name__ == '__main__': main()

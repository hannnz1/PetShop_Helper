"""Run annotated Ch06 paths offline, with explicit upstream boundaries."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "tests" / "data" / "ch06_samples.jsonl"
REPORT = ROOT / "data" / "ch06" / "reports" / "offline_eval.json"


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_labels():
    coref = load_jsonl(ROOT / "tests" / "data" / "ch06_coref_samples.jsonl")
    policy = load_jsonl(ROOT / "tests" / "data" / "ch06_policy_queries.jsonl")
    assert len(coref) == 5 and len(policy) == 3
    for item in coref:
        assert item["query"] and item["expected"]
        if item["kind"] in {"complete", "no_history"}:
            assert item["query"] == item["expected"]
    for item in policy:
        assert item["seed"] and set(item["must_keep"]).isdisjoint(item["must_not_add"])
        assert all(token in item["seed"] for token in item["must_keep"])
    return {"coreference_samples": len(coref), "policy_samples": len(policy),
            "live_semantics": "pending_upstream"}


def main():
    samples = load_jsonl(SAMPLES)
    assert len(samples) == 4 and len({item["id"] for item in samples}) == 4
    labels = validate_labels()
    paths = []
    for sample in samples:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", sample["test"]],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
            check=False,
        )
        path = {key: sample[key] for key in ("id", "path", "expected", "test")}
        path["status"] = sample["status_if_pass"] if result.returncode == 0 else "failed_offline"
        path["result"] = next((line.strip() for line in result.stdout.splitlines()
                               if " passed" in line or " failed" in line), result.stdout[-400:])
        paths.append(path)
        print(f"{sample['id']}: {path['status']} ({path['result']})")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Ch06 deterministic offline paths; no paid model or embedding calls",
        "paths": paths, "annotated_prompts": labels,
        "upstream": {"status": "pending_upstream", "reason": "glm-5.2 balance exhausted; user chose not to recharge"},
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"report: {REPORT}")
    return 0 if all(item["status"] == "passed_offline" for item in paths) else 1


if __name__ == "__main__":
    raise SystemExit(main())

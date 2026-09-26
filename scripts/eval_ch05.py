"""Run the five offline Chapter 5 path checks and write an honest report."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "knowledge_strong_weak": "tests/graph/test_knowledge_route.py",
    "logistics_multistep": "tests/graph/test_react.py::test_second_tool_decision_sees_first_result_and_final_answer_is_separate",
    "chitchat": "tests/graph/test_eval_ch05.py::test_fixed_graph_paths_do_not_call_model_or_create_ticket[chitchat]",
    "complaint": "tests/graph/test_eval_ch05.py::test_fixed_graph_paths_do_not_call_model_or_create_ticket[complaint]",
    "cross_turn": "tests/test_agent_api_integration.py::test_chat_and_agent_http_use_real_core_and_mysql",
}


def main() -> None:
    results = {}
    for name, selector in CASES.items():
        run = subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest", "-q", selector],
                             cwd=ROOT, capture_output=True, text=True, check=False)
        results[name] = {"status": "passed_offline" if run.returncode == 0 else "failed",
                         "test": selector, "exit_code": run.returncode}
        print(f"{name}: {results[name]['status']}")
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "real_model_status": "pending_upstream", "paths": results}
    out = ROOT / "data" / "ch05" / "reports" / "offline_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report: {out}")
    if any(item["status"] == "failed" for item in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""A current evaluation's hallucination rate must not inherit old ledger rows."""

from app.api import rageval


def test_current_run_and_cumulative_ledger_have_separate_numerators():
    report = {
        "meta": {"question_count": 300},
        "retrieval": {"hybrid_rerank": {
            "A_policy": {"count": 60}, "B_model": {"count": 60},
            "C_colloquial": {"count": 60}, "E_multi": {"count": 60},
        }},
        "generation": {"refusal_rate": 1.0,
                       "faithfulness_cases": [{"id": "A30"}]},
    }
    counts = {"未解决": 1, "已解决": 2, "无需解决": 0}
    statuses = {"A30": "未解决", "C2": "已解决", "E15": "已解决"}
    result = rageval._hallucination(report, counts, statuses)
    assert result["evaluated"] == 300
    assert result["cases_judged"] == 1
    assert result["judged"] == 1
    assert result["judged_rate"] == round(1 / 300, 4)
    assert result["ledger"] == {"total": 3, **counts}


def test_refusal_misses_count_as_confirmed_and_missing_report_has_no_rate():
    report = {
        "meta": {"question_count": 120},
        "retrieval": {"hybrid_rerank": {"A_policy": {"count": 60}}},
        "generation": {"refusal_rate": 57 / 60,
                       "faithfulness_cases": [{"id": "A1"}]},
    }
    result = rageval._hallucination(report, {"未解决": 0, "已解决": 1, "无需解决": 0},
                                    {"A1": "已解决"})
    assert result["refusal_missed"] == 3
    assert result["judged"] == 4
    assert result["confirmed"] == 4
    assert result["confirmed_rate"] == round(4 / 120, 4)
    missing = rageval._hallucination(None, {"未解决": 1, "已解决": 0, "无需解决": 0}, {})
    assert missing["evaluated"] is None and missing["judged_rate"] is None
    assert missing["ledger"]["total"] == 1

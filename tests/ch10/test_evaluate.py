from app.core.taxonomy import NUM_CLASSES


def test_metrics_and_strict_threshold_redlines():
    from scripts.ch10.evaluate import evaluate_matrix

    gold = [[1] * NUM_CLASSES, [0] * NUM_CLASSES]
    same = [row[:] for row in gold]
    report = evaluate_matrix(gold, same)
    assert report["micro_f1"] == 1.0
    assert report["status"] == "passed"
    empty = [[0] * NUM_CLASSES for _ in gold]
    assert evaluate_matrix(gold, empty)["status"] == "failed"


def test_replay_uses_strict_better_first_tie():
    from scripts.ch10.scan_threshold_replay import replay
    from app.core.taxonomy import TOPIC_NAMES

    rows = [{"labels": [TOPIC_NAMES[0]]}]
    scores = [{name: (0.9 if i == 0 else 0.1)
               for i, name in enumerate(TOPIC_NAMES)}]
    assert replay(rows, scores, 0.30)["status"] == "passed"
    assert replay(rows, scores, 0.50)["status"] == "failed"


def test_pool_report_refuses_duplicate_ids():
    import pytest
    from scripts.ch10.classify_pool import categorize_pool

    with pytest.raises(ValueError, match="duplicate"):
        categorize_pool([{"id": 1}, {"id": 1}],
                        [{"labels": ["物流"]}, {"labels": ["物流"]}])

"""The evaluator's pass labels must reflect assembled context, not only estimates."""

from scripts import eval_ch07


def test_default_case_reports_twenty_retained_original_turns():
    details = eval_ch07.default_twenty_turns()
    assert details["retained_original_turns"] == 20
    assert details["summary_segments"] == 0


def test_handcrafted_snapshot_case_is_labeled_as_rendering_only():
    details = eval_ch07.post_cascade_rendering()
    assert details["scope"] == "post-cascade model rendering from a synthetic snapshot"

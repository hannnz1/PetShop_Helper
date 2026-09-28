import json

from app.core.taxonomy import NUM_CLASSES, TOPIC_NAMES, SEVERITY


def test_report_matches_hand_calculated_confusion():
    from scripts.ch10.evaluate import evaluate_matrix
    gold = [[value] * NUM_CLASSES for value in (1, 1, 0, 0)]
    prediction = [[value] * NUM_CLASSES for value in (1, 0, 1, 0)]
    report = evaluate_matrix(gold, prediction)
    assert report['micro_precision'] == report['micro_recall'] == .5
    assert report['macro_precision'] == report['macro_recall'] == .5
    for name, row in report['per_class'].items():
        assert (row['tn'], row['fp'], row['fn'], row['tp']) == (1, 1, 1, 1)
        assert row['precision'] == row['recall'] == row['f1'] == .5
        assert row['passed'] is (None if SEVERITY[name] == '宽' else False)


def test_error_samples_and_written_reports_share_masked_content(tmp_path):
    from scripts.ch10.evaluate import build_error_samples, write_evaluation
    rows = [{'text': '请联系13800138000', 'labels': [TOPIC_NAMES[0]], 'review_id': 'pool-1'}]
    errors = build_error_samples(rows, [{'labels': [TOPIC_NAMES[1]]}])
    assert errors[0]['missed'] == [TOPIC_NAMES[0]]
    assert errors[0]['extra'] == [TOPIC_NAMES[1]]
    assert errors[0]['kind'] == 'mixed'
    assert '13800138000' not in errors[0]['text']
    report = {'status': 'failed', 'metadata': {'test_hash': 'abc'}, 'errors': errors,
              'micro_f1': .5, 'macro_f1': .5, 'per_class': {}}
    target = tmp_path/'evaluation.json'
    write_evaluation(report, target)
    assert json.loads(target.read_text(encoding='utf-8')) == report
    markdown = target.with_suffix('.md').read_text(encoding='utf-8')
    assert 'abc' in markdown and '0.5' in markdown
    assert '13800138000' not in markdown


def test_no_support_never_passes_critical_class():
    from scripts.ch10.evaluate import evaluate_matrix
    report = evaluate_matrix([[0]*NUM_CLASSES], [[0]*NUM_CLASSES])
    for name, row in report['per_class'].items():
        if SEVERITY[name] != '宽':
            assert row['passed'] is False

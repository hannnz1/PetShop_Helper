import pytest

from app.core.evidence_confidence import compute_evidence_confidence


def test_four_signal_score():
    result = compute_evidence_confidence([{'rerank_score': .8}, {'rerank_score': .6}])
    assert result['score'] == .5733
    assert result['signals']['valid_count'] == 2


def test_empty_and_dense_scores_are_not_evidence():
    assert compute_evidence_confidence([])['score'] == 0
    with pytest.raises(ValueError):
        compute_evidence_confidence([{'score': .99}])


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -float('inf'), True])
def test_nonfinite_or_boolean_scores_rejected(score):
    with pytest.raises(ValueError):
        compute_evidence_confidence([{'rerank_score': score}])


def test_keyword_only_considers_top_three():
    hits = [{'rerank_score': .5} for _ in range(4)]
    hits[3]['answer'] = '不支持退货'
    assert not compute_evidence_confidence(hits)['signals']['key_clause_hit']
    hits[0]['answer'] = '不支持退货'
    assert compute_evidence_confidence(hits)['signals']['key_clause_hit']

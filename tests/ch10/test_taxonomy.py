from app.core.taxonomy import (
    ID2LABEL, LABEL2ID, NUM_CLASSES, SEVERITY, TOPIC_CLASSES, TOPIC_NAMES,
    terminology_table,
)


def test_single_taxonomy_has_original_seventeen_ordered_classes():
    assert NUM_CLASSES == 17
    assert TOPIC_NAMES[:5] == ("退换货", "物流", "尺码", "发票", "质量问题")
    assert TOPIC_NAMES[-1] == "其他"
    assert len(set(TOPIC_NAMES)) == NUM_CLASSES
    assert all(LABEL2ID[name] == index and ID2LABEL[index] == name
               for index, name in enumerate(TOPIC_NAMES))
    assert SEVERITY["退换货"] == "严" and SEVERITY["评价"] == "宽"
    assert all(topic.boundary and topic.examples for topic in TOPIC_CLASSES)
    table = terminology_table()
    assert all(topic.name in table and topic.boundary in table for topic in TOPIC_CLASSES)


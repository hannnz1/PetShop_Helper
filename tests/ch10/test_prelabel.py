import pytest


class FakeModel:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.prompts = []

    def with_structured_output(self, schema, *, method):
        self.schema = schema
        return self

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.schema(labels=self.response)


@pytest.mark.asyncio
async def test_prelabel_masks_question_and_accepts_multilabel():
    from scripts.ch10.prelabel import prelabel_one

    model = FakeModel(["尺码", "退换货"])
    result = await prelabel_one("订单202601180001234567 买大了想退", model)
    assert result.status == "labeled" and result.labels == ("尺码", "退换货")
    assert "202601180001234567" not in str(model.prompts[0])


@pytest.mark.asyncio
async def test_prelabel_failure_and_invalid_label_are_not_other():
    from scripts.ch10.prelabel import prelabel_one

    broken = await prelabel_one("猫粮怎么买", FakeModel(error=RuntimeError("offline")))
    invalid = await prelabel_one("猫粮怎么买", FakeModel(["虚构类别"]))
    assert broken.status == "failed" and broken.labels == ()
    assert invalid.status == "invalid_labels" and invalid.labels == ()


def test_golden_exact_set_gate():
    from scripts.ch10.validate_golden import validate_golden

    rows = [{"text": "买大了想退", "labels": ["尺码", "退换货"]},
            {"text": "退款", "labels": ["退换货"]}]
    assert validate_golden(rows, [("尺码", "退换货"), ("退换货",)]).passed
    assert not validate_golden(rows, [("尺码",), ("退换货",)]).passed


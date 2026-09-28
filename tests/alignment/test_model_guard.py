import pytest

from app.core.model_guard import unsupported_models, repair_once


def test_unsupported_model_is_mechanical():
    assert unsupported_models('MH-CAD1', 'MH-CAM1') == ['MH-CAD1']
    assert unsupported_models('MH-cam1 mh-CAM1', 'MH-CAM1') == ['MH-cam1', 'mh-CAM1']
    assert unsupported_models('MH-CAM1 MH-CAM1', 'MH-CAM1') == []
    assert unsupported_models('正常描述和订单 1001', '') == []


@pytest.mark.asyncio
async def test_failed_repair_does_not_retry():
    calls = []

    async def repair(hint):
        calls.append(hint)
        return 'MH-CAD1 还是错的'

    result = await repair_once('MH-CAD1', 'MH-CAM1', repair)
    assert len(calls) == 1
    assert result['guard_passed'] is False
    assert result['original_answer'] == 'MH-CAD1'
    assert result['repair_status'] == 'failed'


@pytest.mark.asyncio
async def test_successful_repair_and_unavailable_result():
    async def repaired(hint):
        return 'MH-CAM1'

    assert (await repair_once('MH-CAD1', 'MH-CAM1', repaired))['guard_passed'] is True

    async def unavailable(hint):
        return None

    result = await repair_once('MH-CAD1', 'MH-CAM1', unavailable)
    assert result['answer'] == 'MH-CAD1'
    assert result['repair_status'] == 'unavailable'

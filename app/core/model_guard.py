"""Exact model-name grounding for evaluation answers, with one bounded repair."""

import re

GUARD_VERSION = 'exact-model-v1'
_MODEL = re.compile(r'(?<![A-Za-z0-9])MH-[A-Za-z]{1,4}\d{1,4}(?![A-Za-z0-9])', re.IGNORECASE)


def unsupported_models(answer: str, evidence: str) -> list[str]:
    supported = set(_MODEL.findall(evidence))
    return list(dict.fromkeys(model for model in _MODEL.findall(answer) if model not in supported))


def repair_hint(bad: list[str]) -> str:
    return ('重新回答原问题。上一版出现了证据不支持的型号：' + '、'.join(bad)
            + '。只使用证据中逐字出现的型号（大小写也必须一致），不要猜测或替换成相似型号；'
              '证据不足时明确说明无法回答。')


async def repair_once(answer: str, evidence: str, repair) -> dict:
    bad = unsupported_models(answer, evidence)
    result = {'original_answer': answer, 'answer': answer, 'unsupported_models': bad,
              'repair_status': 'not_needed', 'guard_passed': not bad, 'guard_version': GUARD_VERSION}
    if not bad:
        return result
    repaired = await repair(repair_hint(bad))
    if repaired is None:
        result['repair_status'] = 'unavailable'
        return result
    remaining = unsupported_models(repaired, evidence)
    result.update(answer=repaired, remaining_unsupported_models=remaining,
                  guard_passed=not remaining, repair_status='failed' if remaining else 'repaired')
    return result

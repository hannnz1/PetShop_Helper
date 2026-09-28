"""Deterministic explanations tied to the exact displayed report."""

import hashlib
import json


def build_read_note(kind: str, report: dict) -> dict:
    identity = hashlib.sha256(json.dumps([kind, report], sort_keys=True, ensure_ascii=False,
                                         default=str, allow_nan=False).encode()).hexdigest()
    if kind == 'usage':
        text = (f"共 {report.get('calls', 0)} 次调用，其中 {report.get('available_calls', 0)} 次有用量数据，"
                f"{report.get('unavailable_calls', 0)} 次缺失。均值仅按有用量的调用计算；未配置价格，不估算金额。")
    elif kind == 'calibration':
        text = ('校准已匹配当前数据与配置。' if report.get('status') == 'ready' else
                '当前没有匹配的数据集和配置校准，继续使用原有 Top-1 阈值。')
    else:
        text = '仅比较相同数据集、策略和指标分母的已完成批次；部分结果不作为达标结论。'
    return {'report_id': identity, 'text': text}


def matching_read_note(kind: str, report: dict, note: dict) -> str | None:
    return note.get('text') if note.get('report_id') == build_read_note(kind, report)['report_id'] else None

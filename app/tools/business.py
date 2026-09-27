"""Mock business lookups and semantic FAQ lookup."""

import random
import hashlib
from typing import Annotated, Literal

from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core import query_understanding, retrieval, selfcheck
from app.db import repository


class OrderInput(BaseModel):
    order_id: str = Field(description="订单号，例如 1001")


class ProductInput(BaseModel):
    product_name: str = Field(description="商品名称或关键词，例如 猫粮")


class LogisticsInput(BaseModel):
    order_id: str = Field(description="订单号，用于查询该订单的物流轨迹")


class FaqInput(BaseModel):
    keyword: str = Field(description="完整的知识库检索问题；保留用户提到的商品型号（如 MH-W40）、规格与政策条件，不要只传宽泛关键词")
    category: str | None = Field(default=None, description="可选知识品类过滤")


_LOGISTICS_PHASES = ("已揽件", "运输中", "派送中", "已签收")
_PHASE_LOCATIONS = ("揽收网点", "分拨中心", "派送站", "收货地址")


def _mock_order_state(order_id: str) -> tuple[int, str]:
    """Choose one shared shipment phase and city for both order lookups."""

    rng = random.Random(f"order:{order_id}")
    return rng.randrange(len(_LOGISTICS_PHASES)), rng.choice(
        ["深圳", "广州", "杭州", "上海", "成都"]
    )


@tool(args_schema=OrderInput)
async def query_order(order_id: str) -> dict:
    """查询演示订单信息和物流单号；查轨迹时把 tracking_no 交给 query_logistics。"""

    phase, _ = _mock_order_state(order_id)
    rng = random.Random(f"order:{order_id}")
    return {
        "order_id": order_id,
        "status": "已签收" if phase == 3 else "已发货",
        "amount": rng.randint(50, 2000),
        "created_at": f"2026-07-{rng.randint(1, 12):02d} 10:00",
        "product": rng.choice(["智能猫砂盆", "猫粮 5kg", "猫爬架", "自动饮水机"]),
        "tracking_no": "SF" + str(int.from_bytes(
            hashlib.sha256(order_id.encode("utf-8")).digest()[:6], "big") % 10**12).zfill(12),
    }


@tool(args_schema=ProductInput)
async def query_product(product_name: str) -> dict:
    """查询商品价格、库存和规格。用户咨询某个商品时使用。"""

    rng = random.Random(f"product:{product_name}")
    return {
        "product_name": product_name,
        "price": rng.randint(20, 999),
        "stock": rng.randint(0, 500),
        "spec": rng.choice(["标准装", "家庭装", "试用装"]),
    }


@tool(args_schema=LogisticsInput)
async def query_logistics(order_id: str) -> dict:
    """查询订单物流状态、当前位置和轨迹。用户询问快递进度时使用。"""

    phase, city = _mock_order_state(order_id)
    location = f"{city}{_PHASE_LOCATIONS[phase]}"
    return {
        "order_id": order_id,
        "status": _LOGISTICS_PHASES[phase],
        "location": location,
        "timeline": [
            f"{city}{_PHASE_LOCATIONS[index]} {_LOGISTICS_PHASES[index]}"
            for index in range(phase + 1)
        ],
    }


@tool(args_schema=FaqInput)
async def query_faq(keyword: str, category: str | None = None) -> dict:
    """查询常见问题、政策及商品手册/型号规格知识库；型号问题须在 keyword 中保留完整型号。"""

    settings = get_settings()
    if not settings.milvus_uri.startswith(("http://", "https://")):
        hits = await retrieval.search_knowledge(keyword)
        if not hits:
            return {"hits": [], "message": f"未找到与「{keyword}」相关的常见问题"}
        return {"hits": [{"question": hit["question"], "answer": hit["answer"]} for hit in hits]}

    understood = await query_understanding.understand(keyword)
    standard = understood["standard"]
    expanded = understood["expanded"]
    search_query = standard + (" " + " ".join(expanded) if expanded else "")
    hits = await retrieval.search_knowledge(
        standard, strategy="hybrid_rerank", category=category, bm25_query=search_query,
    )
    top_score = hits[0]["rerank_score"] if hits else 0.0
    if not hits or top_score < settings.rerank_min_score:
        return {
            "sufficient": False, "source": "retrieval_low_conf",
            "reason": f"检索证据不足(top={top_score:.3f})", "citations": [],
        }

    # Retrieval evaluates Top-10, but a 2k-token chat turn cannot carry ten
    # full chunks (including duplicated citation metadata) to the answer model.
    answer_hits = hits[:3]
    evidence_texts = [f"{hit['question']} {hit['answer']}" for hit in answer_hits]
    check = await selfcheck.check_sufficient(standard, evidence_texts)
    if not check["useful"]:
        return {
            "sufficient": False, "source": "self_check",
            "reason": check["reason"], "citations": [],
        }

    arranged = retrieval.arrange_head_tail(answer_hits)
    citations = [
        {
            "n": index, "id": hit["id"], "section_path": hit["section_path"],
            "question": hit["question"], "answer": hit["answer"],
            "content_type": hit["content_type"],
        }
        for index, hit in enumerate(arranged, 1)
    ]
    evidence = "\n".join(
        f"[{item['n']}] {item['question']}: {item['answer']}" for item in citations
    )
    return {"sufficient": True, "evidence": evidence, "citations": citations}


@tool
async def create_ticket(
    description: str,
    ticket_type: Literal["售后", "投诉", "咨询"],
    conversation_id: Annotated[int, InjectedToolArg],
) -> dict:
    """用户明确要求人工、提出投诉或问题无法自助解决时，创建人工工单。

    description 填写用户问题，ticket_type 从售后、投诉、咨询中选择。
    """

    ticket_no = await repository.create_ticket(conversation_id, description, ticket_type)
    return {"ticket_no": ticket_no, "status": "已转人工"}

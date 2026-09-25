"""Extract reusable customer-service QA from historical conversations."""

import hashlib
import json
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.llm import get_chat_model
from app.core.prompts import MINING_PROMPT
from app.db import repository


class QaPair(BaseModel):
    question: str = Field(description="通用问法，不含具体订单号或个人信息")
    answer: str = Field(description="仅保留客服原答明确支持的通用事实及限定条件")
    source_index: int | None = Field(default=None, description="输入中从 1 开始的来源会话编号")
    source_quote: str | None = Field(default=None, description="同一会话客服原答的逐字依据")


class QaExtraction(BaseModel):
    pairs: list[QaPair] = Field(default_factory=list, description="可复用问答，没有则为空")


async def extract_qa(conversation_texts: list[str], model=None) -> list[QaPair]:
    if not conversation_texts:
        return []
    chat_model = model if model is not None else get_chat_model()
    chain = MINING_PROMPT | chat_model.with_structured_output(
        QaExtraction, method=get_settings().structured_output_method
    )
    result: QaExtraction = await chain.ainvoke({
        "conversations": "\n---\n".join(
            f"[会话 {index}]\n{text}" for index, text in enumerate(conversation_texts, 1)
        )
    })
    return result.pairs


@dataclass(frozen=True)
class _Source:
    source_ref: str
    prompt_text: str
    assistant_texts: tuple[str, ...]


async def _load_conversation_texts() -> list[_Source]:
    conversations = await repository.list_conversations_with_messages()
    out: list[_Source] = []
    for conversation_id, messages in conversations:
        # JSON string encoding prevents a user-authored newline from visually
        # manufacturing a new trusted `assistant:` role in the model input.
        lines = [f"{message.role}: {json.dumps(message.content, ensure_ascii=False)}"
                 for message in messages
                 if message.role in {"user", "assistant"} and message.content]
        if lines:
            out.append(_Source(
                source_ref=f"conv:{conversation_id}",
                prompt_text="\n".join(lines),
                assistant_texts=tuple(message.content for message in messages
                                      if message.role == "assistant" and message.content),
            ))
    return out


def _batch_no(group: list[_Source]) -> str:
    snapshot = hashlib.sha256("\0".join(
        f"{source.source_ref}\0{source.prompt_text}" for source in group
    ).encode("utf-8")).hexdigest()[:32]
    return f"mine-{snapshot}"


_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_ORDER_NEAR_LABEL = re.compile(
    r"订单(?:编号|号)?\s*[:：#]?\s*([A-Za-z0-9][A-Za-z0-9_-]{3,})", re.I
)
_ORDER_STYLE = re.compile(r"\b[A-Za-z]{1,10}[-_]?\d{4,}\b")
_FACT_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _sensitive_tokens(text: str) -> set[str]:
    return set(_PHONE.findall(text)) | set(_EMAIL.findall(text)) | set(
        _ORDER_NEAR_LABEL.findall(text)
    ) | set(_ORDER_STYLE.findall(text))


def _safe_for_source(pair: QaPair, source: _Source, all_sources: list[_Source]) -> bool:
    """Reject ungrounded attribution and concrete identifiers before staging."""
    quote = (pair.source_quote or "").strip()
    assistant_answers = source.assistant_texts
    # Keep the answer verbatim within a trusted assistant message. A cited
    # quote alone cannot justify extra, possibly cross-source narrative.
    if not quote or not pair.answer.strip() or not any(
        quote in answer and pair.answer.strip() in answer for answer in assistant_answers
    ):
        return False
    output = f"{pair.question}\n{pair.answer}\n{quote}"
    if (_PHONE.search(output) or _EMAIL.search(output) or _ORDER_STYLE.search(output)
            or _ORDER_NEAR_LABEL.search(output)):
        return False
    # A source quote alone does not justify numeric facts copied from another
    # conversation in the same batch.
    supported_numbers = set(_FACT_NUMBER.findall("\n".join(assistant_answers)))
    if not set(_FACT_NUMBER.findall(pair.answer)).issubset(supported_numbers):
        return False
    known = set().union(*(_sensitive_tokens(source.prompt_text) for source in all_sources))
    return not any(token.casefold() in output.casefold() for token in known)


async def mine(batch_size: int = 20, model=None) -> dict[str, int]:
    """Extract sources in batches while preserving each source_ref.

    `batch_size` sets the number of sources in one structured model call.
    Explicit source indices and source quotes prevent cross-source attribution.
    Finalization is atomic and also recovers legacy kept rows after interruption.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    sources = await _load_conversation_texts()
    extracted = 0
    for start in range(0, len(sources), batch_size):
        group = sources[start:start + batch_size]
        batch_no = _batch_no(group)
        missing = [not await repository.staging_batch_exists(batch_no, source.source_ref)
                   for source in group]
        if not any(missing):
            continue
        texts = [source.prompt_text for source in group]
        pairs = await extract_qa(texts, model=model)
        grouped: list[list[tuple[str, str]]] = [[] for _ in group]
        for pair in pairs:
            index = pair.source_index
            if index is None or not 1 <= index <= len(group):
                continue
            if _safe_for_source(pair, group[index - 1], group):
                grouped[index - 1].append((pair.question, pair.answer))
        for index, source in enumerate(group):
            if missing[index]:
                extracted += await repository.insert_staging_batch(
                    batch_no, source.source_ref, grouped[index]
                )
    classified = await repository.finalize_mined_staging()
    return {"sources": len(sources), "extracted": extracted, **classified}

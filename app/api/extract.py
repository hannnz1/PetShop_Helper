"""Structured after-sales ticket extraction."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from langchain_core.runnables import Runnable

from app.core.memory import estimate_tokens
from app.core.prompts import EXTRACT_PROMPT
from app.schemas.extract import AfterSalesTicket, ExtractRequest

logger = logging.getLogger(__name__)
router = APIRouter()


def get_extractor(request: Request) -> Runnable:
    """Bind the app's shared model to the configured structured output mode."""
    config = request.app.state.settings
    model = request.app.state.model
    return EXTRACT_PROMPT | model.with_structured_output(
        AfterSalesTicket, method=config.structured_output_method
    )


@router.post("/api/extract", response_model=AfterSalesTicket)
async def extract(
    req: ExtractRequest,
    request: Request,
    extractor: Runnable = Depends(get_extractor),
) -> AfterSalesTicket:
    rendered = EXTRACT_PROMPT.format_messages(text=req.text)
    if estimate_tokens(rendered) > request.app.state.settings.token_budget:
        raise HTTPException(status_code=422, detail="消息超出上下文预算")

    try:
        result = await extractor.ainvoke({"text": req.text})
        return AfterSalesTicket.model_validate(result)
    except Exception:
        # Upstream exceptions can include credentials or full response bodies.
        logger.warning("Extract upstream call or output validation failed")
        raise HTTPException(
            status_code=502, detail="上游模型暂时不可用，请稍后重试"
        ) from None

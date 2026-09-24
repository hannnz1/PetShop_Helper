"""Real glm-5.2 structured tool-call gate; exits without a request on unsafe config."""

import asyncio
import sys
from pathlib import Path
from urllib.parse import urlsplit

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Support both ``uv run python scripts/smoke_toolcall.py`` and module import.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings, get_settings
from app.core.llm import get_chat_model


class AddInput(BaseModel):
    a: int = Field(description="第一个加数")
    b: int = Field(description="第二个加数")


@tool(args_schema=AddInput)
def add(a: int, b: int) -> int:
    """把两个整数相加。"""
    return a + b


def is_glm_endpoint(base_url: str) -> bool:
    """Accept documented Z.ai chat API roots, including its coding-plan route."""
    try:
        url = urlsplit(base_url)
        return (
            url.scheme == "https"
            and url.hostname == "api.z.ai"
            and url.port in (None, 443)
            and url.username is None
            and url.password is None
            and url.query == ""
            and url.fragment == ""
            and url.path.rstrip("/") in {"/api/paas/v4", "/api/coding/paas/v4"}
        )
    except ValueError:
        return False


def has_expected_tool_call(response: object) -> bool:
    """Require a parsed LangChain tool call with the requested name and integers."""
    calls = getattr(response, "tool_calls", None)
    if not isinstance(calls, list):
        return False
    for call in calls:
        if not isinstance(call, dict) or call.get("name") != "add":
            continue
        args = call.get("args")
        if (
            isinstance(args, dict)
            and set(args) == {"a", "b"}
            and type(args["a"]) is int
            and type(args["b"]) is int
            and args["a"] == 23
            and args["b"] == 19
        ):
            return True
    return False


async def run_smoke(settings: Settings, model_factory=get_chat_model) -> bool:
    """Run the live gate; never print the key, request, or raw upstream data."""
    if settings.chat_model != "glm-5.2" or not is_glm_endpoint(settings.chat_base_url):
        print("NO-GO: configure CHAT_MODEL=glm-5.2 and a Z.ai API base URL.")
        return False

    try:
        model = model_factory(streaming=False, settings=settings)
        response = await model.bind_tools([add]).ainvoke("请调用 add 工具计算 23 加 19。")
    except Exception:
        print("NO-GO: model request failed; inspect private diagnostics separately.")
        return False

    if has_expected_tool_call(response):
        print("GO: glm-5.2 returned structured add(a=23, b=19) tool_calls.")
        return True
    print("NO-GO: response did not contain the expected structured add tool_call.")
    return False


async def main() -> int:
    try:
        settings = get_settings()
    except Exception:
        print("NO-GO: CHAT_* configuration is unavailable or invalid.")
        return 1
    return 0 if await run_smoke(settings) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

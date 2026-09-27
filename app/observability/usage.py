"""Buffer actual model usage until the Graph turn's final intent is known."""

from dataclasses import dataclass
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult


@dataclass(frozen=True)
class UsageEvent:
    run_id: str
    turn_id: str
    conversation_id: int
    intent: str
    model_name: str
    input_tokens: int | None
    output_tokens: int | None
    usage_status: str
    turn_status: str
    is_estimated: bool = False


class UsageCollector(BaseCallbackHandler):
    """One callback per turn; duplicate terminal callbacks never double count."""

    def __init__(self, turn_id: str, conversation_id: int) -> None:
        self.turn_id = turn_id
        self.conversation_id = conversation_id
        self._runs: dict[str, tuple[str, int | None, int | None]] = {}

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs) -> None:
        key = str(run_id)
        if key in self._runs:
            return
        generation = next((group[0] for group in response.generations if group), None)
        message = getattr(generation, "message", None)
        metadata = message.usage_metadata if isinstance(message, AIMessage) else None
        model_name = (
            (message.response_metadata or {}).get("model_name") if isinstance(message, AIMessage)
            else None
        ) or (response.llm_output or {}).get("model_name") or "unknown"
        if (isinstance(metadata, dict)
                and type(metadata.get("input_tokens")) is int
                and type(metadata.get("output_tokens")) is int):
            counts = (metadata["input_tokens"], metadata["output_tokens"])
        else:
            counts = (None, None)
        self._runs[key] = (str(model_name)[:128], *counts)

    def finalize(self, intent: str, *, outcome: str = "completed") -> list[UsageEvent]:
        settled_intent = intent or "unknown" if outcome == "completed" else "unknown"
        return [UsageEvent(
            run_id=run_id, turn_id=self.turn_id, conversation_id=self.conversation_id,
            intent=settled_intent, model_name=model_name,
            input_tokens=input_tokens, output_tokens=output_tokens,
            usage_status="available" if input_tokens is not None else "unavailable",
            turn_status=outcome,
        ) for run_id, (model_name, input_tokens, output_tokens) in self._runs.items()]

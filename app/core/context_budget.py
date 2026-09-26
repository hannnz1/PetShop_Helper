"""Conservative allocation of a model window to current work and chat history."""

from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.config import Settings
from app.core.memory import estimate_tokens


class ContextBudgetExceeded(ValueError):
    """Even an empty-history turn cannot fit without dropping required context."""


@dataclass(frozen=True)
class FixedCosts:
    """Measured or configured costs that must survive history trimming."""

    system_tools: int
    retrieval_evidence: int
    summary: int
    safety: int

    @classmethod
    def measure(
        cls, settings: Settings, *, system_tools: list[BaseMessage],
        retrieval_evidence: list[BaseMessage], summary: list[BaseMessage],
        safety: int,
    ) -> "FixedCosts":
        """Count actual rendered messages with the same CJK calibration as history."""
        calibration = settings.cjk_chars_per_token
        return cls(
            system_tools=estimate_tokens(system_tools, chars_per_token=calibration),
            retrieval_evidence=estimate_tokens(retrieval_evidence, chars_per_token=calibration),
            summary=estimate_tokens(summary, chars_per_token=calibration),
            safety=safety,
        )

    @property
    def total(self) -> int:
        return self.system_tools + self.retrieval_evidence + self.summary + self.safety


@dataclass(frozen=True)
class ContextBudget:
    history_total: int
    layer1: int
    layer2: int
    current_peak: int


def _effective_window(settings: Settings) -> int:
    # A caller that explicitly supplied the old cap retains it as a hard limit.
    # Its historical default of 2000 is not an implicit cap on the new allocator.
    if "token_budget" in settings.model_fields_set:
        return min(settings.model_context_window, settings.token_budget)
    return settings.model_context_window


def derive_budget(settings: Settings, fixed: FixedCosts) -> ContextBudget:
    """Allocate at most the measured available space and steady-turn target."""
    if min(vars(fixed).values()) < 0:
        raise ValueError("fixed costs must be nonnegative")
    # The last model call may see every earlier tool exchange in this turn.
    # Each earlier assistant generation (including tool-call arguments) can
    # reach max_output_tokens; tool_result_max_tokens is the per-step aggregate.
    prior_exchanges = settings.max_agent_steps - 1
    current_peak = settings.max_user_input_tokens + prior_exchanges * (
        settings.max_output_tokens + settings.tool_result_max_tokens
    )
    available = max(0, _effective_window(settings) - fixed.total - settings.max_output_tokens - current_peak)
    representative = [HumanMessage("问" * (settings.steady_turn_chars // 2)),
                      AIMessage("答" * (settings.steady_turn_chars - settings.steady_turn_chars // 2))]
    target = settings.expected_history_turns * estimate_tokens(
        representative, chars_per_token=settings.cjk_chars_per_token,
    )
    history_total = min(available, target)
    layer1 = round(history_total * 0.7)
    return ContextBudget(history_total, layer1, history_total - layer1, current_peak)


def validate_context_budget(settings: Settings, fixed: FixedCosts) -> None:
    """Fail startup when a peak turn cannot fit without losing required inputs."""
    budget = derive_budget(settings, fixed)
    required = fixed.total + settings.max_output_tokens + budget.current_peak
    window = _effective_window(settings)
    if required > window:
        raise ContextBudgetExceeded(
            f"一轮上下文预算不足 (one turn): required={required}, window={window}"
        )

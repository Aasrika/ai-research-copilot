"""
token_tracking.py
------------------
Shared helper for extracting token usage from a Groq response and
estimating an equivalent production cost. Every agent that calls Groq
uses this, so the extraction/estimation logic exists in exactly one
place rather than being reimplemented six times.
"""

from core.config import MODEL_COST_PER_MILLION_TOKENS, DEFAULT_COST_PER_MILLION_TOKENS


def extract_usage(response, prompt_text: str = "", completion_text: str = "") -> dict:
    """
    Prefers Groq's real usage metadata — response.usage.prompt_tokens /
    .completion_tokens / .total_tokens, present on essentially every real
    Groq chat completion (its API is OpenAI-compatible, including this
    field naming — NOT "input_tokens"/"output_tokens").

    Falls back to a rough chars/4 estimate of the prompt/completion text
    when usage is missing, so a malformed or mocked response still
    produces a number instead of crashing the caller.
    """
    usage = getattr(response, "usage", None)
    if usage is not None:
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "estimated": False,
        }

    prompt_tokens = max(1, len(prompt_text) // 4) if prompt_text else 0
    completion_tokens = max(1, len(completion_text) // 4) if completion_text else 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "estimated": True,
    }


def estimate_cost(usage: dict, model: str) -> float:
    """
    $ estimate of what this call would cost against the equivalent
    OpenAI-tier model (see config.MODEL_COST_PER_MILLION_TOKENS) — Groq
    itself is far cheaper/free for this project's actual usage; this
    number exists purely for production-cost-thinking on the dashboard.
    """
    rates = MODEL_COST_PER_MILLION_TOKENS.get(model, DEFAULT_COST_PER_MILLION_TOKENS)
    cost = (
        usage["prompt_tokens"] / 1_000_000 * rates["input"]
        + usage["completion_tokens"] / 1_000_000 * rates["output"]
    )
    return round(cost, 6)


def empty_totals() -> dict:
    """Zeroed accumulator for threading through LangGraph state across
    multiple LLM calls within one pipeline run (QA retries, or
    critique's one-call-per-claim classification loop)."""
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "tokens_estimated": False,
    }


def read_totals(state) -> dict:
    """Reads the running totals back out of a LangGraph state dict,
    defaulting to zero for keys not yet set (first call in a run)."""
    return {
        "prompt_tokens": state.get("prompt_tokens", 0),
        "completion_tokens": state.get("completion_tokens", 0),
        "total_tokens": state.get("total_tokens", 0),
        "estimated_cost_usd": state.get("estimated_cost_usd", 0.0),
        "tokens_estimated": state.get("tokens_estimated", False),
    }


def accumulate(totals: dict, response, model: str, prompt_text: str = "", completion_text: str = "") -> dict:
    """Returns a NEW totals dict with this call's usage/cost folded in —
    the same accumulator pattern already used for retry_count (state.get
    ('retry_count', 0) + 1) elsewhere in this codebase."""
    usage = extract_usage(response, prompt_text, completion_text)
    cost = estimate_cost(usage, model)
    return {
        "prompt_tokens": totals["prompt_tokens"] + usage["prompt_tokens"],
        "completion_tokens": totals["completion_tokens"] + usage["completion_tokens"],
        "total_tokens": totals["total_tokens"] + usage["total_tokens"],
        "estimated_cost_usd": round(totals["estimated_cost_usd"] + cost, 6),
        "tokens_estimated": totals["tokens_estimated"] or usage["estimated"],
    }

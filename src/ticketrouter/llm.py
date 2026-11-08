"""Tier 3: LLM classifiers behind one interface.

`ClaudeTier` calls the Claude API with a JSON schema in `output_config.format`, so the reply is
always an object with intent, priority, team, confidence, and a short reason. The `anthropic`
package is imported only when a ClaudeTier is built without an injected client.

`SimulatedLLM` is the offline default. It does not call any model. Its accuracy, confidence,
latency, and token usage come from a config so the cascade economics can be studied for free.
Every number it produces is simulated and is labeled that way in reports and the UI.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Protocol

from .pricing import call_cost
from .taxonomy import BY_NAME, INTENT_NAMES, PRIORITIES, TEAMS, team_for

UNKNOWN = "unknown"

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": [*INTENT_NAMES, UNKNOWN]},
        "priority": {"type": "string", "enum": list(PRIORITIES)},
        "team": {"type": "string", "enum": [*TEAMS, "support"]},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["intent", "priority", "team", "confidence", "reason"],
    "additionalProperties": False,
}


class StructuredOutputError(ValueError):
    """The model reply did not match the schema."""


class LLMRefusal(RuntimeError):
    """The model declined to classify. The cascade sends these to a human."""


@dataclass
class LLMResult:
    intent: str
    priority: str
    team: str
    confidence: float
    reason: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    simulated: bool

    @property
    def cost(self) -> float:
        return call_cost(self.model, self.input_tokens, self.output_tokens)

    def to_dict(self) -> dict:
        return asdict(self) | {"cost": round(self.cost, 8)}


def validate_output(obj: object) -> dict:
    """Check a parsed reply against OUTPUT_SCHEMA without a schema library."""
    if not isinstance(obj, dict):
        raise StructuredOutputError("reply is not a JSON object")
    missing = [k for k in OUTPUT_SCHEMA["required"] if k not in obj]
    if missing:
        raise StructuredOutputError(f"missing fields: {missing}")
    extra = set(obj) - set(OUTPUT_SCHEMA["properties"])
    if extra:
        raise StructuredOutputError(f"unexpected fields: {sorted(extra)}")
    for key in ("intent", "priority", "team"):
        allowed = OUTPUT_SCHEMA["properties"][key]["enum"]
        if obj[key] not in allowed:
            raise StructuredOutputError(f"{key}={obj[key]!r} is not one of the allowed values")
    conf = obj["confidence"]
    if isinstance(conf, bool) or not isinstance(conf, int | float) or not 0.0 <= conf <= 1.0:
        raise StructuredOutputError(f"confidence must be a number from 0 to 1, got {conf!r}")
    if not isinstance(obj["reason"], str) or not obj["reason"].strip():
        raise StructuredOutputError("reason must be a non-empty string")
    return {**obj, "confidence": float(conf), "reason": obj["reason"].strip()[:300]}


def system_prompt() -> str:
    lines = [
        "You classify support tickets for Corvane, a B2B analytics platform.",
        "Pick the one intent that best describes what the customer needs. If a ticket mentions a "
        "security problem or a full outage, choose that intent even if other topics appear.",
        "If no intent fits, use 'unknown'. Set priority from urgency and business impact.",
        "Use the team listed next to the intent. Confidence is your probability that the intent is right.",
        "Keep the reason under 20 words.",
        "",
        "Intents (intent: team, usual priority):",
    ]
    lines += [f"- {name}: {BY_NAME[name].team}, {BY_NAME[name].priority}" for name in INTENT_NAMES]
    return "\n".join(lines)


SYSTEM_PROMPT = system_prompt()
SYSTEM_TOKENS = len(SYSTEM_PROMPT) // 4 + 60  # schema and framing overhead


class LLMClassifier(Protocol):
    model: str

    def classify(self, text: str, gold=None, hint: str | None = None) -> LLMResult: ...


class ClaudeTier:
    """Real Claude call. Inject `client` in tests; otherwise the SDK is imported here."""

    def __init__(self, model: str, client=None, max_tokens: int = 400, effort: str = "low"):
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on install extras
                raise RuntimeError("Install the Claude extra first: uv sync --extra claude") from exc
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort

    def request(self, text: str) -> dict:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": f"<ticket>\n{text}\n</ticket>"}],
            "output_config": {"effort": self.effort, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        }

    def classify(self, text: str, gold=None, hint: str | None = None) -> LLMResult:
        start = time.perf_counter()
        resp = self.client.messages.create(**self.request(text))
        latency = (time.perf_counter() - start) * 1000
        if getattr(resp, "stop_reason", None) == "refusal":
            raise LLMRefusal(f"{self.model} declined to classify this ticket")
        raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise StructuredOutputError(f"reply is not valid JSON: {raw[:80]!r}") from exc
        out = validate_output(parsed)
        return LLMResult(
            intent=out["intent"],
            priority=out["priority"],
            team=out["team"],
            confidence=out["confidence"],
            reason=out["reason"],
            model=self.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=round(latency, 1),
            simulated=False,
        )


@dataclass(frozen=True)
class SimConfig:
    """Knobs for a simulated LLM tier. All values are assumptions, not measurements.

    Accuracy falls linearly from `acc_easy` to `acc_hard` as ticket difficulty goes from 0 to 1.
    Confidence is drawn from a range that depends on whether the answer is right, with some
    overlap, because LLM self-reported confidence is only loosely calibrated.
    """

    model: str
    acc_easy: float
    acc_hard: float
    priority_acc: float
    conf_right: tuple[float, float]
    conf_wrong: tuple[float, float]
    latency_base_ms: float
    latency_per_output_token_ms: float
    output_tokens: tuple[int, int] = (45, 90)
    system_tokens: int = SYSTEM_TOKENS


HAIKU_SIM = SimConfig(
    model="claude-haiku-5-5",
    acc_easy=0.97,
    acc_hard=0.74,
    priority_acc=0.84,
    conf_right=(0.72, 0.99),
    conf_wrong=(0.45, 0.9),
    latency_base_ms=320,
    latency_per_output_token_ms=4,
)
OPUS_SIM = SimConfig(
    model="claude-opus-5-5",
    acc_easy=0.99,
    acc_hard=0.9,
    priority_acc=0.9,
    conf_right=(0.8, 0.99),
    conf_wrong=(0.5, 0.85),
    latency_base_ms=1100,
    latency_per_output_token_ms=14,
)


def _u(seed: str, salt: str) -> float:
    """Deterministic uniform number in [0, 1) from a string."""
    h = hashlib.sha256(f"{seed}|{salt}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


class SimulatedLLM:
    def __init__(self, config: SimConfig):
        self.config = config
        self.model = config.model

    def classify(self, text: str, gold=None, hint: str | None = None) -> LLMResult:
        """With a labeled ticket (`gold`), the answer is right with the configured probability.
        With free text there is no ground truth, so it returns `hint` (the best local guess) and a
        simulated confidence. That still shows routing, cost, and latency, not LLM quality."""
        c = self.config
        key = f"{c.model}|{getattr(gold, 'id', '')}|{text}"
        out_tokens = c.output_tokens[0] + int(_u(key, "out") * (c.output_tokens[1] - c.output_tokens[0]))
        in_tokens = c.system_tokens + max(1, len(text) // 4)
        latency = c.latency_base_ms + c.latency_per_output_token_ms * out_tokens * (0.8 + 0.4 * _u(key, "lat"))
        if gold is None:
            intent = hint if hint in BY_NAME else UNKNOWN
            right = True
            priority = BY_NAME[intent].priority if intent in BY_NAME else "normal"
        else:
            p_right = c.acc_easy - (c.acc_easy - c.acc_hard) * gold.difficulty
            right = _u(key, "right") < p_right
            if gold.true_intent not in INTENT_NAMES:
                # Out-of-taxonomy ticket: a careful model says unknown, sometimes it forces a fit.
                intent = UNKNOWN if right else BY_NAME[gold.true_intent].confusable[0]
            elif right:
                intent = gold.true_intent
            else:
                options = [gold.secondary] if gold.secondary else []
                options += list(BY_NAME[gold.true_intent].confusable)
                intent = options[int(_u(key, "pick") * len(options))] if options else UNKNOWN
            if _u(key, "prio") < c.priority_acc:
                priority = gold.true_priority
            else:
                i = PRIORITIES.index(gold.true_priority)
                step = -1 if (_u(key, "dir") < 0.5 and i > 0) or i == len(PRIORITIES) - 1 else 1
                priority = PRIORITIES[i + step]
        lo, hi = c.conf_right if right else c.conf_wrong
        conf = lo + (hi - lo) * _u(key, "conf")
        reason = f"simulated {c.model.split('-')[1]} reading: ticket matches {intent.replace('_', ' ')}"
        return LLMResult(
            intent=intent,
            priority=priority,
            team=team_for(intent),
            confidence=round(conf, 4),
            reason=reason,
            model=c.model,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            latency_ms=round(latency, 1),
            simulated=True,
        )

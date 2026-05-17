"""Input / output safety wrappers.

Minimal pattern-matching defaults. Real deployments should plug in a model-
based filter (Llama Guard, ShieldGemma, NeMo Guardrails) by subclassing.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SafetyVerdict:
    allow: bool
    reason: str = ""


class _PatternFilter:
    def __init__(self, patterns: tuple[str, ...]) -> None:
        self._patterns = tuple(p.lower() for p in patterns)

    def check(self, text: str) -> SafetyVerdict:
        low = text.lower()
        for p in self._patterns:
            if p in low:
                return SafetyVerdict(False, f"matched pattern: {p!r}")
        return SafetyVerdict(True)


class InputSafety(_PatternFilter):
    """Blocks obvious prompt-injection / jailbreak markers before the pipeline runs."""

    def __call__(self, text: str) -> SafetyVerdict:
        return self.check(text)


class OutputSafety(_PatternFilter):
    """Blocks the agent from emitting matching content. Last line of defense."""

    def __call__(self, text: str) -> SafetyVerdict:
        return self.check(text)

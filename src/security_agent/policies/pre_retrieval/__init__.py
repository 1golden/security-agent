from security_agent.policies.pre_retrieval.rule import RulePrePolicy
from security_agent.policies.pre_retrieval.llm import LLMPrePolicy
from security_agent.policies.pre_retrieval.rl import RLPrePolicy

__all__ = ["RulePrePolicy", "LLMPrePolicy", "RLPrePolicy"]


def build(name: str, **kwargs):
    name = name.lower()
    if name == "rule":
        return RulePrePolicy()
    if name == "llm":
        return LLMPrePolicy(**kwargs)
    if name == "rl":
        return RLPrePolicy(**kwargs)
    raise ValueError(f"Unknown pre policy: {name}")

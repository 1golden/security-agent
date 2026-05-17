from security_agent.policies.memory_write.rule import RuleWritePolicy
from security_agent.policies.memory_write.llm import LLMWritePolicy
from security_agent.policies.memory_write.rl import RLWritePolicy

__all__ = ["RuleWritePolicy", "LLMWritePolicy", "RLWritePolicy"]


def build(name: str, **kwargs):
    name = name.lower()
    if name == "rule":
        return RuleWritePolicy()
    if name == "llm":
        return LLMWritePolicy(**kwargs)
    if name == "rl":
        return RLWritePolicy(**kwargs)
    raise ValueError(f"Unknown write policy: {name}")

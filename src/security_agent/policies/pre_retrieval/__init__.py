from security_agent.policies.pre_retrieval.rule import RulePrePolicy
from security_agent.policies.pre_retrieval.llm import LLMPrePolicy
from security_agent.policies.pre_retrieval.rl import RLPrePolicy
from security_agent.policies.pre_retrieval.router import RouterPrePolicy

__all__ = ["RulePrePolicy", "LLMPrePolicy", "RLPrePolicy", "RouterPrePolicy"]


def build(name: str, **kwargs):
    name = name.lower()
    if name == "rule":
        return RulePrePolicy()
    if name == "llm":
        return LLMPrePolicy(**kwargs)
    if name == "router":
        # router wraps both inner policies; needs the LLM backend to wire LLMPrePolicy
        llm = kwargs.get("llm")
        return RouterPrePolicy(
            rule=RulePrePolicy(),
            llm_policy=LLMPrePolicy(llm=llm) if llm is not None else None,
        )
    if name == "rl":
        return RLPrePolicy(**kwargs)
    if name == "cql":
        from security_agent.policies.staged_router import CQLPrePolicy
        llm = kwargs.get("llm")
        return CQLPrePolicy(
            llm_arm=LLMPrePolicy(llm=llm) if llm is not None else None,
        )
    raise ValueError(f"Unknown pre policy: {name}")

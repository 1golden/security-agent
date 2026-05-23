from security_agent.policies.post_retrieval.rule import RulePostPolicy
from security_agent.policies.post_retrieval.llm import LLMPostPolicy
from security_agent.policies.post_retrieval.rl import RLPostPolicy

__all__ = ["RulePostPolicy", "LLMPostPolicy", "RLPostPolicy"]


def build(name: str, **kwargs):
    name = name.lower()
    if name == "rule":
        return RulePostPolicy(threshold=kwargs.get("threshold", 0.5))
    if name == "llm":
        return LLMPostPolicy(llm=kwargs["llm"], threshold=kwargs.get("threshold", 0.5))
    if name == "rl":
        return RLPostPolicy(checkpoint=kwargs.get("checkpoint"))
    if name == "cql":
        from security_agent.policies.staged_router import CQLPostPolicy
        thr = kwargs.get("threshold", 0.5)
        return CQLPostPolicy(
            rule_arm=RulePostPolicy(threshold=thr),
            llm_arm=LLMPostPolicy(llm=kwargs["llm"], threshold=thr),
        )
    raise ValueError(f"Unknown post policy: {name}")

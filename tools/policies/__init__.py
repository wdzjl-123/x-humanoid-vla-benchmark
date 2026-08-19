# All policy classes are resolved lazily in runner.py to avoid importing
# heavy dependencies (torch, IPython, jax) at module load time.
# runner.py reads POLICY_MAP and resolves None entries on demand.

POLICY_MAP = {
    "act":    None,   # resolved lazily: ACTPolicy
    "act_v1": None,   # resolved lazily: ACTPolicy (640×480)
    "multitask_act": None,  # resolved lazily: one RGB-D task-conditioned ACT checkpoint
    "flow":   None,   # resolved lazily: FlowPolicy
}

__all__ = ["POLICY_MAP"]

# R7 Model Card

## Identity

- Policy: task-conditioned RGB-D ACT behavior-cloning policy
- Checkpoint: `submission/models/r7/policy_last.ckpt`
- Training step: 21,000
- Tasks: `ind_task_01`, `ind_task_02`, `ind_task_03`, `lab_task_01`, `lab_task_03`
- Action dimension: 26 (`left_arm[7]`, `left_hand[6]`, `right_arm[7]`, `right_hand[6]`)
- Action chunk: 50 steps

## Inputs

- Head-camera RGB and depth observations
- 26D robot joint/hand state
- 14D left/right end-effector pose
- Five-way task identifier

## Training and evaluation

The development comparison used deterministic episode-level `80/10/10`
partitions and training-only normalization statistics. R7 was subsequently
trained on all public demonstrations for deployment, so it must not be ranked
against the held-out split used for R5.

## Limitations

- This is not a language-encoder or VLM/LLM-based general-purpose VLA model.
- Offline action error does not prove physical task success.
- The included videos are local Isaac Sim diagnostics, not real-robot trials
  and not official score certificates.
- Safety limits, collision handling, and sim-to-real transfer require separate
  validation before robot deployment.

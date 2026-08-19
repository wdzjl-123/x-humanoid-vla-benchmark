# Results

## Reported official evaluation

The following percentages were reported for 25 inference attempts per task:

| Task | Reported score | Approx. successes |
| --- | ---: | ---: |
| `ind_task_01` | 92% | 23/25 |
| `ind_task_02` | 24% | 6/25 |
| `ind_task_03` | 48% | 12/25 |
| `lab_task_01` | 80% | 20/25 |
| `lab_task_03` | 68% | 17/25 |
| Overall | 62.4% | 78/125 |

These numbers are recorded as a user-provided benchmark report. The platform
did not provide a complete machine-readable per-episode artifact in this
workspace, so the table should not be treated as a downloadable score log.

## Offline development comparison

On the fixed 80/10/10 test manifest (101 episodes, 28,180 timesteps), R5
normalized-action L1 was `0.126258`, compared with `0.138117` for the plain
split baseline, a relative reduction of `8.59%`. R7 was trained on all public
demonstrations and therefore has no unbiased score on that same test set.

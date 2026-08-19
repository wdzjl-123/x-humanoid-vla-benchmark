# Video Provenance

The five MP4 files in `videos/` are local Isaac Sim closed-loop recordings of
the R7 checkpoint, one task per file. They execute predicted actions in a
physics simulator; they are not replays of recorded HDF5 frames.

The local diagnostic runner did not emit the organizer's official success
predicate. Visual evidence in a video must therefore be described as a
rollout observation, not as a formally certified success rate.

| Task | Video | Source run | Scope |
| --- | --- | --- | --- |
| `ind_task_01` | `videos/ind_task_01_r7_normal300.mp4` | local Isaac Sim, 300 s limit | visual diagnostic |
| `ind_task_02` | `videos/ind_task_02_r7_normal300.mp4` | local Isaac Sim, 300 s limit | visual diagnostic |
| `ind_task_03` | `videos/ind_task_03_r7_normal300.mp4` | local Isaac Sim, 300 s limit | visual diagnostic |
| `lab_task_01` | `videos/lab_task_01_r7_normal300.mp4` | local Isaac Sim, 300 s limit | visual diagnostic |
| `lab_task_03` | `videos/lab_task_03_r7_normal300.mp4` | local Isaac Sim, 300 s limit | visual diagnostic |

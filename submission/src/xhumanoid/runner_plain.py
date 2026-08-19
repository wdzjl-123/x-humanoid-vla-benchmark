"""
ZMQ inference loop — receives obs from simulation, sends back actions.
Policy implementation is in act_policy.py.
"""
import os
import sys
import argparse
import time
import cv2
import numpy as np

# Ensure project root is on sys.path regardless of how this module is imported
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from common.utils.zmq_utils import ZmqPublisher, ZmqReceiver
from tools.policies import POLICY_MAP


def parse_args():
    parser = argparse.ArgumentParser(description="Policy inference server")
    parser.add_argument(
        "--policy", choices=list(POLICY_MAP.keys()), required=True,
        help="Policy type: act | act_v1 | multitask_act | flow"
    )
    parser.add_argument(
        "--model-path", required=False, default=None,
        help="Path to model checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Inference device: cuda | cpu  (default: cuda)"
    )
    parser.add_argument("--zmq-recv-port", type=int, default=5556)
    parser.add_argument("--zmq-send-port", type=int, default=5557)
    parser.add_argument(
        "--zmq-bind-host", default="0.0.0.0",
        help="Local interface used to publish actions (default: 0.0.0.0 for remote simulator access)",
    )
    parser.add_argument("--sim-host", default="127.0.0.1",
                        help="Organizer sim host to receive obs from "
                             "(cross-machine eval; default same-machine 127.0.0.1)")
    parser.add_argument(
        "--task", default="",
        help="Task name published via ZMQ topic 'task' before inference"
    )
    parser.add_argument(
        "--task-handshake-timeout", type=float, default=60.0,
        help="Maximum seconds to wait for task_cbd when --wait-for-task-cbd is set (default: 60)",
    )
    parser.add_argument(
        "--wait-for-task-cbd", action="store_true",
        help="Wait for the optional legacy task_cbd acknowledgement after publishing --task",
    )
    # ACT 参数
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Action chunk size (ACT default: 50; Flow default: checkpoint config or 16)"
    )
    parser.add_argument(
        "--temporal-agg", action="store_true", default=None,
        help="Enable temporal aggregation (ACT default: True)"
    )
    parser.add_argument(
        "--no-temporal-agg", dest="temporal_agg", action="store_false",
        help="Disable temporal aggregation"
    )
    parser.add_argument(
        "--act-action-horizon", type=int, default=None,
        help="When temporal aggregation is disabled, execute this many leading chunk actions before replanning",
    )
    parser.add_argument(
        "--act-temporal-decay", type=float, default=None,
        help="ACT temporal aggregation decay (default: 0.01)",
    )
    parser.add_argument(
        "--act-temporal-priority", choices=("oldest", "newest", "uniform"), default=None,
        help="Which covering ACT chunk predictions receive higher aggregation weight (default: oldest)",
    )
    parser.add_argument(
        "--act-hand-state", choices=("task_home", "legacy_home", "measured"),
        default="legacy_home",
        help="ACT hand state source: legacy_home (default), task_home, or measured",
    )
    parser.add_argument(
        "--act-hand-feedback", choices=("commanded", "measured"), default=None,
        help="ACT subsequent hand feedback source; defaults from --act-hand-state",
    )
    parser.add_argument(
        "--act-debug-every", type=int, default=50,
        help="ACT telemetry interval in policy steps (default: 50)",
    )
    parser.add_argument(
        "--act-image-color-order", choices=("rgb", "bgr"), default="rgb",
        help="ACT simulator image channel order (default: rgb; bgr reproduces the historical runner)",
    )
    parser.add_argument(
        "--multitask-hand-state", choices=("task_home", "legacy_home", "measured"),
        default="measured",
        help="Unified RGB-D ACT hand state source (default: measured, matching its training state)",
    )
    parser.add_argument(
        "--multitask-hand-feedback", choices=("commanded", "measured"),
        default="measured",
        help="Unified RGB-D ACT subsequent hand feedback source (default: measured)",
    )
    # ACT image resize override (defaults set per policy name: act=320x240, act_v1=640x480)
    parser.add_argument("--img-w", type=int, default=None,
                        help="ACT inference image resize width (override per-policy default)")
    parser.add_argument("--img-h", type=int, default=None,
                        help="ACT inference image resize height (override per-policy default)")
    # Flow Matching 参数
    parser.add_argument("--flow-sample-steps", type=int, default=None,
                        help="ODE Euler steps for FlowPolicy sampling (default: 8)")
    parser.add_argument("--flow-action-horizon", type=int, default=None,
                        help="Number of sampled FlowPolicy actions to execute before replanning (default: 4)")
    parser.add_argument("--policy-seed", type=int, default=1,
                        help="Sampling seed for stochastic policies")
    return parser.parse_args()


def build_action_dict(np_action: np.ndarray, robot_type: str) -> dict:
    """
    将 action array 按末端执行器类型拆分为仿真侧期望的 dict。

    robot_type="brainco2": [left_arm(7), left_hand(6), right_arm(7), right_hand(6)]
    robot_type="gripper":  [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]
    """
    if robot_type == "brainco2":
        return {
            "left_arm":  np_action[:7].tolist(),
            "left_hand": np_action[7:13].tolist(),
            "right_arm": np_action[13:20].tolist(),
            "right_hand": np_action[20:26].tolist(),
        }
    elif robot_type == "gripper":
        return {
            "left_arm":  np_action[:7].tolist(),
            "left_hand": np_action[7:8].tolist(),
            "right_arm": np_action[8:15].tolist(),
            "right_hand": np_action[15:16].tolist(),
        }
    else:
        raise ValueError(f"Unknown robot_type: {robot_type}")


def wait_for_task_ack(
    zmq_receiver: ZmqReceiver,
    zmq_publisher: ZmqPublisher,
    task_name: str,
    timeout_seconds: float,
) -> bool:
    """Publish a task until the simulator confirms it, without waiting forever."""
    if timeout_seconds <= 0:
        raise ValueError("task-handshake-timeout must be positive")

    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            return False
        zmq_publisher.send_msg(task_name, topic=b"task")
        result = zmq_receiver.receive_envelope(
            timeout=max(1, min(500, int(remaining_seconds * 1000))),
        )
        if result is not None and str(result.get("topic", "")) == "task_cbd":
            return True


def run_inference_server():
    args = parse_args()

    print(f"[runner] Loading {args.policy} policy from {args.model_path} on {args.device}")
    extra_kwargs = {}
    if args.policy in ("act", "act_v1", "multitask_act"):
        # Baseline ACT variants share RGB-only inference. multitask_act has a
        # distinct RGB-D checkpoint but uses the same runner and ZMQ contract.
        if args.policy == "multitask_act":
            from tools.policies.multitask_act_policy import MultiTaskACTPolicy as _PolicyCls
        else:
            from tools.policies.act_policy import ACTPolicy as _PolicyCls
        extra_kwargs["task_name"] = args.task
        if args.policy == "multitask_act":
            extra_kwargs["hand_state_mode"] = args.multitask_hand_state
            extra_kwargs["hand_feedback"] = args.multitask_hand_feedback
        else:
            extra_kwargs["hand_state_mode"] = args.act_hand_state
            extra_kwargs["hand_feedback"] = args.act_hand_feedback
        extra_kwargs["image_color_order"] = args.act_image_color_order
        extra_kwargs["debug_every"] = args.act_debug_every
        extra_kwargs["action_horizon"] = args.act_action_horizon
        extra_kwargs["temporal_decay"] = args.act_temporal_decay
        extra_kwargs["temporal_priority"] = args.act_temporal_priority
        if args.policy == "act":
            _PolicyCls.IMG_W, _PolicyCls.IMG_H = 320, 240
        elif args.policy == "act_v1":
            _PolicyCls.IMG_W, _PolicyCls.IMG_H = 640, 480
        # Unified checkpoint resolution is part of its serialized config.
        if args.policy != "multitask_act" and args.img_w is not None:
            _PolicyCls.IMG_W = args.img_w
        if args.policy != "multitask_act" and args.img_h is not None:
            _PolicyCls.IMG_H = args.img_h
        if args.chunk_size is not None:
            _PolicyCls.CHUNK_SIZE = args.chunk_size
        if args.temporal_agg is not None:
            _PolicyCls.TEMPORAL_AGG = args.temporal_agg
    elif args.policy == "flow":
        from tools.policies.flow_policy import FlowPolicy as _PolicyCls
        extra_kwargs["task_name"] = args.task
        extra_kwargs["seed"] = args.policy_seed
        if args.img_w is not None:
            _PolicyCls.IMG_W = args.img_w
        if args.img_h is not None:
            _PolicyCls.IMG_H = args.img_h
        if args.chunk_size is not None:
            _PolicyCls.CHUNK_SIZE = args.chunk_size
        if args.flow_sample_steps is not None:
            _PolicyCls.SAMPLE_STEPS = args.flow_sample_steps
        if args.flow_action_horizon is not None:
            _PolicyCls.ACTION_HORIZON = args.flow_action_horizon
    else:
        raise ValueError(f"Unknown policy: {args.policy}")
    policy = _PolicyCls(args.model_path, args.device, **extra_kwargs)

    time.sleep(2)  # 等待仿真侧 ZMQ bind 完成

    zmq_receiver  = ZmqReceiver(port=args.zmq_recv_port, host=args.sim_host)
    zmq_publisher = ZmqPublisher(port=args.zmq_send_port, bind_host=args.zmq_bind_host)
    print(f"[runner] Ready — recv:{args.sim_host}:{args.zmq_recv_port}  send:{args.zmq_bind_host}:{args.zmq_send_port}")

    try:
        simulation_running = False
        robot_type = "brainco2"

        # The organizer can be temporarily busy after a prior task. Bound the
        # wait so a stale handshake never monopolizes the action port.
        if args.task:
            if args.wait_for_task_cbd:
                print(f"[runner] Publishing task {args.task!r}, waiting for task_cbd...")
                if not wait_for_task_ack(
                    zmq_receiver,
                    zmq_publisher,
                    args.task,
                    args.task_handshake_timeout,
                ):
                    print(
                        "[runner] task_cbd was not received within "
                        f"{args.task_handshake_timeout:.1f}s; exiting cleanly"
                    )
                    return
                print("[runner] task_cbd received, starting inference loop")
            else:
                # Official evaluation reports no completion callback. Publish the
                # task trigger, then preserve every incoming test/start/obs frame.
                print(f"[runner] Publishing task {args.task!r}")
                zmq_publisher.send_msg(args.task, topic=b"task")

        while True:
            result = zmq_receiver.receive_envelope()
            if result is None:
                continue

            topic      = str(result.get("topic", "")).encode("utf-8")
            data       = result.get("payload")
            episode_id = int(result.get("episode_id", -1))
            step_id    = int(result.get("step_id", -1))

            if topic == b"start":
                print(f"[runner] Episode {episode_id} started, robot={data}")
                simulation_running = True
                if isinstance(data, dict):
                    robot_type = data.get("end_effector", "brainco2")
                policy.reset()

            elif topic == b"obs":
                if not simulation_running:
                    print(f"[runner] Auto-start on first obs (episode {episode_id}) — start msg was lost")
                    simulation_running = True
                    policy.reset()
                if data is None:
                    print("[runner] Error: obs payload is None, skipping")
                    continue

                np_action = policy.infer(obs=data)
                action_dict = build_action_dict(np_action, robot_type)
                zmq_publisher.send_msg(
                    action_dict, topic=b"action",
                    episode_id=episode_id, step_id=step_id
                )
                # Keep the outbound command auditable during remote evaluation.
                if step_id % 10 == 0:
                    has_nonfinite = not np.all(np.isfinite(np_action))
                    msg = (
                        f"[runner|episode={episode_id}|step={step_id}] "
                        f"l_arm={np.round(np_action[:7],4).tolist()} "
                        f"l_hand={np.round(np_action[7:13],4).tolist()} "
                        f"r_arm={np.round(np_action[13:20],4).tolist()} "
                        f"r_hand={np.round(np_action[20:26],4).tolist()} "
                        f"nonfinite={has_nonfinite}"
                    )
                    print(msg)
                    if hasattr(policy, '_dbg'):
                        policy._dbg(msg)

            elif topic == b"reset":
                print(f"[runner] Episode {episode_id} ended, resetting")
                simulation_running = False
                policy.reset()

            elif topic == b"test":
                zmq_publisher.send_msg(
                    data=None, topic=b"test",
                    episode_id=episode_id, step_id=step_id
                )

            else:
                print(f"[runner] Unknown topic: {topic}")

    except KeyboardInterrupt:
        print("\n[runner] Shutting down...")
    except Exception as e:
        print(f"[runner] Fatal error: {e}")
        raise
    finally:
        zmq_receiver.close()
        zmq_publisher.close()
        cv2.destroyAllWindows()
        print("[runner] Done")


if __name__ == "__main__":
    run_inference_server()

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from tools.multitask import MULTITASK_STATE_DIM, TASK_NAMES, depth_to_meters, task_one_hot
from tools.policies.act_policy import build_act_model
from tools.policies.runner import parse_args, run_inference_server, wait_for_task_ack
from tools.train.multitask_dataset import (
    MultiTaskEpisodicDataset,
    RobustAugmentationConfig,
    build_episode_splits,
    compute_multitask_stats,
    create_episode_split_manifest,
    build_task_schedule,
)


class MultiTaskACTTests(unittest.TestCase):
    def test_task_condition_is_fixed_and_rejects_unknown_tasks(self):
        encoded = task_one_hot("ind_task_03")
        self.assertEqual(encoded.shape, (len(TASK_NAMES),))
        self.assertEqual(int(encoded.argmax()), TASK_NAMES.index("ind_task_03"))
        self.assertEqual(float(encoded.sum()), 1.0)
        with self.assertRaises(ValueError):
            task_one_hot("not_a_benchmark_task")

    def test_task_sampling_schedule_weights_hard_tasks(self):
        weights = dict(zip(TASK_NAMES, (2, 3, 3, 3, 2)))
        schedule = build_task_schedule(TASK_NAMES, weights)
        self.assertEqual(len(schedule), 13)
        self.assertEqual(schedule.count("ind_task_02"), 3)
        self.assertEqual(schedule.count("lab_task_01"), 3)
        self.assertEqual(schedule.count("ind_task_01"), 2)
        invalid = dict.fromkeys(TASK_NAMES, 1)
        invalid["ind_task_01"] = 0
        with self.assertRaises(ValueError):
            build_task_schedule(TASK_NAMES, invalid)

    def test_depth_conversion_handles_mm_and_meters(self):
        millimeters = depth_to_meters(np.array([[0.0, 730.0, 2600.0]], dtype=np.float32))
        meters = depth_to_meters(np.array([[0.0, 0.73, 2.6]], dtype=np.float32))
        np.testing.assert_allclose(millimeters, meters, atol=1e-5)
        self.assertEqual(float(depth_to_meters(np.array([[9000.0]], dtype=np.float32))[0, 0]), 5.0)

    def test_episode_split_manifest_is_deterministic_and_disjoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for task_name in TASK_NAMES:
                task_dir = root / task_name / "train"
                task_dir.mkdir(parents=True)
                for episode_index in range(10):
                    (task_dir / f"episode_{episode_index:03d}.hdf5").touch()

            first = create_episode_split_manifest(str(root), split_seed=17)
            second = create_episode_split_manifest(str(root), split_seed=17)
            self.assertEqual(first, second)
            for task_name in TASK_NAMES:
                task_splits = first["tasks"][task_name]
                self.assertEqual(len(task_splits["train"]), 8)
                self.assertEqual(len(task_splits["val"]), 1)
                self.assertEqual(len(task_splits["test"]), 1)
                members = set().union(*[set(task_splits[name]) for name in ("train", "val", "test")])
                self.assertEqual(len(members), 10)

    def test_depth_enabled_act_forward_has_expected_shape(self):
        with patch("tools.policies.detr.models.backbone.is_main_process", return_value=False):
            policy = build_act_model(
                chunk_size=4,
                hidden_dim=64,
                dim_feedforward=128,
                enc_layers=1,
                dec_layers=1,
                nheads=4,
                state_dim=MULTITASK_STATE_DIM,
                device="cpu",
                use_depth_image=True,
            )
        policy.eval()
        with torch.inference_mode():
            output = policy(
                torch.zeros(1, MULTITASK_STATE_DIM),
                torch.zeros(1, 1, 3, 48, 64),
                torch.ones(1, 1, 48, 64),
            )
        self.assertEqual(tuple(output.shape), (1, 4, 26))
        self.assertTrue(torch.isfinite(output).all())

    def test_runner_multitask_defaults_match_training_state(self):
        argv = [
            "runner.py", "--policy", "multitask_act", "--model-path", "policy.ckpt",
            "--task", "ind_task_01",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.multitask_hand_state, "measured")
        self.assertEqual(args.multitask_hand_feedback, "measured")
        self.assertEqual(args.task_handshake_timeout, 60.0)
        self.assertEqual(args.task_publish_delay, 5.0)

    def test_task_publish_waits_until_after_pub_bind_and_sends_once(self):
        events = []

        class Policy:
            def __init__(self, *_args, **_kwargs):
                events.append("policy")

            def reset(self):
                events.append("reset")

        class Receiver:
            def __init__(self, **_kwargs):
                events.append("receiver")

            def receive_envelope(self):
                raise KeyboardInterrupt

            def close(self):
                events.append("receiver_close")

        class Publisher:
            def __init__(self, **_kwargs):
                events.append("publisher")

            def send_msg(self, payload, topic, **_kwargs):
                events.append(("send", payload, topic))

            def close(self):
                events.append("publisher_close")

        def record_sleep(seconds):
            events.append(("sleep", seconds))

        argv = [
            "runner.py", "--policy", "multitask_act", "--model-path", "policy.ckpt",
            "--task", "ind_task_02", "--task-publish-delay", "5",
        ]
        with patch.object(sys, "argv", argv), \
             patch("tools.policies.multitask_act_policy.MultiTaskACTPolicy", Policy), \
             patch("tools.policies.runner.ZmqReceiver", Receiver), \
             patch("tools.policies.runner.ZmqPublisher", Publisher), \
             patch("tools.policies.runner.time.sleep", side_effect=record_sleep), \
             patch("tools.policies.runner.cv2.destroyAllWindows"):
            run_inference_server()

        publisher_index = events.index("publisher")
        delay_index = events.index(("sleep", 5.0))
        send_events = [event for event in events if isinstance(event, tuple) and event[0] == "send"]
        self.assertLess(publisher_index, delay_index)
        self.assertEqual(send_events, [("send", "ind_task_02", b"task")])

    def test_task_handshake_stops_after_ack(self):
        class Publisher:
            def __init__(self):
                self.sent = []

            def send_msg(self, payload, topic):
                self.sent.append((payload, topic))

        class Receiver:
            def receive_envelope(self, timeout):
                self.timeout = timeout
                return {"topic": "task_cbd"}

        publisher = Publisher()
        receiver = Receiver()
        self.assertTrue(wait_for_task_ack(receiver, publisher, "ind_task_01", 1.0))
        self.assertEqual(publisher.sent, [("ind_task_01", b"task")])
        self.assertGreaterEqual(receiver.timeout, 1)

    def test_task_handshake_timeout_is_bounded(self):
        class Publisher:
            def __init__(self):
                self.sent = 0

            def send_msg(self, payload, topic):
                del payload, topic
                self.sent += 1

        class Receiver:
            def receive_envelope(self, timeout):
                del timeout
                return None

        publisher = Publisher()
        with patch("tools.policies.runner.time.monotonic", side_effect=(0.0, 0.0, 1.1)):
            self.assertFalse(wait_for_task_ack(Receiver(), publisher, "ind_task_01", 1.0))
        self.assertEqual(publisher.sent, 1)

    def test_task_handshake_timeout_must_be_positive(self):
        with self.assertRaises(ValueError):
            wait_for_task_ack(object(), object(), "ind_task_01", 0.0)

    @unittest.skipUnless(Path("data/ind_task_01/train").is_dir(), "challenge data is not available")
    def test_real_hdf5_sample_has_rgbd_and_45d_state(self):
        train_refs, val_refs = build_episode_splits(
            "data", task_names=TASK_NAMES, val_ratio=0, max_episodes_per_task=1
        )
        self.assertEqual(len(val_refs), 0)
        stats = compute_multitask_stats(train_refs)
        dataset = MultiTaskEpisodicDataset(
            train_refs,
            stats,
            chunk_size=4,
            image_width=64,
            image_height=48,
            max_depth_meters=5.0,
        )
        image, depth, state, action, is_pad = dataset[0]
        self.assertEqual(tuple(image.shape), (1, 3, 48, 64))
        self.assertEqual(tuple(depth.shape), (1, 48, 64))
        self.assertEqual(tuple(state.shape), (MULTITASK_STATE_DIM,))
        self.assertEqual(tuple(action.shape), (4, 26))
        self.assertEqual(tuple(is_pad.shape), (4,))
        self.assertGreater(float(depth.max()), 0.0)

        robust_dataset = MultiTaskEpisodicDataset(
            train_refs,
            stats,
            chunk_size=4,
            image_width=64,
            image_height=48,
            max_depth_meters=5.0,
            use_aug=True,
            augmentation_config=RobustAugmentationConfig(),
            motion_sampling_alpha=2.0,
        )
        robust_image, robust_depth, robust_state, _, _ = robust_dataset[0]
        self.assertEqual(tuple(robust_image.shape), (1, 3, 48, 64))
        self.assertEqual(tuple(robust_depth.shape), (1, 48, 64))
        self.assertTrue(torch.isfinite(robust_depth).all())
        self.assertTrue(torch.isfinite(robust_state).all())
        self.assertGreaterEqual(float(robust_depth.min()), 0.0)
        self.assertLessEqual(float(robust_depth.max()), 5.0)
        self.assertEqual(set(robust_dataset.motion_cdfs), set(TASK_NAMES))


if __name__ == "__main__":
    unittest.main()

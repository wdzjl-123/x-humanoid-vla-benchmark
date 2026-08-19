import sys
import unittest
from unittest.mock import patch

import numpy as np

from tools.policies.act_policy import ACTPolicy
from tools.policies.runner import parse_args
from tools.eval_act_closed_loop import build_replay_observation


def _policy(mode: str, task_name: str = "ind_task_01") -> ACTPolicy:
    policy = ACTPolicy.__new__(ACTPolicy)
    policy.task_name = task_name
    policy.hand_state_mode = mode
    policy.hand_feedback = "measured" if mode == "measured" else "commanded"
    policy.image_color_order = "rgb"
    policy.norm_stats = {
        "qpos_mean": np.zeros(26, dtype=np.float32),
    }
    policy._last_l_hand, policy._last_r_hand = policy._select_hand_home()
    return policy


def _obs(left_hand, right_hand):
    return {
        "puppet": {
            "arm_left_position_raw": {"data": np.arange(7, dtype=np.float32)},
            "arm_right_position_raw": {"data": -np.arange(7, dtype=np.float32)},
            "end_effector_left_position_raw": {"data": left_hand},
            "end_effector_right_position_raw": {"data": right_hand},
        }
    }


class ACTHandStateTests(unittest.TestCase):
    def test_task_home_uses_task_specific_seed(self):
        policy = _policy("task_home")
        expected_l, expected_r = ACTPolicy._TASK_HAND_HOME["ind_task_01"]
        np.testing.assert_allclose(policy._last_l_hand, expected_l)
        np.testing.assert_allclose(policy._last_r_hand, expected_r)

    def test_legacy_home_reproduces_historical_seed(self):
        policy = _policy("legacy_home")
        np.testing.assert_allclose(policy._last_l_hand, ACTPolicy._LEGACY_L_HAND_HOME)
        np.testing.assert_allclose(policy._last_r_hand, ACTPolicy._LEGACY_R_HAND_HOME)

    def test_measured_mode_uses_finite_measurements_and_falls_back_per_joint(self):
        policy = _policy("measured")
        left = np.array([1.57, np.nan, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        right = np.array([1.57, 0.1, 0.2, np.inf, 0.4, 0.5], dtype=np.float32)
        qpos = policy._get_qpos(_obs(left, right))

        self.assertEqual(qpos.shape, (26,))
        self.assertAlmostEqual(qpos[7], 1.57)
        self.assertAlmostEqual(qpos[8], policy._last_l_hand[1])
        self.assertAlmostEqual(qpos[20], 1.57)
        self.assertAlmostEqual(qpos[23], policy._last_r_hand[3])
        self.assertTrue(np.all(np.isfinite(qpos)))

    def test_task_home_ignores_unreliable_measurements(self):
        policy = _policy("task_home")
        qpos = policy._get_qpos(
            _obs(np.full(6, 9.0, dtype=np.float32), np.full(6, 8.0, dtype=np.float32))
        )
        np.testing.assert_allclose(qpos[7:13], policy._last_l_hand)
        np.testing.assert_allclose(qpos[20:26], policy._last_r_hand)

    def test_legacy_seed_can_use_measured_feedback_after_start(self):
        policy = _policy("legacy_home")
        policy.hand_feedback = "measured"
        left = np.linspace(0.1, 0.6, 6, dtype=np.float32)
        right = np.linspace(0.2, 0.7, 6, dtype=np.float32)
        qpos = policy._get_qpos(_obs(left, right))

        np.testing.assert_allclose(qpos[7:13], left)
        np.testing.assert_allclose(qpos[20:26], right)

    def test_image_color_order_is_an_explicit_ab_switch(self):
        policy = _policy("task_home")
        policy.CAM_NAME = "camera_head"
        policy.IMG_W = 1
        policy.IMG_H = 1
        obs = {
            "camera_observations": {
                "color_images": {
                    "camera_head": np.array([[[10, 20, 30]]], dtype=np.uint8),
                },
            },
        }

        rgb = (policy._get_image(obs).squeeze().numpy() * 255).round().astype(int)
        policy.image_color_order = "bgr"
        bgr = (policy._get_image(obs).squeeze().numpy() * 255).round().astype(int)

        np.testing.assert_array_equal(rgb, np.array([10, 20, 30]))
        np.testing.assert_array_equal(bgr, np.array([30, 20, 10]))

    def test_temporal_priority_weights_are_ordered_as_configured(self):
        oldest = ACTPolicy.temporal_weights(3, decay=1.0, priority="oldest")
        newest = ACTPolicy.temporal_weights(3, decay=1.0, priority="newest")
        uniform = ACTPolicy.temporal_weights(3, decay=1.0, priority="uniform")

        self.assertGreater(oldest[0], oldest[-1])
        self.assertLess(newest[0], newest[-1])
        np.testing.assert_allclose(uniform, np.full(3, 1 / 3))

    def test_replay_observation_uses_aligned_state_and_rgb_image(self):
        target = np.arange(26, dtype=np.float32)[None]
        image_dict = {
            "color_images": {
                "camera_head": np.array([[[10, 20, 30]]], dtype=np.uint8),
            },
        }
        ctrl = {
            "puppet": {
                "arm_left_position_align": target[:, :7],
                "end_effector_left_position_align": target[:, 7:13],
                "arm_right_position_align": target[:, 13:20],
                "end_effector_right_position_align": target[:, 20:26],
            },
        }
        obs, replay_target = build_replay_observation(image_dict, ctrl, "camera_head")

        np.testing.assert_allclose(replay_target, target[0])
        np.testing.assert_allclose(obs["puppet"]["arm_right_position_raw"]["data"], target[0, 13:20])
        np.testing.assert_array_equal(
            obs["camera_observations"]["color_images"]["camera_head"],
            np.array([[[30, 20, 10]]], dtype=np.uint8),
        )

    def test_cli_exposes_the_hand_state_ab_switch(self):
        argv = [
            "runner.py", "--policy", "act", "--model-path", "policy.ckpt",
            "--act-hand-state", "legacy_home", "--act-debug-every", "17",
            "--act-hand-feedback", "measured",
            "--act-image-color-order", "bgr",
            "--act-action-horizon", "8", "--act-temporal-decay", "0.2",
            "--act-temporal-priority", "newest",
            "--task-publish-delay", "7.5",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.act_hand_state, "legacy_home")
        self.assertEqual(args.act_hand_feedback, "measured")
        self.assertEqual(args.act_debug_every, 17)
        self.assertEqual(args.act_image_color_order, "bgr")
        self.assertEqual(args.act_action_horizon, 8)
        self.assertEqual(args.act_temporal_decay, 0.2)
        self.assertEqual(args.act_temporal_priority, "newest")
        self.assertEqual(args.task_publish_delay, 7.5)

    def test_cli_defaults_to_the_validated_legacy_seed(self):
        argv = ["runner.py", "--policy", "act", "--model-path", "policy.ckpt"]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.act_hand_state, "legacy_home")
        self.assertIsNone(args.act_hand_feedback)
        self.assertEqual(args.task_publish_delay, 5.0)


if __name__ == "__main__":
    unittest.main()

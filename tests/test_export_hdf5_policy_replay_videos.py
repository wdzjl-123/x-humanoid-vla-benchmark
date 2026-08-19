import unittest

import numpy as np

from tools.export_r7_hdf5_policy_replay_videos import (
    ACTION_DIM,
    ExportError,
    derive_source_timing,
    validate_action_arrays,
)


class HDF5PolicyReplayVideoTests(unittest.TestCase):
    def test_33_34_millisecond_observation_timestamps_export_at_30_fps(self):
        timing = derive_source_timing(np.array([33, 66, 100, 133, 166], dtype=np.int64))
        self.assertEqual(timing.output_fps, 30)
        self.assertEqual(timing.median_interval_ms, 33.0)
        self.assertAlmostEqual(timing.source_hz, 1000.0 / 33.0)

    def test_action_diagnostics_respect_all_four_26d_groups(self):
        actions = np.zeros((2, ACTION_DIM), dtype=np.float32)
        targets = np.zeros((2, ACTION_DIM), dtype=np.float32)
        targets[0, :7] = 1.0
        targets[0, 7:13] = 2.0
        targets[0, 13:20] = 3.0
        targets[0, 20:] = 4.0
        trace = validate_action_arrays(actions, targets, np.array([0.01, 0.02]))
        self.assertEqual(trace.actions.shape, (2, ACTION_DIM))
        self.assertAlmostEqual(float(trace.group_mae["left_arm"][0]), 1.0)
        self.assertAlmostEqual(float(trace.group_mae["left_hand"][0]), 2.0)
        self.assertAlmostEqual(float(trace.group_mae["right_arm"][0]), 3.0)
        self.assertAlmostEqual(float(trace.group_mae["right_hand"][0]), 4.0)
        self.assertAlmostEqual(float(trace.frame_mae[0]), (7 + 12 + 21 + 24) / 26, places=6)

    def test_action_diagnostics_reject_timeline_mismatch(self):
        actions = np.zeros((3, ACTION_DIM), dtype=np.float32)
        with self.assertRaises(ExportError):
            validate_action_arrays(actions, np.zeros((2, ACTION_DIM), dtype=np.float32), np.zeros(3))
        with self.assertRaises(ExportError):
            validate_action_arrays(actions, actions, np.zeros(2))

    def test_timestamp_validation_rejects_non_30hz_sources(self):
        with self.assertRaises(ExportError):
            derive_source_timing(np.array([0, 100, 200], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()

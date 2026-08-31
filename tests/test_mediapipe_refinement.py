import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


def straight_hand():
    points = np.zeros((21, 3), dtype=np.float64)
    points[0] = [0.0, 0.0, 0.0]
    points[1:5] = [[0.3, 0.4, 0.0], [0.6, 0.7, 0.0], [0.9, 1.0, 0.0], [1.2, 1.3, 0.0]]
    points[5:9] = [[0.5, 1.0, 0.0], [0.5, 1.5, 0.0], [0.5, 2.0, 0.0], [0.5, 2.5, 0.0]]
    points[9:13] = [[0.0, 1.1, 0.0], [0.0, 1.7, 0.0], [0.0, 2.3, 0.0], [0.0, 2.9, 0.0]]
    points[13:17] = [[-0.3, 1.0, 0.0], [-0.3, 1.5, 0.0], [-0.3, 2.0, 0.0], [-0.3, 2.5, 0.0]]
    points[17:21] = [[-0.5, 0.9, 0.0], [-0.5, 1.3, 0.0], [-0.5, 1.7, 0.0], [-0.5, 2.1, 0.0]]
    return points


class GeometryTests(unittest.TestCase):
    def test_unmirrored_video_swaps_selfie_handedness(self):
        from mediapipe_refinement import source_side

        self.assertEqual(source_side("Left", input_mirrored=False), "right")
        self.assertEqual(source_side("Right", input_mirrored=False), "left")
        self.assertEqual(source_side("Left", input_mirrored=True), "left")

    def test_matching_landmarks_produce_neutral_hand_pose(self):
        from mediapipe_refinement import landmarks_to_hand_pose

        pose = landmarks_to_hand_pose(straight_hand(), straight_hand())

        np.testing.assert_allclose(pose, np.zeros((15, 3)), atol=1e-8)

    def test_index_bend_changes_only_index_base_joint(self):
        from mediapipe_refinement import landmarks_to_hand_pose

        reference = straight_hand()
        target = reference.copy()
        target[6:9] = [[1.0, 1.0, 0.0], [1.5, 1.0, 0.0], [2.0, 1.0, 0.0]]

        pose = landmarks_to_hand_pose(target, reference)

        np.testing.assert_allclose(pose[0], [0.0, 0.0, -np.pi / 2], atol=1e-7)
        np.testing.assert_allclose(pose[1:], np.zeros((14, 3)), atol=1e-7)

    def test_degenerate_palm_is_rejected(self):
        from mediapipe_refinement import landmarks_to_hand_pose

        with self.assertRaisesRegex(ValueError, "palm"):
            landmarks_to_hand_pose(np.zeros((21, 3)), straight_hand())


class FusionTests(unittest.TestCase):
    def test_fusion_replaces_only_requested_hand_frames(self):
        from mediapipe_refinement import fuse_hand_poses

        motion = {
            "body_pose": np.arange(2 * 21 * 3).reshape(2, 21, 3),
            "left_hand_pose": np.zeros((2, 15, 3)),
            "right_hand_pose": np.zeros((2, 15, 3)),
            "source_frame_number": np.array([1, 2]),
        }
        left = np.full((15, 3), 0.25)
        right = np.full((15, 3), -0.5)

        refined = fuse_hand_poses(motion, {2: {"left": left, "right": right}}, required_frames=[2])

        np.testing.assert_array_equal(refined["body_pose"], motion["body_pose"])
        np.testing.assert_array_equal(refined["left_hand_pose"][0], np.zeros((15, 3)))
        np.testing.assert_array_equal(refined["left_hand_pose"][1], left)
        np.testing.assert_array_equal(refined["right_hand_pose"][1], right)

    def test_fusion_preserves_unpredicted_side(self):
        from mediapipe_refinement import fuse_hand_poses

        motion = {
            "left_hand_pose": np.zeros((1, 15, 3)),
            "right_hand_pose": np.zeros((1, 15, 3)),
            "source_frame_number": np.array([1]),
        }
        refined = fuse_hand_poses(motion, {1: {"left": np.ones((15, 3))}})
        np.testing.assert_array_equal(refined["left_hand_pose"][0], np.ones((15, 3)))
        np.testing.assert_array_equal(refined["right_hand_pose"][0], np.zeros((15, 3)))


class ProgressTests(unittest.TestCase):
    def test_corrupt_landmark_file_is_not_complete(self):
        from mediapipe_refinement import reconcile_landmark_progress

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "landmarks").mkdir()
            np.savez_compressed(root / "landmarks/00001_left.npz", world_landmarks=np.zeros((20, 3)))

            progress = reconcile_landmark_progress(root, [1])

            self.assertEqual(progress["frames"]["1"]["left"]["status"], "error")
            self.assertEqual(progress["frames"]["1"]["right"]["status"], "pending")


class CommandTests(unittest.TestCase):
    def test_help_does_not_import_mediapipe_or_smplx(self):
        script = Path(__file__).parents[1] / "mediapipe_refinement.py"

        result = subprocess.run([sys.executable, str(script), "--help"], text=True, capture_output=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("detect", result.stdout)
        self.assertIn("fit", result.stdout)


if __name__ == "__main__":
    unittest.main()

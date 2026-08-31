import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


class GeometryTests(unittest.TestCase):
    def test_unmirrored_video_swaps_selfie_handedness(self):
        from mediapipe_refinement import source_side

        self.assertEqual(source_side("Left", input_mirrored=False), "right")
        self.assertEqual(source_side("Right", input_mirrored=False), "left")
        self.assertEqual(source_side("Left", input_mirrored=True), "left")

class OptimizationTests(unittest.TestCase):
    def test_project_points_returns_normalized_image_coordinates(self):
        import torch
        from mediapipe_refinement import project_points

        joints = torch.tensor([[[1.0, 2.0, 2.0]]])
        focal = torch.tensor([[100.0, 200.0]])
        princpt = torch.tensor([[20.0, 40.0]])

        projected = project_points(joints, focal, princpt, image_size=(200, 400))

        torch.testing.assert_close(projected, torch.tensor([[[0.35, 0.60]]]))

    def test_sequence_objective_is_zero_for_an_exact_unchanged_pose(self):
        import torch
        from mediapipe_refinement import sequence_objective

        projected = torch.zeros((2, 2, 21, 2))
        hand_pose = torch.zeros((2, 2, 15, 3))
        wrist_pose = torch.zeros((2, 2, 3))

        losses = sequence_objective(
            projected=projected,
            target=projected,
            weights=torch.ones((2, 2, 21)),
            hand_pose=hand_pose,
            initial_hand_pose=hand_pose,
            wrist_pose=wrist_pose,
            initial_wrist_pose=wrist_pose,
            frame_numbers=torch.tensor([1.0, 2.0]),
            maximum_joint_degrees=120.0,
            initial_pose_weight=0.02,
            temporal_weight=0.05,
            wrist_weight=0.1,
        )

        self.assertEqual(float(losses["total"]), 0.0)

    def test_sequence_objective_ignores_missing_hand_observations(self):
        import torch
        from mediapipe_refinement import sequence_objective

        projected = torch.ones((1, 2, 21, 2))
        target = torch.zeros_like(projected)
        hand_pose = torch.zeros((1, 2, 15, 3))
        wrist_pose = torch.zeros((1, 2, 3))
        losses = sequence_objective(
            projected=projected,
            target=target,
            weights=torch.zeros((1, 2, 21)),
            hand_pose=hand_pose,
            initial_hand_pose=hand_pose,
            wrist_pose=wrist_pose,
            initial_wrist_pose=wrist_pose,
            frame_numbers=torch.tensor([1.0]),
            maximum_joint_degrees=120.0,
            initial_pose_weight=0.02,
            temporal_weight=0.05,
            wrist_weight=0.1,
        )

        self.assertEqual(float(losses["reprojection"]), 0.0)

    def test_sequence_objective_penalizes_temporal_jumps_and_joint_limits(self):
        import torch
        from mediapipe_refinement import sequence_objective

        hand_pose = torch.zeros((3, 2, 15, 3))
        hand_pose[1, 0, 0, 0] = torch.pi
        projected = torch.zeros((3, 2, 21, 2))
        wrist_pose = torch.zeros((3, 2, 3))
        losses = sequence_objective(
            projected=projected,
            target=projected,
            weights=torch.ones((3, 2, 21)),
            hand_pose=hand_pose,
            initial_hand_pose=torch.zeros_like(hand_pose),
            wrist_pose=wrist_pose,
            initial_wrist_pose=wrist_pose,
            frame_numbers=torch.tensor([1.0, 2.0, 4.0]),
            maximum_joint_degrees=120.0,
            initial_pose_weight=0.0,
            temporal_weight=1.0,
            wrist_weight=0.0,
        )

        self.assertGreater(float(losses["temporal"]), 0.0)
        self.assertGreater(float(losses["angle_limit"]), 0.0)

    def test_optimizer_moves_an_observed_hand_and_preserves_an_unobserved_hand(self):
        import torch
        from types import SimpleNamespace
        from mediapipe_refinement import optimize_hand_sequence

        class FakeModel(torch.nn.Module):
            def forward(self, left_hand_pose, right_hand_pose, **_):
                def joints(pose):
                    x = pose[:, :1].expand(-1, 21)
                    return torch.stack((x, torch.zeros_like(x), torch.ones_like(x)), dim=-1)

                return SimpleNamespace(joints=torch.cat((joints(left_hand_pose), joints(right_hand_pose)), dim=1))

        motion = {
            "global_orient": np.zeros((1, 3)),
            "body_pose": np.zeros((1, 21, 3)),
            "left_hand_pose": np.zeros((1, 15, 3)),
            "right_hand_pose": np.zeros((1, 15, 3)),
            "jaw_pose": np.zeros((1, 3)),
            "left_eye_pose": np.zeros((1, 3)),
            "right_eye_pose": np.zeros((1, 3)),
            "betas": np.zeros((1, 10)),
            "expression": np.zeros((1, 10)),
            "translation": np.zeros((1, 3)),
        }
        target = torch.zeros((1, 2, 21, 2))
        target[:, 0, :, 0] = 0.1
        weights = torch.zeros((1, 2, 21))
        weights[:, 0] = 1.0

        result = optimize_hand_sequence(
            model=FakeModel(),
            motion=motion,
            indexes=[0],
            joint_indexes=np.arange(42).reshape(2, 21),
            target=target,
            weights=weights,
            focal=torch.tensor([[100.0, 100.0]]),
            princpt=torch.zeros((1, 2)),
            image_size=(200, 200),
            frame_numbers=torch.tensor([1.0]),
            optimization_steps=30,
            learning_rate=0.1,
            maximum_joint_degrees=120.0,
            initial_pose_weight=0.0,
            temporal_weight=0.0,
            wrist_weight=0.1,
            device="cpu",
        )

        self.assertGreater(result["left_hand_pose"][0, 0, 0], 0.05)
        np.testing.assert_allclose(result["right_hand_pose"], np.zeros((1, 15, 3)), atol=1e-7)

    def test_optimizer_rejects_non_finite_loss(self):
        import torch
        from types import SimpleNamespace
        from mediapipe_refinement import optimize_hand_sequence

        class InvalidModel(torch.nn.Module):
            def forward(self, left_hand_pose, **_):
                joints = left_hand_pose[:, :1, None].expand(-1, 42, 3) * torch.nan
                return SimpleNamespace(joints=joints)

        motion = {
            "global_orient": np.zeros((1, 3)), "body_pose": np.zeros((1, 21, 3)),
            "left_hand_pose": np.zeros((1, 15, 3)), "right_hand_pose": np.zeros((1, 15, 3)),
            "jaw_pose": np.zeros((1, 3)), "left_eye_pose": np.zeros((1, 3)),
            "right_eye_pose": np.zeros((1, 3)), "betas": np.zeros((1, 10)),
            "expression": np.zeros((1, 10)), "translation": np.zeros((1, 3)),
        }
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            optimize_hand_sequence(
                InvalidModel(), motion, [0], np.arange(42).reshape(2, 21),
                torch.zeros((1, 2, 21, 2)), torch.ones((1, 2, 21)),
                torch.ones((1, 2)), torch.zeros((1, 2)), (1, 1), torch.tensor([1.0]),
                1, 0.1, 120.0, 0.02, 0.05, 0.1, "cpu",
            )


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

        fit_help = subprocess.run(
            [sys.executable, str(script), "fit", "--help"], text=True, capture_output=True
        )
        self.assertEqual(fit_help.returncode, 0, fit_help.stderr)
        self.assertIn("--meta-dir", fit_help.stdout)
        self.assertIn("--optimization-steps", fit_help.stdout)
        self.assertIn("--temporal-weight", fit_help.stdout)


if __name__ == "__main__":
    unittest.main()

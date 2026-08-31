import tempfile
import unittest
from pathlib import Path
import subprocess
import sys

import numpy as np


class ProjectionTests(unittest.TestCase):
    def test_project_points_matches_smplerx_camera(self):
        from hamer_refinement import project_points

        points = np.array([[1.0, 2.0, 4.0], [-1.0, -2.0, 2.0]])

        actual = project_points(points, focal=[100, 200], princpt=[10, 20])

        np.testing.assert_allclose(actual, [[35, 120], [-40, -180]])

    def test_project_points_rejects_non_positive_depth(self):
        from hamer_refinement import project_points

        with self.assertRaisesRegex(ValueError, "depth"):
            project_points(np.array([[1.0, 2.0, 0.0]]), [100, 100], [0, 0])

    def test_square_box_stays_square_at_image_boundary(self):
        from hamer_refinement import square_box_from_points

        actual = square_box_from_points(
            np.array([[0.0, 10.0], [30.0, 40.0]]),
            width=1280,
            height=720,
            padding=2.0,
            minimum=32,
        )

        np.testing.assert_allclose(actual, [0.0, 0.0, 60.0, 60.0])

    def test_hand_boxes_use_left_and_right_vertex_mappings(self):
        from hamer_refinement import hand_boxes_from_vertices

        vertices = np.array(
            [
                [-1.0, -1.0, 10.0],
                [1.0, 1.0, 10.0],
                [3.0, 2.0, 10.0],
                [5.0, 4.0, 10.0],
            ]
        )
        mapping = {"left_hand": [0, 1], "right_hand": [2, 3]}

        boxes = hand_boxes_from_vertices(
            vertices,
            mapping,
            focal=[100, 100],
            princpt=[100, 100],
            width=300,
            height=300,
            padding=1.0,
            minimum=20,
        )

        np.testing.assert_allclose(boxes["left"], [90, 90, 110, 110])
        np.testing.assert_allclose(boxes["right"], [130, 120, 150, 140])


class ProgressTests(unittest.TestCase):
    @staticmethod
    def write_prediction(path, frame, is_right):
        np.savez_compressed(
            path,
            hand_pose=np.repeat(np.eye(3)[None], 15, axis=0),
            global_orient=np.eye(3)[None],
            box=np.array([10.0, 20.0, 50.0, 60.0]),
            is_right=np.asarray(is_right),
            frame_number=np.asarray(frame),
        )

    def test_progress_recovers_completed_hand_files(self):
        from hamer_refinement import reconcile_hand_progress

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mano").mkdir()
            self.write_prediction(root / "mano/00001_left.npz", 1, False)
            self.write_prediction(root / "mano/00001_right.npz", 1, True)

            progress = reconcile_hand_progress(root, [1, 2])

            self.assertEqual(progress["frames"]["1"]["left"]["status"], "ok")
            self.assertEqual(progress["frames"]["1"]["right"]["status"], "ok")
            self.assertEqual(progress["frames"]["2"]["left"]["status"], "pending")
            self.assertTrue((root / "progress.json").is_file())

    def test_progress_marks_corrupt_prediction_as_error(self):
        from hamer_refinement import reconcile_hand_progress

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mano").mkdir()
            np.savez_compressed(
                root / "mano/00001_left.npz",
                hand_pose=np.zeros((14, 3, 3)),
            )

            progress = reconcile_hand_progress(root, [1])

            item = progress["frames"]["1"]["left"]
            self.assertEqual(item["status"], "error")
            self.assertIn("hand_pose", item["error"])


class CommandTests(unittest.TestCase):
    def test_help_does_not_require_hamer_dependencies(self):
        script = Path(__file__).parents[1] / "hamer_refinement.py"

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--checkpoint", result.stdout)
        self.assertIn("--vertex-ids", result.stdout)


if __name__ == "__main__":
    unittest.main()

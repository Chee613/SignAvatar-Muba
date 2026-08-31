import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class ProgressTests(unittest.TestCase):
    def test_first_unprocessed_frame_skips_terminal_statuses(self):
        from suhu_pilot import first_unprocessed_frame

        progress = {
            "frames": {
                "1": {"status": "ok"},
                "2": {"status": "no_detection"},
                "3": {"status": "error"},
            }
        }

        self.assertEqual(first_unprocessed_frame(progress, total_frames=5), 4)

    def test_reconcile_progress_reads_persisted_artifacts(self):
        from suhu_pilot import reconcile_progress

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            for name in ("smplx", "meta", "overlays"):
                (run_dir / name).mkdir()
            (run_dir / "smplx/00001_0.npz").touch()
            (run_dir / "meta/00001_0.json").touch()
            (run_dir / "overlays/000001.jpg").touch()
            (run_dir / "overlays/000002.jpg").touch()

            progress = reconcile_progress(run_dir, total_frames=3)

            self.assertEqual(progress["frames"]["1"]["status"], "ok")
            self.assertEqual(progress["frames"]["2"]["status"], "no_detection")
            self.assertEqual(progress["frames"]["3"]["status"], "pending")
            self.assertEqual(
                json.loads((run_dir / "progress.json").read_text())["frames"],
                progress["frames"],
            )


class ConfigTests(unittest.TestCase):
    def test_repository_config_matches_suhu_contract(self):
        from suhu_pilot import load_config

        config = load_config(Path(__file__).parents[1] / "config/suhu_pilot.json")

        self.assertEqual(config["sign_id"], "SUHU")
        self.assertEqual(config["source_video"], "Suhu.mp4")
        self.assertEqual(config["processing_fps"], 30)
        self.assertEqual(config["model"], "smpler_x_h32")
        self.assertEqual(config["expected_source_frames"], 146)
        self.assertEqual(config["reviewer_count"], 3)

    def test_source_probe_must_match_config(self):
        from suhu_pilot import validate_source_probe

        config = {
            "source_frame_width": 1280,
            "source_frame_height": 720,
            "source_fps": 50,
            "expected_source_frames": 146,
        }
        probe = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1280,
                    "height": 720,
                    "avg_frame_rate": "50/1",
                    "nb_frames": "146",
                    "duration": "2.920000",
                }
            ]
        }

        self.assertEqual(validate_source_probe(probe, config)["frame_count"], 146)
        probe["streams"][0]["width"] = 640
        with self.assertRaisesRegex(ValueError, "width"):
            validate_source_probe(probe, config)

    def test_validate_file_accepts_expected_size_and_sha256(self):
        from suhu_pilot import validate_file

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.bin"
            path.write_bytes(b"abc")

            result = validate_file(
                path,
                expected_size=3,
                expected_sha256="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

        self.assertEqual(result["size"], 3)

    def test_validate_file_rejects_corrupt_download(self):
        from suhu_pilot import validate_file

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.bin"
            path.write_bytes(b"abc")

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                validate_file(path, expected_size=3, expected_sha256="0" * 64)


class MotionTests(unittest.TestCase):
    @staticmethod
    def write_frame(path, beta):
        np.savez(
            path,
            global_orient=np.zeros((1, 3)),
            body_pose=np.zeros((21, 3)),
            left_hand_pose=np.zeros((15, 3)),
            right_hand_pose=np.zeros((15, 3)),
            jaw_pose=np.zeros((1, 3)),
            leye_pose=np.zeros((1, 3)),
            reye_pose=np.zeros((1, 3)),
            betas=np.full((1, 10), beta),
            expression=np.zeros((1, 10)),
            transl=np.zeros((1, 3)),
        )

    def test_consolidate_motion_stacks_frames_and_freezes_shape(self):
        from suhu_pilot import consolidate_motion

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "smplx").mkdir()
            for frame, beta in enumerate((1.0, 2.0, 9.0), start=1):
                self.write_frame(run_dir / f"smplx/{frame:05d}_0.npz", beta)

            outputs = consolidate_motion(run_dir, total_frames=3, fps=30)
            raw = dict(np.load(outputs["raw"]))
            clean = dict(np.load(outputs["clean"]))

            self.assertEqual(raw["body_pose"].shape, (3, 21, 3))
            np.testing.assert_array_equal(raw["source_frame_number"], [1, 2, 3])
            np.testing.assert_array_equal(raw["betas"][:, 0], [1.0, 2.0, 9.0])
            np.testing.assert_array_equal(clean["betas"][:, 0], [2.0, 2.0, 2.0])
            self.assertEqual(int(clean["fps"]), 30)

    def test_consolidate_motion_rejects_missing_frames(self):
        from suhu_pilot import consolidate_motion

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "smplx").mkdir()
            self.write_frame(run_dir / "smplx/00001_0.npz", 1.0)

            with self.assertRaisesRegex(ValueError, "2"):
                consolidate_motion(run_dir, total_frames=2, fps=30)

    def test_motion_qa_flags_large_hand_jump(self):
        from suhu_pilot import analyze_motion

        frames = 5
        motion = {
            "global_orient": np.zeros((frames, 3)),
            "body_pose": np.zeros((frames, 21, 3)),
            "left_hand_pose": np.zeros((frames, 15, 3)),
            "right_hand_pose": np.zeros((frames, 15, 3)),
            "betas": np.zeros((frames, 10)),
        }
        motion["right_hand_pose"][2:, 0, 0] = np.pi / 2
        motion["body_pose"][2:, 19, 1] = np.pi / 2

        report = analyze_motion(motion)

        self.assertEqual(report["invalid_value_count"], 0)
        self.assertEqual(report["jumps"]["left_wrist_pose"]["outlier_frames"], [3])
        self.assertEqual(report["jumps"]["right_hand_pose"]["outlier_frames"], [3])
        self.assertAlmostEqual(
            report["jumps"]["right_hand_pose"]["maximum_degrees"], 90.0
        )


class ReviewTests(unittest.TestCase):
    def test_three_recognized_reviews_pass(self):
        from suhu_pilot import assess_reviews

        rows = [
            {
                "reviewer_id": str(index),
                "identified_word": "SUHU",
                "handshape": "2",
                "orientation": "2",
                "location": "2",
                "movement": "2",
                "non_manual": "1" if index == 1 else "2",
            }
            for index in range(1, 4)
        ]

        result = assess_reviews(rows, sign_id="SUHU", reviewer_count=3)

        self.assertTrue(result["passed"])
        self.assertEqual(result["recognized_count"], 3)
        self.assertAlmostEqual(result["category_averages"]["non_manual"], 5 / 3)

    def test_zero_scores_and_wrong_gloss_fail(self):
        from suhu_pilot import assess_reviews

        rows = [
            {
                "reviewer_id": str(index),
                "identified_word": "SUHU" if index < 3 else "PANAS",
                "handshape": "0" if index < 3 else "2",
                "orientation": "2",
                "location": "2",
                "movement": "2",
                "non_manual": "2",
            }
            for index in range(1, 4)
        ]

        result = assess_reviews(rows, sign_id="SUHU", reviewer_count=3)

        self.assertFalse(result["passed"])
        self.assertIn("recognition", result["failures"])
        self.assertIn("handshape_zeroes", result["failures"])


class NotebookTests(unittest.TestCase):
    def test_colab_notebook_is_complete_and_python_cells_compile(self):
        notebook_path = Path(__file__).parents[1] / "notebooks/suhu_pilot_colab.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        required_ids = {
            "settings",
            "drive",
            "preflight",
            "environment",
            "public-assets-download",
            "private-assets",
            "source-validation",
            "preprocess",
            "representative-frame",
            "resumable-inference",
            "missing-policy",
            "consolidate-qa",
            "comparison-renders",
            "avatar-render",
            "blind-review",
        }
        cells = {cell["id"]: cell for cell in notebook["cells"]}

        self.assertEqual(required_ids - cells.keys(), set())
        self.assertLess(
            list(cells).index("public-assets-download"),
            list(cells).index("private-assets"),
        )
        public_assets_source = "".join(cells["public-assets-download"]["source"])
        self.assertIn("--continue-at", public_assets_source)
        self.assertIn("smpler_x_h32.pth.tar", public_assets_source)
        self.assertIn("faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth", public_assets_source)
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            self.assertNotIn("TODO", source)
            self.assertNotIn("TBD", source)
            if cell["cell_type"] == "code":
                compile(source, cell["id"], "exec")


if __name__ == "__main__":
    unittest.main()

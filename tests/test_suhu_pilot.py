import json
import subprocess
import sys
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
        self.assertEqual(config["hand_refinement"]["preview_frames"], [16, 25, 38, 55, 73])
        self.assertEqual(config["hand_refinement"]["method"], "mediapipe_smplx_optimization")
        self.assertIs(config["hand_refinement"]["input_mirrored"], False)
        self.assertEqual(config["hand_refinement"]["minimum_detection_confidence"], 0.5)
        self.assertEqual(config["hand_refinement"]["minimum_tracking_confidence"], 0.5)
        self.assertEqual(config["hand_refinement"]["maximum_joint_degrees"], 120)
        self.assertEqual(config["hand_refinement"]["optimization_steps"], 600)
        self.assertEqual(config["hand_refinement"]["learning_rate"], 0.01)
        self.assertEqual(config["hand_refinement"]["initial_pose_weight"], 0.02)
        self.assertEqual(config["hand_refinement"]["temporal_weight"], 0.05)
        self.assertEqual(config["hand_refinement"]["wrist_weight"], 0.1)

    def test_hand_confidence_must_be_a_probability(self):
        from suhu_pilot import load_config

        config_path = Path(__file__).parents[1] / "config/suhu_pilot.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["hand_refinement"].update(
            method="mediapipe_smplx_optimization",
            minimum_detection_confidence=1.1,
            minimum_tracking_confidence=0.5,
            maximum_joint_degrees=150,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "minimum_detection_confidence"):
                load_config(path)

    def test_hand_optimizer_requires_positive_steps(self):
        from suhu_pilot import load_config

        config_path = Path(__file__).parents[1] / "config/suhu_pilot.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["hand_refinement"]["optimization_steps"] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "optimization_steps"):
                load_config(path)

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

    def test_validate_file_records_hash_when_only_size_is_known(self):
        from suhu_pilot import validate_file

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.bin"
            path.write_bytes(b"abc")

            result = validate_file(path, expected_size=3)

        self.assertEqual(
            result["sha256"],
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )


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


class HandRefinementTests(unittest.TestCase):
    @staticmethod
    def motion(frames=2):
        return {
            "global_orient": np.arange(frames * 3).reshape(frames, 3),
            "body_pose": np.arange(frames * 21 * 3).reshape(frames, 21, 3),
            "left_hand_pose": np.zeros((frames, 15, 3)),
            "right_hand_pose": np.zeros((frames, 15, 3)),
            "betas": np.ones((frames, 10)),
            "source_frame_number": np.arange(1, frames + 1),
        }

    @staticmethod
    def rotations(z_degrees):
        angle = np.radians(z_degrees)
        matrix = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return np.repeat(matrix[None], 15, axis=0)

    def test_left_mano_rotations_are_mirrored(self):
        from suhu_pilot import mano_hand_pose_to_axis_angle

        actual = mano_hand_pose_to_axis_angle(self.rotations(90), is_right=False)

        np.testing.assert_allclose(actual, np.tile([0.0, 0.0, -np.pi / 2], (15, 1)))

    def test_right_mano_rotations_keep_their_coordinate_system(self):
        from suhu_pilot import mano_hand_pose_to_axis_angle

        actual = mano_hand_pose_to_axis_angle(self.rotations(90), is_right=True)

        np.testing.assert_allclose(actual, np.tile([0.0, 0.0, np.pi / 2], (15, 1)))

    def test_fusion_changes_only_hand_pose_arrays(self):
        from suhu_pilot import fuse_hand_predictions

        motion = self.motion()
        predictions = {
            frame: {"left": self.rotations(30), "right": self.rotations(60)}
            for frame in (1, 2)
        }

        refined = fuse_hand_predictions(motion, predictions)

        for key in motion:
            if key not in {"left_hand_pose", "right_hand_pose"}:
                np.testing.assert_array_equal(refined[key], motion[key])
        np.testing.assert_allclose(refined["left_hand_pose"][:, :, 2], -np.pi / 6)
        np.testing.assert_allclose(refined["right_hand_pose"][:, :, 2], np.pi / 3)
        self.assertFalse(np.shares_memory(refined["body_pose"], motion["body_pose"]))

    def test_fusion_rejects_a_missing_side(self):
        from suhu_pilot import fuse_hand_predictions

        with self.assertRaisesRegex(ValueError, "frame 1 right"):
            fuse_hand_predictions(
                self.motion(frames=1),
                {1: {"left": self.rotations(30)}},
            )

    def test_mano_conversion_rejects_invalid_values(self):
        from suhu_pilot import mano_hand_pose_to_axis_angle

        rotations = self.rotations(30)
        rotations[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            mano_hand_pose_to_axis_angle(rotations, is_right=True)


class RendererCommandTests(unittest.TestCase):
    def test_help_exposes_preview_and_flat_hand_options_without_render_dependencies(self):
        script = Path(__file__).parents[1] / "render_smplx.py"

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        for option in (
            "--motion",
            "--model-root",
            "--meta-dir",
            "--output-dir",
            "--frames",
            "--flat-hand-mean",
        ):
            self.assertIn(option, result.stdout)

    def test_renderer_rejects_duplicate_frame_numbers_before_loading_dependencies(self):
        script = Path(__file__).parents[1] / "render_smplx.py"

        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--motion",
                "missing.npz",
                "--model-root",
                "missing-models",
                "--meta-dir",
                "missing-meta",
                "--output-dir",
                "output",
                "--frames",
                "1,1",
            ],
            text=True,
            capture_output=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unique positive integers", result.stderr)


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
    def test_code_cells_have_visible_numbered_labels(self):
        notebook_path = Path(__file__).parents[1] / "notebooks/suhu_pilot_colab.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

        for number, cell in enumerate(code_cells, start=1):
            source = "".join(cell["source"])
            self.assertTrue(
                source.startswith(f"# Cell {number} — "),
                f'{cell["id"]} is missing its visible cell label',
            )

    def test_avatar_render_uses_model_root_and_persists_subprocess_errors(self):
        notebook_path = Path(__file__).parents[1] / "notebooks/suhu_pilot_colab.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cells = {cell["id"]: cell for cell in notebook["cells"]}
        avatar_source = "".join(cells["avatar-render"]["source"])

        self.assertIn("SMPLERX_DIR / 'common/utils/human_model_files'", avatar_source)
        self.assertNotIn("SMPLERX_DIR / 'common/utils/human_model_files/smplx'", avatar_source)
        self.assertIn("capture_output=True", avatar_source)
        self.assertIn("avatar_render.log", avatar_source)
        self.assertIn("avatar_result.stderr", avatar_source)

    def test_colab_repairs_torchgeometry_boolean_mask_operations(self):
        notebook_path = Path(__file__).parents[1] / "notebooks/suhu_pilot_colab.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cells = {cell["id"]: cell for cell in notebook["cells"]}
        environment_source = "".join(cells["environment"]["source"])

        self.assertIn("torchgeometry/core/conversions.py", environment_source)
        self.assertIn("mask_d2 * (~mask_d0_d1)", environment_source)
        self.assertIn("torchgeometry-bool-mask=v1", environment_source)

    def test_colab_pins_yapf_for_mmcv_compatibility(self):
        notebook_path = Path(__file__).parents[1] / "notebooks/suhu_pilot_colab.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cells = {cell["id"]: cell for cell in notebook["cells"]}
        environment_source = "".join(cells["environment"]["source"])

        self.assertIn("'yapf==0.40.1'", environment_source)
        self.assertIn("yapf=0.40.1", environment_source)

    def test_representative_inference_persists_subprocess_error_output(self):
        notebook_path = Path(__file__).parents[1] / "notebooks/suhu_pilot_colab.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cells = {cell["id"]: cell for cell in notebook["cells"]}
        representative_source = "".join(cells["representative-frame"]["source"])

        self.assertIn("capture_output=True", representative_source)
        self.assertIn("representative_frame.log", representative_source)
        self.assertIn("result.stderr", representative_source)

    def test_colab_anchors_upstream_inference_working_directory(self):
        notebook_path = Path(__file__).parents[1] / "notebooks/suhu_pilot_colab.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cells = {cell["id"]: cell for cell in notebook["cells"]}
        environment_source = "".join(cells["environment"]["source"])

        self.assertIn("inference_script = SMPLERX_DIR / 'main/inference.py'", environment_source)
        self.assertIn("os.chdir(osp.dirname(osp.abspath(__file__)))", environment_source)

    def test_colab_environment_pins_mkl_and_repairs_existing_environment(self):
        notebook_path = Path(__file__).parents[1] / "notebooks/suhu_pilot_colab.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cells = {cell["id"]: cell for cell in notebook["cells"]}
        environment_source = "".join(cells["environment"]["source"])

        self.assertGreaterEqual(environment_source.count("'mkl=2024.0'"), 2)
        self.assertIn("install_marker.read_text", environment_source)

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
            "mediapipe-setup",
            "hand-preview",
            "hand-full-refinement",
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

    def test_colab_gates_two_hand_refinement_before_final_render(self):
        notebook_path = Path(__file__).parents[1] / "notebooks/suhu_pilot_colab.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cells = {cell["id"]: cell for cell in notebook["cells"]}
        order = list(cells)

        self.assertLess(order.index("mediapipe-setup"), order.index("hand-preview"))
        self.assertLess(order.index("hand-preview"), order.index("hand-full-refinement"))
        self.assertLess(order.index("hand-full-refinement"), order.index("avatar-render"))

        setup = "".join(cells["mediapipe-setup"]["source"])
        self.assertIn("Python 3.10", setup)
        self.assertIn("mediapipe==1.0.1", setup)
        self.assertIn("hand_landmarker.task", setup)
        self.assertNotIn("HaMeR", setup)
        self.assertNotIn("MANO_RIGHT.pkl", setup)

        preview = "".join(cells["hand-preview"]["source"])
        for frame in (16, 25, 38, 55, 73):
            self.assertIn(str(frame), preview)
        self.assertIn("mediapipe_refinement.py", preview)
        self.assertIn("'detect'", preview)
        self.assertIn("'fit'", preview)
        self.assertIn("input_mirrored", preview)
        self.assertIn("'--meta-dir'", preview)
        self.assertIn("'--optimization-steps'", preview)
        self.assertIn("'--temporal-weight'", preview)
        self.assertIn("render_smplx.py", preview)
        self.assertIn("HAND_REFINEMENT_CLEARED", preview)
        self.assertNotIn("--flat-hand-mean", preview)

        full = "".join(cells["hand-full-refinement"]["source"])
        self.assertIn("cleared.json", full)
        self.assertIn("mediapipe_refinement.py", full)
        self.assertIn("'detect'", full)
        self.assertIn("'fit'", full)
        self.assertIn("'--meta-dir'", full)
        self.assertIn("'--optimization-steps'", full)
        self.assertIn("suhu_motion_hand_refined.npz", full)
        self.assertNotIn("interpolate", full.lower())
        self.assertIn("optimization.json", full)

        avatar = "".join(cells["avatar-render"]["source"])
        self.assertIn("suhu_motion_hand_refined.npz", avatar)
        self.assertNotIn("--flat-hand-mean", avatar)


if __name__ == "__main__":
    unittest.main()

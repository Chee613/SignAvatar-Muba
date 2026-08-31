# HaMeR Hand Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a resumable HaMeR stage that replaces only the SUHU finger poses after a five-frame visual gate.

**Architecture:** Keep SMPLer-X and HaMeR in separate Colab environments. A lightweight runner projects SMPL-X hand vertices to obtain crops, sends those crops through HaMeR, and persists MANO rotation matrices; pure NumPy helpers mirror the left hand and fuse both 15-joint poses into a copy of the consolidated SMPL-X motion. A reusable renderer produces original, preview, and final refined avatar frames.

**Tech Stack:** Python 3.10, NumPy, SciPy, PyTorch 2.0.0, CUDA 11.7, HaMeR, MANO, SMPL-X, OpenCV, pyrender, FFmpeg, Google Colab, Google Drive.

**Spec:** `docs/superpowers/specs/2026-08-31-hamer-hand-refinement-design.md`

## Global Constraints

- Pin HaMeR to `3a01849f4148352e9260b69bf28b65d1671a4905` and Mano2Smpl-X reference logic to `901c124b9163294449b44821a642ae8e99d41cb0`.
- Keep the existing SMPLer-X commit and environment unchanged.
- Keep `MANO_RIGHT.pkl`, checkpoints, source video, and generated artifacts out of Git.
- Process preview frames `[16, 25, 38, 55, 73]` before the full clip.
- Preserve every SMPL-X field except `left_hand_pose` and `right_hand_pose`.
- Do not smooth or interpolate missing hand predictions.
- Render refined MANO poses with `use_pca=False` and `flat_hand_mean=True`.

---

### Task 1: Pure MANO conversion and fusion

**Files:**
- Modify: `suhu_pilot.py`
- Modify: `config/suhu_pilot.json`
- Test: `tests/test_suhu_pilot.py`

**Interfaces:**
- Consumes: HaMeR hand rotations shaped `[15, 3, 3]` and consolidated motion dictionaries.
- Produces: `mano_hand_pose_to_axis_angle(rotations, is_right) -> np.ndarray[15, 3]` and `fuse_hand_predictions(motion, predictions, required_frames=None) -> dict[str, np.ndarray]`.

- [ ] **Step 1: Write failing conversion tests**

```python
def test_left_mano_rotations_are_mirrored_before_axis_angle(self):
    from suhu_pilot import _axis_angle_to_matrix, mano_hand_pose_to_axis_angle
    source = np.zeros((15, 3)); source[0] = [0.2, -0.3, 0.4]
    rotations = _axis_angle_to_matrix(source)
    actual = mano_hand_pose_to_axis_angle(rotations, is_right=False)
    mirror = np.diag([-1.0, 1.0, 1.0])
    expected_matrix = mirror @ rotations[0] @ mirror
    np.testing.assert_allclose(_axis_angle_to_matrix(actual[0]), expected_matrix, atol=1e-7)

def test_right_mano_rotations_keep_their_coordinate_system(self):
    from suhu_pilot import _axis_angle_to_matrix, mano_hand_pose_to_axis_angle
    source = np.zeros((15, 3)); source[0] = [0.2, -0.3, 0.4]
    actual = mano_hand_pose_to_axis_angle(_axis_angle_to_matrix(source), is_right=True)
    np.testing.assert_allclose(_axis_angle_to_matrix(actual), _axis_angle_to_matrix(source), atol=1e-7)
```

- [ ] **Step 2: Run the conversion tests and verify `ImportError`**

Run: `python -m unittest discover -s tests -p "test_suhu_pilot.py"`

Expected: FAIL because `mano_hand_pose_to_axis_angle` does not exist.

- [ ] **Step 3: Implement matrix validation and conversion**

```python
def mano_hand_pose_to_axis_angle(rotations, is_right):
    from scipy.spatial.transform import Rotation
    matrices = np.asarray(rotations, dtype=np.float64)
    if matrices.shape != (15, 3, 3) or not np.isfinite(matrices).all():
        raise ValueError("MANO hand pose must be finite with shape (15, 3, 3)")
    if not is_right:
        mirror = np.diag([-1.0, 1.0, 1.0])
        matrices = mirror @ matrices @ mirror
    return Rotation.from_matrix(matrices).as_rotvec()
```

Use SciPy's tested rotation conversion and verify round trips through `_axis_angle_to_matrix`.

- [ ] **Step 4: Write failing fusion tests**

```python
def test_fusion_changes_only_hand_pose_arrays(self):
    from suhu_pilot import _axis_angle_to_matrix, fuse_hand_predictions
    motion = self.motion_fixture(frames=2)
    rotations = _axis_angle_to_matrix(np.full((15, 3), 0.1))
    predictions = {
        1: {"left": rotations, "right": rotations},
        2: {"left": rotations, "right": rotations},
    }
    refined = fuse_hand_predictions(motion, predictions)
    for key in motion:
        if key not in {"left_hand_pose", "right_hand_pose"}:
            np.testing.assert_array_equal(refined[key], motion[key])

def test_fusion_rejects_a_missing_side(self):
    from suhu_pilot import fuse_hand_predictions
    with self.assertRaisesRegex(ValueError, "frame 1 right"):
        fuse_hand_predictions(self.motion_fixture(1), {1: {"left": np.eye(3)[None].repeat(15, 0)}})
```

- [ ] **Step 5: Implement minimal fusion**

Copy every motion array, require both sides for every requested 1-based frame, convert each prediction, and assign only the corresponding hand-pose row. Reject missing frames, missing sides, invalid matrices, and frame numbers outside the motion range.

- [ ] **Step 6: Add stable configuration values**

```json
"hand_refinement": {
  "preview_frames": [16, 25, 38, 55, 73],
  "box_padding": 2.0,
  "minimum_box_size": 32
}
```

Validate the object and its positive values in `load_config`.

- [ ] **Step 7: Run all tests and commit**

Run: `python -m unittest discover -s tests -p "test_suhu_pilot.py"`

Expected: PASS.

Commit: `git commit -am "Add MANO hand fusion utilities"`

---

### Task 2: Hand-box projection and resumable HaMeR runner

**Files:**
- Create: `hamer_refinement.py`
- Test: `tests/test_hamer_refinement.py`

**Interfaces:**
- Consumes: consolidated motion NPZ, source JPEG frames, camera JSON files, SMPL-X models, MANO/SMPL-X vertex IDs, HaMeR checkpoint, and output directory.
- Produces: `project_points`, `square_box_from_points`, `hand_boxes_from_vertices`, `reconcile_hand_progress`, and CLI-created `hands/mano/{frame:05d}_{side}.npz`, overlays, and `progress.json`.

- [ ] **Step 1: Write failing geometry tests**

```python
def test_project_points_matches_smplerx_camera(self):
    from hamer_refinement import project_points
    points = np.array([[1.0, 2.0, 4.0]])
    np.testing.assert_allclose(project_points(points, [100, 200], [10, 20]), [[35, 120]])

def test_square_box_clips_to_image(self):
    from hamer_refinement import square_box_from_points
    box = square_box_from_points(np.array([[0, 10], [30, 40]]), 1280, 720, padding=2.0, minimum=32)
    self.assertEqual(box.shape, (4,))
    self.assertGreaterEqual(box[0], 0); self.assertGreaterEqual(box[1], 0)
    self.assertLessEqual(box[2], 1279); self.assertLessEqual(box[3], 719)
```

- [ ] **Step 2: Run and verify module-not-found failure**

Run: `python -m unittest discover -s tests -p "test_hamer_refinement.py"`

Expected: FAIL because `hamer_refinement.py` does not exist.

- [ ] **Step 3: Implement pure projection helpers**

Use the SMPLer-X projection `x * fx / z + cx`, `y * fy / z + cy`; reject non-positive depth and non-finite values. Load `left_hand` and `right_hand` arrays from `MANO_SMPLX_vertex_ids.pkl`, calculate square padded boxes, and return boxes ordered left then right.

- [ ] **Step 4: Write failing progress tests**

```python
def test_progress_recovers_completed_hand_files(self):
    from hamer_refinement import reconcile_hand_progress
    (root / "mano").mkdir(parents=True)
    self.write_prediction(root / "mano/00001_left.npz", False)
    self.write_prediction(root / "mano/00001_right.npz", True)
    progress = reconcile_hand_progress(root, [1, 2])
    self.assertEqual(progress["frames"]["1"]["left"]["status"], "ok")
    self.assertEqual(progress["frames"]["2"]["right"]["status"], "pending")
```

- [ ] **Step 5: Implement atomic progress reconciliation**

Reuse `atomic_write_json`; mark a side `ok` only when its NPZ contains finite `hand_pose` `[15,3,3]`, `global_orient` `[1,3,3]`, matching handedness, and matching frame number. Mark corrupt files `error` and missing files `pending`.

- [ ] **Step 6: Implement the HaMeR CLI without detector dependencies**

Parse:

```text
--motion --frames-dir --meta-dir --smplx-model-root --vertex-ids
--checkpoint --output-dir --frames 16,25,38,55,73
--box-padding 2.0 --minimum-box-size 32
```

Import Torch, OpenCV, SMPL-X, and HaMeR only inside `run_inference`. Load the models once, reconstruct one original SMPL-X frame at a time, project both mapped hands, create `ViTDetDataset(model_cfg, image, boxes, np.array([0,1]))`, and run batch size two. Persist each side immediately. Use HaMeR's renderer to save one crop/mesh overlay per side. On restart, skip sides already marked `ok`.

- [ ] **Step 7: Run tests and commit**

Run: `python -m unittest discover -s tests`

Expected: PASS without HaMeR installed because heavy imports are lazy.

Commit: `git add hamer_refinement.py tests/test_hamer_refinement.py && git commit -m "Add resumable HaMeR inference runner"`

---

### Task 3: Reusable SMPL-X renderer

**Files:**
- Create: `render_smplx.py`
- Modify: `tests/test_suhu_pilot.py`

**Interfaces:**
- Consumes: motion NPZ, SMPL-X model root, camera metadata, output folder, optional comma-separated frame list, and `--flat-hand-mean`.
- Produces: numbered PNG avatar frames for preview or full-video assembly.

- [ ] **Step 1: Add a failing renderer CLI contract test**

Read `render_smplx.py` as source and require arguments `--motion`, `--model-root`, `--meta-dir`, `--output-dir`, `--frames`, and `--flat-hand-mean`; compile the source so the test does not require the private model.

- [ ] **Step 2: Verify the missing-file failure**

Run: `python -m unittest discover -s tests -p "test_suhu_pilot.py"`

Expected: FAIL because `render_smplx.py` does not exist.

- [ ] **Step 3: Move the existing renderer into the CLI**

Preserve the existing camera, material, light, and 180-degree X rotation. Add selected-frame rendering and pass `flat_hand_mean=args.flat_hand_mean` to `smplx.create`. Keep heavy imports inside `main` so repository tests compile without pyrender.

- [ ] **Step 4: Run tests and commit**

Run: `python -m unittest discover -s tests`

Expected: PASS.

Commit: `git add render_smplx.py tests/test_suhu_pilot.py && git commit -m "Extract reusable SMPL-X renderer"`

---

### Task 4: Colab setup, preview gate, and full refinement

**Files:**
- Modify: `notebooks/suhu_pilot_colab.ipynb`
- Modify: `tests/test_suhu_pilot.py`

**Interfaces:**
- Consumes: Tasks 1-3 APIs and private Drive assets.
- Produces: configured HaMeR environment, five-frame preview gate, resumable full inference, refined motion/QA, and refined comparison videos.

- [ ] **Step 1: Add failing notebook contract tests**

Require code-cell IDs `hamer-setup`, `hand-preview`, and `hand-full-refinement`; pinned commit and versions; exact MANO Drive path; official demo archive URL; preview frame gate; runner invocations; `HAND_REFINEMENT_CLEARED`; `flat_hand_mean=True`; and final renderer input `suhu_motion_hand_refined.npz`. Continue requiring visible sequential `# Cell N —` labels.

- [ ] **Step 2: Run and verify missing-cell failures**

Run: `python -m unittest discover -s tests -p "test_suhu_pilot.py"`

Expected: FAIL because the three cells are absent.

- [ ] **Step 3: Add `hamer-setup`**

Clone and pin HaMeR, create `/content/envs/hamer`, install the minimal inference dependencies without Detectron2 or ViTPose, and verify CUDA. Require `MyDrive/BIM-Avatar/models/mano/MANO_RIGHT.pkl`. If the three public assets are missing from Drive, resume-download the official 6,037,554,929-byte demo archive to `/content`, extract it, and copy only `hamer.ckpt`, `model_config.yaml`, and `mano_mean_params.npz` to Drive. Recreate HaMeR's expected `_DATA` layout in `/content` and record file hashes and dependency versions in the manifest.

- [ ] **Step 4: Add `hand-preview`**

Run `hamer_refinement.py` for `[16,25,38,55,73]`, load its predictions, call `fuse_hand_predictions` for those frames, and save `suhu_motion_hand_preview.npz`. Render original and refined selected frames with `render_smplx.py`; assemble the five-panel 1-FPS comparison with FFmpeg. On exact input `HAND_REFINEMENT_CLEARED`, atomically write `hands/preview/cleared.json`; otherwise raise before full inference.

- [ ] **Step 5: Add `hand-full-refinement`**

Require `cleared.json`, run the resumable HaMeR CLI for frames 1 through `total_frames`, require both valid sides for all frames, fuse and save `suhu_motion_hand_refined.npz`, run `analyze_motion`, and write `qa/hand_refinement.json`. Do not interpolate or smooth.

- [ ] **Step 6: Replace the embedded avatar renderer**

Call `render_smplx.py` with the refined motion and `--flat-hand-mean`, produce `suhu_avatar_hand_refined.mp4`, the blind-review copy, and a source/original-overlay/refined-avatar comparison. Renumber all code-cell labels sequentially.

- [ ] **Step 7: Run tests and commit**

Run: `python -m unittest discover -s tests`

Expected: PASS.

Commit: `git add notebooks/suhu_pilot_colab.ipynb tests/test_suhu_pilot.py && git commit -m "Add Colab HaMeR refinement workflow"`

---

### Task 5: Documentation and final verification

**Files:**
- Modify: `README.md`
- Modify: `.gitignore`
- Test: `tests/test_suhu_pilot.py`

**Interfaces:**
- Consumes: completed notebook and artifact names.
- Produces: exact user setup instructions, licences, recovery steps, and final repository verification.

- [ ] **Step 1: Update documentation**

Document the two new private Drive folders, `MANO_RIGHT.pkl`, the separate HaMeR environment, the 6.04 GB temporary archive download, the five-frame gate, full-clip resumption, output names, HaMeR/Mano2Smpl-X attribution, and the rule that full vocabulary remains blocked until BIM review passes.

- [ ] **Step 2: Extend ignore rules**

Ignore MANO files, HaMeR checkpoints/data archives, hand predictions, previews, and refined videos without ignoring source code, tests, specs, or plans.

- [ ] **Step 3: Run complete verification**

Run:

```text
python -m unittest discover -s tests
python -m py_compile suhu_pilot.py hamer_refinement.py render_smplx.py
git diff --check
git status --short
```

Expected: every test passes, every Python file compiles, no whitespace errors appear, and only intended files are modified.

- [ ] **Step 4: Commit and push**

Commit: `git add .gitignore README.md config notebooks suhu_pilot.py hamer_refinement.py render_smplx.py tests docs && git commit -m "Implement HaMeR hand refinement pilot"`

Push: `git push origin main`

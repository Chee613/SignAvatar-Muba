# SUHU BIM avatar pilot

This repository implements the one-word hackathon gate: convert `Suhu.mp4` to per-frame SMPL-X motion with SMPLer-X-H32, render that motion on a neutral SMPL-X avatar, and require three BIM-fluent reviewers to recognize **SUHU** before processing more vocabulary.

The guided workflow is [notebooks/suhu_pilot_colab.ipynb](notebooks/suhu_pilot_colab.ipynb). It is designed for a free Google Colab Tesla T4 and persists each completed frame in private Google Drive. It performs pretrained inference only; there is no training or fine-tuning.

## Scope

Included:

- pinned SMPLer-X revision `064baef0e4ab5277a3297691bc1d46ea5412586f`;
- strict validation of the original 1280×720, 50 FPS, 146-frame H.264 clip;
- a silent 30 FPS derivative and approximately 88 sequential frames;
- H32 resource testing on frame 38;
- restart-safe, single-person inference with immediate Drive persistence;
- one retry for missing detections at `bbox_thr=20`;
- raw and consolidated SMPL-X motion, automated rotation-jump QA, and comparison videos;
- direct neutral SMPL-X avatar rendering without cross-rig retargeting;
- a blind three-reviewer recognition and quality gate.

Not included: a custom avatar, Malay-to-BIM translation, word sequencing, hand-model fusion, frontend work, training, or full-dataset batching.

## Private Drive setup

Use one Google account. Create this layout without changing the filenames:

```text
MyDrive/BIM-Avatar/
├── inputs/
│   └── Suhu.mp4
├── models/
│   ├── smplerx/
│   │   ├── smpler_x_h32.pth.tar        # notebook downloads and verifies
│   │   └── smpler_x_l32.pth.tar        # optional OOM fallback
│   ├── mmdet/
│   │   ├── faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth  # automatic
│   │   └── mmdet_faster_rcnn_r50_fpn_coco.py                  # automatic
│   └── smplx/
│       ├── MANO_SMPLX_vertex_ids.pkl
│       ├── SMPL-X__FLAME_vertex_ids.npy
│       ├── SMPLX_NEUTRAL.pkl
│       ├── SMPLX_to_J14.pkl
│       ├── SMPLX_NEUTRAL.npz
│       ├── SMPLX_MALE.npz
│       ├── SMPLX_FEMALE.npz
│       └── smpl/
│           ├── SMPL_NEUTRAL.pkl
│           ├── SMPL_MALE.pkl
│           └── SMPL_FEMALE.pkl
├── runs/SUHU/                         # notebook creates this
└── reviews/                           # notebook creates this
```

Obtain the files from their official sources:

- [SMPLer-X model and setup instructions](https://github.com/caizhongang/SMPLer-X/tree/064baef0e4ab5277a3297691bc1d46ea5412586f)
- [SMPL-X registration and downloads](https://smpl-x.is.tue.mpg.de/)
- [SMPL registration and downloads](https://smpl.is.tue.mpg.de/)
- [MMDetection Faster R-CNN checkpoint](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth)
- [Inference detector configuration](https://github.com/openxrlab/xrmocap/blob/main/configs/modules/human_perception/mmdet_faster_rcnn_r50_fpn_coco.py)

Models, videos, parameters, extracted frames, and generated media are ignored by Git. Do not override those exclusions or redistribute private assets.

The notebook downloads H32 (7.94 GB) and the two public MMDetection files directly into Drive using pinned official URLs. Interrupted downloads resume, and exact file size plus SHA-256 must pass before inference. Make sure the connected Drive has enough free storage. SMPL-X, SMPL, and the optional L32 fallback remain manual because they are restricted or selected only after the H32 resource test.

## Run the pilot

1. Merge or push this implementation so Colab can clone it. If it is not on `main`, change `PILOT_REPO_REF` in the notebook settings cell to the pushed branch name.
2. Open the notebook in Colab and select a **T4 GPU** runtime.
3. Run cells in order. The preflight requires at least 14 GiB free VRAM, 10 GiB available system RAM, and 20 GiB temporary disk.
4. Let the notebook build the isolated Python 3.8 / PyTorch 1.12 / CUDA 11.3 environment and download the verified public checkpoints. The first setup is slow; interrupted H32 downloads resume from the bytes already stored in Drive.
5. Upload the personally licensed SMPL-X and SMPL files listed above, then inspect the frame-38 overlay. If H32 reports CUDA out-of-memory, restart the runtime, change `MODEL_NAME` to `smpler_x_l32`, provide its checkpoint, and rerun from the private-assets cell. Do not modify mixed precision for this pilot.
6. Run full inference. Disconnects are recoverable: completed artifacts have already been copied to Drive and the next runtime starts at the first unfinished frame.
7. Resolve any flagged motion frame against the source. The notebook never smooths or interpolates hands automatically.
8. Inspect the source, overlay, and avatar videos at normal and half speed.
9. Give only `reviews/SUHU_blind_review.mp4` to each reviewer. Do not show the intended gloss or source filename. Fill `reviews/SUHU_review.csv` and rerun the review cell.

Do not rotate Google accounts to bypass Colab limits, and do not run this workload on the company GPU hosting eKYC/OCR services.

If the pinned Python environment cannot be made stable in Colab within one engineering day, stop changing package versions and use the upstream inference Docker image on a separate GPU VM. This is a fallback, not a reason to move the workload onto the production server.

## Persistent outputs

```text
runs/SUHU/
├── manifest.json
├── progress.json
├── frames/
├── smplx/
├── meta/
├── overlays/
├── motion/
│   ├── suhu_motion_raw.npz
│   └── suhu_motion.npz
├── qa/
│   ├── suhu_motion_qa.json
│   └── suhu_review_result.json
├── avatar/
│   ├── suhu_avatar.mp4
│   └── suhu_three_way_comparison.mp4
├── suhu_original_30fps.mp4
├── suhu_smplerx_overlay.mp4
└── suhu_source_vs_smplx.mp4
```

`suhu_motion.npz` contains:

```text
fps                 scalar
global_orient       [N, 3]
body_pose           [N, 21, 3]
left_hand_pose      [N, 15, 3]
right_hand_pose     [N, 15, 3]
jaw_pose            [N, 3]
left_eye_pose       [N, 3]
right_eye_pose      [N, 3]
betas               [N, 10]
expression          [N, 10]
translation         [N, 3]
valid_frame_mask    [N]
source_frame_number [N]
```

The clean motion changes only `betas`, replacing per-frame values with their temporal median to prevent body-shape flicker. Body and hand motion are not smoothed.

## Acceptance gate

The pilot passes only when all three BIM-fluent reviewers independently identify `SUHU`, every category average is at least 1.5/2, and no category receives zero from two or more reviewers. Categories are handshape, palm/finger orientation, location, movement, and non-manual cues.

Failure routing:

- source overlay wrong → reconstruction/model issue;
- overlay correct but avatar wrong → playback/rendering issue;
- fingers wrong while body is correct → consider a later hand-refinement stage;
- engineering review passes but BIM review fails → linguistic failure; do not scale.

After SUHU passes, run the same full-body pipeline on `Makan`, `Rumah`, `Doa`, `Jurukamera`, `Taman Negara Endau-Rompin`, `Ais`, `Wang tunai`, and `Balik ke rumah`. Route `1` and `A` to a separate hand-only reconstruction experiment. Do not process all 2,811 clips before this canary succeeds.

## Licences and attribution

A hackathon exemption does not remove licence obligations. Confirm that the event and public demonstration are genuinely non-commercial and that the person accepting each model licence is permitted to do so for the team or company.

- SMPLer-X uses the [S-Lab License 1.0](https://github.com/caizhongang/SMPLer-X/blob/064baef0e4ab5277a3297691bc1d46ea5412586f/LICENSE), which permits non-commercial use subject to its conditions and requires contact for commercial use.
- SMPL-X and SMPL model files are restricted assets. Review and accept the current [SMPL-X model licence](https://smpl-x.is.tue.mpg.de/modellicense.html); it allows specified non-commercial uses, is single-user/non-transferable, and prohibits sharing the model files with third parties.
- MMDetection 2.26.0 is [Apache-2.0 licensed](https://github.com/open-mmlab/mmdetection/blob/v2.26.0/LICENSE). Check the separate terms and provenance of every downloaded checkpoint.
- Obtain written permission for the BIM video and signer appearance. Credit the BIM source, SMPLer-X paper, and SMPL-X paper in the hackathon README and presentation.

This section is an engineering checklist, not legal advice. Recheck the linked terms before the demonstration because licences can change.

## Local checks

The lightweight logic tests do not need a GPU:

```powershell
python -m unittest discover -s tests -v
```

They cover source validation, resume state, upstream parameter schema, shape stabilization, rotation-jump QA, notebook structure, and the blind-review rubric. Full model and visual acceptance must be run in Colab with the private assets.

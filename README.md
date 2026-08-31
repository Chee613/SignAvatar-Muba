# SUHU BIM avatar pilot

This repository implements the one-word hackathon gate for **SUHU**. It reconstructs the signer with SMPLer-X-H32, refines both finger poses with MediaPipe Hand Landmarker, renders a neutral SMPL-X avatar, and requires three BIM-fluent reviewers to recognize the sign before any vocabulary expansion.

The guided workflow is [notebooks/suhu_pilot_colab.ipynb](notebooks/suhu_pilot_colab.ipynb). It targets a free Google Colab Tesla T4, persists every completed prediction in private Google Drive, and performs pretrained inference only—there is no training or fine-tuning.

## What the pilot does

- Pins SMPLer-X at `064baef0e4ab5277a3297691bc1d46ea5412586f`.
- Pins MediaPipe at `1.0.1` and downloads Google's public Hand Landmarker model.
- Validates the original 1280×720, 50 FPS, 146-frame H.264 clip.
- Produces a 30 FPS derivative, resumable SMPLer-X outputs, QA, and comparison videos.
- Runs MediaPipe on preview frames `16, 25, 38, 55, 73` before the full clip.
- Replaces only `left_hand_pose` and `right_hand_pose`; body, wrists, timing, face, shape, and translation remain from SMPLer-X.
- Renders refined hands with `use_pca=False` and `flat_hand_mean=True`.
- Stops before the full vocabulary unless the blind BIM review passes.

Custom avatars, Malay-to-BIM translation, word sequencing, frontend work, training, and full-dataset batching are outside this hackathon pilot.

## Private Google Drive setup

Use one Google account and keep this exact layout:

```text
MyDrive/BIM-Avatar/
├── inputs/
│   └── Suhu.mp4
├── models/
│   ├── smplerx/
│   │   ├── smpler_x_h32.pth.tar
│   │   └── smpler_x_l32.pth.tar       # optional OOM fallback
│   ├── mmdet/
│   │   ├── faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
│   │   └── mmdet_faster_rcnn_r50_fpn_coco.py
│   ├── smplx/
│   │   ├── MANO_SMPLX_vertex_ids.pkl
│   │   ├── SMPL-X__FLAME_vertex_ids.npy
│   │   ├── SMPLX_NEUTRAL.pkl
│   │   ├── SMPLX_to_J14.pkl
│   │   ├── SMPLX_NEUTRAL.npz
│   │   ├── SMPLX_MALE.npz
│   │   ├── SMPLX_FEMALE.npz
│   │   └── smpl/
│   │       ├── SMPL_NEUTRAL.pkl
│   │       ├── SMPL_MALE.pkl
│   │       └── SMPL_FEMALE.pkl
│   └── mediapipe/                      # notebook creates this
│       └── hand_landmarker.task
├── runs/SUHU/                          # notebook creates this
└── reviews/                            # notebook creates this
```

No separate MANO or HaMeR download is required. MediaPipe detects 21 landmarks for each hand; the pilot aligns them to a neutral SMPL-X hand and derives the 15 local finger rotations for each side.

Obtain restricted files only from their official sources:

- [SMPLer-X setup and checkpoint instructions](https://github.com/caizhongang/SMPLer-X/tree/064baef0e4ab5277a3297691bc1d46ea5412586f)
- [SMPL-X registration and downloads](https://smpl-x.is.tue.mpg.de/)
- [SMPL registration and downloads](https://smpl.is.tue.mpg.de/)

The notebook downloads public SMPLer-X, MMDetection, and MediaPipe assets. Interrupted checkpoint downloads resume automatically.

Never commit or redistribute videos, model files, checkpoints, extracted frames, predictions, or generated media.

## Run in Colab

1. Push this repository to `main`, open the notebook in Colab, and select a **T4 GPU** runtime.
2. Run Cells 1–14 in order. Two isolated environments are created:
   - SMPLer-X: Python 3.8, PyTorch 1.12, CUDA 11.3 packages.
   - MediaPipe: Python 3.10 and MediaPipe 1.0.1; CPU inference is sufficient.
3. In Cell 15, inspect all five preview frames. Compare the source, SMPLer-X overlay, original avatar, MediaPipe landmarks, and refined avatar. Continue only by typing `HAND_REFINEMENT_CLEARED` exactly.
4. Run Cell 16. It copies the approved preview landmarks into the full run, skips valid files already in Drive, and resumes missing left/right hands after a disconnect.
5. Run Cell 17 and inspect the refined three-way video before typing `AVATAR_CLEARED`.
6. Give only `reviews/SUHU_blind_review.mp4` to each BIM reviewer. Do not reveal the gloss or source filename. Complete `reviews/SUHU_review.csv`, then rerun Cell 18.

The supplied `Suhu.mp4` is unmirrored, so `hand_refinement.input_mirrored` is `false`. Set it to `true` only when a preview labels the visible left and right hands backwards.

Do not rotate Google accounts to bypass Colab limits. Do not run this workload on the company GPU hosting eKYC/OCR services. If the pinned environment cannot be stabilized within one engineering day, use the upstream image on a separate GPU VM.

## Resumption and failure handling

- SMPLer-X progress is stored in `runs/SUHU/progress.json`.
- MediaPipe progress is stored separately in `hands/preview/progress.json` and `hands/full/progress.json`.
- A hand side is complete only when its NPZ has valid finite 21-point image/world landmarks and matching frame/side metadata.
- Corrupt or missing hand files are rerun; successful files are skipped.
- Full hand refinement cannot start without `hands/preview/cleared.json`.
- Missing hand predictions are never silently filled.
- Review flagged hand jumps against the source before accepting the avatar.

## Main outputs

```text
runs/SUHU/
├── manifest.json
├── progress.json
├── frames/
├── smplx/
├── meta/
├── overlays/
├── hands/
│   ├── preview/
│   │   ├── cleared.json
│   │   ├── progress.json
│   │   └── suhu_hand_refinement_preview.mp4
│   └── full/
│       ├── landmarks/
│       └── progress.json
├── motion/
│   ├── suhu_motion_raw.npz
│   ├── suhu_motion.npz
│   ├── suhu_motion_hand_preview.npz
│   └── suhu_motion_hand_refined.npz
├── qa/
│   ├── suhu_motion_qa.json
│   ├── hand_refinement.json
│   └── suhu_review_result.json
└── avatar/
    ├── suhu_avatar_hand_refined.mp4
    └── suhu_three_way_hand_refined_comparison.mp4
```

`suhu_motion_hand_refined.npz` retains the consolidated SMPL-X schema. Only the two `[N, 15, 3]` hand-pose arrays differ from `suhu_motion.npz`.

## Acceptance gate

The pilot passes only when all three BIM-fluent reviewers independently identify `SUHU`, every category averages at least 1.5/2, and no category receives zero from two or more reviewers. The categories are handshape, palm/finger orientation, location, movement, and non-manual cues.

Failure routing:

- Source overlay wrong → SMPLer-X reconstruction issue.
- MediaPipe hand overlay wrong → landmark detection or handedness issue.
- Hand overlay correct but refined avatar wrong → conversion or rendering issue.
- Engineering review passes but BIM review fails → linguistic failure; do not scale.

After SUHU passes, run the defined canary set. Do not process all 2,811 clips until the full-body canary passes BIM review and the hand-only alphabet/number route is defined.

## Licences and attribution

A hackathon exemption does not remove licence obligations. Confirm non-commercial eligibility and permission for the BIM videos and signer appearance.

- [SMPLer-X S-Lab License 1.0](https://github.com/caizhongang/SMPLer-X/blob/064baef0e4ab5277a3297691bc1d46ea5412586f/LICENSE)
- [MediaPipe Apache-2.0 licence](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE)
- [SMPL-X model licence](https://smpl-x.is.tue.mpg.de/modellicense.html) and the separately accepted MANO/SMPL terms
- [MMDetection Apache-2.0 licence](https://github.com/open-mmlab/mmdetection/blob/v2.26.0/LICENSE)

Keep the copyright notices and credit the BIM source, SMPLer-X, SMPL-X, MediaPipe, and MMDetection in the hackathon presentation. This is an engineering checklist, not legal advice.

## Local verification

The lightweight checks do not need a GPU or private models:

```powershell
python -m unittest discover -s tests -v
python -m py_compile suhu_pilot.py mediapipe_refinement.py hamer_refinement.py render_smplx.py
```

Full model inference and visual acceptance must be performed in Colab.

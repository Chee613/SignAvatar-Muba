# SUHU BIM avatar pilot

This repository implements the one-word hackathon gate for **SUHU**. It reconstructs the signer with SMPLer-X-H32, refines both finger poses with HaMeR, renders a neutral SMPL-X avatar, and requires three BIM-fluent reviewers to recognize the sign before any vocabulary expansion.

The guided workflow is [notebooks/suhu_pilot_colab.ipynb](notebooks/suhu_pilot_colab.ipynb). It targets a free Google Colab Tesla T4, persists every completed prediction in private Google Drive, and performs pretrained inference only—there is no training or fine-tuning.

## What the pilot does

- Pins SMPLer-X at `064baef0e4ab5277a3297691bc1d46ea5412586f`.
- Pins HaMeR at `3a01849f4148352e9260b69bf28b65d1671a4905`.
- Uses the Mano2Smpl-X left/right conversion from reference commit `901c124b9163294449b44821a642ae8e99d41cb0`.
- Validates the original 1280×720, 50 FPS, 146-frame H.264 clip.
- Produces a 30 FPS derivative, resumable SMPLer-X outputs, QA, and comparison videos.
- Runs HaMeR on preview frames `16, 25, 38, 55, 73` before the full clip.
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
│   ├── mano/
│   │   └── MANO_RIGHT.pkl             # private; upload manually
│   └── hamer/                          # notebook fills from official archive
│       ├── hamer.ckpt
│       ├── model_config.yaml
│       └── mano_mean_params.npz
├── runs/SUHU/                          # notebook creates this
└── reviews/                            # notebook creates this
```

Only `MANO_RIGHT.pkl` is required. HaMeR mirrors left-hand crops and the pilot converts the resulting left-hand rotations back into the SMPL-X coordinate system.

Obtain restricted files only from their official sources:

- [SMPLer-X setup and checkpoint instructions](https://github.com/caizhongang/SMPLer-X/tree/064baef0e4ab5277a3297691bc1d46ea5412586f)
- [SMPL-X registration and downloads](https://smpl-x.is.tue.mpg.de/)
- [SMPL registration and downloads](https://smpl.is.tue.mpg.de/)
- [MANO registration and downloads](https://mano.is.tue.mpg.de/)

The notebook downloads public SMPLer-X, MMDetection, and HaMeR assets. The HaMeR setup temporarily downloads the official 6.04 GB demo archive, extracts only the three required public files into Drive, then removes the temporary archive. Interrupted downloads resume automatically.

Never commit or redistribute videos, model files, checkpoints, extracted frames, predictions, or generated media.

## Run in Colab

1. Push this repository to `main`, open the notebook in Colab, and select a **T4 GPU** runtime.
2. Upload `MANO_RIGHT.pkl` to `MyDrive/BIM-Avatar/models/mano/` before running Cell 7.
3. Run Cells 1–14 in order. Two isolated environments are created:
   - SMPLer-X: Python 3.8, PyTorch 1.12, CUDA 11.3 packages.
   - HaMeR: Python 3.10, PyTorch 2.0.0, torchvision 0.15.1, CUDA 11.7 wheels.
4. In Cell 15, inspect all five preview frames. Compare the source, SMPLer-X overlay, original avatar, both HaMeR hand overlays, and refined avatar. Continue only by typing `HAND_REFINEMENT_CLEARED` exactly.
5. Run Cell 16. It copies the approved preview predictions into the full run, skips valid files already in Drive, and resumes missing left/right hands after a disconnect.
6. Run Cell 17 and inspect the refined three-way video before typing `AVATAR_CLEARED`.
7. Give only `reviews/SUHU_blind_review.mp4` to each BIM reviewer. Do not reveal the gloss or source filename. Complete `reviews/SUHU_review.csv`, then rerun Cell 18.

Do not rotate Google accounts to bypass Colab limits. Do not run this workload on the company GPU hosting eKYC/OCR services. If the pinned environment cannot be stabilized within one engineering day, use the upstream image on a separate GPU VM.

## Resumption and failure handling

- SMPLer-X progress is stored in `runs/SUHU/progress.json`.
- HaMeR progress is stored separately in `hands/preview/progress.json` and `hands/full/progress.json`.
- A hand side is complete only when its NPZ has valid finite rotation matrices and matching frame/hand metadata.
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
│       ├── mano/
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
- HaMeR hand overlay wrong → hand crop/model issue.
- Hand overlay correct but refined avatar wrong → conversion or rendering issue.
- Engineering review passes but BIM review fails → linguistic failure; do not scale.

After SUHU passes, run the defined canary set. Do not process all 2,811 clips until the full-body canary passes BIM review and the hand-only alphabet/number route is defined.

## Licences and attribution

A hackathon exemption does not remove licence obligations. Confirm non-commercial eligibility and permission for the BIM videos and signer appearance.

- [SMPLer-X S-Lab License 1.0](https://github.com/caizhongang/SMPLer-X/blob/064baef0e4ab5277a3297691bc1d46ea5412586f/LICENSE)
- [HaMeR MIT licence](https://github.com/geopavlakos/hamer/blob/3a01849f4148352e9260b69bf28b65d1671a4905/LICENSE.md) and [CVPR 2024 paper](https://arxiv.org/abs/2312.05251)
- [Mano2Smpl-X MIT licence](https://github.com/VincentHu19/Mano2Smpl-X/blob/901c124b9163294449b44821a642ae8e99d41cb0/LICENSE)
- [SMPL-X model licence](https://smpl-x.is.tue.mpg.de/modellicense.html) and the separately accepted MANO/SMPL terms
- [MMDetection Apache-2.0 licence](https://github.com/open-mmlab/mmdetection/blob/v2.26.0/LICENSE)

Keep the copyright notices and credit the BIM source, SMPLer-X, SMPL-X, MANO, HaMeR, and Mano2Smpl-X in the hackathon presentation. This is an engineering checklist, not legal advice.

## Local verification

The lightweight checks do not need a GPU or private models:

```powershell
python -m unittest discover -s tests -v
python -m py_compile suhu_pilot.py hamer_refinement.py render_smplx.py
```

Full model inference and visual acceptance must be performed in Colab.

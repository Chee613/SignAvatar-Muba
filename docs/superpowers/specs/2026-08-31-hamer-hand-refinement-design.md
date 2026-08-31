# HaMeR Hand Refinement Design

## Objective

Improve the SUHU avatar's finger articulation while preserving the SMPLer-X body, arm, wrist, face, camera, and timing results that already match the source. The refinement must prove itself on five representative frames before it can modify the full 88-frame clip.

## Scope

The pilot adds pretrained HaMeR inference and MANO-to-SMPL-X finger-pose conversion. It does not train either model, add a frontend, process other words, automatically smooth sign-critical motion, or replace the working SMPLer-X body estimator.

## Pinned Sources

- HaMeR commit `3a01849f4148352e9260b69bf28b65d1671a4905`.
- Mano2Smpl-X reference commit `901c124b9163294449b44821a642ae8e99d41cb0`.
- Existing SMPLer-X commit `064baef0e4ab5277a3297691bc1d46ea5412586f`.
- HaMeR runs in a separate micromamba environment using Python 3.10, PyTorch 2.0.0, torchvision 0.15.1, and CUDA 11.7 runtime packages. The existing SMPLer-X environment remains unchanged.

## Private and Public Assets

Google Drive remains the private persistence layer:

```text
MyDrive/BIM-Avatar/models/
├── mano/
│   └── MANO_RIGHT.pkl
└── hamer/
    └── _DATA/
        ├── data/mano_mean_params.npz
        └── hamer_ckpts/
            ├── model_config.yaml
            └── checkpoints/hamer.ckpt
```

`MANO_RIGHT.pkl` is downloaded by the user from the official MANO site and is never committed, copied to a public location, or redistributed. HaMeR's official demo archive is downloaded to Colab temporary storage. The notebook copies only the checkpoint, model config, and mean-parameter file into private Drive; detector and ViTPose assets are not persisted because this pipeline supplies its own hand boxes.

## Data Flow

1. Load the consolidated SMPL-X motion, per-frame camera metadata, neutral SMPL-X model, and `MANO_SMPLX_vertex_ids.pkl`.
2. Reconstruct each SMPL-X mesh and project the mapped left- and right-hand vertices into the 1280x720 source frame using the stored focal length and principal point.
3. Build one square hand box per side with a 2.0 padding factor, clipped to image bounds. Reject non-finite boxes, boxes smaller than 32 pixels, and boxes extending entirely outside the source image.
4. Pass the original frame, both boxes, and handedness values `[0, 1]` to HaMeR's `ViTDetDataset`. HaMeR mirrors the left crop internally and predicts both hands with `MANO_RIGHT.pkl`.
5. Save each prediction immediately as a per-frame, per-side NPZ containing rotation matrices, box coordinates, handedness, frame number, and finite-value status.
6. Convert right-hand finger rotation matrices directly to axis-angle. Convert left-hand matrices with `M @ R @ M`, where `M = diag(-1, 1, 1)`, before converting to axis-angle.
7. Replace only `left_hand_pose` and `right_hand_pose` in a copy of the SMPLer-X motion. Preserve body pose, including both wrist joints, because the source comparison shows that SMPLer-X already captures arm and wrist orientation correctly. Wrist fusion remains a separate follow-up only if BIM review identifies a palm-orientation failure.
8. Render refined motion with `use_pca=False` and `flat_hand_mean=True`, which is required when transferring direct MANO finger rotations into SMPL-X.

## Five-Frame Gate

The preview processes source frames `[16, 25, 38, 55, 73]`, representing entry, first contact, held sign, release, and exit. It produces:

```text
runs/SUHU/hands/preview/
├── crops/
├── mano/
├── overlays/
├── refined_avatar/
└── suhu_hand_preview_comparison.mp4
```

The side-by-side preview shows source, original SMPLer-X overlay, original avatar, HaMeR crop overlay, and refined avatar. The user must enter `HAND_REFINEMENT_CLEARED` only when:

- both index fingers remain clearly extended where present in the source;
- folded fingers do not become open or claw-like;
- left and right hands are not swapped or mirrored incorrectly;
- palm/finger orientation agrees with the source;
- fingertip contact is at least as accurate as the original reconstruction; and
- no wrist discontinuity or mesh inversion is introduced.

If the preview fails, the notebook stops before full-clip inference and preserves all original motion and renders.

## Full-Clip Processing

After preview approval, HaMeR loads once and processes both hands for every frame sequentially with batch size two. Results are written immediately under `runs/SUHU/hands/mano`; `runs/SUHU/hands/progress.json` records `ok` or `error` for each frame and side. A reconnected runtime skips completed predictions and resumes at the first incomplete hand.

Full fusion requires valid predictions for both hands on every frame. It creates:

```text
runs/SUHU/motion/suhu_motion_hand_refined.npz
runs/SUHU/qa/hand_refinement.json
runs/SUHU/avatar/suhu_avatar_hand_refined.mp4
runs/SUHU/avatar/suhu_hand_refined_comparison.mp4
```

No missing hand is silently interpolated, and no temporal smoothing is applied during this pilot. Existing rotation-jump QA runs again on the refined motion. A new report records source boxes, invalid predictions, handedness, per-hand maximum adjacent angular change, and the frames whose refined hand poses differ by more than 45 degrees from the original.

## Failure Handling

- Missing `MANO_RIGHT.pkl`: stop with the exact expected Drive path.
- Missing or incomplete HaMeR public assets: use resumable download into Colab temporary storage and do not mark setup complete.
- Invalid projected box: record the frame and side and stop the preview or full fusion.
- CUDA out of memory: restart the runtime and retry batch size one; do not alter precision or the SMPLer-X environment.
- Invalid or non-finite HaMeR rotations: write an error status and preserve the original motion.
- Preview rejected: retain the original pipeline and use manual SUHU keyframes only as an explicitly labelled hackathon fallback.

## Tests

- Unit-test left-hand mirroring against `M @ R @ M` and verify right-hand rotations remain unchanged.
- Unit-test fusion to prove it changes only the two hand-pose arrays.
- Unit-test rejection of missing sides, invalid shapes, and NaN/Inf values.
- Notebook tests require the pinned commits, separate environment, five-frame gate, resumable progress, `flat_hand_mean=True`, and a renderer input pointing to the refined motion only after approval.
- Run the existing 19-test suite plus the new hand-refinement tests.

## Acceptance

The engineering gate passes when all five preview frames satisfy the visual criteria, the full refined clip contains no missing/invalid hand predictions or unresolved rotation jumps, and the three BIM reviewers still meet the existing SUHU recognition rubric. The project does not proceed to other words until this gate passes.

## Licensing and Provenance

The manifest records the HaMeR and Mano2Smpl-X commits, checkpoint hashes, MANO file hash, environment versions, preview decision, and all generated paths. HaMeR and Mano2Smpl-X code attribution is added to the README and presentation. The MANO licence applies even for a hackathon, and its model file remains private and single-user.

# Project Version History and Architecture Evolution

This document tracks all versions of the Bahasa Isyarat Malaysia (BIM) 3D Avatar reconstruction pipeline.

---

## Version Overview Matrix

| Version | Name | Primary Technique | Hand Quality | Colab Runtime | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **v1.0** | SMPLer-X Baseline | Monocular SMPLer-X end-to-end | Low (Claw-like fingers, contact issues) | ~15 min | Superseded |
| **v2.0** | HaMeR MANO Pilot | ViTDet Crop + HaMeR MANO prediction | Medium-High | ~25 min (6GB download) | Archived |
| **v3.0** | MediaPipe Heuristic | MediaPipe 3D -> Bone Angle Math | Low-Medium (Z-depth ambiguity, jitter) | ~16 min (10MB download) | Superseded |
| **v4.0** | Sequence Reprojection Optimizer | PyTorch Differentiable 2D Reprojection + Temporal + Bio Limits | **High** (Anatomically bounded, smooth) | **~17 min** | **Active Production** |

---

## Detailed Version Changelog

### Version 1.0: SMPLer-X Monocular Baseline
- **Folder**: `versions/v1_smplerx_baseline/`
- **Method**: Direct extraction of SMPL-X (54 joints) from monocular RGB video using SMPLer-X (`smpler_x_h32`).
- **Verdict**: Body, arms, head, and gross timing tracked accurately, but hand shapes failed the BIM linguistic gate (claw-like curled fingers, inaccurate thumb placement, inconsistent hand contact).

### Version 2.0: HaMeR MANO Hand Refinement Pilot
- **Folder**: `versions/v2_hamer_pilot/`
- **Script**: `versions/v2_hamer_pilot/hamer_refinement.py`
- **Method**: Bounding-box hand crops generated from wrist joints, evaluated with HaMeR Vision Transformer predicting MANO joint parameters, converted to SMPL-X via `flat_hand_mean`.
- **Verdict**: Improved single-hand shapes, but incurred high memory/download overhead (~6GB demo data archive) and complex left-hand mirroring issues during two-handed crossings.

### Version 3.0: Direct MediaPipe 3D Landmark Conversion
- **Folder**: `versions/v3_mediapipe_direct_heuristic/`
- **Script**: `versions/v3_mediapipe_direct_heuristic/mediapipe_direct_heuristic.py`
- **Method**: Lightweight MediaPipe HandLandmarker (10MB) estimating 21 3D landmarks per hand, directly calculating joint rotation vectors via geometric bone formulas (`landmarks_to_hand_pose`).
- **Verdict**: Ultra-fast setup, but suffered from 3D monocular Z-depth distortion, axial twist ambiguity, absence of camera reprojection feedback, and frame-to-frame jitter.

### Version 4.0: Sequence-Level Differentiable Reprojection Optimizer (Current)
- **Folder**: `versions/v4_sequence_reprojection_optimizer/`
- **Script**: `mediapipe_refinement.py`
- **Notebook**: `notebooks/suhu_pilot_colab.ipynb`
- **Method**: Differentiable PyTorch gradient descent (Adam) optimizing SMPL-X hand parameters against 2D MediaPipe keypoints using a multi-term objective (2D Smooth-L1 Reprojection + Temporal Acceleration Smoothness + Biomechanical Flexion/Abduction Boundaries + SMPLer-X Base Regularization).
- **Verdict**: Resolves finger twist ambiguities, eliminates temporal jitter, guarantees anatomical validity, and runs in ~60 seconds on Tesla T4 GPU.

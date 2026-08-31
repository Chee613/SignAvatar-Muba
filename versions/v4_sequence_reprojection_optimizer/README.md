# Version 4.0: Sequence-Level Differentiable Reprojection Optimizer (Current Production)

## Description
State-of-the-art hand refinement pipeline inspired by the SignAvatars (ECCV 2024) formulation. It treats SMPL-X hand parameters as trainable PyTorch variables and optimizes them across the entire sequence using differentiable 2D reprojection, temporal velocity/acceleration regularization, and anatomical joint limits.

## Architecture
- Body Initializer: SMPLer-X (smpler_x_h32) with camera, body pose, translation, and shape frozen
- 2D Observation Source: MediaPipe 2D image coordinates (x, y) with handedness confidence weights
- Optimizer: PyTorch Adam gradient descent on left_hand_pose [T, 15, 3], right_hand_pose [T, 15, 3], and delta wrist pose [T, 2, 3]
- Objective Function:
  1. 2D Reprojection Loss: Smooth L1 pixel distance between projected 3D avatar joints and MediaPipe 2D detections
  2. Temporal Smoothness Loss: Velocity and acceleration regularization eliminating frame-to-frame popping
  3. Biomechanical Joint Limits: Anatomical flexion and abduction constraints preventing unnatural bends
  4. Initial Pose Anchor: Regularization keeping hands close to the SMPLer-X baseline
- Missing Frame Handling: Automatically preserves baseline SMPLer-X tracking when hands are lowered or at rest

## Strengths
- High linguistic and anatomical precision.
- No claw-like twisting (axial twist resolved by 2D reprojection).
- Smooth temporal motion without jitter.
- Lightweight and fast: Runs in ~45-90 seconds on a Tesla T4 GPU in Google Colab.

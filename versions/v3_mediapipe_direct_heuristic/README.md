# Version 3.0: Direct MediaPipe 3D Landmark Conversion

## Description
A lightweight hand refinement alternative that replaced the heavy 6GB HaMeR model with Google MediaPipe (10MB hand_landmarker.task) and directly converted 3D landmark vectors into 15 SMPL-X joint rotations using heuristic bone vector math.

## Architecture
- Body estimator: SMPLer-X
- Hand landmark detector: MediaPipe HandLandmarker (21 3D landmarks per hand)
- Joint fitting: Direct vector angle calculation (landmarks_to_hand_pose)
- Dependencies: Lightweight Python 3.10 environment (10MB download, 5s setup)

## Strengths
- Ultra-fast download and inference (10MB model vs 6GB HaMeR).
- Completely decoupled Python 3.10 environment.

## Known Limitations
- Axial Twist Ambiguity: A 3D bone direction vector only has 2 degrees of freedom, leaving the 3rd axial roll rotation undetermined (leading to claw-like twisting).
- Unreliable Monocular Z-Depth: MediaPipe estimates depth from flat 2D images without LiDAR, distorting finger coordinates when hands touch or overlap.
- No Reprojection Loop: Never verifies whether avatar fingers align with the 2D video.
- Frame-to-frame jitter: Lacks temporal acceleration smoothing across consecutive frames.

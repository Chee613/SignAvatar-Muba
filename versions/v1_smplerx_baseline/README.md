# Version 1.0: SMPLer-X Monocular Baseline

## Description
The initial baseline pipeline for Bahasa Isyarat Malaysia (BIM) sign language avatar reconstruction. It uses monocular SMPLer-X (smpler_x_h32) with MMDetection Faster R-CNN bounding box tracking.

## Architecture
- Full-body pose estimator: SMPLer-X (smpler_x_h32)
- Body mesh: SMPL-X neutral model (54 joints)
- Hands: SMPLer-X integrated 15-joint estimation

## Strengths
- Fast end-to-end full body and arm tracking.
- Captures global body orientation, torso, neck, and elbow timing accurately.

## Known Limitations
- Handshape failure: Distorted, claw-like finger curls on fast or overlapping motions.
- Finger contact issues: Inability to resolve precise fingertip touching (e.g., SUHU at 0.8s to 1.8s).
- Thumb position inaccuracies.

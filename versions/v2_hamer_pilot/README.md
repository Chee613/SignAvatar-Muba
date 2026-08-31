# Version 2.0: HaMeR MANO Hand Refinement Pilot

## Description
The second-generation refinement pipeline introducing a dedicated hand-crop neural network (HaMeR) to estimate MANO hand parameters on high-resolution crops, then merging them into the SMPL-X full-body mesh.

## Architecture
- Body estimator: SMPLer-X
- Hand detector: ViTDet hand bounding box crops from wrist locations
- Hand estimator: HaMeR (Vision Transformer predicting MANO hand parameters)
- Fusion: MANO to SMPL-X rotation conversion with flat_hand_mean enabled

## Strengths
- High-fidelity single-hand pose predictions.
- Better individual finger curl modeling on unoccluded crops.

## Known Limitations
- Heavy checkpoint footprint: Requires downloading ~6GB demo data archive (hamer_demo_data.tar.gz).
- Slower setup time on Colab instances.
- Left-hand flipping coordinate complexities during close two-handed interactions.

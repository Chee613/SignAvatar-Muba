import argparse
import json
import os
from pathlib import Path

import numpy as np

from suhu_pilot import atomic_write_json


SIDES = ("left", "right")
FINGER_CHAINS = (
    ((5, 6, 7, 8), 0),
    ((9, 10, 11, 12), 3),
    ((17, 18, 19, 20), 6),
    ((13, 14, 15, 16), 9),
    ((1, 2, 3, 4), 12),
)
HAND_CONNECTIONS = (
    (0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (17, 0),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
)
REFERENCE_CHAINS = ("index", "middle", "pinky", "ring", "thumb")
BODY_WRIST_INDEXES = (19, 20)


def project_points(joints, focal, princpt, image_size):
    import torch

    if joints.shape[-1] != 3 or focal.shape[-1] != 2 or princpt.shape[-1] != 2:
        raise ValueError("joints and camera parameters have invalid shapes")
    depth = joints[..., 2:3].clamp_min(1e-4)
    pixels = joints[..., :2] / depth * focal[:, None, :] + princpt[:, None, :]
    size = torch.as_tensor(image_size, dtype=joints.dtype, device=joints.device)
    return pixels / size


def sequence_objective(
    projected,
    target,
    weights,
    hand_pose,
    initial_hand_pose,
    wrist_pose,
    initial_wrist_pose,
    frame_numbers,
    maximum_joint_degrees,
    initial_pose_weight,
    temporal_weight,
    wrist_weight,
):
    import torch
    import torch.nn.functional as functional

    point_error = functional.smooth_l1_loss(projected, target, reduction="none", beta=0.01).sum(-1)
    reprojection = (point_error * weights).sum() / weights.sum().clamp_min(1.0)
    initial_pose = (hand_pose - initial_hand_pose).square().mean()
    wrist = (wrist_pose - initial_wrist_pose).square().mean()
    temporal = hand_pose.new_zeros(())
    if len(hand_pose) > 1:
        gaps = (frame_numbers[1:] - frame_numbers[:-1]).clamp_min(1.0)
        hand_velocity = (hand_pose[1:] - hand_pose[:-1]) / gaps[:, None, None, None]
        wrist_velocity = (wrist_pose[1:] - wrist_pose[:-1]) / gaps[:, None, None]
        temporal = hand_velocity.square().mean() + wrist_velocity.square().mean()
        if len(hand_pose) > 2:
            temporal = temporal + (hand_velocity[1:] - hand_velocity[:-1]).square().mean()
    maximum_radians = torch.deg2rad(hand_pose.new_tensor(maximum_joint_degrees))
    angle_limit = torch.relu(torch.linalg.vector_norm(hand_pose, dim=-1) - maximum_radians).square().mean()
    total = (
        reprojection
        + initial_pose_weight * initial_pose
        + temporal_weight * temporal
        + wrist_weight * wrist
        + 0.05 * angle_limit
    )
    return {
        "total": total,
        "reprojection": reprojection,
        "initial_pose": initial_pose,
        "temporal": temporal,
        "wrist": wrist,
        "angle_limit": angle_limit,
    }


def optimize_hand_sequence(
    model,
    motion,
    indexes,
    joint_indexes,
    target,
    weights,
    focal,
    princpt,
    image_size,
    frame_numbers,
    optimization_steps,
    learning_rate,
    maximum_joint_degrees,
    initial_pose_weight,
    temporal_weight,
    wrist_weight,
    device,
):
    import torch

    device = torch.device(device)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def values(name):
        return torch.as_tensor(np.asarray(motion[name])[indexes], dtype=torch.float32, device=device)

    initial_hand = torch.stack((values("left_hand_pose"), values("right_hand_pose")), dim=1)
    base_body = values("body_pose")
    initial_wrist = base_body[:, BODY_WRIST_INDEXES].clone()
    hand_pose = torch.nn.Parameter(initial_hand.clone())
    wrist_pose = torch.nn.Parameter(initial_wrist.clone())
    optimizer = torch.optim.Adam((hand_pose, wrist_pose), lr=learning_rate)
    target = target.to(device)
    weights = weights.to(device)
    focal = focal.to(device)
    princpt = princpt.to(device)
    frame_numbers = frame_numbers.to(device)
    joint_indexes = torch.as_tensor(joint_indexes, dtype=torch.long, device=device)
    fixed = {
        "global_orient": values("global_orient"),
        "jaw_pose": values("jaw_pose"),
        "leye_pose": values("left_eye_pose"),
        "reye_pose": values("right_eye_pose"),
        "betas": values("betas"),
        "expression": values("expression"),
        "transl": values("translation"),
    }
    initial_losses = None
    losses = None
    for _ in range(optimization_steps):
        optimizer.zero_grad()
        body_pose = base_body.clone()
        body_pose[:, BODY_WRIST_INDEXES] = wrist_pose
        output = model(
            body_pose=body_pose.reshape(len(indexes), -1),
            left_hand_pose=hand_pose[:, 0].reshape(len(indexes), -1),
            right_hand_pose=hand_pose[:, 1].reshape(len(indexes), -1),
            return_verts=False,
            **fixed,
        )
        joints = output.joints[:, joint_indexes].reshape(len(indexes), -1, 3)
        projected = project_points(joints, focal, princpt, image_size).reshape(len(indexes), 2, 21, 2)
        losses = sequence_objective(
            projected,
            target,
            weights,
            hand_pose,
            initial_hand,
            wrist_pose,
            initial_wrist,
            frame_numbers,
            maximum_joint_degrees,
            initial_pose_weight,
            temporal_weight,
            wrist_weight,
        )
        if not torch.isfinite(losses["total"]):
            raise RuntimeError("hand optimization produced a non-finite loss")
        if initial_losses is None:
            initial_losses = {name: float(value.detach().cpu()) for name, value in losses.items()}
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_((hand_pose, wrist_pose), 10.0)
        optimizer.step()
    return {
        "left_hand_pose": hand_pose[:, 0].detach().cpu().numpy(),
        "right_hand_pose": hand_pose[:, 1].detach().cpu().numpy(),
        "wrist_pose": wrist_pose.detach().cpu().numpy(),
        "initial_losses": initial_losses,
        "final_losses": {name: float(value.detach().cpu()) for name, value in losses.items()},
    }


def source_side(label, input_mirrored=False):
    normalized = str(label).strip().casefold()
    if normalized not in SIDES:
        raise ValueError(f"unknown MediaPipe handedness: {label}")
    if input_mirrored:
        return normalized
    return "right" if normalized == "left" else "left"


def _landmark_error(path, frame, side):
    try:
        with np.load(path) as values:
            required = {"world_landmarks", "image_landmarks", "frame_number", "side", "handedness_score"}
            missing = sorted(required - set(values.files))
            if missing:
                return "missing " + ", ".join(missing)
            for key in ("world_landmarks", "image_landmarks"):
                value = np.asarray(values[key])
                if value.shape != (21, 3) or not np.isfinite(value).all():
                    return f"{key} must be finite with shape (21, 3)"
            if int(np.asarray(values["frame_number"]).item()) != frame:
                return "frame_number does not match the filename"
            if str(np.asarray(values["side"]).item()) != side:
                return "side does not match the filename"
    except Exception as error:
        return str(error)
    return None


def reconcile_landmark_progress(output_dir, frames):
    output_dir = Path(output_dir)
    (output_dir / "landmarks").mkdir(parents=True, exist_ok=True)
    progress = {"frames": {}}
    for frame in frames:
        progress["frames"][str(frame)] = {}
        for side in SIDES:
            path = output_dir / "landmarks" / f"{frame:05d}_{side}.npz"
            item = {"path": str(path)}
            if not path.exists():
                item["status"] = "pending"
            else:
                error = _landmark_error(path, frame, side)
                item["status"] = "error" if error else "ok"
                if error:
                    item["error"] = error
            progress["frames"][str(frame)][side] = item
    atomic_write_json(output_dir / "progress.json", progress)
    return progress


def _parse_frames(value):
    frames = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not frames or any(frame <= 0 for frame in frames) or len(set(frames)) != len(frames):
        raise argparse.ArgumentTypeError("frames must be unique positive integers")
    return frames


def _write_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def _detect(args):
    import cv2
    import mediapipe as mp

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "overlays").mkdir(exist_ok=True)
    progress = reconcile_landmark_progress(args.output_dir, args.frames)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=args.minimum_detection_confidence,
        min_hand_presence_confidence=args.minimum_detection_confidence,
        min_tracking_confidence=args.minimum_tracking_confidence,
    )
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        for frame in args.frames:
            statuses = progress["frames"][str(frame)]
            overlay_path = args.output_dir / "overlays" / f"{frame:06d}.jpg"
            if all(statuses[side]["status"] == "ok" for side in SIDES) and overlay_path.is_file():
                continue
            image_path = args.frames_dir / f"{frame:06d}.jpg"
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(image_path)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = detector.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            )
            detected = {}
            for handedness, image_points, world_points in zip(
                result.handedness, result.hand_landmarks, result.hand_world_landmarks
            ):
                category = handedness[0]
                side = source_side(category.category_name, args.input_mirrored)
                score = float(category.score)
                if side in detected and detected[side][0] >= score:
                    continue
                detected[side] = (
                    score,
                    np.asarray([[point.x, point.y, point.z] for point in image_points]),
                    np.asarray([[point.x, point.y, point.z] for point in world_points]),
                )
            for side, (score, image_points, world_points) in detected.items():
                path = args.output_dir / "landmarks" / f"{frame:05d}_{side}.npz"
                if _landmark_error(path, frame, side) is not None:
                    _write_npz(
                        path,
                        world_landmarks=world_points,
                        image_landmarks=image_points,
                        handedness_score=np.asarray(score),
                        frame_number=np.asarray(frame),
                        side=np.asarray(side),
                    )
                height, width = image.shape[:2]
                pixels = np.rint(image_points[:, :2] * [width, height]).astype(int)
                color = (255, 180, 40) if side == "left" else (40, 220, 255)
                for start, end in HAND_CONNECTIONS:
                    cv2.line(image, tuple(pixels[start]), tuple(pixels[end]), color, 2, cv2.LINE_AA)
                for point in pixels:
                    cv2.circle(image, tuple(point), 3, color, -1, cv2.LINE_AA)
                cv2.putText(image, side.upper(), tuple(pixels[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if not cv2.imwrite(str(overlay_path), image):
                raise OSError(f"failed to write {overlay_path}")
            progress = reconcile_landmark_progress(args.output_dir, args.frames)
    incomplete = [
        f"{frame} {side}"
        for frame in args.frames
        for side in SIDES
        if progress["frames"][str(frame)][side]["status"] != "ok"
    ]
    if incomplete:
        print(f"Notice: MediaPipe skipped {len(incomplete)} hand detections on rest/lowered frames ({', '.join(incomplete[:6])}). Baseline SMPLer-X pose is preserved for these frames.")


def _fit(args):
    import smplx
    import torch
    from smplx.joint_names import JOINT_NAMES

    with np.load(args.motion) as archive:
        motion = {key: archive[key] for key in archive.files}
    source_frames = [int(frame) for frame in motion["source_frame_number"]]
    frame_indexes = {frame: index for index, frame in enumerate(source_frames)}
    missing_frames = [frame for frame in args.frames if frame not in frame_indexes]
    if missing_frames:
        raise ValueError(f"frames are outside the motion: {missing_frames}")
    if args.optimization_steps <= 0 or args.learning_rate <= 0:
        raise ValueError("optimization steps and learning rate must be positive")

    selected_indexes = [frame_indexes[frame] for frame in args.frames]
    target = np.zeros((len(args.frames), 2, 21, 2), dtype=np.float32)
    weights = np.zeros((len(args.frames), 2, 21), dtype=np.float32)
    observed = np.zeros((len(args.frames), 2), dtype=bool)
    for row, frame in enumerate(args.frames):
        for side_index, side in enumerate(SIDES):
            path = args.landmarks_dir / f"{frame:05d}_{side}.npz"
            if not path.is_file() or _landmark_error(path, frame, side):
                continue
            with np.load(path) as values:
                target[row, side_index] = values["image_landmarks"][:, :2]
                weights[row, side_index] = float(values["handedness_score"])
                observed[row, side_index] = True
    if not observed.any():
        raise RuntimeError("MediaPipe did not provide any valid hand observations")

    metas = [json.loads((args.meta_dir / f"{frame:05d}_0.json").read_text()) for frame in args.frames]
    focal = torch.tensor(np.asarray([item["focal"] for item in metas]), dtype=torch.float32)
    princpt = torch.tensor(np.asarray([item["princpt"] for item in metas]), dtype=torch.float32)
    names = {name: index for index, name in enumerate(JOINT_NAMES)}
    joint_indexes = np.zeros((2, 21), dtype=np.int64)
    for side_index, side in enumerate(SIDES):
        joint_indexes[side_index, 0] = names[f"{side}_wrist"]
        for (chain, _), finger in zip(FINGER_CHAINS, REFERENCE_CHAINS):
            joint_names = [f"{side}_{finger}{number}" for number in (1, 2, 3)] + [f"{side}_{finger}"]
            for landmark_index, name in zip(chain, joint_names):
                joint_indexes[side_index, landmark_index] = names[name]

    model = smplx.create(
        str(args.model_root),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        use_pca=False,
        flat_hand_mean=False,
        num_betas=10,
        num_expression_coeffs=10,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = optimize_hand_sequence(
        model=model,
        motion=motion,
        indexes=selected_indexes,
        joint_indexes=joint_indexes,
        target=torch.tensor(target),
        weights=torch.tensor(weights),
        focal=focal,
        princpt=princpt,
        image_size=(args.image_width, args.image_height),
        frame_numbers=torch.tensor(args.frames, dtype=torch.float32),
        optimization_steps=args.optimization_steps,
        learning_rate=args.learning_rate,
        maximum_joint_degrees=args.maximum_joint_degrees,
        initial_pose_weight=args.initial_pose_weight,
        temporal_weight=args.temporal_weight,
        wrist_weight=args.wrist_weight,
        device=device,
    )
    refined = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in motion.items()}
    for row, index in enumerate(selected_indexes):
        for side_index, side in enumerate(SIDES):
            if observed[row, side_index]:
                refined[f"{side}_hand_pose"][index] = result[f"{side}_hand_pose"][row]
                refined["body_pose"][index, BODY_WRIST_INDEXES[side_index]] = result["wrist_pose"][row, side_index]
    _write_npz(args.output_motion, **refined)
    qa = {
        "method": "mediapipe_smplx_optimization",
        "device": device,
        "optimization_steps": args.optimization_steps,
        "observed_left_frames": int(observed[:, 0].sum()),
        "observed_right_frames": int(observed[:, 1].sum()),
        "requested_frames": len(args.frames),
        "initial_losses": result["initial_losses"],
        "final_losses": result["final_losses"],
    }
    if args.qa_output:
        atomic_write_json(args.qa_output, qa)
    print(json.dumps(qa, indent=2))


def _build_parser():
    parser = argparse.ArgumentParser(description="MediaPipe-to-SMPL-X hand refinement")
    commands = parser.add_subparsers(dest="command", required=True)
    detect = commands.add_parser("detect", help="detect both hands and save 21 landmarks")
    detect.add_argument("--frames-dir", type=Path, required=True)
    detect.add_argument("--model", type=Path, required=True)
    detect.add_argument("--output-dir", type=Path, required=True)
    detect.add_argument("--frames", type=_parse_frames, required=True)
    detect.add_argument("--fps", type=float, required=True)
    detect.add_argument("--minimum-detection-confidence", type=float, default=0.5)
    detect.add_argument("--minimum-tracking-confidence", type=float, default=0.5)
    detect.add_argument("--input-mirrored", action="store_true")
    fit = commands.add_parser("fit", help="convert saved landmarks into SMPL-X hand poses")
    fit.add_argument("--motion", type=Path, required=True)
    fit.add_argument("--model-root", type=Path, required=True)
    fit.add_argument("--landmarks-dir", type=Path, required=True)
    fit.add_argument("--meta-dir", type=Path, required=True)
    fit.add_argument("--output-motion", type=Path, required=True)
    fit.add_argument("--qa-output", type=Path)
    fit.add_argument("--frames", type=_parse_frames, required=True)
    fit.add_argument("--image-width", type=int, required=True)
    fit.add_argument("--image-height", type=int, required=True)
    fit.add_argument("--optimization-steps", type=int, default=600)
    fit.add_argument("--learning-rate", type=float, default=0.01)
    fit.add_argument("--maximum-joint-degrees", type=float, default=120.0)
    fit.add_argument("--initial-pose-weight", type=float, default=0.02)
    fit.add_argument("--temporal-weight", type=float, default=0.05)
    fit.add_argument("--wrist-weight", type=float, default=0.1)
    return parser


def main():
    args = _build_parser().parse_args()
    if args.command == "detect":
        _detect(args)
    else:
        _fit(args)


if __name__ == "__main__":
    main()

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


def _unit(vector, label):
    vector = np.asarray(vector, dtype=np.float64)
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length < 1e-8:
        raise ValueError(f"{label} is degenerate")
    return vector / length


def _palm_local(landmarks):
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("hand landmarks must be finite with shape (21, 3)")
    across = _unit(points[5] - points[17], "palm width")
    forward = _unit(points[9] - points[0], "palm length")
    normal = _unit(np.cross(across, forward), "palm normal")
    forward = _unit(np.cross(normal, across), "palm frame")
    frame = np.column_stack((across, forward, normal))
    return (points - points[0]) @ frame


def _rotation_between(source, target):
    source = _unit(source, "reference bone")
    target = _unit(target, "detected bone")
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if cosine > 1.0 - 1e-10:
        return np.eye(3)
    if cosine < -1.0 + 1e-10:
        basis = np.eye(3)[int(np.argmin(np.abs(source)))]
        axis = _unit(np.cross(source, basis), "opposite bone axis")
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    cross = np.cross(source, target)
    skew = np.array(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]]
    )
    return np.eye(3) + skew + skew @ skew / (1.0 + cosine)


def landmarks_to_hand_pose(landmarks, reference_landmarks, maximum_joint_degrees=150.0):
    from scipy.spatial.transform import Rotation

    target = _palm_local(landmarks)
    reference = _palm_local(reference_landmarks)
    matrices = np.repeat(np.eye(3)[None], 15, axis=0)
    for chain, pose_start in FINGER_CHAINS:
        cumulative = np.eye(3)
        for offset, (parent, child) in enumerate(zip(chain[:-1], chain[1:])):
            reference_direction = _unit(reference[child] - reference[parent], "reference bone")
            target_direction = _unit(target[child] - target[parent], "detected bone")
            local_target = cumulative.T @ target_direction
            local_rotation = _rotation_between(reference_direction, local_target)
            matrices[pose_start + offset] = local_rotation
            cumulative = cumulative @ local_rotation
    pose = Rotation.from_matrix(matrices).as_rotvec()
    degrees = np.degrees(np.linalg.norm(pose, axis=1))
    if np.any(degrees > maximum_joint_degrees):
        joint = int(np.argmax(degrees))
        raise ValueError(f"joint {joint} rotation is {degrees[joint]:.1f} degrees")
    return pose


def source_side(label, input_mirrored=False):
    normalized = str(label).strip().casefold()
    if normalized not in SIDES:
        raise ValueError(f"unknown MediaPipe handedness: {label}")
    if input_mirrored:
        return normalized
    return "right" if normalized == "left" else "left"


def fuse_hand_poses(motion, predictions, required_frames=None):
    refined = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in motion.items()
    }
    frame_indexes = {
        int(frame): index for index, frame in enumerate(np.asarray(motion["source_frame_number"]))
    }
    frames = list(frame_indexes) if required_frames is None else list(required_frames)
    for frame in frames:
        if frame not in frame_indexes:
            raise ValueError(f"frame {frame} is outside the motion")
        index = frame_indexes[frame]
        if frame in predictions:
            for side in SIDES:
                if side in predictions[frame]:
                    pose = np.asarray(predictions[frame][side], dtype=np.float64)
                    if pose.shape != (15, 3) or not np.isfinite(pose).all():
                        raise ValueError(f"frame {frame} {side} pose must be finite with shape (15, 3)")
                    refined[f"{side}_hand_pose"][index] = pose
    return refined


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


def _reference_landmarks(model_root, betas):
    import smplx
    import torch
    from smplx.joint_names import JOINT_NAMES

    model = smplx.create(
        str(model_root),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
        num_expression_coeffs=10,
    )
    with torch.no_grad():
        output = model(betas=torch.tensor(np.asarray(betas)[None], dtype=torch.float32))
    joints = output.joints[0].detach().cpu().numpy()
    indexes = {name: index for index, name in enumerate(JOINT_NAMES[: len(joints)])}
    references = {}
    for side in SIDES:
        points = np.zeros((21, 3), dtype=np.float64)
        points[0] = joints[indexes[f"{side}_wrist"]]
        for (chain, _), finger in zip(FINGER_CHAINS, REFERENCE_CHAINS):
            names = [f"{side}_{finger}{number}" for number in (1, 2, 3)] + [f"{side}_{finger}"]
            for landmark_index, name in zip(chain, names):
                points[landmark_index] = joints[indexes[name]]
        references[side] = points
    return references


def _fit(args):
    with np.load(args.motion) as archive:
        motion = {key: archive[key] for key in archive.files}
    references = _reference_landmarks(args.model_root, np.median(motion["betas"], axis=0))
    predictions = {}
    for frame in args.frames:
        predictions[frame] = {}
        for side in SIDES:
            path = args.landmarks_dir / f"{frame:05d}_{side}.npz"
            if path.is_file():
                error = _landmark_error(path, frame, side)
                if error:
                    continue
                try:
                    with np.load(path) as values:
                        predictions[frame][side] = landmarks_to_hand_pose(
                            values["world_landmarks"],
                            references[side],
                            maximum_joint_degrees=args.maximum_joint_degrees,
                        )
                except Exception as exc:
                    print(f"Notice: frame {frame} {side} skipped ({exc}); baseline SMPLer-X pose preserved.")
    refined = fuse_hand_poses(motion, predictions, required_frames=args.frames)
    _write_npz(args.output_motion, **refined)


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
    fit.add_argument("--output-motion", type=Path, required=True)
    fit.add_argument("--frames", type=_parse_frames, required=True)
    fit.add_argument("--maximum-joint-degrees", type=float, default=150.0)
    return parser


def main():
    args = _build_parser().parse_args()
    if args.command == "detect":
        _detect(args)
    else:
        _fit(args)


if __name__ == "__main__":
    main()

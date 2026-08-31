import json
import hashlib
import os
from fractions import Fraction
from pathlib import Path

import numpy as np


TERMINAL_STATUSES = {"ok", "no_detection", "error"}
REQUIRED_CONFIG_KEYS = {
    "sign_id",
    "source_video",
    "processing_fps",
    "model",
    "source_frame_width",
    "source_frame_height",
    "source_fps",
    "expected_source_frames",
    "temporary_avatar",
    "reviewer_count",
    "hand_refinement",
}
UPSTREAM_MOTION_FIELDS = {
    "global_orient": ("global_orient", (1, 3)),
    "body_pose": ("body_pose", (21, 3)),
    "left_hand_pose": ("left_hand_pose", (15, 3)),
    "right_hand_pose": ("right_hand_pose", (15, 3)),
    "jaw_pose": ("jaw_pose", (1, 3)),
    "left_eye_pose": ("leye_pose", (1, 3)),
    "right_eye_pose": ("reye_pose", (1, 3)),
    "betas": ("betas", (1, 10)),
    "expression": ("expression", (1, 10)),
    "translation": ("transl", (1, 3)),
}
REVIEW_CATEGORIES = (
    "handshape",
    "orientation",
    "location",
    "movement",
    "non_manual",
)


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_CONFIG_KEYS - config.keys())
    if missing:
        raise ValueError(f"Missing configuration keys: {', '.join(missing)}")
    for key in (
        "processing_fps",
        "source_frame_width",
        "source_frame_height",
        "source_fps",
        "expected_source_frames",
        "reviewer_count",
    ):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    hand_refinement = config["hand_refinement"]
    if not isinstance(hand_refinement, dict):
        raise ValueError("hand_refinement must be an object")
    preview_frames = hand_refinement.get("preview_frames")
    if (
        not isinstance(preview_frames, list)
        or not preview_frames
        or any(not isinstance(frame, int) or frame <= 0 for frame in preview_frames)
        or len(set(preview_frames)) != len(preview_frames)
    ):
        raise ValueError("hand_refinement.preview_frames must contain unique positive integers")
    if hand_refinement.get("method") != "mediapipe":
        raise ValueError("hand_refinement.method must be mediapipe")
    if not isinstance(hand_refinement.get("input_mirrored"), bool):
        raise ValueError("hand_refinement.input_mirrored must be boolean")
    for key in ("minimum_detection_confidence", "minimum_tracking_confidence"):
        value = hand_refinement.get(key)
        if not isinstance(value, (int, float)) or not 0.0 < value <= 1.0:
            raise ValueError(f"hand_refinement.{key} must be greater than 0 and at most 1")
    maximum_degrees = hand_refinement.get("maximum_joint_degrees")
    if not isinstance(maximum_degrees, (int, float)) or not 0.0 < maximum_degrees <= 180.0:
        raise ValueError("hand_refinement.maximum_joint_degrees must be greater than 0 and at most 180")
    return config


def validate_file(path, expected_size, expected_sha256=None):
    path = Path(path)
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"{path.name} size is {path.stat().st_size}, expected {expected_size}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256.lower():
        raise ValueError(
            f"{path.name} SHA-256 is {actual_sha256}, expected {expected_sha256}"
        )
    return {"path": str(path), "size": expected_size, "sha256": actual_sha256}


def validate_source_probe(probe, config):
    streams = [stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"]
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream, found {len(streams)}")
    stream = streams[0]
    expected = {
        "codec_name": "h264",
        "width": config["source_frame_width"],
        "height": config["source_frame_height"],
        "nb_frames": str(config["expected_source_frames"]),
    }
    errors = [
        f"{key}: expected {value}, got {stream.get(key)}"
        for key, value in expected.items()
        if stream.get(key) != value
    ]
    try:
        fps = float(Fraction(stream["avg_frame_rate"]))
        duration = float(stream.get("duration", probe.get("format", {}).get("duration")))
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(f"Invalid FPS or duration in source probe: {error}") from error
    if abs(fps - config["source_fps"]) > 0.01:
        errors.append(f"fps: expected {config['source_fps']}, got {fps}")
    expected_duration = config["expected_source_frames"] / config["source_fps"]
    if abs(duration - expected_duration) > 0.1:
        errors.append(f"duration: expected about {expected_duration:.2f}, got {duration}")
    if errors:
        raise ValueError("Source video mismatch: " + "; ".join(errors))
    return {"codec": stream["codec_name"], "fps": fps, "frame_count": int(stream["nb_frames"]), "duration": duration}


def first_unprocessed_frame(progress, total_frames):
    frames = progress.get("frames", {})
    for frame in range(1, total_frames + 1):
        if frames.get(str(frame), {}).get("status") not in TERMINAL_STATUSES:
            return frame
    return None


def atomic_write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def reconcile_progress(run_dir, total_frames):
    run_dir = Path(run_dir)
    progress_path = run_dir / "progress.json"
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.exists()
        else {"frames": {}}
    )
    frames = progress.setdefault("frames", {})

    for frame in range(1, total_frames + 1):
        smplx = run_dir / "smplx" / f"{frame:05d}_0.npz"
        meta = run_dir / "meta" / f"{frame:05d}_0.json"
        overlay = run_dir / "overlays" / f"{frame:06d}.jpg"
        paths = {"smplx": str(smplx), "meta": str(meta), "overlay": str(overlay)}

        if smplx.exists() and meta.exists() and overlay.exists():
            frames[str(frame)] = {"status": "ok", "outputs": paths}
        elif overlay.exists() and not smplx.exists() and not meta.exists():
            frames[str(frame)] = {"status": "no_detection", "outputs": paths}
        elif smplx.exists() or meta.exists() or overlay.exists():
            frames[str(frame)] = {
                "status": "error",
                "outputs": paths,
                "error": "incomplete frame artifacts",
            }
        elif frames.get(str(frame), {}).get("status") not in TERMINAL_STATUSES:
            frames[str(frame)] = {"status": "pending", "outputs": paths}

    atomic_write_json(progress_path, progress)
    return progress


def consolidate_motion(run_dir, total_frames, fps):
    run_dir = Path(run_dir)
    missing = [
        frame
        for frame in range(1, total_frames + 1)
        if not (run_dir / "smplx" / f"{frame:05d}_0.npz").exists()
    ]
    if missing:
        raise ValueError(f"Missing SMPL-X frames: {missing}")

    motion = {field: [] for field in UPSTREAM_MOTION_FIELDS}
    for frame in range(1, total_frames + 1):
        path = run_dir / "smplx" / f"{frame:05d}_0.npz"
        with np.load(path) as values:
            for target, (source, shape) in UPSTREAM_MOTION_FIELDS.items():
                if source not in values:
                    raise ValueError(f"{path.name} is missing {source}")
                value = np.asarray(values[source])
                if value.shape != shape:
                    raise ValueError(
                        f"{path.name} {source} has shape {value.shape}, expected {shape}"
                    )
                if not np.isfinite(value).all():
                    raise ValueError(f"{path.name} {source} contains NaN or Inf")
                motion[target].append(value[0] if shape[0] == 1 else value)

    raw = {field: np.stack(values) for field, values in motion.items()}
    raw.update(
        fps=np.asarray(fps, dtype=np.int32),
        valid_frame_mask=np.ones(total_frames, dtype=bool),
        source_frame_number=np.arange(1, total_frames + 1, dtype=np.int32),
    )
    clean = {field: value.copy() for field, value in raw.items()}
    clean["betas"][:] = np.median(raw["betas"], axis=0)

    motion_dir = run_dir / "motion"
    motion_dir.mkdir(exist_ok=True)
    raw_path = motion_dir / "suhu_motion_raw.npz"
    clean_path = motion_dir / "suhu_motion.npz"
    np.savez_compressed(raw_path, **raw)
    np.savez_compressed(clean_path, **clean)
    return {"raw": raw_path, "clean": clean_path}


def _axis_angle_to_matrix(values):
    vectors = np.asarray(values, dtype=np.float64)
    flat = vectors.reshape(-1, 3)
    theta_squared = np.sum(flat * flat, axis=1)
    theta = np.sqrt(theta_squared)
    small = theta_squared < 1e-12
    a = np.empty_like(theta)
    b = np.empty_like(theta)
    a[small] = 1.0 - theta_squared[small] / 6.0
    b[small] = 0.5 - theta_squared[small] / 24.0
    a[~small] = np.sin(theta[~small]) / theta[~small]
    b[~small] = (1.0 - np.cos(theta[~small])) / theta_squared[~small]

    skew = np.zeros((len(flat), 3, 3), dtype=np.float64)
    skew[:, 0, 1] = -flat[:, 2]
    skew[:, 0, 2] = flat[:, 1]
    skew[:, 1, 0] = flat[:, 2]
    skew[:, 1, 2] = -flat[:, 0]
    skew[:, 2, 0] = -flat[:, 1]
    skew[:, 2, 1] = flat[:, 0]
    matrices = (
        np.eye(3)[None, :, :]
        + a[:, None, None] * skew
        + b[:, None, None] * (skew @ skew)
    )
    return matrices.reshape(vectors.shape[:-1] + (3, 3))


def mano_hand_pose_to_axis_angle(rotations, is_right):
    from scipy.spatial.transform import Rotation

    matrices = np.asarray(rotations, dtype=np.float64)
    if matrices.shape != (15, 3, 3) or not np.isfinite(matrices).all():
        raise ValueError("MANO hand pose must be finite with shape (15, 3, 3)")
    if not is_right:
        mirror = np.diag([-1.0, 1.0, 1.0])
        matrices = mirror @ matrices @ mirror
    return Rotation.from_matrix(matrices).as_rotvec()


def fuse_hand_predictions(motion, predictions, required_frames=None):
    refined = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in motion.items()
    }
    source_frames = np.asarray(motion["source_frame_number"])
    frame_indexes = {int(frame): index for index, frame in enumerate(source_frames)}
    frames = list(frame_indexes) if required_frames is None else list(required_frames)

    for frame in frames:
        if frame not in frame_indexes:
            raise ValueError(f"frame {frame} is outside the motion")
        frame_predictions = predictions.get(frame, {})
        index = frame_indexes[frame]
        for side, is_right in (("left", False), ("right", True)):
            if side not in frame_predictions:
                raise ValueError(f"frame {frame} {side} prediction is missing")
            refined[f"{side}_hand_pose"][index] = mano_hand_pose_to_axis_angle(
                frame_predictions[side], is_right=is_right
            )
    return refined


def _jump_report(values):
    rotations = _axis_angle_to_matrix(values)
    if rotations.ndim == 3:
        rotations = rotations[:, None, :, :]
    relative = np.swapaxes(rotations[:-1], -1, -2) @ rotations[1:]
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1, 1)
    jumps = np.degrees(np.arccos(cosine))
    median = float(np.median(jumps)) if jumps.size else 0.0
    mad = float(np.median(np.abs(jumps - median))) if jumps.size else 0.0
    robust_threshold = median + 6.0 * mad
    flagged = (jumps > robust_threshold) & (jumps > 45.0)
    outlier_frames = (np.flatnonzero(np.any(flagged, axis=1)) + 2).tolist()
    return {
        "maximum_degrees": float(np.max(jumps)) if jumps.size else 0.0,
        "robust_threshold_degrees": robust_threshold,
        "outlier_frames": outlier_frames,
    }


def analyze_motion(motion):
    invalid = sum(
        int(np.size(value) - np.count_nonzero(np.isfinite(value)))
        for value in motion.values()
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number)
    )
    jumps = {
        field: _jump_report(motion[field])
        for field in (
            "global_orient",
            "body_pose",
            "left_hand_pose",
            "right_hand_pose",
        )
    }
    jumps["left_wrist_pose"] = _jump_report(motion["body_pose"][:, 19])
    jumps["right_wrist_pose"] = _jump_report(motion["body_pose"][:, 20])
    return {
        "invalid_value_count": invalid,
        "jumps": jumps,
        "shape_variance_max": float(np.max(np.var(motion["betas"], axis=0))),
    }


def assess_reviews(rows, sign_id, reviewer_count):
    if len(rows) != reviewer_count:
        raise ValueError(f"Expected {reviewer_count} reviews, found {len(rows)}")
    reviewer_ids = [str(row.get("reviewer_id", "")).strip() for row in rows]
    if any(not value for value in reviewer_ids) or len(set(reviewer_ids)) != len(rows):
        raise ValueError("Reviewer IDs must be present and unique")

    scores = {}
    for category in REVIEW_CATEGORIES:
        try:
            values = [float(row[category]) for row in rows]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid {category} score") from error
        if any(value not in (0.0, 1.0, 2.0) for value in values):
            raise ValueError(f"{category} scores must be 0, 1, or 2")
        scores[category] = values

    recognized_count = sum(
        str(row.get("identified_word", "")).strip().casefold() == sign_id.casefold()
        for row in rows
    )
    averages = {category: sum(values) / reviewer_count for category, values in scores.items()}
    failures = []
    if recognized_count != reviewer_count:
        failures.append("recognition")
    for category, values in scores.items():
        if averages[category] < 1.5:
            failures.append(f"{category}_average")
        if values.count(0.0) >= 2:
            failures.append(f"{category}_zeroes")
    return {
        "passed": not failures,
        "recognized_count": recognized_count,
        "category_averages": averages,
        "failures": failures,
    }

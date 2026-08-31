import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np

from suhu_pilot import atomic_write_json


SIDES = (("left", False), ("right", True))


def project_points(points, focal, princpt):
    points = np.asarray(points, dtype=np.float64)
    focal = np.asarray(focal, dtype=np.float64)
    princpt = np.asarray(princpt, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be finite with shape (N, 3)")
    if focal.shape != (2,) or princpt.shape != (2,):
        raise ValueError("focal and princpt must have shape (2,)")
    if np.any(points[:, 2] <= 0):
        raise ValueError("hand vertices must have positive camera depth")
    return points[:, :2] * focal / points[:, 2:3] + princpt


def square_box_from_points(points, width, height, padding=2.0, minimum=32):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("projected hand points must be finite with shape (N, 2)")
    if width <= 0 or height <= 0 or padding <= 0 or minimum <= 0:
        raise ValueError("image dimensions, padding, and minimum must be positive")
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = (low + high) / 2.0
    side = max(float(np.max(high - low)) * padding, float(minimum))
    side = min(side, float(width), float(height))
    x1 = np.clip(center[0] - side / 2.0, 0.0, width - side)
    y1 = np.clip(center[1] - side / 2.0, 0.0, height - side)
    return np.array([x1, y1, x1 + side, y1 + side], dtype=np.float32)


def hand_boxes_from_vertices(
    vertices,
    vertex_ids,
    focal,
    princpt,
    width,
    height,
    padding=2.0,
    minimum=32,
):
    vertices = np.asarray(vertices)
    boxes = {}
    for side, _ in SIDES:
        key = f"{side}_hand"
        if key not in vertex_ids:
            raise ValueError(f"vertex mapping is missing {key}")
        indexes = np.asarray(vertex_ids[key], dtype=np.int64)
        if indexes.ndim != 1 or indexes.size == 0 or np.any(indexes < 0) or np.any(indexes >= len(vertices)):
            raise ValueError(f"vertex mapping for {key} is invalid")
        points = project_points(vertices[indexes], focal, princpt)
        boxes[side] = square_box_from_points(
            points,
            width,
            height,
            padding=padding,
            minimum=minimum,
        )
    return boxes


def _prediction_error(path, frame, is_right):
    try:
        with np.load(path) as values:
            required = {"hand_pose", "global_orient", "box", "is_right", "frame_number"}
            missing = sorted(required - set(values.files))
            errors = [f"missing {', '.join(missing)}"] if missing else []
            expected_shapes = {
                "hand_pose": (15, 3, 3),
                "global_orient": (1, 3, 3),
                "box": (4,),
            }
            for key, shape in expected_shapes.items():
                if key not in values:
                    continue
                value = np.asarray(values[key])
                if value.shape != shape:
                    errors.append(f"{key} has shape {value.shape}, expected {shape}")
                elif not np.isfinite(value).all():
                    errors.append(f"{key} contains NaN or Inf")
            if errors:
                return "; ".join(errors)
            if int(np.asarray(values["frame_number"]).item()) != frame:
                return "frame_number does not match the filename"
            if bool(np.asarray(values["is_right"]).item()) != is_right:
                return "is_right does not match the filename"
    except Exception as error:
        return str(error)
    return None


def reconcile_hand_progress(output_dir, frames):
    output_dir = Path(output_dir)
    (output_dir / "mano").mkdir(parents=True, exist_ok=True)
    progress = {"frames": {}}
    for frame in frames:
        frame_status = {}
        for side, is_right in SIDES:
            path = output_dir / "mano" / f"{frame:05d}_{side}.npz"
            item = {"path": str(path)}
            if not path.exists():
                item["status"] = "pending"
            else:
                error = _prediction_error(path, frame, is_right)
                item["status"] = "error" if error else "ok"
                if error:
                    item["error"] = error
            frame_status[side] = item
        progress["frames"][str(frame)] = frame_status
    atomic_write_json(output_dir / "progress.json", progress)
    return progress


def _parse_frames(value):
    frames = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not frames or any(frame <= 0 for frame in frames) or len(set(frames)) != len(frames):
        raise argparse.ArgumentTypeError("frames must be unique positive integers")
    return frames


def _build_parser():
    parser = argparse.ArgumentParser(description="Resumable HaMeR refinement for SMPL-X hands")
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--meta-dir", type=Path, required=True)
    parser.add_argument("--smplx-model-root", type=Path, required=True)
    parser.add_argument("--vertex-ids", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=_parse_frames, required=True)
    parser.add_argument("--box-padding", type=float, default=2.0)
    parser.add_argument("--minimum-box-size", type=float, default=32.0)
    return parser


def run_inference(args):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import cv2
    import smplx
    import torch
    from hamer.datasets.vitdet_dataset import DEFAULT_MEAN, DEFAULT_STD, ViTDetDataset
    from hamer.models import load_hamer
    from hamer.utils import recursive_to
    from hamer.utils.renderer import Renderer

    for path in (
        args.motion,
        args.frames_dir,
        args.meta_dir,
        args.smplx_model_root,
        args.vertex_ids,
        args.checkpoint,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("mano", "crops", "overlays"):
        (args.output_dir / name).mkdir(exist_ok=True)

    with np.load(args.motion) as archive:
        motion = {key: archive[key] for key in archive.files}
    with args.vertex_ids.open("rb") as stream:
        vertex_ids = pickle.load(stream)

    body_model = smplx.create(
        str(args.smplx_model_root),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        use_pca=False,
        num_betas=10,
        num_expression_coeffs=10,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("HaMeR refinement requires a CUDA runtime")
    hamer_model, model_cfg = load_hamer(str(args.checkpoint))
    hamer_model = hamer_model.to(device).eval()
    renderer = Renderer(model_cfg, faces=hamer_model.mano.faces)
    source_indexes = {
        int(frame): index for index, frame in enumerate(motion["source_frame_number"])
    }

    progress = reconcile_hand_progress(args.output_dir, args.frames)
    for frame in args.frames:
        if frame not in source_indexes:
            raise ValueError(f"frame {frame} is outside the motion")
        missing_sides = [
            (side, is_right)
            for side, is_right in SIDES
            if progress["frames"][str(frame)][side]["status"] != "ok"
        ]
        if not missing_sides:
            continue

        index = source_indexes[frame]
        tensor = lambda name, flatten=False: torch.tensor(
            motion[name][index : index + 1].reshape(1, -1)
            if flatten
            else motion[name][index : index + 1],
            dtype=torch.float32,
        )
        with torch.no_grad():
            output = body_model(
                global_orient=tensor("global_orient"),
                body_pose=tensor("body_pose", True),
                left_hand_pose=tensor("left_hand_pose", True),
                right_hand_pose=tensor("right_hand_pose", True),
                jaw_pose=tensor("jaw_pose"),
                leye_pose=tensor("left_eye_pose"),
                reye_pose=tensor("right_eye_pose"),
                betas=tensor("betas"),
                expression=tensor("expression"),
                transl=tensor("translation"),
            )
        vertices = output.vertices[0].detach().cpu().numpy()
        meta = json.loads((args.meta_dir / f"{frame:05d}_0.json").read_text())
        image_path = args.frames_dir / f"{frame:06d}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"cannot decode {image_path}")
        boxes = hand_boxes_from_vertices(
            vertices,
            vertex_ids,
            meta["focal"],
            meta["princpt"],
            image.shape[1],
            image.shape[0],
            args.box_padding,
            args.minimum_box_size,
        )
        selected_boxes = np.stack([boxes[side] for side, _ in missing_sides])
        handedness = np.asarray([is_right for _, is_right in missing_sides], dtype=np.float32)
        # The projected boxes already include the configured padding.
        dataset = ViTDetDataset(model_cfg, image, selected_boxes, handedness, rescale_factor=1.0)
        batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)))
        batch = recursive_to(batch, device)
        with torch.no_grad():
            result = hamer_model(batch)

        for item_index, (side, is_right) in enumerate(missing_sides):
            prediction_path = args.output_dir / "mano" / f"{frame:05d}_{side}.npz"
            patch = batch["img"][item_index].detach().cpu()
            patch_rgb = patch * torch.tensor(DEFAULT_STD)[:, None, None] / 255
            patch_rgb = patch_rgb + torch.tensor(DEFAULT_MEAN)[:, None, None] / 255
            patch_rgb = np.clip(patch_rgb.permute(1, 2, 0).numpy(), 0, 1)
            rendered = renderer(
                result["pred_vertices"][item_index].detach().cpu().numpy(),
                result["pred_cam_t"][item_index].detach().cpu().numpy(),
                batch["img"][item_index],
                mesh_base_color=(0.65, 0.74, 0.86),
                scene_bg_color=(1, 1, 1),
            )
            crop_path = args.output_dir / "crops" / f"{frame:05d}_{side}.jpg"
            overlay_path = args.output_dir / "overlays" / f"{frame:05d}_{side}.jpg"
            if not cv2.imwrite(str(crop_path), np.uint8(255 * patch_rgb[:, :, ::-1])):
                raise OSError(f"failed to write {crop_path}")
            if not cv2.imwrite(str(overlay_path), np.uint8(255 * rendered[:, :, ::-1])):
                raise OSError(f"failed to write {overlay_path}")
            temporary = prediction_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    hand_pose=result["pred_mano_params"]["hand_pose"][item_index].detach().cpu().numpy(),
                    global_orient=result["pred_mano_params"]["global_orient"][item_index].detach().cpu().numpy(),
                    betas=result["pred_mano_params"]["betas"][item_index].detach().cpu().numpy(),
                    box=boxes[side],
                    is_right=np.asarray(is_right),
                    frame_number=np.asarray(frame),
                )
            os.replace(temporary, prediction_path)
            progress = reconcile_hand_progress(args.output_dir, args.frames)

    progress = reconcile_hand_progress(args.output_dir, args.frames)
    incomplete = [
        f"{frame}:{side}"
        for frame in args.frames
        for side, _ in SIDES
        if progress["frames"][str(frame)][side]["status"] != "ok"
    ]
    if incomplete:
        raise RuntimeError("Incomplete HaMeR predictions: " + ", ".join(incomplete))
    print(json.dumps({"frames": args.frames, "output_dir": str(args.output_dir)}))


def main():
    args = _build_parser().parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()

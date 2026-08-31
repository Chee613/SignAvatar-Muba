import argparse
import json
import os
from pathlib import Path

import numpy as np


def _parse_frames(value):
    frames = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not frames or any(frame <= 0 for frame in frames) or len(set(frames)) != len(frames):
        raise argparse.ArgumentTypeError("frames must be unique positive integers")
    return frames


def _build_parser():
    parser = argparse.ArgumentParser(description="Render consolidated SMPL-X motion")
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--meta-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=_parse_frames)
    parser.add_argument("--flat-hand-mean", action="store_true")
    return parser


def main():
    args = _build_parser().parse_args()
    for path in (args.motion, args.model_root, args.meta_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import cv2
    import pyrender
    import smplx
    import torch
    import trimesh

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.motion) as archive:
        motion = {key: archive[key] for key in archive.files}
    source_frames = [int(frame) for frame in motion["source_frame_number"]]
    if len(source_frames) != len(set(source_frames)):
        raise ValueError("source_frame_number must contain unique values")
    indexes = {frame: index for index, frame in enumerate(source_frames)}
    frames = args.frames or source_frames
    missing = [frame for frame in frames if frame not in indexes]
    if missing:
        raise ValueError(f"frames are outside the motion: {missing}")

    metas = [
        json.loads((args.meta_dir / f"{frame:05d}_0.json").read_text())
        for frame in source_frames
    ]
    focal = np.median(np.asarray([item["focal"] for item in metas]), axis=0)
    princpt = np.median(np.asarray([item["princpt"] for item in metas]), axis=0)
    model = smplx.create(
        str(args.model_root),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        use_pca=False,
        flat_hand_mean=args.flat_hand_mean,
        num_betas=10,
        num_expression_coeffs=10,
    )
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        roughnessFactor=0.75,
        alphaMode="OPAQUE",
        baseColorFactor=(0.18, 0.72, 0.78, 1.0),
    )
    renderer = pyrender.OffscreenRenderer(viewport_width=1280, viewport_height=720)
    rotation = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])

    def tensor(name, index, flatten=False):
        values = motion[name][index : index + 1]
        if flatten:
            values = values.reshape(1, -1)
        return torch.tensor(values, dtype=torch.float32)

    try:
        for frame in frames:
            index = indexes[frame]
            with torch.no_grad():
                output = model(
                    global_orient=tensor("global_orient", index),
                    body_pose=tensor("body_pose", index, True),
                    left_hand_pose=tensor("left_hand_pose", index, True),
                    right_hand_pose=tensor("right_hand_pose", index, True),
                    jaw_pose=tensor("jaw_pose", index),
                    leye_pose=tensor("left_eye_pose", index),
                    reye_pose=tensor("right_eye_pose", index),
                    betas=tensor("betas", index),
                    expression=tensor("expression", index),
                    transl=tensor("translation", index),
                )
            mesh = trimesh.Trimesh(output.vertices[0].numpy(), model.faces, process=False)
            mesh.apply_transform(rotation)
            scene = pyrender.Scene(
                bg_color=(0.025, 0.035, 0.08, 1.0),
                ambient_light=(0.45, 0.45, 0.45),
            )
            scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True))
            scene.add(
                pyrender.IntrinsicsCamera(
                    fx=float(focal[0]),
                    fy=float(focal[1]),
                    cx=float(princpt[0]),
                    cy=float(princpt[1]),
                )
            )
            for position in ((0, -1, 1), (0, 1, 1), (1, 1, 2)):
                pose = np.eye(4)
                pose[:3, 3] = position
                scene.add(
                    pyrender.DirectionalLight(color=np.ones(3), intensity=1.1),
                    pose=pose,
                )
            rgb, _ = renderer.render(scene)
            output_path = args.output_dir / f"{frame:06d}.png"
            if not cv2.imwrite(str(output_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise OSError(f"failed to write {output_path}")
    finally:
        renderer.delete()


if __name__ == "__main__":
    main()

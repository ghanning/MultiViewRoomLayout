import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import omnicv
import tqdm


def Ry(a: float) -> np.ndarray:
    return np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])


def create_perspective_views(pano_path: Path, pose_path: Path, output_dir: Path) -> None:
    """! Create perspective views from panorama image.

    @see https://kaustubh-sadekar.github.io/OmniCV-Lib/Equirectangular-to-perspective.html#example-code-for-equirectangular-to-perspective-conversion

    @param pano_path Path to panorama image.
    @param pose_path Path to panorama pose file.
    @param output_dir Output directory.
    """
    img_pano = cv2.imread(pano_path)

    with open(pose_path) as f:
        pose = json.load(f)

    Rt = np.array(pose["camera_rt_matrix"])
    R, t = Rt[:3, :3], Rt[:, 3]

    mapper = omnicv.fisheyeImgConv()
    fov = 90
    phi = 0

    size = 1024
    c = size / 2.0
    f = c / np.tan(np.deg2rad(fov / 2.0))
    K = np.array(
        [
            [f, 0, c],
            [0, f, c],
            [0, 0, 1],
        ]
    )

    for theta in (0, 90, 180, 270):
        img_persp = mapper.eqruirect2persp(img_pano, fov, theta, phi, size, size)
        rgb_dir = output_dir / "rgb"
        rgb_dir.mkdir(exist_ok=True, parents=True)
        name = pano_path.stem.replace("equirectangular", str(theta))
        cv2.imwrite(rgb_dir / f"{name}.jpg", img_persp)

        R_ = Ry(-np.deg2rad(theta))
        pose = dict(
            camera_rt_matrix=np.c_[R_ @ R, R_ @ t].tolist(),
            camera_k_matrix=K.tolist(),
            image_width=size,
            image_height=size,
        )
        pose_dir = output_dir / "pose"
        pose_dir.mkdir(exist_ok=True, parents=True)
        name = name.replace("rgb", "pose")
        with open(pose_dir / f"{name}.json", "w") as f:
            json.dump(pose, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert panoramas to perspective images")
    parser.add_argument("--root_dir", "-rd", type=Path, required=True, help="Path to 2D-3D-Semantics root directory")
    parser.add_argument("--area", "-a", type=str, required=True, help="Area")
    args = parser.parse_args()

    pano_dir = args.root_dir / args.area / "pano"
    persp_dir = args.root_dir / args.area / "persp"

    for img_path in tqdm.tqdm(list((pano_dir / "rgb").glob("*.png"))):
        pose_path = pano_dir / "pose" / (img_path.stem.replace("rgb", "pose") + ".json")
        create_perspective_views(img_path, pose_path, persp_dir)

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import imageio
import mat73
import numpy as np
import torch
import tqdm

from .cuboid import Cuboid
from .fit_cuboid import cuboid_distance, fit_cuboid, render_cuboid


def main(disjoint_space: Dict, scene: str, output_dir: Path) -> bool:
    logging.info(f"Processing scene {scene}")

    verts, vert_colors, names = list(), list(), list()
    objects = disjoint_space["object"]
    num_objects = len(objects["name"])
    for i in range(num_objects):
        verts.append(objects["points"][i])
        vert_colors.append(objects["RGB_color"][i])
        names.append(objects["name"][i])

    fwc_verts = np.concatenate(
        [v for v, n in zip(verts, names) if "floor" in n or "wall" in n or "ceiling" in n]
    ).astype(np.float32)
    R, t, s = fit_cuboid(torch.from_numpy(fwc_verts))

    cuboid_path = output_dir / f"{scene}.json"
    with open(cuboid_path, "w") as f:
        data = dict(R=R.numpy().tolist(), t=t.numpy().tolist(), s=s.numpy().tolist())
        json.dump(data, f)

    cuboid = Cuboid(R.numpy(), t.numpy(), s.numpy())
    all_verts = np.concatenate(verts).astype(np.float32)
    all_colors = np.concatenate(vert_colors).astype(np.float32)
    frames = render_cuboid(all_verts, all_colors / 255.0, cuboid.corners[cuboid.edges].reshape(-1, 3))
    anim_path = output_dir / f"{scene}.gif"
    imageio.mimsave(anim_path, frames, fps=5, loop=0)

    dist = cuboid_distance(R, t, s, torch.from_numpy(fwc_verts))
    success = torch.sum(dist > 0.1) / dist.shape[0] <= 0.1
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit cuboids to point clouds")
    parser.add_argument("--root_dir", "-rd", type=Path, required=True, help="Path to 2D-3D Semantics root directory")
    parser.add_argument("--output_dir", "-od", type=Path, required=True, help="Path to output directory")
    parser.add_argument("--output_split_path", "-osp", type=Path, help="Path to output list of scenes")
    parser.add_argument("--area", "-a", type=str, required=True, help="Area")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="[%(asctime)s %(module)s %(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    pointcloud_path = args.root_dir / args.area / "3d" / "pointcloud.mat"
    pointcloud = mat73.loadmat(pointcloud_path)
    key = args.area[0].upper() + args.area[1:]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    successful_scenes = list()

    for disjoint_space in tqdm.tqdm(pointcloud[key]["Disjoint_Space"]):
        scene = f"{args.area}:{disjoint_space['name']}"
        if main(disjoint_space, scene, args.output_dir):
            successful_scenes.append(scene)

    if args.output_split_path:
        args.output_split_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_split_path, "w") as f:
            for scene in successful_scenes:
                f.write(f"{scene}\n")

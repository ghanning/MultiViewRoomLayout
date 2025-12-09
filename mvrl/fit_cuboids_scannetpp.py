import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import imageio
import numpy as np
import torch
import tqdm
from plyfile import PlyData

from .cuboid import Cuboid
from .fit_cuboid import cuboid_distance, fit_cuboid, render_cuboid


def read_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """! Read mesh.

    @param path Path to .ply file.
    @return The faces, vertices, vertex colors and vertex labels.
    """
    mesh = PlyData.read(path, known_list_len={"face": {"vertex_indices": 3}})
    faces = np.vstack(mesh["face"].data["vertex_indices"])
    verts = np.stack([mesh["vertex"]["x"], mesh["vertex"]["y"], mesh["vertex"]["z"]]).T
    vert_colors = np.stack([mesh["vertex"]["red"], mesh["vertex"]["green"], mesh["vertex"]["blue"]]).T
    vert_labels = np.array(mesh["vertex"]["label"])
    return faces, verts, vert_colors, vert_labels


def read_classes(path: Path) -> Dict:
    """! Read semantic classes.

    @param Path to semantic class list.
    @return The mapping from class name to index.
    """
    with open(path) as f:
        classes = [line.strip() for line in f]
    class2idx = {class_: idx for idx, class_ in enumerate(classes)}
    return class2idx


def main(root_dir: Path, scene: str, output_dir: Path) -> bool:
    logging.info(f"Processing scene {scene}")
    scene_dir = root_dir / "data" / scene

    mesh_path = scene_dir / "scans" / "mesh_aligned_0.05_semantic.ply"
    _, verts, vert_colors, vert_labels = read_mesh(mesh_path)
    logging.info(f"Read semantic mesh with {verts.shape[0]} vertices")

    classes_path = root_dir / "metadata" / "semantic_classes.txt"
    class2idx = read_classes(classes_path)

    mask = np.zeros(verts.shape[0], dtype=bool)
    for class_ in ("floor", "wall", "ceiling"):
        mask[vert_labels == class2idx[class_]] = True

    R, t, s = fit_cuboid(torch.from_numpy(verts[mask]))

    cuboid_path = output_dir / f"{scene}.json"
    with open(cuboid_path, "w") as f:
        data = dict(R=R.numpy().tolist(), t=t.numpy().tolist(), s=s.numpy().tolist())
        json.dump(data, f)

    cuboid = Cuboid(R.numpy(), t.numpy(), s.numpy())
    frames = render_cuboid(verts, vert_colors / 255.0, cuboid.corners[cuboid.edges].reshape(-1, 3))
    anim_path = output_dir / f"{scene}.gif"
    imageio.mimsave(anim_path, frames, fps=5, loop=0)

    dist = cuboid_distance(R, t, s, torch.from_numpy(verts[mask]))
    success = torch.sum(dist > 0.1) / dist.shape[0] <= 0.1
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit cuboids to semantic meshes")
    parser.add_argument("--root_dir", "-rd", type=Path, required=True, help="Path to ScanNet++ v2 root directory")
    parser.add_argument("--output_dir", "-od", type=Path, required=True, help="Path to output directory")
    parser.add_argument("--output_split_path", "-osp", type=Path, help="Path to output list of scenes")
    parser.add_argument("--scenes", "-sc", type=str, nargs="+", help="Scenes")
    parser.add_argument("--split_path", "-sp", type=Path, help="Path to text file with list of scenes")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="[%(asctime)s %(module)s %(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    if args.split_path:
        with open(args.split_path, "r") as f:
            scenes = [line.strip() for line in f]
    else:
        scenes = args.scenes

    args.output_dir.mkdir(parents=True, exist_ok=True)

    successful_scenes = list()

    for scene in tqdm.tqdm(scenes):
        if main(args.root_dir, scene, args.output_dir):
            successful_scenes.append(scene)

    if args.output_split_path:
        args.output_split_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_split_path, "w") as f:
            for scene in successful_scenes:
                f.write(f"{scene}\n")

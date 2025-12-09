import argparse
import json
import random
from pathlib import Path

import tqdm

from .cuboid import Cuboid
from .utils import read_pose_2d3ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample image tuples")
    parser.add_argument("--root_dir", "-rd", type=Path, required=True, help="Path to 2D-3D-Semantics root directory")
    parser.add_argument("--output_path", "-op", type=Path, required=True, help="Path to output JSON file")
    parser.add_argument("--split_path", "-sp", type=Path, required=True, help="Path to text file with list of scenes")
    parser.add_argument("--cuboid_path", "-cp", type=Path, required=True, help="Path to file with ground truth cuboids")
    parser.add_argument("--num_panos", "-np", type=int, required=True, help="Number of panorama images per tuple")
    parser.add_argument("--seed", "-s", type=int, default=1, help="Random seed")
    args = parser.parse_args()

    with open(args.split_path, "r") as f:
        scenes = [line.strip() for line in f]

    with open(args.cuboid_path) as f:
        cuboid_params = json.load(f)

    random.seed(args.seed)
    image_tuples = list()

    for scene in tqdm.tqdm(scenes):
        area, space = scene.split(":")

        cuboid = Cuboid.from_dict(cuboid_params[scene])

        pano_dir = args.root_dir / area / "pano"
        pano_names = list()

        for img_path in (pano_dir / "rgb").glob(f"*_{space}_*"):
            pose_path = pano_dir / "pose" / (img_path.stem.replace("rgb", "pose") + ".json")
            R, t, _, _ = read_pose_2d3ds(pose_path)
            pano_center = -R.T @ t

            if not cuboid.inside(pano_center[None]):
                continue

            pano_names.append(img_path.name)

        images = random.sample(pano_names, args.num_panos)

        persp_img_dir = args.root_dir / area / "persp" / "rgb"
        perspective_images = [
            p.name for i in images for p in sorted(persp_img_dir.glob(f"camera_{i.split('_')[1]}_*.jpg"))
        ]

        image_tuples.append(dict(scene=scene, images=images, perspective_images=perspective_images))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_path, "w") as f:
        json.dump(image_tuples, f)

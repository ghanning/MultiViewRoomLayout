import argparse
import json
import random
from pathlib import Path

import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample image tuples")
    parser.add_argument("--root_dir", "-rd", type=Path, required=True, help="Path to ScanNet++ v2 root directory")
    parser.add_argument("--output_path", "-op", type=Path, required=True, help="Path to output JSON file")
    parser.add_argument("--split_path", "-sp", type=Path, required=True, help="Path to text file with list of scenes")
    parser.add_argument("--num_images", "-ni", type=int, required=True, help="Number of images per tuple")
    parser.add_argument("--num_tuples", "-nt", type=int, required=True, help="Number of image tuples per scene")
    parser.add_argument("--seed", "-s", type=int, default=1, help="Random seed")
    args = parser.parse_args()

    with open(args.split_path, "r") as f:
        scenes = [line.strip() for line in f]

    random.seed(args.seed)
    image_tuples = list()

    for scene in tqdm.tqdm(scenes):
        image_dir = args.root_dir / "data" / scene / "dslr" / "undistorted_images"
        image_names = sorted([i.name for i in image_dir.glob("*.JPG")])
        for _ in range(args.num_tuples):
            image_tuples.append(dict(scene=scene, images=random.sample(image_names, args.num_images)))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_path, "w") as f:
        json.dump(image_tuples, f)

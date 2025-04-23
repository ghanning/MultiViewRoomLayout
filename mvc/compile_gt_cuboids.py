import argparse
import json
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create file with ground truth cuboids")
    parser.add_argument("--cuboid_dir", "-cd", type=Path, required=True, help="Path to directory with fitted cuboids")
    parser.add_argument("--output_path", "-op", type=Path, required=True, help="Path to output JSON file")
    parser.add_argument("--split_path", "-sp", type=Path, required=True, help="Path to text file with list of scenes")
    args = parser.parse_args()

    with open(args.split_path, "r") as f:
        scenes = [line.strip() for line in f]

    cuboids = dict()

    for scene in scenes:
        cuboid_path = args.cuboid_dir / f"{scene}.json"
        with open(cuboid_path) as f:
            cuboids[scene] = json.load(f)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_path, "w") as f:
        json.dump(cuboids, f)

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from meshlib import mrmeshnumpy, mrmeshpy

from .cuboid import Cuboid
from .utils import chunk, dataset_dir, get_images_2d3ds, get_images_scannetpp


def orthogonal_vector(x: np.ndarray) -> np.ndarray:
    """! Create a vector orthogonal to another vector.

    @param x The input vector.
    @return A vector orthogonal to x.
    """
    y = np.zeros_like(x)
    m = (x != 0).argmax()
    n = m + 1 if m + 1 < x.shape[0] else 0
    y[n] = x[m]
    y[m] = -x[n]
    return y


def random_rotation() -> np.ndarray:
    """! Generate a random rotation matrix.

    @return The rotation matrix (3, 3).
    """
    x = np.random.rand(3)
    x /= np.linalg.norm(x)
    y = orthogonal_vector(x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return np.c_[x, y, z]


def make_prediction(images: List[Tuple], margin: float = 1.0) -> Cuboid:
    """! Predict cuboid room layout.

    @param images Input images.
    @param margin Margin between camera centers and the faces of the cuboid.
    @return The predicted cuboid.
    """
    R = random_rotation()
    c = np.stack([-Ri.T @ ti for Ri, ti, _, _ in images]) @ R.T
    t = -(np.min(c, axis=0) + np.max(c, axis=0)) / 2.0
    s = 2.0 * np.max(np.abs(c + t) + margin, axis=0)
    return Cuboid(R, t, s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate example room layout predictions")
    parser.add_argument(
        "--root_dir", "-rd", type=Path, required=True, help="Path to ScanNet++ v2 or 2D-3D-Semantics root directory"
    )
    parser.add_argument("--dataset", "-d", required=True, choices=("scannetpp", "2d3ds"), help="Dataset")
    parser.add_argument("--split", "-s", required=True, choices=("train", "val", "test", "all"), help="Data split")
    parser.add_argument("--num_pred", "-np", type=int, default=1, help="Number of predictions per image tuple")
    parser.add_argument("--num_images", "-ni", type=int, help="Number of images per tuple (ScanNet++)")
    parser.add_argument("--output_path", "-op", type=Path, required=True, help="Path to output JSON file")
    parser.add_argument(
        "--mesh_dir", "-md", type=Path, help="Mesh directory (if set the predictions are saved as meshes)"
    )
    args = parser.parse_args()

    with open(dataset_dir() / args.dataset / f"images_{args.split}.json") as f:
        image_tuples = json.load(f)

    predictions = list()
    transforms_cache = dict()
    pred_num = 0

    for image_tuple in image_tuples:
        scene = image_tuple["scene"]

        if args.dataset == "scannetpp":
            images, _ = get_images_scannetpp(
                args.root_dir, scene, image_tuple["images"][: args.num_images], transforms_cache
            )
            assert args.num_pred in (1, len(images))
        else:
            images, _ = get_images_2d3ds(args.root_dir, scene, image_tuple["perspective_images"])
            assert args.num_pred in (1, 2, len(images))

        preds_tuple = list()
        for image_chunk in chunk(images, len(images) // args.num_pred):
            cuboid = make_prediction(image_chunk)
            if args.mesh_dir:
                mesh = mrmeshnumpy.meshFromFacesVerts(cuboid.faces, cuboid.corners)
                path = args.mesh_dir / f"mesh{pred_num}.obj"
                pred_num += 1
                args.mesh_dir.mkdir(parents=True, exist_ok=True)
                mrmeshpy.saveMesh(mesh, path)
                preds_tuple.append(str(path))
            else:
                preds_tuple.append(dict(R=cuboid.R.tolist(), t=cuboid.t.tolist(), s=cuboid.s.tolist()))

        predictions.append(preds_tuple)

    with open(args.output_path, "w") as f:
        json.dump(predictions, f)

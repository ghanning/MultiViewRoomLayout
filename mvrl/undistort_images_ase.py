import argparse
import json
from pathlib import Path

import numpy as np
import pycolmap
import tqdm
from projectaria_tools.projects import ase

from .utils import dataset_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Undistort ASE images")
    parser.add_argument("--root_dir", "-rd", type=Path, required=True, help="Path to ASE root directory")
    args = parser.parse_args()

    scenes = []
    for split in ("val", "test"):
        with open(dataset_dir() / "ase" / f"scenes_{split}.txt") as f:
            scenes.extend([line.strip() for line in f.readlines()])

    device = ase.get_ase_rgb_calibration()
    params = device.get_projection_params()
    focal_length = params[0]
    params = np.insert(params, 0, focal_length)
    width, height = device.get_image_size()

    camera = pycolmap.Camera(model="RAD_TAN_THIN_PRISM_FISHEYE", width=width, height=height, params=params.tolist())
    options = pycolmap.UndistortCameraOptions()
    options.min_scale = 1.0
    options.max_scale = 1.0
    camera_undistorted = pycolmap.undistort_camera(options, camera)

    # Write undistorted camera
    data = camera_undistorted.todict()
    data["model"] = data["model"].name
    data["params"] = data["params"].tolist()
    with open(args.root_dir / "camera_undistorted.json", "w") as f:
        json.dump(data, f)

    # Undistort images
    for scene in tqdm.tqdm(scenes, desc="scenes"):
        input_dir = args.root_dir / scene / "rgb"
        output_dir = args.root_dir / scene / "rgb_undistorted"
        output_dir.mkdir(exist_ok=True)

        for image_path in tqdm.tqdm(list(input_dir.glob("*.jpg")), desc="images"):
            image = pycolmap.Bitmap().read(image_path, True)
            image_undistorted, cam = pycolmap.undistort_image(options, image, camera)
            assert cam.model == camera_undistorted.model and np.all(cam.params == camera_undistorted.params)
            ok = image_undistorted.write(output_dir / image_path.name)
            assert ok, f"Failed to write undistorted image {image_path.name}"

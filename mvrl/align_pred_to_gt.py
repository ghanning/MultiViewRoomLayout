import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pycolmap
import tqdm
from meshlib import mrmeshnumpy

from .metric import Metric
from .utils import (
    DATASETS,
    Image,
    dataset_dir,
    flatten_multi_room,
    get_images,
    get_layout,
    layout_to_mesh,
)


def create_reconstruction(images: List[Image]) -> pycolmap.Reconstruction:
    reconstruction = pycolmap.Reconstruction()

    # Assume all images have the same intrinsics
    K = images[0].K
    camera = pycolmap.Camera(
        camera_id=1,
        model="PINHOLE",
        width=images[0].width,
        height=images[0].height,
        params=[K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
    )
    reconstruction.add_camera(camera)

    rig = pycolmap.Rig(rig_id=1)
    sensor = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=camera.camera_id)
    rig.add_ref_sensor(sensor)
    reconstruction.add_rig(rig)

    for idx, image in enumerate(images):
        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(image.R), image.t)

        frame = pycolmap.Frame(frame_id=idx + 1, rig_id=rig.rig_id)
        image = pycolmap.Image(
            image_id=idx + 1, camera_id=camera.camera_id, frame_id=frame.frame_id, name=image.path.name
        )
        frame.add_data_id(image.data_id)
        reconstruction.add_frame(frame)
        reconstruction.frame(frame.frame_id).set_cam_from_world(
            camera_id=camera.camera_id, cam_from_world=cam_from_world
        )
        reconstruction.add_image(image)
        reconstruction.register_frame(frame.frame_id)

    return reconstruction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align predicted layouts to ground truth using COLMAP")
    parser.add_argument("--root_dir", "-rd", type=Path, required=True, help="Path to dataset root directory")
    parser.add_argument("--input_pred", "-ip", type=Path, required=True, help="Path to file with input predictions")
    parser.add_argument("--output_pred", "-op", type=Path, required=True, help="Path to file with output predictions")
    parser.add_argument("--dataset", "-d", required=True, choices=DATASETS, help="Dataset")
    parser.add_argument("--split", "-s", required=True, help="Data split ('train', 'val', 'test' etc.)")
    parser.add_argument("--max_error", "-me", type=float, help="Maximum alignment error (m)")
    args = parser.parse_args()

    with open(dataset_dir() / args.dataset / f"images_{args.split}.json") as f:
        image_tuples = json.load(f)

    with open(args.input_pred) as f:
        layouts = json.load(f)

    if args.dataset == "ase" or args.split == "multi_room":
        image_tuples, _, layouts = flatten_multi_room(image_tuples, None, layouts)
    assert len(layouts) == len(image_tuples)

    mean_metric = Metric("Mean alignment error", unit="m")
    median_metric = Metric("Median alignment error", unit="m")

    cache = {}
    layouts_aligned = []

    for image_tuple, layout in tqdm.tqdm(list(zip(image_tuples, layouts))):
        if isinstance(layout, list):
            raise NotImplementedError("Multiple predictions per image tuple not supported")

        num_images = len(layout["poses"])
        images_gt = get_images(args.dataset, args.root_dir, image_tuple, cache, num_images)

        images_pred = []
        for i, p in zip(images_gt, layout["poses"]):
            R, t = np.array(p["R"]), np.array(p["t"])
            images_pred.append(Image(R, t, i.K, i.width, i.height, i.path))
        reconstruction_pred = create_reconstruction(images_pred)

        tgt_image_names = [img.path.name for img in images_gt]
        tgt_locations = np.stack([-img.R.T @ img.t for img in images_gt])
        min_common_images = 3
        ransac_options = pycolmap.RANSACOptions()
        if args.max_error is not None:
            ransac_options.max_error = args.max_error
        transform = pycolmap.align_reconstruction_to_locations(
            reconstruction_pred, tgt_image_names, tgt_locations, min_common_images, ransac_options
        )
        if transform is None:
            raise RuntimeError("Alignment failed")

        pred_locations = np.stack([-img.R.T @ img.t for img in images_pred])
        aligned_locations = transform * pred_locations
        error = np.linalg.norm(aligned_locations - tgt_locations, axis=1)
        mean_metric.add(np.mean(error))
        median_metric.add(np.median(error))

        layout_aligned = {"poses": []}
        for img in images_pred:
            cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(img.R), img.t)
            cam_from_world = transform.transform_camera_world(cam_from_world)
            layout_aligned["poses"].append(
                {"R": cam_from_world.rotation.matrix().tolist(), "t": cam_from_world.translation.tolist()}
            )
        mesh = layout_to_mesh(get_layout(layout, args.input_pred.parent))
        faces = mrmeshnumpy.getNumpyFaces(mesh.topology)
        verts = transform * mrmeshnumpy.getNumpyVerts(mesh)
        layout_aligned["faces"] = faces.tolist()
        layout_aligned["verts"] = verts.tolist()
        layouts_aligned.append(layout_aligned)

    mean_metric.print()
    median_metric.print()

    args.output_pred.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_pred, "w") as f:
        json.dump(layouts_aligned, f)

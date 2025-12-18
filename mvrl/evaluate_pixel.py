import argparse
import json
from pathlib import Path

import numpy as np
import tqdm

from .metric import Metric
from .metrics import depth_normal_error
from .renderer import Renderer
from .utils import (
    dataset_dir,
    flatten_multi_room,
    get_images_2d3ds,
    get_images_scannetpp,
    get_layout,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predicted layouts (pixel-wise metrics)")
    parser.add_argument(
        "--root_dir", "-rd", type=Path, required=True, help="Path to ScanNet++ v2 or 2D-3D-Semantics root directory"
    )
    parser.add_argument("--pred", "-p", type=Path, required=True, help="Path to file with layout predictions")
    parser.add_argument("--dataset", "-d", required=True, choices=("scannetpp", "2d3ds"), help="Dataset")
    parser.add_argument("--split", "-s", required=True, help="Data split ('train', 'val', 'test' etc.)")
    parser.add_argument("--num_images", "-ni", type=int, help="Number of images per tuple (ScanNet++)")
    parser.add_argument(
        "--normal_angle_threshold", "-nat", type=float, default=10.0, help="Normal angle error threshold"
    )
    parser.add_argument("--skip", "-sk", action="store_true", help="Skip views for which metrics could not be computed")
    args = parser.parse_args()

    with open(dataset_dir() / args.dataset / f"images_{args.split}.json") as f:
        image_tuples = json.load(f)

    with open(dataset_dir() / args.dataset / f"layouts_{args.split}.json") as f:
        layouts_gt = json.load(f)

    with open(args.pred) as f:
        layout_preds_per_tuple = json.load(f)

    if args.split == "multi_room":
        image_tuples, layouts_gt, layout_preds_per_tuple = flatten_multi_room(
            image_tuples, layouts_gt, layout_preds_per_tuple
        )
    assert len(layout_preds_per_tuple) == len(image_tuples)

    depth_metric = Metric("Depth RMSE", unit="m")
    normal_metric = Metric(f"Normal angle error (recall @ {args.normal_angle_threshold} deg)")

    transforms_cache = dict()
    renderer = None
    num_skip, num_tot = 0, 0

    for image_tuple, layouts_pred in tqdm.tqdm(list(zip(image_tuples, layout_preds_per_tuple))):
        scene = image_tuple["scene"]
        layout_gt = get_layout(layouts_gt[scene])
        scene = scene.split(":")[0]

        if args.dataset == "scannetpp":
            images, image_size = get_images_scannetpp(
                args.root_dir, scene, image_tuple["images"][: args.num_images], transforms_cache
            )
        else:
            images, image_size = get_images_2d3ds(args.root_dir, scene, image_tuple["perspective_images"])

        if not isinstance(layouts_pred, list):
            layouts_pred = [layouts_pred]
        layouts_pred = [get_layout(p, args.pred.parent) for p in layouts_pred]

        if renderer is None or renderer.fbo.size != image_size:
            renderer = Renderer(image_size)

        for image_idx, (R, t, K, path) in enumerate(images):
            if len(layouts_pred) == 1:  # Single prediction
                pred_idx = 0
            elif len(layouts_pred) == len(images):  # One prediction per perspective image
                pred_idx = image_idx
            else:  # One prediction per panorama (for 2D-3D-Semantics)
                assert args.dataset == "2d3ds"
                pred_idx = image_idx // (len(images) // len(image_tuple["images"]))
            depth_rmse, normal_error = depth_normal_error(
                layout_gt, layouts_pred[pred_idx], renderer, R, t, K, np.deg2rad(args.normal_angle_threshold), path
            )
            if np.isnan(depth_rmse) or np.isnan(normal_error):
                if args.skip:
                    num_skip += 1
                else:
                    raise RuntimeError("Failed to compute metrics")
            else:
                depth_metric.add(depth_rmse)
                normal_metric.add(normal_error)

        num_tot += len(images)

    print(depth_metric.summary())
    print(normal_metric.summary())

    if num_skip > 0:
        print(f"Skipped {num_skip} out of {num_tot} views")

    del renderer

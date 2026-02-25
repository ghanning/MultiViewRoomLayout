import argparse
import json
from pathlib import Path

import numpy as np
import tqdm

from .metric import Metric
from .metrics import depth_normal_error
from .renderer import Renderer
from .utils import DATASETS, dataset_dir, flatten_multi_room, get_images, get_layout

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predicted layouts (pixel-wise metrics)")
    parser.add_argument("--root_dir", "-rd", type=Path, required=True, help="Path to dataset root directory")
    parser.add_argument("--pred", "-p", type=Path, required=True, help="Path to file with layout predictions")
    parser.add_argument("--dataset", "-d", required=True, choices=DATASETS, help="Dataset")
    parser.add_argument("--split", "-s", required=True, help="Data split ('train', 'val', 'test' etc.)")
    parser.add_argument("--num_images", "-ni", type=int, help="Number of images per tuple (ScanNet++/ASE)")
    parser.add_argument("--flatten", "-f", action="store_true", help="Flatten multi-room layouts")
    parser.add_argument(
        "--normal_angle_threshold", "-nat", type=float, default=10.0, help="Normal angle error threshold"
    )
    args = parser.parse_args()

    with open(dataset_dir() / args.dataset / f"images_{args.split}.json") as f:
        image_tuples = json.load(f)

    with open(dataset_dir() / args.dataset / f"layouts_{args.split}.json") as f:
        layouts_gt = json.load(f)

    with open(args.pred) as f:
        layout_preds_per_tuple = json.load(f)

    if args.flatten:  # Flatten lists to compute metrics per room instead of per scene
        image_tuples, layouts_gt, layout_preds_per_tuple = flatten_multi_room(
            image_tuples, layouts_gt, layout_preds_per_tuple
        )
    assert len(layout_preds_per_tuple) == len(image_tuples)

    depth_metric = Metric("Depth RMSE", unit="m")
    normal_metric = Metric(f"Normal angle error (recall @ {args.normal_angle_threshold} deg)")

    cache = dict()
    renderer = None

    for image_tuple, layouts_pred in tqdm.tqdm(list(zip(image_tuples, layout_preds_per_tuple))):
        layout_gt = get_layout(layouts_gt[image_tuple["scene"]])
        images = get_images(args.dataset, args.root_dir, image_tuple, cache, args.num_images)

        if not isinstance(layouts_pred, list):
            layouts_pred = [layouts_pred]
        layouts_pred = [get_layout(p, args.pred.parent) for p in layouts_pred]

        image_size = (images[0].width, images[0].height)  # Assume all images have the same size
        if renderer is None or renderer.fbo.size != image_size:
            renderer = Renderer(image_size)

        for image_idx, image in enumerate(images):
            if len(layouts_pred) == 1:  # Single prediction
                pred_idx = 0
            elif len(layouts_pred) == len(images):  # One prediction per perspective image
                pred_idx = image_idx
            else:  # One prediction per panorama (for 2D-3D-Semantics)
                assert args.dataset == "2d3ds"
                pred_idx = image_idx // (len(images) // len(image_tuple["images"]))
            depth_rmse, normal_error = depth_normal_error(
                layout_gt, layouts_pred[pred_idx], renderer, image, np.deg2rad(args.normal_angle_threshold)
            )
            depth_metric.add(depth_rmse)
            normal_metric.add(normal_error)

    depth_metric.print()
    normal_metric.print()
    del renderer

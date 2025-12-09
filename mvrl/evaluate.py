import argparse
import json
from pathlib import Path

import numpy as np
import tqdm

from .cuboid import Cuboid
from .metric import Metric
from .metrics import chamfer_distance, iou3d, rotation_error
from .utils import dataset_dir, get_layout

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predicted layouts")
    parser.add_argument("--pred", "-p", type=Path, required=True, help="Path to file with layout predictions")
    parser.add_argument("--dataset", "-d", required=True, choices=("scannetpp", "2d3ds"), help="Dataset")
    parser.add_argument("--split", "-s", required=True, choices=("train", "val", "test"), help="Data split")
    parser.add_argument("--use_best", "-ub", action="store_true", help="Use prediction with highest IoU for each scene")
    args = parser.parse_args()

    with open(dataset_dir() / args.dataset / f"images_{args.split}.json") as f:
        image_tuples = json.load(f)

    with open(dataset_dir() / args.dataset / f"layouts_{args.split}.json") as f:
        cuboid_params_gt = json.load(f)

    with open(args.pred) as f:
        layout_preds_per_tuple = json.load(f)
    assert len(layout_preds_per_tuple) == len(image_tuples)

    iou_metric = Metric("IoU")
    rot_metric = Metric("Rotation error", unit="deg")
    chamfer_metric = Metric("Chamfer distance", unit="m")
    seed = 1234

    for image_tuple, layouts_pred in tqdm.tqdm(list(zip(image_tuples, layout_preds_per_tuple))):
        scene = image_tuple["scene"]
        cuboid_gt = Cuboid.from_dict(cuboid_params_gt[scene])

        if not isinstance(layouts_pred, list):
            layouts_pred = [layouts_pred]
        layouts_pred = [get_layout(p, args.pred.parent) for p in layouts_pred]

        if args.use_best:
            ious = [iou3d(cuboid_gt, layout_pred) for layout_pred in layouts_pred]
            idx = np.argmax(ious)
            iou_metric.add(ious[idx])
            if isinstance(layouts_pred[idx], Cuboid):
                rot_metric.add(np.rad2deg(rotation_error(cuboid_gt, layouts_pred[idx])))
            chamfer_metric.add(chamfer_distance(cuboid_gt, layouts_pred[idx], seed))
        else:
            for layout_pred in layouts_pred:
                iou_metric.add(iou3d(cuboid_gt, layout_pred))
                if isinstance(layout_pred, Cuboid):
                    rot_metric.add(np.rad2deg(rotation_error(cuboid_gt, layout_pred)))
                chamfer_metric.add(chamfer_distance(cuboid_gt, layout_pred, seed))

    print(iou_metric.summary())
    if rot_metric.values:
        print(rot_metric.summary(auc_thr=[1, 5, 10, 20]))
    print(chamfer_metric.summary())

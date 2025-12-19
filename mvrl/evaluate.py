import argparse
import json
from pathlib import Path

import numpy as np
import tqdm

from .cuboid import Cuboid
from .metric import Metric
from .metrics import chamfer_distance, iou3d, rotation_error, wall_recall
from .utils import DATASETS, dataset_dir, flatten_multi_room, get_layout

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predicted layouts")
    parser.add_argument("--pred", "-p", type=Path, required=True, help="Path to file with layout predictions")
    parser.add_argument("--dataset", "-d", required=True, choices=DATASETS, help="Dataset")
    parser.add_argument("--split", "-s", required=True, help="Data split ('train', 'val', 'test' etc.)")
    parser.add_argument(
        "--metrics", "-m", nargs="+", default=["iou", "rotation", "chamfer", "recall"], help="Metrics to evaluate"
    )
    parser.add_argument("--use_best", "-ub", action="store_true", help="Use prediction with highest IoU for each scene")
    args = parser.parse_args()

    with open(dataset_dir() / args.dataset / f"images_{args.split}.json") as f:
        image_tuples = json.load(f)

    with open(dataset_dir() / args.dataset / f"layouts_{args.split}.json") as f:
        layouts_gt = json.load(f)

    with open(args.pred) as f:
        layout_preds_per_tuple = json.load(f)

    if args.dataset == "ase" or args.split == "multi_room":
        image_tuples, layouts_gt, layout_preds_per_tuple = flatten_multi_room(
            image_tuples, layouts_gt, layout_preds_per_tuple
        )
    assert len(layout_preds_per_tuple) == len(image_tuples)

    iou_metric = Metric("IoU")
    rot_metric = Metric("Rotation error", unit="deg")
    chamfer_metric = Metric("Chamfer distance", unit="m")
    wall_metric = Metric("Wall recall")
    room_metric = Metric("Room recall")
    seed = 1234

    for image_tuple, layouts_pred in tqdm.tqdm(list(zip(image_tuples, layout_preds_per_tuple))):
        scene = image_tuple["scene"]
        layout_gt = get_layout(layouts_gt[scene])

        if not isinstance(layouts_pred, list):
            layouts_pred = [layouts_pred]
        layouts_pred = [get_layout(p, args.pred.parent) for p in layouts_pred]

        if args.use_best:
            assert "iou" in args.metrics, "IoU metric required for best layout selection"
            ious = [iou3d(layout_gt, layout_pred) for layout_pred in layouts_pred]
            idx = np.argmax(ious)
            iou_metric.add(ious[idx])
            if isinstance(layout_gt, Cuboid) and isinstance(layouts_pred[idx], Cuboid) and "rotation" in args.metrics:
                rot_metric.add(np.rad2deg(rotation_error(layout_gt, layouts_pred[idx])))
            if "chamfer" in args.metrics:
                chamfer_metric.add(chamfer_distance(layout_gt, layouts_pred[idx], seed))
            if "recall" in args.metrics:
                recall = wall_recall(layout_gt, layouts_pred[idx])
                wall_metric.add(recall)
                room_metric.add(all(recall))
        else:
            for layout_pred in layouts_pred:
                if "iou" in args.metrics:
                    iou_metric.add(iou3d(layout_gt, layout_pred))
                if isinstance(layout_gt, Cuboid) and isinstance(layout_pred, Cuboid) and "rotation" in args.metrics:
                    rot_metric.add(np.rad2deg(rotation_error(layout_gt, layout_pred)))
                if "chamfer" in args.metrics:
                    chamfer_metric.add(chamfer_distance(layout_gt, layout_pred, seed))
                if "recall" in args.metrics:
                    recall = wall_recall(layout_gt, layout_pred)
                    wall_metric.add(recall)
                    room_metric.add(all(recall))

    iou_metric.print()
    rot_metric.print(auc_thr=[1, 5, 10, 20])
    chamfer_metric.print()
    wall_metric.print()
    room_metric.print()

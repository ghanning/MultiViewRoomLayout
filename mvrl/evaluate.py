import argparse
import json
from pathlib import Path

import numpy as np
import tqdm
from meshlib import mrmeshpy

from .cuboid import Cuboid
from .metric import Metric
from .metrics import chamfer_distance, iou3d, rotation_error, wall_f1, wall_recall
from .utils import (
    DATASETS,
    Layout,
    dataset_dir,
    get_layout,
    layout_to_mesh,
    remove_floor_ceiling,
    unflatten_predictions,
)


def multi_room_wall_recall(layout_gt: Layout, layout_pred: Layout, wall_metric: Metric, room_metric: Metric):
    """! Compute the wall & room recall. Handles multi-room layouts.

    @param layout_gt Ground truth layout.
    @param layout_pred Predicted layout.
    @param wall_metric Wall recall metric.
    @param room_metric Room recall metric.
    """
    mesh_gt = layout_to_mesh(layout_gt)
    components = mrmeshpy.getAllComponents(mesh_gt)
    for i in range(len(components)):  # Loop over rooms in the ground truth layout
        mesh_comp = mesh_gt.cloneRegion(components[i])
        # Correspondences between rooms in layout_gt & layout_pred not necessarily known, so pass the full prediction
        recall = wall_recall(mesh_comp, layout_pred)
        wall_metric.add(recall)
        room_metric.add(all(recall))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predicted layouts")
    parser.add_argument("--pred", "-p", type=Path, required=True, help="Path to file with layout predictions")
    parser.add_argument("--dataset", "-d", required=True, choices=DATASETS, help="Dataset")
    parser.add_argument("--split", "-s", required=True, help="Data split ('train', 'val', 'test' etc.)")
    parser.add_argument(
        "--metrics", "-m", nargs="+", default=["iou", "rotation", "chamfer", "recall", "f1"], help="Metrics to evaluate"
    )
    parser.add_argument("--f1_iou_thr", "-fiou", type=float, default=0.5, help="IoU threshold for wall F1 score")
    parser.add_argument("--use_best", "-ub", action="store_true", help="Use prediction with highest IoU for each scene")
    parser.add_argument("--unflatten", "-uf", action="store_true", help="Unflatten multi-room layouts")
    parser.add_argument(
        "--only_walls", "-ow", action="store_true", help="Remove floor and ceiling from ground truth layouts"
    )
    parser.add_argument("--image_tuples", "-it", type=Path, help="Path to file with image tuples")
    args = parser.parse_args()

    image_tuples_path = args.image_tuples or dataset_dir() / args.dataset / f"images_{args.split}.json"
    with open(image_tuples_path) as f:
        image_tuples = json.load(f)

    with open(dataset_dir() / args.dataset / f"layouts_{args.split}.json") as f:
        layouts_gt = json.load(f)

    with open(args.pred) as f:
        layout_preds_per_tuple = json.load(f)

    if args.unflatten:  # Unflatten predictions (for single-room method applied to multi-room dataset)
        layout_preds_per_tuple = unflatten_predictions(layout_preds_per_tuple, image_tuples)
    assert len(layout_preds_per_tuple) == len(image_tuples)

    if args.only_walls:
        layouts_gt = remove_floor_ceiling(layouts_gt)

    iou_metric = Metric("IoU")
    rot_metric = Metric("Rotation error", unit="deg")
    chamfer_metric = Metric("Chamfer distance", unit="m")
    wall_metric = Metric("Wall recall")
    room_metric = Metric("Room recall")
    f1_metric = Metric(f"Wall F1 (IoU@{args.f1_iou_thr})")
    seed = 1234

    for image_tuple, layouts_pred in tqdm.tqdm(list(zip(image_tuples, layout_preds_per_tuple))):
        scene = image_tuple["scene"]
        if "scene" in layouts_pred:
            assert layouts_pred["scene"] == scene, f"Mismatching scenes {scene} / {layouts_pred['scene']}"
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
                multi_room_wall_recall(layout_gt, layouts_pred[idx], wall_metric, room_metric)
            if "f1" in args.metrics:
                f1_metric.add(wall_f1(layout_gt, layouts_pred[idx], iou_threshold=args.f1_iou_thr))
        else:
            for layout_pred in layouts_pred:
                if "iou" in args.metrics:
                    iou_metric.add(iou3d(layout_gt, layout_pred))
                if isinstance(layout_gt, Cuboid) and isinstance(layout_pred, Cuboid) and "rotation" in args.metrics:
                    rot_metric.add(np.rad2deg(rotation_error(layout_gt, layout_pred)))
                if "chamfer" in args.metrics:
                    chamfer_metric.add(chamfer_distance(layout_gt, layout_pred, seed))
                if "recall" in args.metrics:
                    multi_room_wall_recall(layout_gt, layout_pred, wall_metric, room_metric)
                if "f1" in args.metrics:
                    f1_metric.add(wall_f1(layout_gt, layout_pred, iou_threshold=args.f1_iou_thr))

    iou_metric.print()
    rot_metric.print(auc_thr=[1, 5, 10, 20])
    chamfer_metric.print()
    wall_metric.print()
    room_metric.print()
    f1_metric.print()

from typing import List, Union

import numpy as np
from meshlib import mrmeshpy
from scipy.optimize import linear_sum_assignment
from shapely import Polygon

from ..cuboid import Cuboid
from ..utils import layout_to_mesh
from .wall_recall import get_wall_quads

ZERO_TOLERANCE = 1e-6
LARGE_COST_VALUE = 1e6


def calc_poly_iou(poly1, poly2):
    if poly1.intersects(poly2):
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
        poly_iou = inter_area / union_area if union_area > 0 else 0
    else:
        poly_iou = 0
    return poly_iou


def are_planes_parallel_and_close(
    corners_1: np.ndarray,
    corners_2: np.ndarray,
    parallel_tolerance: float,
    dist_tolerance: float,
):
    p1, p2, p3, _ = corners_1
    q1, q2, q3, _ = corners_2
    n1 = np.cross(np.subtract(p2, p1), np.subtract(p3, p1))
    n2 = np.cross(np.subtract(q2, q1), np.subtract(q3, q1))
    n1_length = np.linalg.norm(n1)
    n2_length = np.linalg.norm(n2)
    assert (
        n1_length * n2_length > ZERO_TOLERANCE
    ), f"Invalid plane corners, corners_1: {corners_1}, corners_2: {corners_2}"

    return (
        np.linalg.norm(np.cross(n1, n2)) / (n1_length * n2_length) < parallel_tolerance
        and np.abs(np.dot(np.subtract(q1, p1), n1)) / n1_length < dist_tolerance
    )


def calc_thin_bbox_iou_2d(
    corners_1: np.ndarray,
    corners_2: np.ndarray,
    parallel_tolerance: float,
    dist_tolerance: float,
):
    if are_planes_parallel_and_close(corners_1, corners_2, parallel_tolerance, dist_tolerance):
        p1, p2, _, p4 = corners_2
        v1 = np.subtract(p2, p1)
        v2 = np.subtract(p4, p1)
        basis1 = v1 / np.linalg.norm(v1)
        basis1_orth = v2 - np.dot(v2, basis1) * basis1
        basis2 = basis1_orth / np.linalg.norm(basis1_orth)

        projected_corners_1 = [
            [
                np.dot(np.subtract(point, p1), basis1),
                np.dot(np.subtract(point, p1), basis2),
            ]
            for point in corners_1
        ]
        projected_corners_2 = [
            [
                np.dot(np.subtract(point, p1), basis1),
                np.dot(np.subtract(point, p1), basis2),
            ]
            for point in corners_2
        ]
        box1 = Polygon(projected_corners_1)
        box2 = Polygon(projected_corners_2)

        return calc_poly_iou(box1, box2)
    else:
        return 0


def wall_f1(
    layout_gt: Union[Cuboid, mrmeshpy.Mesh],
    layout_pred: Union[Cuboid, mrmeshpy.Mesh],
    angle_thr: float = np.deg2rad(1),
    parallel_tolerance: float = np.sin(np.deg2rad(5)),
    dist_tolerance: float = 0.2,
    up: np.ndarray = np.array([0, 0, 1]),
    iou_threshold: float = 0.25,
) -> float:
    """! Compute the wall F1 score between the ground truth and predicted layouts.

    The metric was proposed in 'SPATIALLM: Training Large Language Models for Structured Indoor Modeling' by Mao et al.
    @see https://github.com/manycore-research/SpatialLM/blob/029808be90daef80cf2b850e254716ca7d64254f/eval.py

    @param layout_gt The ground truth layout.
    @param layout_pred The predicted layout.
    @param angle_thr The angle threshold (in radians) for clustering faces.
    @param parallel_tolerance The tolerance for checking if planes are parallel.
    @param dist_tolerance The tolerance for checking if planes are close.
    @param up The up vector.
    @param iou_threshold The IoU threshold to consider a predicted wall as a true positive.
    @return The wall F1 score.
    """
    mesh_gt, mesh_pred = layout_to_mesh(layout_gt), layout_to_mesh(layout_pred)
    walls_gt, walls_pred = get_wall_quads(mesh_gt, angle_thr, up), get_wall_quads(mesh_pred, angle_thr, up)
    num_gt, num_pred = len(walls_gt), len(walls_pred)
    assert num_gt > 0, "Ground truth layout should have at least one wall"

    iou_matrix = np.zeros((num_pred, num_gt))
    for i, wall_pred in enumerate(walls_pred):
        for j, wall_gt in enumerate(walls_gt):
            iou_matrix[i, j] = calc_thin_bbox_iou_2d(
                wall_pred,
                wall_gt,
                parallel_tolerance=parallel_tolerance,
                dist_tolerance=dist_tolerance,
            )

    cost_matrix = np.full((num_pred, num_gt), LARGE_COST_VALUE)
    cost_matrix[iou_matrix > iou_threshold] = -1
    indices = linear_sum_assignment(cost_matrix)

    debug = False
    if debug:
        import matplotlib.colors as mcolors
        import rerun as rr

        rr.init("wall_f1_debug", spawn=True)
        rr.log("pred", rr.Clear(recursive=True))
        rr.log("gt", rr.Clear(recursive=True))
        colors = [mcolors.to_rgb(v) for v in mcolors.TABLEAU_COLORS.values()]

        for i, wall in enumerate(walls_pred):
            m = np.where(indices[0] == i)[0]
            iou = iou_matrix[indices[0][m[0]], indices[1][m[0]]] if len(m) > 0 else 0
            color = colors[m[0] % len(colors)] if iou >= iou_threshold else (127, 127, 127)
            rr.log(f"pred/{i}", rr.LineStrips3D(wall[[0, 1, 2, 3, 0]], colors=color))
        for i, wall in enumerate(walls_gt):
            m = np.where(indices[1] == i)[0]
            iou = iou_matrix[indices[0][m[0]], indices[1][m[0]]] if len(m) > 0 else 0
            color = colors[m[0] % len(colors)] if iou >= iou_threshold else (127, 127, 127)
            rr.log(f"gt/{i}", rr.LineStrips3D(wall[[0, 1, 2, 3, 0]], colors=color))

        breakpoint()

    tp_percent = iou_matrix[indices[0], indices[1]]
    tp = np.sum(tp_percent >= iou_threshold)

    precision = tp / num_pred if num_pred > 0 else 0.0
    recall = tp / num_gt if num_gt > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

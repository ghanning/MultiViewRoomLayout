from typing import Optional, Union

import numpy as np
import open3d
from meshlib import mrmeshnumpy, mrmeshpy

from ..cuboid import Cuboid


def layout_to_open3d_mesh(layout: Union[Cuboid, mrmeshpy.Mesh]) -> open3d.t.geometry.TriangleMesh:
    """! Convert layout to Open3D mesh.

    @param layout The layout.
    @return The mesh.
    """
    if isinstance(layout, Cuboid):
        verts, faces = layout.corners, layout.faces
    else:
        verts, faces = mrmeshnumpy.getNumpyVerts(layout), mrmeshnumpy.getNumpyFaces(layout.topology)
    return open3d.t.geometry.TriangleMesh(open3d.core.Tensor(verts.astype(np.float32)), open3d.core.Tensor(faces))


def chamfer_distance(
    layout1: Union[Cuboid, mrmeshpy.Mesh], layout2: Union[Cuboid, mrmeshpy.Mesh], seed: Optional[int] = None
) -> float:
    """! Compute the Chamfer distance between two room layouts.

    @param layout1, layout2 The layouts (either cuboids or triangle meshes).
    @param seed Random seed value.
    @return The Chamfer distance.
    """
    if any(isinstance(x, mrmeshpy.Mesh) and x.topology.numValidFaces() == 0 for x in (layout1, layout2)):
        return np.nan
    mesh1, mesh2 = layout_to_open3d_mesh(layout1), layout_to_open3d_mesh(layout2)
    params = open3d.t.geometry.MetricParameters()
    if seed is not None:
        open3d.utility.random.seed(seed)
    metrics = mesh1.compute_metrics(
        mesh2,
        [
            open3d.t.geometry.Metric.ChamferDistance,
        ],
        params,
    )
    return metrics[0].item()

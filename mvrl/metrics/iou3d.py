from typing import Union

import numpy as np
from meshlib import mrmeshpy

from ..cuboid import Cuboid
from ..utils import layout_to_mesh


def unary_union(mesh: mrmeshpy.Mesh) -> mrmeshpy.Mesh:
    """! Compute the unary union of a mesh with multiple components.

    @param mesh The input mesh.
    @return The unionized mesh.
    """
    components = mrmeshpy.getAllComponents(mesh)
    if len(components) == 1:
        return mesh

    union = mrmeshpy.Mesh()
    for i in range(len(components)):
        mi = mesh.cloneRegion(components[i])
        union = mrmeshpy.boolean(union, mi, mrmeshpy.BooleanOperation.Union).mesh

    return union


def multi_component_intersection(mesh1: mrmeshpy.Mesh, mesh2: mrmeshpy.Mesh) -> mrmeshpy.Mesh:
    """! Compute the intersection of two meshes with multiple components.

    @param mesh1, mesh2 The input meshes.
    @return The intersection mesh.
    """
    components1 = mrmeshpy.getAllComponents(mesh1)
    components2 = mrmeshpy.getAllComponents(mesh2)

    intersection = mrmeshpy.Mesh()
    for i in range(len(components1)):
        mi = mesh1.cloneRegion(components1[i])
        for j in range(len(components2)):
            mj = mesh2.cloneRegion(components2[j])
            isect = mrmeshpy.boolean(mi, mj, mrmeshpy.BooleanOperation.Intersection)
            intersection.addMesh(isect.mesh)

    return intersection


def iou3d(layout1: Union[Cuboid, mrmeshpy.Mesh], layout2: Union[Cuboid, mrmeshpy.Mesh]) -> float:
    """! Compute the 3D intersection-over-union (IoU) between two room layouts.

    @param layout1, layout2 The layouts (either cuboids or triangle meshes).
    @return The IoU.
    """
    mesh1, mesh2 = layout_to_mesh(layout1), layout_to_mesh(layout2)
    if mesh1.topology.numValidFaces() == 0 or mesh2.topology.numValidFaces() == 0:
        return np.nan
    assert mesh1.volume() > 0.0 and mesh2.volume() > 0.0, "Zero or negative volume"

    # Unionize each layout in case they contain multiple, overlapping components (rooms)
    mesh1, mesh2 = unary_union(mesh1), unary_union(mesh2)

    # Compute the union and intersection of the two layouts
    union = mrmeshpy.Mesh(mesh1)
    union.addMesh(mesh2)
    union = unary_union(union)
    intersection = multi_component_intersection(mesh1, mesh2)

    debug = False
    if debug:
        import rerun as rr

        from .visualization import visualize_layout, visualize_mesh

        rr.init("iou3d", spawn=True)
        visualize_layout("layout1", layout1, color=[246, 205, 97, 128])
        visualize_layout("layout2", layout2, color=[14, 154, 167, 128])
        visualize_mesh("intersection", intersection, albedo_factor=[254, 138, 113, 128])
        visualize_mesh("union", union, albedo_factor=[74, 78, 77, 128])
        breakpoint()

    return intersection.volume() / union.volume()

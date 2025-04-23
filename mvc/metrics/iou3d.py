from typing import Union

from meshlib import mrmeshnumpy, mrmeshpy

from ..cuboid import Cuboid


def layout_to_mesh(layout: Union[Cuboid, mrmeshpy.Mesh]) -> mrmeshpy.Mesh:
    if isinstance(layout, Cuboid):
        return mrmeshnumpy.meshFromFacesVerts(layout.faces, layout.corners)
    else:
        return layout


def iou3d(layout1: Union[Cuboid, mrmeshpy.Mesh], layout2: Union[Cuboid, mrmeshpy.Mesh]) -> float:
    """! Compute the 3D intersection-over-union (IoU) between two room layouts.

    @param layout1, layout2 The layouts (either cuboids or triangle meshes).
    @return The IoU.
    """
    mesh1, mesh2 = layout_to_mesh(layout1), layout_to_mesh(layout2)
    intersection = mrmeshpy.boolean(mesh1, mesh2, mrmeshpy.BooleanOperation.Intersection)
    union = mrmeshpy.boolean(mesh1, mesh2, mrmeshpy.BooleanOperation.Union)

    debug = False
    if debug:
        import rerun as rr

        from .visualization import visualize_layout, visualize_mesh

        rr.init("iou3d", spawn=True)
        visualize_layout("layout1", layout1)
        visualize_layout("layout2", layout2)
        visualize_mesh("intersection", intersection.mesh)
        visualize_mesh("union", union.mesh)
        breakpoint()

    return intersection.mesh.volume() / union.mesh.volume()

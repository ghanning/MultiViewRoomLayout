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
    assert mesh1.volume() > 0.0 and mesh2.volume() > 0.0, "Zero or negative volume"
    intersection = mrmeshpy.boolean(mesh1, mesh2, mrmeshpy.BooleanOperation.Intersection)
    union = mrmeshpy.boolean(mesh1, mesh2, mrmeshpy.BooleanOperation.Union)

    debug = False
    if debug:
        import rerun as rr

        from .visualization import visualize_layout, visualize_mesh

        rr.init("iou3d", spawn=True)
        visualize_layout("layout1", layout1, color=[246, 205, 97, 128])
        visualize_layout("layout2", layout2, color=[14, 154, 167, 128])
        visualize_mesh("intersection", intersection.mesh, albedo_factor=[254, 138, 113, 128])
        visualize_mesh("union", union.mesh, albedo_factor=[74, 78, 77, 128])
        breakpoint()

    return intersection.mesh.volume() / union.mesh.volume()

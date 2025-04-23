from typing import Union

import rerun as rr
from meshlib import mrmeshnumpy, mrmeshpy

from ..cuboid import Cuboid


def visualize_camera(name: str, R, t, K, width, height) -> None:
    rr.log(name, rr.Pinhole(image_from_camera=K, width=width, height=height))
    rr.log(name, rr.Transform3D(mat3x3=R, translation=t, from_parent=True))


def visualize_cuboid(name: str, cuboid: Cuboid) -> None:
    rr.log(name, rr.Transform3D(translation=cuboid.t, mat3x3=cuboid.R, from_parent=True))
    rr.log(name, rr.Boxes3D(sizes=cuboid.s))


def visualize_mesh(name: str, mesh: mrmeshpy.Mesh) -> None:
    verts, faces = mrmeshnumpy.getNumpyVerts(mesh), mrmeshnumpy.getNumpyFaces(mesh.topology)
    rr.log(name, rr.Mesh3D(vertex_positions=verts, triangle_indices=faces))


def visualize_layout(name: str, layout: Union[Cuboid, mrmeshpy.Mesh]) -> None:
    if isinstance(layout, Cuboid):
        visualize_cuboid(name, layout)
    else:
        visualize_mesh(name, layout)

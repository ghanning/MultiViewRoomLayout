from typing import Optional, Union

import rerun as rr
from meshlib import mrmeshnumpy, mrmeshpy
from rerun.datatypes import Rgba32Like

from ..cuboid import Cuboid


def visualize_camera(name: str, R, t, K, width, height) -> None:
    rr.log(name, rr.Pinhole(image_from_camera=K, width=width, height=height))
    rr.log(name, rr.Transform3D(mat3x3=R, translation=t, from_parent=True))


def visualize_cuboid(name: str, cuboid: Cuboid, color: Optional[Rgba32Like] = None) -> None:
    rr.log(name, rr.Transform3D(translation=cuboid.t, mat3x3=cuboid.R, from_parent=True))
    rr.log(name, rr.Boxes3D(sizes=cuboid.s, colors=color))


def visualize_mesh(name: str, mesh: mrmeshpy.Mesh, albedo_factor: Optional[Rgba32Like] = None) -> None:
    verts, faces = mrmeshnumpy.getNumpyVerts(mesh), mrmeshnumpy.getNumpyFaces(mesh.topology)
    rr.log(name, rr.Mesh3D(vertex_positions=verts, triangle_indices=faces, albedo_factor=albedo_factor))


def visualize_layout(name: str, layout: Union[Cuboid, mrmeshpy.Mesh], color: Optional[Rgba32Like] = None) -> None:
    if isinstance(layout, Cuboid):
        visualize_cuboid(name, layout, color=color)
    else:
        visualize_mesh(name, layout, albedo_factor=color)

from pathlib import Path
from typing import Optional, Tuple, Union

import moderngl
import numpy as np
from meshlib import mrmeshnumpy, mrmeshpy

from ..cuboid import Cuboid
from ..renderer import Renderer


def perspective_projection(
    fx: float, fy: float, cx: float, cy: float, width: int, height: int, znear: float = 0.01, zfar: float = 1000.0
) -> np.ndarray:
    """! Calculate OpenGL perspective matrix from OpenCV/COLMAP camera intrinsics.

    @see http://kgeorge.github.io/2014/03/08/calculating-opengl-perspective-matrix-from-opencv-intrinsic-matrix

    @param fx, fy Focal lengths.
    @param cx, cy Principal point.
    @param width, height Image dimensions.
    @param znear, zfar Position of the near and far planes.
    @return The projection matrix (4, 4).
    """
    cy = height - cy  # Flip y-axis
    return np.array(
        [
            [2 * fx / width, 0, 1 - 2 * cx / width, 0],
            [0, 2 * fy / height, 1 - 2 * cy / height, 0],
            [0, 0, -(zfar + znear) / (zfar - znear), -2 * zfar * znear / (zfar - znear)],
            [0, 0, -1, 0],
        ]
    )


def render_layout(
    layout: Union[Cuboid, mrmeshpy.Mesh],
    renderer: Renderer,
    world_to_view: np.ndarray,
    view_to_clip: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """! Render room layout.

    @param layout The layout to render.
    @param renderer Renderer.
    @param world_to_view World-to-view transformation (4, 4).
    @param view_to_clip View-to-clip space transformation (4, 4).
    @return Normal and depth maps.
    """
    renderer.clear()

    if isinstance(layout, Cuboid):
        verts, faces = layout.corners, layout.faces
    else:
        verts, faces = mrmeshnumpy.getNumpyVerts(layout), mrmeshnumpy.getNumpyFaces(layout.topology)

    face_normals = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]], axis=1)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)

    verts = verts[faces[:, [1, 0, 2]]].reshape(-1, 3)
    vert_normals = np.repeat(face_normals, 3, axis=0)

    renderer.add(verts, vert_normals, moderngl.TRIANGLES)
    renderer.program["normalize_color"].value = True
    img, depth = renderer.render(world_to_view, view_to_clip, return_depth=True, as_float=True)
    return img, depth


def depth_normal_error(
    layout1: Union[Cuboid, mrmeshpy.Mesh],
    layout2: Union[Cuboid, mrmeshpy.Mesh],
    renderer: Renderer,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    normal_angle_thr: float,
    image_path: Optional[Path] = None,
) -> Tuple[float, float]:
    """! Compute depth and normal errors.

    @param layout1, layout2 The ground truth and predicted layout.
    @param renderer Renderer.
    @param R The world-to-camera rotation matrix (3, 3).
    @param t The world-to-camera translation vector (3).
    @param K The camera intrinsics (3, 3).
    @param normal_angle_thr Normal angle threshold in radians.
    @param image_path Image path (for debugging purposes).
    @return The depth RMSE and the normal angle error (ratio of pixels with normal angle error < normal_angle_thr).
    """
    if any(isinstance(x, mrmeshpy.Mesh) and x.topology.numValidFaces() == 0 for x in (layout1, layout2)):
        return np.nan, np.nan

    world_to_camera = np.eye(4)
    world_to_camera[:3, :3] = R
    world_to_camera[:3, 3] = t
    opencv_to_opengl = np.eye(4)
    opencv_to_opengl[1, 1] = opencv_to_opengl[2, 2] = -1
    world_to_view = opencv_to_opengl @ world_to_camera

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    width, height = renderer.fbo.size
    znear, zfar = 0.01, 1000.0
    view_to_clip = perspective_projection(fx, fy, cx, cy, width, height, znear=znear, zfar=zfar)

    normals1, depth1 = render_layout(layout1, renderer, world_to_view, view_to_clip)
    normals2, depth2 = render_layout(layout2, renderer, world_to_view, view_to_clip)

    mask = np.isclose(depth1, zfar) | np.isclose(depth2, zfar)
    if np.all(mask):
        return np.nan, np.nan

    depth_rmse = np.sqrt(np.mean((depth1[~mask] - depth2[~mask]) ** 2))

    dot = np.sum(normals1 * normals2, axis=2)
    angle_diff = np.arccos(np.clip(dot, -1.0, 1.0))
    normal_error = np.mean(angle_diff[~mask] < normal_angle_thr)

    debug = False
    if debug:
        import rerun as rr

        from .visualization import visualize_camera, visualize_layout

        rr.init("depth_normal_error", spawn=True)
        visualize_layout("layout1", layout1, color=[246, 205, 97, 128])
        visualize_layout("layout2", layout2, color=[14, 154, 167, 128])
        visualize_camera("camera", R, t, K, depth1.shape[1], depth1.shape[0])
        depth1[np.isclose(depth1, zfar)] = 0.0
        depth2[np.isclose(depth2, zfar)] = 0.0
        rr.log("camera/depth", rr.DepthImage(depth1))
        rr.log("camera/depth2", rr.DepthImage(depth2))
        rr.log("camera/normals1", rr.Image(normals1))
        rr.log("camera/normals2", rr.Image(normals2))
        if image_path:
            rr.log("camera/image", rr.EncodedImage(path=image_path))
        breakpoint()

    return depth_rmse, normal_error

import logging
from typing import List, Tuple

import glm
import moderngl
import numpy as np
import torch

from .renderer import Renderer


def Rx(a):
    """! Create rotation matrix for rotation around the x axis.

    @param a Rotation angle in radians.
    @return The rotation matrix (3, 3).
    """
    zero, one = torch.tensor(0.0, device=a.device), torch.tensor(1.0, device=a.device)
    ca, sa = torch.cos(a), torch.sin(a)
    return torch.stack([torch.stack([one, zero, zero]), torch.stack([zero, ca, -sa]), torch.stack([zero, sa, ca])])


def Ry(a):
    """! Create rotation matrix for rotation around the y axis.

    @param a Rotation angle in radians.
    @return The rotation matrix (3, 3).
    """
    zero, one = torch.tensor(0.0, device=a.device), torch.tensor(1.0, device=a.device)
    ca, sa = torch.cos(a), torch.sin(a)
    return torch.stack([torch.stack([ca, zero, sa]), torch.stack([zero, one, zero]), torch.stack([-sa, zero, ca])])


def Rz(a):
    """! Create rotation matrix for rotation around the z axis.

    @param a Rotation angle in radians.
    @return The rotation matrix (3, 3).
    """
    zero, one = torch.tensor(0.0, device=a.device), torch.tensor(1.0, device=a.device)
    ca, sa = torch.cos(a), torch.sin(a)
    return torch.stack([torch.stack([ca, -sa, zero]), torch.stack([sa, ca, zero]), torch.stack([zero, zero, one])])


def euler_to_matrix(r: torch.Tensor) -> torch.Tensor:
    """! Convert Euler angles to rotation matrix.

    @param r The Euler angles (roll, pitch, yaw) (3).
    @return The rotation matrix (3, 3).
    """
    R = Rz(r[2]) @ Ry(r[1]) @ Rx(r[0])
    return R


def initialize_cuboid(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """! Initialize cuboid from a set of points.

    @param points The points (N, 3).
    @return The cuboid parameters r (3), t (3) and s (3).
    """
    r = torch.zeros(3)
    q = 0.02
    t = -(torch.quantile(points, q, 0) + torch.quantile(points, 1.0 - q, 0)) / 2.0
    s = 2.0 * torch.quantile(torch.abs(points + t), 0.95, dim=0)
    return r, t, s


def cuboid_distance(R: torch.Tensor, t: torch.Tensor, s: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """! Compute the distance between a cuboid and a set of points.

    @param R, t, s The cuboid parameters.
    @param points The points (N, 3).
    @return The distance between each point and the faces of the cuboid (N).
    """
    points = points @ R.T + t
    half_size = s / 2.0
    dist_inside = torch.min(half_size - torch.abs(points), dim=1)[0]
    dist_outside = torch.linalg.norm(points.clamp(min=-half_size, max=half_size) - points, dim=1)
    inside = torch.all(torch.abs(points) <= half_size, dim=1)
    return torch.where(inside, dist_inside, dist_outside)


def fit_cuboid(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """! Fit a cuboid to a set of points.

    @param points The points (N, 3),
    @return The cuboid parameters R (3, 3), t (3) and s (3).
    """
    if points.shape[0] == 0:
        return torch.eye(3), torch.zeros(3), torch.ones(3)

    r, t, s = initialize_cuboid(points)
    for x in (r, t, s):
        x.requires_grad = True

    logging.debug(f"Initial distance: {torch.mean(cuboid_distance(euler_to_matrix(r), t, s, points)):.3f}")

    optimizer = torch.optim.LBFGS([r, t, s], lr=0.5, max_iter=1000)

    def cost_func():
        optimizer.zero_grad()
        dist = cuboid_distance(euler_to_matrix(r), t, s, points)
        loss = loss_fnc(dist, torch.zeros_like(dist))
        loss = torch.mean(loss)
        loss.backward()
        return loss

    loss_fncs = [torch.nn.HuberLoss(reduction="none", delta=d) for d in (10.0, 1.0, 0.1, 0.01)]

    for loss_fnc in loss_fncs:
        with torch.set_grad_enabled(True):
            optimizer.step(cost_func)

    logging.debug(f"Final distance: {torch.mean(cuboid_distance(euler_to_matrix(r), t, s, points)):.3f}")

    if torch.any(s <= 0.0):
        logging.warning(f"Invalid size {s}")
        s = torch.abs(s)

    return euler_to_matrix(r.detach()), t.detach(), s.detach()


def render_cuboid(
    points: np.ndarray,
    colors: np.ndarray,
    cuboid_lines: np.ndarray,
    image_size: Tuple[int, int] = (800, 600),
    num_frames: int = 50,
) -> List[np.ndarray]:
    """! Render point cloud and fitted cuboid, with the camera rotating around the scene.

    @param points Points (N, 3).
    @param colors Point colors (N, 3).
    @param cuboid_lines Cuboid lines (24, 3).
    @param image_size Image size (width, height).
    @param num_frames Number of frames to render.
    @return A list of rendered frames.
    """
    renderer = Renderer(image_size, samples=32, line_width=5.0)
    renderer.add(points, colors, moderngl.POINTS)
    renderer.add(cuboid_lines, np.ones((cuboid_lines.shape[0], 3)), moderngl.LINES)
    frames = list()

    proj = np.array(glm.perspective(45.0, 1.0, 0.1, 1000.0))
    center = np.mean(points, axis=0)
    size = np.max(points, axis=0) - np.min(points, axis=0)
    radius = 1.2 * np.max(size[:2])
    height = 1.5 * size[2]
    up = (0.0, 0.0, 1.0)

    for i in range(num_frames):
        a = i * 2.0 * np.pi / num_frames
        eye = center + np.asarray((radius * np.cos(a), radius * np.sin(a), height)).astype(np.float32)
        look = np.array(glm.lookAt(eye, center, up))
        frame = renderer.render(look, proj)
        frames.append(frame)

    return frames

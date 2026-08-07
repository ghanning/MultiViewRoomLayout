from typing import Dict, Self

import numpy as np


class Cuboid:
    """! Cuboid class.

    The pose of the cuboid is parameterized by the rotation matrix R and translation vector such that a point x in
    world coordinates is transformed to the cuboid's local frame via R * x + t.

    The size of the cuboid in the local frame is given by the size vector s.
    """

    def __init__(self, R: np.ndarray, t: np.ndarray, s: np.ndarray) -> None:
        """! Class initializer.

        @param R Rotation matrix (3, 3).
        @param t Translation vector (3).
        @param s Size vector (3).
        """
        if np.any(s <= 0.0):
            raise ValueError(f"Invalid size {s}")
        if np.linalg.det(R) < 0.0:
            raise ValueError(f"Invalid orientation {R}")
        self.R = R.copy()
        self.t = t.copy()
        self.s = s.copy()

    @classmethod
    def from_dict(cls, dict_: Dict) -> Self:
        """! Create cuboid from dictionary.

        @param dict_ Dictionary with keys "R", "t" and "s".
        @return The cuboid.
        """
        return cls(*(np.array(dict_[key]) for key in ("R", "t", "s")))

    @classmethod
    def from_Rd(cls, R: np.ndarray, d: np.ndarray) -> Self:
        """! Create cuboid from rotation matrix and plane offset vector.

        This is an alternative representation where world coordinates are transformed to the local frame by R * x, and
        the cuboid planes are defined by x = d[0], y = d[1], z = d[2] (lower bounds) and x = d[3], y = d[4], z = d[5]
        (upper bounds).

        @param R Rotation matrix (3, 3).
        @param d Plane offset vector (6).
        @return The cuboid.
        """
        t = -(d[:3] + d[3:]) / 2.0
        s = d[3:] - d[:3]
        return cls(R, t, s)

    @property
    def d(self) -> np.ndarray:
        """! Get plane offset vector.

        @return The plane offset vector (6).
        """
        t, s = self.t, self.s
        d = np.concatenate([-t - s / 2.0, -t + s / 2.0], axis=-1)
        return d

    @property
    def corners(self) -> np.ndarray:
        """! Get the corners of the cuboid in world coordinates.

        @return The cuboid corners (8, 3).
        """
        corners = (
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0],
                ]
            )
            - 0.5
        )
        return (corners * self.s - self.t) @ self.R

    @property
    def edges(self) -> np.ndarray:
        """! Get the cuboid edge indices (into the corners array).

        @return The edge indices (12, 2).
        """
        return np.array(
            [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
        )

    @property
    def faces(self) -> np.ndarray:
        """! Get the cuboid face indices (into the corners array).

        @return The face indices (12, 3).
        """
        return np.array(
            [
                [0, 1, 2],
                [3, 2, 1],
                [4, 6, 5],
                [7, 5, 6],
                [0, 4, 1],
                [5, 1, 4],
                [2, 3, 6],
                [7, 6, 3],
                [0, 2, 4],
                [6, 4, 2],
                [1, 5, 3],
                [7, 3, 5],
            ]
        )

    def inside(self, points: np.ndarray) -> np.ndarray:
        """! Check if points are inside the cuboid.

        @param points Points to check (N, 3).
        @return A boolean array indicating whether each point is inside (N).
        """
        points = points @ self.R.T + self.t
        half_size = self.s / 2.0
        return np.all(np.abs(points) <= half_size, axis=1)

from typing import List, Tuple, Union

import numpy as np
from meshlib import mrmeshpy

from ..cuboid import Cuboid
from ..utils import layout_to_mesh


def dot(v1: mrmeshpy.Vector3f, v2: mrmeshpy.Vector3f) -> float:
    return sum(v1[k] * v2[k] for k in range(3))


def cluster_faces(
    topology: mrmeshpy.MeshTopology, normals: mrmeshpy.FaceNormals, angle_thr: float
) -> List[List[mrmeshpy.FaceId]]:
    """! Cluster faces based on connectivity and normals.

    @param topology The mesh topology.
    @param normals The per-face normals.
    @param angle_thr The angle threshold (in radians) for clustering.
    @return A list of clusters, each containing a list of FaceIds.
    """
    face_ids = topology.getFaceIds(None)
    clusters = []
    visited = set()
    cos_thr = np.cos(angle_thr)

    for id in face_ids:
        if id.get() in visited:
            continue
        visited.add(id.get())
        cluster = []
        stack = [id]
        while stack:
            fi = stack.pop()
            cluster.append(fi)
            ei = [mrmeshpy.EdgeId() for _ in range(3)]
            topology.getTriEdges(fi, *ei)
            for i in range(3):
                fj = topology.right(ei[i])
                if fj.get() not in visited and dot(normals[fi], normals[fj]) > cos_thr:
                    visited.add(fj.get())
                    stack.append(fj)

        clusters.append(cluster)

    return clusters


def sample_points(
    mesh: mrmeshpy.Mesh, face_ids: List[mrmeshpy.FaceId], dist: float = 0.25
) -> Tuple[np.ndarray, np.ndarray]:
    """! Sample points on the quad defined by two faces.

    @param mesh The mesh.
    @param face_ids The two face IDs defining the quad.
    @param dist The approximate distance between sampled points.
    @return The sampled points and the quad vertices.
    """
    assert len(face_ids) == 2  # We only support quads formed by two triangles
    f0, f1 = face_ids

    vi0 = set([v.get() for v in mesh.topology.getTriVerts(f0)])
    vi1 = set([v.get() for v in mesh.topology.getTriVerts(f1)])
    shared_verts = vi0 & vi1
    assert len(shared_verts) == 2, "Faces should share exactly one edge (two vertices)"
    si0, si1 = shared_verts
    vi = list(vi0 - shared_verts) + [si0] + list(vi1 - shared_verts) + [si1]

    points = [mesh.points[mrmeshpy.VertId(id)] for id in vi]
    quad = np.array([[p.x, p.y, p.z] for p in points])

    width = np.linalg.norm(quad[1] - quad[0])
    height = np.linalg.norm(quad[3] - quad[0])
    num_u = max(1, int(np.ceil(width / dist)))
    num_v = max(1, int(np.ceil(height / dist)))
    u = (np.arange(num_u) + 0.5) / num_u
    v = (np.arange(num_v) + 0.5) / num_v
    uu, vv = np.meshgrid(u, v)
    uu = uu.flatten()[:, None]
    vv = vv.flatten()[:, None]

    # Bilinear interpolation on quad
    return (quad[0] * (1 - uu) * (1 - vv) + quad[1] * uu * (1 - vv) + quad[2] * uu * vv + quad[3] * (1 - uu) * vv), quad


def wall_recall(
    layout_gt: Union[Cuboid, mrmeshpy.Mesh],
    layout_pred: Union[Cuboid, mrmeshpy.Mesh],
    angle_thr: float = np.deg2rad(1),
    up: np.ndarray = np.array([0, 0, 1]),
    dist_thr: float = 0.25,
    ratio_thr: float = 0.9,
) -> List[bool]:
    """! Compute the wall recall for a predicted layout against the ground truth layout.

    @param layout_gt The ground truth layout.
    @param layout_pred The predicted layout.
    @param angle_thr The angle threshold (in radians) for clustering faces.
    @param up The up vector.
    @param dist_thr The threshold for the distance between ground truth wall and predicted layout.
    @param ratio_thr The threshold for the ratio of points on the wall that must be within dist_thr.
    @return A list of booleans indicating success for each wall.
    """
    mesh_gt, mesh_pred = layout_to_mesh(layout_gt), layout_to_mesh(layout_pred)
    normals = mrmeshpy.computePerFaceNormals(mesh_gt)
    clusters = cluster_faces(mesh_gt.topology, normals, angle_thr)

    debug = False
    if debug:
        import matplotlib.colors as mcolors
        import rerun as rr

        rr.init("wall_recall_clusters", spawn=True)
        colors = [mcolors.to_rgb(v) for v in mcolors.TABLEAU_COLORS.values()]
        for idx, c in enumerate(clusters):
            verts = []
            for f in c:
                verts.extend([[vi.x, vi.y, vi.z] for vi in mesh_gt.getTriPoints(f)])
            cols = np.tile(colors[idx % len(colors)], (len(verts), 1))
            rr.log(f"cluster_{idx}", rr.Mesh3D(vertex_positions=verts, vertex_colors=cols))

        breakpoint()

    clusters = [c for c in clusters if abs(dot(normals[c[0]], up)) < 0.5]  # Remove floor/ceiling

    success = []
    for c in clusters:
        points, quad = sample_points(mesh_gt, c)
        if mesh_pred.topology.numValidFaces() > 0:
            dist = np.array([mesh_pred.signedDistance(mrmeshpy.Vector3f(p[0], p[1], p[2])) for p in points])
        else:
            dist = np.array([np.inf for _ in points])
        success.append(np.sum(np.abs(dist) < dist_thr) / len(dist) > ratio_thr)

        debug = False
        if debug:
            import rerun as rr

            from .visualization import visualize_mesh

            rr.init("wall_recall", spawn=True)
            rr.log("quad", rr.LineStrips3D(quad[[0, 1, 2, 3, 0]]))
            colors = np.array([[255, 0, 0] if abs(d) > dist_thr else [0, 255, 0] for d in dist])
            rr.log("points", rr.Points3D(points, colors=colors, radii=0.05))
            visualize_mesh("mesh_pred", mesh_pred, albedo_factor=[255, 255, 255, 128])
            breakpoint()

    return success

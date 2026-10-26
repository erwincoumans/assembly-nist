"""Convex hulls with fewer vertices, within a distance of the original (for collision shapes whose narrowphase cost
grows with their vertex and edge counts, such as Metamorphysics' SAT)."""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull


def simplify_hull(points, tol, max_vertices=None):
    """Vertices of a convex hull inside hull(points) whose surface is within tol of every point: greedily the point
    farthest outside the current hull is added, from the extreme points along the axes and diagonals."""
    v = np.asarray(points, np.float64)
    v = v[ConvexHull(v).vertices]
    if len(v) <= 8:
        return v
    dirs = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)], float)
    keep = list(np.unique(np.argmax(v @ dirs.T, axis=0)))
    while True:
        try:
            h = ConvexHull(v[keep])
        except Exception:      # (coplanar start)
            keep = list(range(len(v)))
            break
        out = (v @ h.equations[:, :3].T + h.equations[:, 3]).max(1)
        k = int(np.argmax(out))
        if out[k] <= tol or (max_vertices and len(keep) >= max_vertices):
            break
        keep.append(k)
    s = v[keep]
    return s[ConvexHull(s).vertices]

"""Convex collision pieces for the parts and fixtures whose CPU models collide signed-distance fields (nut and bolt
threads, connectors), for simulators that collide convex shapes only (Metamorphysics), or to compare with them.

pieces(task, geometry) -> (part pieces, fixture pieces), each a list of (vertices (n, 3), position, quaternion wxyz)
in the part's / fixture's frame. geometry:
  mjfit   threads swept from ISO profiles by mjfit (github.com/younghyopark/mjfit): the bolt a threaded rod from the
          head's top to its chamfered tip, the nut a hex with its thread; their helices in the phase of the plugin's
          thread SDFs (threads.py), so the assembled pose is threaded as it is on the CPU
  none    no pieces (the SDF geoms are left out)
"""
from __future__ import annotations

import functools
import math

import numpy as np

import os

SEGMENTS = int(os.environ.get("ASSEMBLENIST_MJFIT_SEGMENTS", 24))          # wedges per turn
# between facing thread surfaces (the SDF threads' 0.2 mm radial play is 0.1 mm on the flanks)
CLEARANCE = float(os.environ.get("ASSEMBLENIST_MJFIT_CLEARANCE", 1e-4))
SEAM_RELIEF = float(os.environ.get("ASSEMBLENIST_MJFIT_SEAM_RELIEF", 0.0))


def pieces(task, geometry="mjfit"):
    from ..tasks import TASKS
    if geometry in ("none", "mjwarp", "sdfdcd"):
        return [], []
    fam = TASKS[task].family
    if geometry == "mjfit" and fam == "nut":
        return thread_pieces(task)
    raise NotImplementedError(f"convex pieces of {task} ({geometry})")


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def _rot_z(angle):
    return np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])


def _mat(q):
    import mujoco
    m = np.empty(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def _body_pieces(m, body):
    """(vertices, pos, quat wxyz) of a compiled body's collision geoms in its frame (cylinders as 32-gon prisms)."""
    import mujoco
    out = []
    b = m.body(body).id
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != b or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        t = m.geom_type[g]
        if t == mujoco.mjtGeom.mjGEOM_MESH:
            k = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]].astype(np.float64)
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            r, h = m.geom_size[g][:2]
            a = 2 * np.pi * np.arange(32) / 32
            ring = np.c_[r * np.cos(a), r * np.sin(a)]
            v = np.vstack([np.c_[ring, np.full(32, -h)], np.c_[ring, np.full(32, h)]])
        else:
            raise NotImplementedError(f"mjfit geom type {t}")
        out.append((v, m.geom_pos[g].copy(), m.geom_quat[g].copy()))
    return out


def _transform(pieces, pos, quat):
    """Pieces in frame F, F at (pos, quat) in G -> pieces in G."""
    R = _mat(quat)
    return [(v, pos + R @ p, _quat_mul(quat, q)) for v, p, q in pieces]


def _inside(pieces, pts):
    """Whether each point lies inside any convex piece."""
    from scipy.spatial import ConvexHull
    inside = np.zeros(len(pts), bool)
    for v, p, q in pieces:
        w = v @ _mat(q).T + p
        lo, hi = w.min(0), w.max(0)
        sel = np.nonzero(~inside & (pts >= lo).all(1) & (pts <= hi).all(1))[0]
        if not len(sel):
            continue
        eq = ConvexHull(w).equations
        inside[sel] |= (pts[sel] @ eq[:, :3].T + eq[:, 3] <= 1e-9).all(1)
    return inside


def _penetration(movers, fixed, shrink=0.97):
    """How many points of the movers' pieces (their vertices, pulled toward each piece's centre by `shrink`, and
    the centres) lie inside a fixed piece: zero when they do not overlap."""
    pts = []
    for v, p, q in movers:
        w = v @ _mat(q).T + p
        c = w.mean(0)
        pts.append(np.vstack([c + shrink * (w - c), c[None]]))
    return int(_inside(fixed, np.vstack(pts)).sum())


def _cache_path(task):
    import os
    from pathlib import Path
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "assemblenist"
    return root / f"mjfit_{task}_s{SEGMENTS}_c{CLEARANCE:g}_r{SEAM_RELIEF:g}.npz"


@functools.lru_cache(maxsize=None)
def thread_pieces(task):
    """mjfit bolt (fixture frame) and nut (part frame) of a nut task: the bolt in the phase of the plugin's bolt
    SDF, the nut turned so that it meshes with the bolt at the assembled pose (then all along the helix, as the
    banks' starts are). Cached in ~/.cache/assemblenist."""
    cache = _cache_path(task)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return list(z["nut"]), list(z["bolt"])
    import mujoco
    import mjfit
    from ..robot import quat_to_mat
    from ..tasks import TASKS, scene_pose
    from ..threads import THREADS, thread_attributes
    P, crest, root, htop, top, minor, z0, z1, flat = THREADS[task]
    bolt_a, nut_a = thread_attributes(task)
    depth = (5 * math.sqrt(3) / 16) * P
    xml = mjfit.thread(major_diameter=2 * crest, pitch=P, thread_length=top - htop, clearance=CLEARANCE,
                       segments_per_turn=SEGMENTS, seam_relief=SEAM_RELIEF, chamfer=depth, bolt_head_height=0.0, nut_size=2 * flat,
                       nut_height=z1 - z0, driven="nut", fixed=False, standalone=True, floor=False)
    m = mujoco.MjModel.from_xml_string(xml)
    # the bolt's frame is at its tip, its thread along +z: upright in the fixture frame, tip at `top`
    bolt = _transform(_body_pieces(m, "bolt"), np.array([0.0, 0.0, top]), np.array([0.0, 1.0, 0.0, 0.0]))
    nut = _transform(_body_pieces(m, "nut"), np.array([0.0, 0.0, z0]), np.array([1.0, 0.0, 0.0, 0.0]))
    rng = np.random.default_rng(0)

    def turned(parts, psi):
        return _transform(parts, np.zeros(3), _rot_z(psi))

    # the bolt in the SDF bolt's phase: most agreement of material on a cylinder through its teeth
    n = 1500
    th, zz = rng.uniform(0, 2 * np.pi, n), rng.uniform(htop + 0.3 * (top - htop), htop + 0.7 * (top - htop), n)
    r = rng.uniform(bolt_a["root"], bolt_a["crest"], n)
    pts = np.c_[r * np.cos(th), r * np.sin(th), zz]
    want = sdf_numpy("bolt", pts, list(bolt_a.values())) < 0
    agree = lambda psi: (_inside(bolt, pts @ _mat(_rot_z(-psi)).T) == want).mean()
    grid = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    psi_b = max(grid, key=agree)
    psi_b = max(np.linspace(psi_b - np.pi / 48, psi_b + np.pi / 48, 13), key=agree)
    bolt = turned(bolt, psi_b)
    # the nut at the assembled pose, in the fixture frame: turned to where it overlaps the bolt least
    fpos, fquat = scene_pose(TASKS[task].fixture_scene)
    gpos, gquat = scene_pose(TASKS[task].part_scene)
    Rf = quat_to_mat(fquat)
    G = (Rf.T @ (gpos - fpos), Rf.T @ quat_to_mat(gquat))           # the part frame in the fixture frame
    near = [pc for pc in bolt if (pc[1][2] > z0 - 2 * P + G[0][2]) and (pc[1][2] < z1 + 2 * P + G[0][2])]

    def overlap(psi):
        placed = [(v, G[0] + G[1] @ p, _quat_mul(_mat_quat(G[1]), q)) for v, p, q in turned(nut, psi)]
        return _penetration(placed, near)
    grid = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    scores = [overlap(psi) for psi in grid]
    psi_n = grid[int(np.argmin(scores))]
    fine = np.linspace(psi_n - np.pi / 72, psi_n + np.pi / 72, 9)
    psi_n = fine[int(np.argmin([overlap(psi) for psi in fine]))]
    thread_pieces.overlap = getattr(thread_pieces, "overlap", {})
    thread_pieces.overlap[task] = (int(min(scores)), int(overlap(psi_n)), int(max(scores)))
    nut = turned(nut, psi_n)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, nut=np.array(nut, dtype=object), bolt=np.array(bolt, dtype=object))
    return nut, bolt


def _mat_quat(R):
    import mujoco
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).reshape(-1))
    return q


def sdf_numpy(kind, p, a):
    """The plugin's bolt / nut thread SDF (distance only), vectorized: for checks and alignment."""
    p = np.asarray(p, np.float64)
    tan60 = math.sqrt(3.0)
    r = np.hypot(p[:, 0], p[:, 1])
    th = np.arctan2(p[:, 1], p[:, 0])

    def profile(P, top, lo, hi, s):
        return np.clip(top - tan60 * np.abs(s - 0.5 * P), lo, hi)

    if kind == "bolt":
        P, rp, root, crest, tz0, tz1, _, _, ph = a[:9]
        s = P * np.mod(p[:, 2] / P - th / (2 * np.pi) + ph, 1.0)
        rs = profile(P, rp + 0.25 * P * tan60, root, crest, s)
        d = r - rs                                         # (radial, not euclidean: enough for the sign)
        cham = (p[:, 2] - tz1 + (r - root)) * math.sqrt(0.5)
        return np.maximum.reduce([d, tz0 - p[:, 2], p[:, 2] - tz1, cham])
    P, rp, c, minor, major, flat, z0, z1, ph = a[:9]
    s = P * np.mod(p[:, 2] / P - th / (2 * np.pi) + ph, 1.0)
    rh = profile(P, rp + c + 0.25 * P * tan60, minor, major, s)
    hole = rh - r
    # hex prism, flats at `flat` (one flat normal along +x)
    ang = np.mod(th + np.pi / 6, np.pi / 3) - np.pi / 6
    body = np.maximum(r * np.cos(ang) - flat, np.maximum(z0 - p[:, 2], p[:, 2] - z1))
    return np.maximum(hole, body)

"""Point-cloud observations synthesized on the learner from compact poses.

Fig. 3 encodes a fixed-order, flattened point cloud (3 x n_pts) of the movable part, its
mating geometry and the robot (gripper). Points are sampled once per asset (seeded,
area-weighted on the source collision meshes) and expressed in the TCP frame each step.
"""
from __future__ import annotations

import mujoco
import numpy as np
import torch

from ..sim.robot import quat_to_mat
from ..sim.scene import TCP_OFFSET, build_xml
from ..sim.tasks import TASKS, load_obj, scene_pose

POINT_SCALE = 10.0  # metres -> decimetres before the encoder (local choice)


def sample_surface(verts, faces, n, rng):
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    idx = rng.choice(len(faces), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[idx]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


def _box_surface(size, n, rng):
    hx, hy, hz = size
    areas = np.array([hy * hz, hy * hz, hx * hz, hx * hz, hx * hy, hx * hy])
    face = rng.choice(6, n, p=areas / areas.sum())
    p = rng.uniform(-1, 1, (n, 3)) * np.array([hx, hy, hz])
    axis, sign = face // 2, np.where(face % 2 == 0, 1.0, -1.0)
    p[np.arange(n), axis] = sign * np.array([hx, hy, hz])[axis]
    return p


def finger_points(n_per_finger, seed=0):
    """Points on the left finger collision geoms, in the left-finger body frame."""
    rng = np.random.default_rng(seed)
    m = mujoco.MjModel.from_xml_string(build_xml("R16"))
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    pts = []
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body and m.geom_contype[g]]
    for g in geoms:
        R = quat_to_mat(m.geom_quat[g]); p0 = m.geom_pos[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            v = m.mesh_vert[va:va + vn]; f = m.mesh_face[fa:fa + fn]
            local = sample_surface(v, f, 4 * n_per_finger, rng)
        else:
            local = _box_surface(m.geom_size[g], n_per_finger, rng)
        pts.append(local @ R.T + p0)
    pts = np.concatenate(pts)
    return pts[rng.choice(len(pts), n_per_finger, replace=False)], m.body_pos[body].copy()


def mating_points(t, n, rng):
    """World-frame surface samples of the mating geometry (fixture; for gears also the two meshed
    neighbour gears, which are part of the static assembly the gear must mesh with)."""
    meshes = []
    v, f = load_obj(t.fixture_asset)
    fpos, fquat = scene_pose(t.fixture_scene)
    meshes.append((v @ quat_to_mat(fquat).T + fpos, f))
    if t.family == "gear":
        from ..sim.gears import GEARS
        for other in GEARS:
            if other != t.part_asset:
                v, f = load_obj(other)
                gpos, gquat = scene_pose(other)
                meshes.append((v @ quat_to_mat(gquat).T + gpos, f))
    V = np.concatenate([m[0] for m in meshes])
    F = np.concatenate([m[1] + sum(len(x[0]) for x in meshes[:i]) for i, m in enumerate(meshes)])
    return sample_surface(V, F, n, rng)


class PointTables:
    def __init__(self, task_names, n_part=128, n_fix=128, n_grip=64, seed=0, device="cpu"):
        self.n_part, self.n_fix, self.n_grip = n_part, n_fix, n_grip
        part, fix = [], []
        for k, name in enumerate(task_names):
            t = TASKS[name]
            rng = np.random.default_rng(seed + 1000 * k)
            v, f = load_obj(t.part_asset)
            part.append(sample_surface(v, f, n_part, rng))
            fix.append(mating_points(t, n_fix, rng))
        fp, fbody = finger_points(n_grip // 2, seed)
        # in TCP frame: hand frame = TCP shifted by -TCP_OFFSET along z
        base = fp + fbody - np.array([0, 0, TCP_OFFSET])
        right = base * np.array([-1.0, -1.0, 1.0])  # right finger body is rotated 180 deg about z
        dev = torch.device(device)
        self.part = torch.tensor(np.stack(part), dtype=torch.float32, device=dev)
        self.fix = torch.tensor(np.stack(fix), dtype=torch.float32, device=dev)
        self.left = torch.tensor(base, dtype=torch.float32, device=dev)
        self.right = torch.tensor(right, dtype=torch.float32, device=dev)
        self.n_pts = n_part + n_fix + n_grip

    def to(self, device):
        for k in ("part", "fix", "left", "right"):
            setattr(self, k, getattr(self, k).to(device))
        return self

    def __call__(self, task, tcp_p, tcp_R, part_p, part_q, finger):
        """All inputs are torch tensors (B, ...). Returns (B, 3 * n_pts) flattened, scaled."""
        Rt = tcp_R.transpose(1, 2)                                            # world -> tcp
        Rp = quat_to_mat_torch(part_q)
        pw = torch.einsum("bij,bkj->bki", Rp, self.part[task]) + part_p[:, None]
        p_part = torch.einsum("bij,bkj->bki", Rt, pw - tcp_p[:, None])
        p_fix = torch.einsum("bij,bkj->bki", Rt, self.fix[task] - tcp_p[:, None])
        B = task.shape[0]
        y = torch.zeros(B, 1, 3, device=task.device)
        y[:, 0, 1] = 1.0
        p_l = self.left[None] + y * finger[:, 0, None, None]
        p_r = self.right[None] - y * finger[:, 1, None, None]
        pts = torch.cat([p_part, p_fix, p_l, p_r], 1)
        return (pts * POINT_SCALE).reshape(B, -1)


def quat_to_mat_torch(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))

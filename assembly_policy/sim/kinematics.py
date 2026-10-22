"""Vectorized Franka forward kinematics (numpy), read from the compiled MuJoCo model.

Computes TCP pose and twist from post-step qpos/qvel exactly (no lagging sensor data).
"""
from __future__ import annotations

import mujoco
import numpy as np

from .robot import quat_to_mat


class FrankaFK:
    def __init__(self, m: mujoco.MjModel):
        names = [f"link{i}" for i in range(1, 8)] + ["hand"]
        self.body = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
        self.pos = np.array([m.body_pos[b] for b in self.body])          # in parent frame
        self.rot = quat_to_mat(np.array([m.body_quat[b] for b in self.body]))
        base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link0")
        self.base_pos = m.body_pos[base].copy()
        self.base_rot = quat_to_mat(m.body_quat[base])
        tcp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.tcp_pos = m.site_pos[tcp].copy()
        for b in self.body[:7]:
            j = m.body_jntadr[b]
            assert np.allclose(m.jnt_axis[j], [0, 0, 1]) and np.allclose(m.jnt_pos[j], 0)
        lf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.finger_pos = m.body_pos[lf].copy()  # in hand frame

    def forward(self, q):
        """q: (N,7) -> dict with joint origins/axes (N,7,3), hand R/p, tcp pos/R."""
        N = q.shape[0]
        R = np.broadcast_to(self.base_rot, (N, 3, 3)).copy()
        p = np.broadcast_to(self.base_pos, (N, 3)).copy()
        origins, axes = np.empty((N, 7, 3)), np.empty((N, 7, 3))
        for i in range(8):
            p = p + np.einsum("nij,j->ni", R, self.pos[i])
            R = R @ self.rot[i]
            if i < 7:
                c, s = np.cos(q[:, i]), np.sin(q[:, i])
                Rz = np.zeros((N, 3, 3)); Rz[:, 0, 0] = c; Rz[:, 0, 1] = -s; Rz[:, 1, 0] = s; Rz[:, 1, 1] = c
                Rz[:, 2, 2] = 1
                origins[:, i] = p; axes[:, i] = R[:, :, 2]
                R = R @ Rz
        tcp = p + np.einsum("nij,j->ni", R, self.tcp_pos)
        return dict(origins=origins, axes=axes, hand_R=R, hand_p=p, tcp_p=tcp, tcp_R=R)

    def tcp_twist(self, fk, qd):
        """Linear and angular velocity of the TCP in the world/base frame. qd: (N,7)."""
        w = np.einsum("nij,ni->nj", fk["axes"], qd)
        lin = np.einsum("ni,nij->nj", qd, np.cross(fk["axes"], fk["tcp_p"][:, None, :] - fk["origins"]))
        return lin, w

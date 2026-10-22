"""Franka helpers: indices, quaternion math, damped-least-squares IK on the TCP site."""
from __future__ import annotations

import mujoco
import numpy as np

ARM_DOF = 7
HOME_Q = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.785])
FINGER_OPEN = 0.04


class Indices:
    def __init__(self, m: mujoco.MjModel):
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.arm_qpos = np.array([m.jnt_qposadr[jid(f"joint{i+1}")] for i in range(7)])
        self.arm_dof = np.array([m.jnt_dofadr[jid(f"joint{i+1}")] for i in range(7)])
        self.finger_qpos = np.array([m.jnt_qposadr[jid(f"finger_joint{i+1}")] for i in range(2)])
        self.finger_dof = np.array([m.jnt_dofadr[jid(f"finger_joint{i+1}")] for i in range(2)])
        self.part_qpos = m.jnt_qposadr[jid("part_free")]
        self.part_dof = m.jnt_dofadr[jid("part_free")]
        self.tcp_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.part_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "part_frame")
        self.fixture_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "fixture_frame")
        self.part_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "part")
        self.hand_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.finger_bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("left_finger", "right_finger")]
        self.jnt_range = np.array([m.jnt_range[jid(f"joint{i+1}")] for i in range(7)])


# ------------------------------------------------------------------ quaternions (w, x, y, z)
def quat_mul(a, b):
    aw, ax, ay, az = np.moveaxis(np.asarray(a), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b), -1, 0)
    return np.stack([aw*bw - ax*bx - ay*by - az*bz, aw*bx + ax*bw + ay*bz - az*by,
                     aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw], -1)


def quat_conj(q):
    q = np.asarray(q)
    return q * np.array([1, -1, -1, -1])


def quat_to_mat(q):
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], -1),
                     np.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], -1),
                     np.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1)], -2)


def mat_to_quat(R):
    q = np.empty(R.shape[:-2] + (4,))
    for idx in np.ndindex(R.shape[:-2]):
        mujoco.mju_mat2Quat(q[idx], np.ascontiguousarray(R[idx]).ravel())
    return q


def quat_apply(q, v):
    return np.einsum("...ij,...j->...i", quat_to_mat(q), v)


def random_quat(rng, n=None):
    u = rng.random((3,) if n is None else (n, 3))
    u1, u2, u3 = np.moveaxis(u, -1, 0)
    q = np.stack([np.sqrt(1-u1)*np.sin(2*np.pi*u2), np.sqrt(1-u1)*np.cos(2*np.pi*u2),
                  np.sqrt(u1)*np.sin(2*np.pi*u3), np.sqrt(u1)*np.cos(2*np.pi*u3)], -1)
    return q


def axis_angle_quat(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    return np.concatenate([np.cos(angle/2)[..., None] if np.ndim(angle) else [np.cos(angle/2)],
                           np.sin(np.asarray(angle)/2)[..., None] * axis], -1)


# ------------------------------------------------------------------ IK
def solve_ik(m, d, idx: Indices, target_pos, target_quat, q_init=None, iters=200, tol_pos=5e-4,
             tol_rot=5e-3, damping=1e-3, null_gain=0.05, rng=None, restarts=4):
    """DLS IK for the TCP site. Returns (q_arm, success). Leaves d.qpos arm entries at the result."""
    target_R = quat_to_mat(target_quat)
    jac_p = np.zeros((3, m.nv)); jac_r = np.zeros((3, m.nv))
    lo, hi = idx.jnt_range[:, 0], idx.jnt_range[:, 1]
    best = (None, np.inf)
    for attempt in range(restarts):
        if attempt == 0:
            q = np.array(HOME_Q if q_init is None else q_init, dtype=float)
        else:
            rng = rng or np.random.default_rng(0)
            q = np.clip(HOME_Q + rng.normal(0, 0.5, 7), lo + 0.05, hi - 0.05)
        for _ in range(iters):
            d.qpos[idx.arm_qpos] = q
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            pos = d.site_xpos[idx.tcp_site]
            R = d.site_xmat[idx.tcp_site].reshape(3, 3)
            e_p = target_pos - pos
            Re = target_R @ R.T
            e_r = np.zeros(3); mujoco.mju_quat2Vel(e_r, mat_to_quat(Re[None])[0], 1.0)
            err = np.linalg.norm(e_p) + 0.05 * np.linalg.norm(e_r)
            if np.linalg.norm(e_p) < tol_pos and np.linalg.norm(e_r) < tol_rot:
                return q, True
            mujoco.mj_jacSite(m, d, jac_p, jac_r, idx.tcp_site)
            J = np.vstack([jac_p[:, idx.arm_dof], jac_r[:, idx.arm_dof]])
            e = np.r_[e_p, e_r]
            JJt = J @ J.T + damping * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            # null-space pull toward home posture
            N = np.eye(7) - J.T @ np.linalg.solve(JJt, J)
            dq += N @ (null_gain * (HOME_Q - q))
            q = np.clip(q + np.clip(dq, -0.2, 0.2), lo + 1e-3, hi - 1e-3)
        if err < best[1]:
            best = (q.copy(), err)
    d.qpos[idx.arm_qpos] = best[0]
    return best[0], False

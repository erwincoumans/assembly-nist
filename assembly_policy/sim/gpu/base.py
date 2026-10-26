"""The MDP of sim/env.py on the GPU (torch), shared by the GPU simulation backends (MuJoCo Warp, Metamorphysics).

A backend simulates one control step for all environments and exposes the MuJoCo state layout of the task models
(qpos: arm 7, fingers 2, part 7 (pos, quat wxyz); qvel: arm 7, fingers 2, part 6 (world linear velocity of the part
frame, angular velocity in the part frame)), so that reset-bank states load unchanged. Everything else - actions to
PD targets, the gripper latch, proprioception history, rewards and terminations - is the CPU environment's, in torch.

The task of each environment is fixed when the backend is built (its world belongs to that task's model);
set_states may load any bank state of the same task.
"""
from __future__ import annotations

import numpy as np
import torch

from ..env import HISTORY, PROPRIO_DIM, EnvConfig
from ..kinematics import FrankaFK
from ..robot import Indices, quat_to_mat
from ..scene import ARM_KD, ARM_KP, TORQUE_LIMIT, VELOCITY_LIMIT
from ..tasks import TASKS, scene_pose

NQ, NV = 16, 15
ARM_Q, FINGER_Q, PART_Q = slice(0, 7), slice(7, 9), slice(9, 16)
ARM_V, FINGER_V, PART_V = slice(0, 7), slice(7, 9), slice(9, 15)


class TorchFK:
    """sim/kinematics.FrankaFK in torch: TCP pose and twist from the arm joints."""

    def __init__(self, fk, device):
        f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
        self.pos, self.rot = f(fk.pos), f(fk.rot)
        self.base_pos, self.base_rot, self.tcp_pos = f(fk.base_pos), f(fk.base_rot), f(fk.tcp_pos)

    def forward(self, q):
        N = q.shape[0]
        R = self.base_rot.expand(N, 3, 3)
        p = self.base_pos.expand(N, 3)
        origins, axes = [], []
        c, s = torch.cos(q), torch.sin(q)
        for i in range(8):
            p = p + R @ self.pos[i]
            R = R @ self.rot[i]
            if i < 7:
                origins.append(p); axes.append(R[:, :, 2])
                x, y = R[:, :, 0], R[:, :, 1]
                ci, si = c[:, i, None], s[:, i, None]
                R = torch.stack([ci * x + si * y, ci * y - si * x, R[:, :, 2]], 2)     # R @ Rz(q_i)
        tcp = p + R @ self.tcp_pos
        return dict(origins=torch.stack(origins, 1), axes=torch.stack(axes, 1), tcp_p=tcp, tcp_R=R)

    @staticmethod
    def tcp_twist(fk, qd):
        w = (fk["axes"] * qd[:, :, None]).sum(1)
        lin = (qd[:, :, None] * torch.cross(fk["axes"], fk["tcp_p"][:, None, :] - fk["origins"], dim=2)).sum(1)
        return lin, w


def mat_to_quat(R):
    """sim/env._mat_to_quat_batch in torch (Shepperd, w >= 0)."""
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    c = torch.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], 1)
    k = torch.argmax(c, 1)
    s = torch.sqrt(torch.clamp(c.gather(1, k[:, None])[:, 0], min=1e-30)) * 2
    a21, a12, a02, a20, a10, a01 = R[:, 2, 1], R[:, 1, 2], R[:, 0, 2], R[:, 2, 0], R[:, 1, 0], R[:, 0, 1]
    cand = torch.stack([torch.stack([s / 4, (a21 - a12) / s, (a02 - a20) / s, (a10 - a01) / s], 1),
                        torch.stack([(a21 - a12) / s, s / 4, (a01 + a10) / s, (a02 + a20) / s], 1),
                        torch.stack([(a02 - a20) / s, (a01 + a10) / s, s / 4, (a12 + a21) / s], 1),
                        torch.stack([(a10 - a01) / s, (a02 + a20) / s, (a12 + a21) / s, s / 4], 1)], 1)
    q = cand[torch.arange(len(R), device=R.device), k]
    return q * torch.where(q[:, :1] < 0, -1.0, 1.0)


def quat_to_mat_t(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


class GpuAssemblyEnv:
    """BatchedAssemblyEnv's interface (set_states, observe, step, horizon, t, task, N) over a GPU backend.

    backend: built for (task_names, task_of_env), with
      substeps (per environment), qpos (N, 16) and qvel (N, 15) torch views of the current state,
      write(env_ids, qpos, qvel) and simulate(ctrl (N, 8)) -> (velocity violation (N,) bool, mean torque (N, 7)).
    model: a compiled MuJoCo task model (all share the robot), for the kinematics and the joint ranges.
    """

    def __init__(self, backend, task_names, task_of_env, ecfg: EnvConfig, model, device):
        self.backend, self.ecfg, self.dev = backend, ecfg, torch.device(device)
        self.task_names = list(task_names)
        self.task = np.asarray(task_of_env, dtype=np.int64).copy()
        self.N = len(self.task)
        self.horizon = int(round(ecfg.horizon_s / ecfg.control_dt))
        self.t = np.zeros(self.N, dtype=np.int64)
        f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.dev)
        self.task_t = torch.as_tensor(self.task, device=self.dev)
        self.fk = TorchFK(FrankaFK(model), self.dev)
        jnt_range = Indices(model).jnt_range
        self.vlim = f(VELOCITY_LIMIT) * ecfg.velocity_limit_scale
        self.tlim = f(TORQUE_LIMIT)
        self.jlo, self.jhi = f(jnt_range[:, 0]), f(jnt_range[:, 1])
        self.scale = f(ecfg.action_scale_joints) if len(ecfg.action_scale_joints) else ecfg.action_scale
        kp, gk = [], []
        for name in self.task_names:
            t = TASKS[name]
            kp.append(np.asarray(t.keypoints))
            ppos, pquat = scene_pose(t.part_scene)
            gk.append(ppos + np.asarray(t.keypoints) @ quat_to_mat(pquat).T)
        self.keypoints, self.goal_kp = f(np.stack(kp)), f(np.stack(gk))
        self.prev_action = torch.zeros(self.N, 8, device=self.dev)
        self.grip_cmd = torch.zeros(self.N, device=self.dev)
        self.hist = torch.zeros(self.N, HISTORY, PROPRIO_DIM, device=self.dev)

    def close(self):
        self.backend.close()

    # ------------------------------------------------------------- state helpers
    def set_states(self, env_ids, states, task_ids=None):
        env_ids = np.asarray(env_ids)
        if task_ids is not None and (np.asarray(task_ids) != self.task[env_ids]).any():
            raise ValueError("GPU backends keep the task of each environment fixed")
        if len(env_ids) == 0:
            return
        s = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.dev)
        ids = torch.as_tensor(env_ids, device=self.dev)
        self.backend.write(ids, s[:, 1:1 + NQ], s[:, 1 + NQ:1 + NQ + NV])
        self.t[env_ids] = 0
        self.prev_action[ids] = 0.0
        finger = s[:, 1 + 7:1 + 9].mean(1)
        self.grip_cmd[ids] = torch.where(finger > 0.036, 0.04, self.ecfg.grip_close)
        self.hist[ids] = self._proprio(ids)[:, None, :]

    def _proprio(self, ids=None):
        qpos, qvel = self.backend.qpos, self.backend.qvel
        if ids is not None:
            qpos, qvel = qpos[ids], qvel[ids]
        fk = self.fk.forward(qpos[:, ARM_Q])
        lin, ang = self.fk.tcp_twist(fk, qvel[:, ARM_V])
        quat = mat_to_quat(fk["tcp_R"])
        quat = quat * torch.where(quat[:, 1:2] < 0, -1.0, 1.0)
        prev = self.prev_action if ids is None else self.prev_action[ids]
        return torch.cat([fk["tcp_p"], quat, lin, ang, qpos[:, ARM_Q], qpos[:, FINGER_Q], prev], 1)

    def observe_torch(self):
        qpos = self.backend.qpos
        fk = self.fk.forward(qpos[:, ARM_Q])
        return dict(proprio=self.hist.reshape(self.N, -1).clone(), task=self.task_t.clone(),
                    tcp_p=fk["tcp_p"], tcp_R=fk["tcp_R"], part_p=qpos[:, 9:12].clone(), part_q=qpos[:, 12:16].clone(),
                    finger=qpos[:, FINGER_Q].clone())

    def observe(self):
        return {k: v.cpu().numpy() for k, v in self.observe_torch().items()}

    # ------------------------------------------------------------- dynamics
    def step(self, action):
        a = torch.as_tensor(np.asarray(action) if not torch.is_tensor(action) else action, dtype=torch.float32,
                            device=self.dev)
        q = self.backend.qpos[:, ARM_Q]
        ctrl = torch.empty(self.N, 8, device=self.dev)
        ctrl[:, :7] = torch.minimum(torch.maximum(q + self.scale * a[:, :7], self.jlo), self.jhi)
        g = self.ecfg.grip_deadband
        self.grip_cmd = torch.where(a[:, 7] > g, 0.04, torch.where(a[:, 7] < -g, self.ecfg.grip_close, self.grip_cmd))
        ctrl[:, 7] = self.grip_cmd
        viol, tau = self.backend.simulate(ctrl)
        self.t += 1
        qpos, qvel = self.backend.qpos, self.backend.qvel
        nonfinite = ~(torch.isfinite(qpos).all(1) & torch.isfinite(qvel).all(1)) | (qpos.abs() > 1e6).any(1)
        qd = torch.nan_to_num(qvel[:, ARM_V])
        cost = self.ecfg.cost_coef * ((a ** 2).sum(1) + ((qd / self.vlim) ** 2).sum(1)
                                      + ((torch.nan_to_num(tau) / self.tlim) ** 2).sum(1))
        d_asm = torch.nan_to_num(self.assembly_distance(), nan=1e3)
        safety = viol | nonfinite
        success = (d_asm < self.ecfg.success_threshold) & ~safety
        reward = success.float() - cost
        self.prev_action = a.clone()
        prop = torch.nan_to_num(self._proprio())
        self.hist = torch.cat([self.hist[:, 1:], prop[:, None]], 1)
        out = {k: v.cpu().numpy() for k, v in dict(reward=reward, success=success, safety=safety, d_asm=d_asm,
                                                    cost=cost, unstable=nonfinite).items()}
        timeout = (self.t >= self.horizon) & ~out["success"] & ~out["safety"]
        info = dict(success=out["success"], safety=out["safety"], timeout=timeout, d_asm=out["d_asm"].astype(np.float64),
                    cost=out["cost"].astype(np.float64), unstable=out["unstable"])
        return out["reward"].astype(np.float32), out["success"] | out["safety"] | timeout, info

    def assembly_distance(self):
        qpos = self.backend.qpos
        R = quat_to_mat_t(qpos[:, 12:16] / qpos[:, 12:16].norm(dim=1, keepdim=True))
        kp = qpos[:, None, 9:12] + torch.einsum("nij,nkj->nki", R, self.keypoints[self.task_t])
        return (kp - self.goal_kp[self.task_t]).norm(dim=2).amax(1)

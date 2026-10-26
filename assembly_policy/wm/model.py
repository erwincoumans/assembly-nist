"""A deterministic world model of the assembly tasks (after SuperMimic, arXiv:2610.28480): an MLP predicts the change
of the state over one 40 ms control step from the state, the action and the task.

The state is the environment's in the task's goal frame (where the part's frame is when assembled), so that one
model serves every task and the part's pose is what assembly is about:
  q, qd (7, 7) arm joints; f, fd (2, 2) fingers; g (1) the gripper's latched command;
  p (3) the part frame's origin, R (3 x 3) its rotation, v (3) its origin's velocity, w (3) its angular velocity.
What is known exactly is computed, not learned: the PD targets and the gripper latch from the action (sim/env.py),
the TCP pose (forward kinematics), the part in the TCP frame. The network predicts deltas (normalized): of q, qd, f,
fd, p, v, w, and of R as a rotation vector in the goal frame (R' = exp(dtheta) R); with prior=True, of the positions
the change beyond moving at the start's velocities over the step.

Features "geo" add what contact depends on, in the arm's motion as much as in the part's (the policy pushes the part
into the table and the fixture with saturated joint torques): world heights, the assembly keypoints' offsets, the
TCP's twist and the part's slip relative to it. "geo2" adds the keypoints' offsets zoomed (tanh at 0.5, 2 and 8 mm):
the clearances are ~50 um, which the offsets normalized by their spread over the workspace (~0.1 m) cannot resolve.
"geo3" adds the orientation error zoomed (sin(angle) axis; tanh at 0.01, 0.05 and 0.2 rad) and the gears' tooth phase
(sin, cos of teeth x the angle about the gear's axis), which decides whether a gear meshes with its neighbours.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..sim.gpu.base import TorchFK, quat_to_mat_t

STATE = dict(q=7, qd=7, f=2, fd=2, g=1, p=3, R=9, v=3, w=3)       # the state's layout (37)
DELTA = dict(q=7, qd=7, f=2, fd=2, p=3, rot=3, v=3, w=3)          # the predicted deltas (30)


def _slices(layout):
    out, k = {}, 0
    for name, n in layout.items():
        out[name] = slice(k, k + n)
        k += n
    return out, k


S, STATE_DIM = _slices(STATE)
D, DELTA_DIM = _slices(DELTA)


def so3_exp(v):
    """Rotation matrices of rotation vectors (..., 3)."""
    theta = v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    k = v / theta
    K = torch.zeros(v.shape[:-1] + (3, 3), device=v.device, dtype=v.dtype)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = k[..., 2], -k[..., 1], k[..., 0]
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    I = torch.eye(3, device=v.device, dtype=v.dtype).expand_as(K)
    return I + s * K + (1 - c) * (K @ K)


def so3_log(R):
    """Rotation vectors (..., 3) of rotation matrices."""
    cos = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos)
    w = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    return w * (theta / (2 * torch.sin(theta)).clamp(min=1e-7))[..., None]


class TaskFrames:
    """Goal frames, keypoints and the robot's kinematics: the conversions between the environment's MuJoCo state
    (qpos 16, qvel 15, gripper command) and the model's."""

    def __init__(self, task_names, model, ecfg, device, n_engines=1, features="base"):
        from ..sim.robot import Indices, quat_to_mat
        from ..sim.kinematics import FrankaFK
        from ..sim.gears import GEARS
        from ..sim.tasks import TASKS, scene_pose
        f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
        gp, gR, kp = [], [], []
        for name in task_names:
            pos, quat = scene_pose(TASKS[name].part_scene)
            gp.append(pos); gR.append(quat_to_mat(quat)); kp.append(TASKS[name].keypoints)
        self.goal_p, self.goal_R, self.keypoints = f(gp), f(gR), f(kp)
        self.teeth = f([GEARS[TASKS[n].part_asset][1] if TASKS[n].family == "gear" else 0 for n in task_names])
        self.fk = TorchFK(FrankaFK(model), device)
        rng = Indices(model).jnt_range
        self.jlo, self.jhi = f(rng[:, 0]), f(rng[:, 1])
        self.scale = f(ecfg.action_scale_joints) if len(ecfg.action_scale_joints) else ecfg.action_scale
        self.grip_close, self.deadband = ecfg.grip_close, ecfg.grip_deadband
        self.ntasks = len(task_names)
        self.n_engines = n_engines          # (several: the engine is an input, the simulator context of SuperMimic)
        self.feature_set = features
        self.dt = ecfg.control_dt
        self.device = device

    def from_mujoco(self, x, task):
        """(..., 32) MuJoCo state -> (..., 37) model state."""
        G, gp = self.goal_R[task], self.goal_p[task]
        Rw = quat_to_mat_t(x[..., 12:16] / x[..., 12:16].norm(dim=-1, keepdim=True))
        out = torch.empty(x.shape[:-1] + (STATE_DIM,), device=x.device)
        out[..., S["q"]], out[..., S["f"]] = x[..., 0:7], x[..., 7:9]
        out[..., S["qd"]], out[..., S["fd"]] = x[..., 16:23], x[..., 23:25]
        out[..., S["g"]] = x[..., 31:32]
        Gt = G.transpose(-1, -2)
        out[..., S["p"]] = (Gt @ (x[..., 9:12] - gp)[..., None])[..., 0]
        out[..., S["R"]] = (Gt @ Rw).reshape(x.shape[:-1] + (9,))
        out[..., S["v"]] = (Gt @ x[..., 25:28, None])[..., 0]
        out[..., S["w"]] = (Gt @ (Rw @ x[..., 28:31, None]))[..., 0]
        return out

    def act(self, s, a):
        """The PD targets' offsets from the joints and the gripper's next command (sim/env.py)."""
        q = s[..., S["q"]]
        target = torch.minimum(torch.maximum(q + self.scale * a[..., :7], self.jlo), self.jhi)
        g = s[..., S["g"]][..., 0]
        g = torch.where(a[..., 7] > self.deadband, torch.full_like(g, 0.04),
                        torch.where(a[..., 7] < -self.deadband, torch.full_like(g, self.grip_close), g))
        return target - q, g

    def features(self, s, a, task, engine=None):
        """The network's input: the state, the TCP and the part relative to it, the processed action, the task (and
        the engine whose dynamics to predict)."""
        dq, g = self.act(s, a)
        G, gp = self.goal_R[task], self.goal_p[task]
        shape = s.shape[:-1]
        fk = self.fk.forward(s[..., S["q"]].reshape(-1, 7))
        tcp_w, tcpR_w = fk["tcp_p"].reshape(shape + (3,)), fk["tcp_R"].reshape(shape + (3, 3))
        Gt = G.transpose(-1, -2)
        tcp = (Gt @ (tcp_w - gp)[..., None])[..., 0]
        tcpR = Gt @ tcpR_w
        R = s[..., S["R"]].reshape(shape + (3, 3))
        p = s[..., S["p"]]
        p_in_tcp = (tcpR.transpose(-1, -2) @ (p - tcp)[..., None])[..., 0]
        R_in_tcp = tcpR.transpose(-1, -2) @ R
        six = lambda M: M[..., :, :2].reshape(shape + (6,))
        onehot = torch.nn.functional.one_hot(task, self.ntasks).float()
        parts = [s[..., S["q"]], s[..., S["qd"]], s[..., S["f"]], s[..., S["fd"]], g[..., None], dq, a,
                 p, six(R), s[..., S["v"]], s[..., S["w"]], tcp, six(tcpR), p_in_tcp, six(R_in_tcp), onehot]
        if self.feature_set in ("geo", "geo2", "geo3"):
            kp = self.keypoints[task]
            moved = p[..., None, :] + (R @ kp.transpose(-1, -2)).transpose(-1, -2)
            lin_w, ang_w = self.fk.tcp_twist(fk, s[..., S["qd"]].reshape(-1, 7))
            lin = (Gt @ lin_w.reshape(shape + (3, 1)))[..., 0]
            ang = (Gt @ ang_w.reshape(shape + (3, 1)))[..., 0]
            carried = lin + torch.cross(ang, p - tcp, dim=-1)      # (the velocity of the part's origin if rigidly held)
            parts += [tcp_w, (G @ p[..., None])[..., 0] + gp, (moved - kp).reshape(shape + (-1,)), lin, ang,
                      s[..., S["v"]] - carried, s[..., S["w"]] - ang]
            if self.feature_set in ("geo2", "geo3"):
                off = (moved - kp).reshape(shape + (-1,))
                parts += [torch.tanh(off / z) for z in (0.0005, 0.002, 0.008)]
            if self.feature_set == "geo3":
                rv = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1) / 2
                n = self.teeth[task]
                phase = n * torch.atan2(R[..., 1, 0] - R[..., 0, 1], R[..., 0, 0] + R[..., 1, 1])
                parts += [torch.tanh(rv / z) for z in (0.01, 0.05, 0.2)]
                parts += [torch.stack([torch.sin(phase), torch.cos(phase)], -1) * (n > 0)[..., None]]
        if self.n_engines > 1:
            parts.append(torch.nn.functional.one_hot(engine, self.n_engines).float())
        return torch.cat(parts, -1)

    def integrate(self, s, d, g):
        """The next state from the (unnormalized) deltas and the gripper's next command."""
        out = s.clone()
        for k in ("q", "qd", "f", "fd", "p", "v", "w"):
            out[..., S[k]] = s[..., S[k]] + d[..., D[k]]
        R = s[..., S["R"]].reshape(s.shape[:-1] + (3, 3))
        out[..., S["R"]] = (so3_exp(d[..., D["rot"]]) @ R).reshape(s.shape[:-1] + (9,))
        out[..., S["g"]] = g[..., None]
        return out

    def delta(self, s0, s1):
        """The deltas from s0 to s1 (the training targets)."""
        d = torch.empty(s0.shape[:-1] + (DELTA_DIM,), device=s0.device)
        for k in ("q", "qd", "f", "fd", "p", "v", "w"):
            d[..., D[k]] = s1[..., S[k]] - s0[..., S[k]]
        R0 = s0[..., S["R"]].reshape(s0.shape[:-1] + (3, 3))
        R1 = s1[..., S["R"]].reshape(s1.shape[:-1] + (3, 3))
        d[..., D["rot"]] = so3_log(R1 @ R0.transpose(-1, -2))
        return d

    def d_asm(self, s, task):
        """The assembly distance (sim/env.py) of model states."""
        R = s[..., S["R"]].reshape(s.shape[:-1] + (3, 3))
        kp = self.keypoints[task]
        moved = s[..., None, S["p"]] + (R @ kp.transpose(-1, -2)).transpose(-1, -2)
        return (moved - kp).norm(dim=-1).amax(-1)


class Normalizer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def fit(self, x):
        x = x.reshape(-1, x.shape[-1])
        self.mean.copy_(x.mean(0))
        self.std.copy_(x.std(0).clamp(min=1e-6))


class WorldModel(nn.Module):
    """f(s, a, task) -> next state: an MLP of LayerNorm-SiLU blocks on the features, predicting normalized deltas."""

    def __init__(self, frames: TaskFrames, in_dim, hidden=(1024, 1024, 1024), prior=False):
        super().__init__()
        self.frames = frames
        self.use_prior = prior
        self.inp = Normalizer(in_dim)
        self.out = Normalizer(DELTA_DIM)
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU()]
            d = h
        layers.append(nn.Linear(d, DELTA_DIM))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def prior(self, s):
        """The deltas of moving at the start's velocities over the step (zero without the prior)."""
        d = torch.zeros(s.shape[:-1] + (DELTA_DIM,), device=s.device, dtype=s.dtype)
        if self.use_prior:
            h = self.frames.dt
            for pos, vel in (("q", "qd"), ("f", "fd"), ("p", "v"), ("rot", "w")):
                d[..., D[pos]] = h * s[..., S[vel]]
        return d

    def forward(self, s, a, task, engine=None):
        x = self.frames.features(s, a, task, engine)
        _, g = self.frames.act(s, a)
        d = self.net((x - self.inp.mean) / self.inp.std) * self.out.std + self.out.mean + self.prior(s)
        return self.frames.integrate(s, d, g)

    def rollout(self, s0, actions, task, engine=None):
        """States (k + 1, ..., 37) from s0 under actions (k, ..., 8)."""
        out = [s0]
        for a in actions:
            out.append(self(out[-1], a, task, engine))
        return torch.stack(out)

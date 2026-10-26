"""The world model (model.py) as a simulation backend: the environment of sim/gpu/base.py (observations, rewards,
terminations) with the model stepping the state, so that the policy's episodes can be imagined and their success
rates set against the engines':

python -m assembly_policy.evaluate --policy checkpoints/e8192_it12500.pt --backend wm \\
    --backend-kw '{"model": "runs/wm/all/world_model.pt", "engine": "mmp"}'

engine: the engine whose dynamics to imagine, of a model fitted to several (fit.py). The joint velocity limits are
checked at the control steps only (the engines check every substep).
"""
from __future__ import annotations

import mujoco
import numpy as np
import torch

from ..evaluate import env_config
from ..sim.gpu.base import GpuAssemblyEnv, mat_to_quat, quat_to_mat_t
from ..sim.scene import ARM_KD, ARM_KP, TORQUE_LIMIT, VELOCITY_LIMIT, build_xml
from .model import S, TaskFrames, WorldModel


def load_world_model(path, device):
    """The fitted model (fit.py) and its checkpoint."""
    ck = torch.load(path, map_location=device, weights_only=False)
    tasks = ck["tasks"]
    pol = ck.get("policy") or "checkpoints/e8192_it12500.pt"
    ecfg = env_config(torch.load(pol, map_location="cpu", weights_only=False)["args"])
    frames = TaskFrames(tasks, mujoco.MjModel.from_xml_string(build_xml(tasks[0])), ecfg, device,
                        len(ck.get("engines", [None])), ck.get("features", "base"))
    model = WorldModel(frames, ck["model"]["inp.mean"].shape[0], [int(h) for h in ck["hidden"].split(",")],
                       bool(ck.get("prior", False))).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def to_mujoco(frames, s, task):
    """(..., 37) model state -> MuJoCo qpos (..., 16), qvel (..., 15) (the inverse of TaskFrames.from_mujoco)."""
    G, gp = frames.goal_R[task], frames.goal_p[task]
    shape = s.shape[:-1]
    Rw = G @ s[..., S["R"]].reshape(shape + (3, 3))
    qpos = torch.cat([s[..., S["q"]], s[..., S["f"]], (G @ s[..., S["p"], None])[..., 0] + gp,
                      mat_to_quat(Rw.reshape(-1, 3, 3)).reshape(shape + (4,))], -1)
    w_local = (Rw.transpose(-1, -2) @ (G @ s[..., S["w"], None]))[..., 0]
    qvel = torch.cat([s[..., S["qd"]], s[..., S["fd"]], (G @ s[..., S["v"], None])[..., 0], w_local], -1)
    return qpos, qvel


class WMBackend:
    def __init__(self, task_names, task_of_env, ecfg, device, model, engine=None):
        self.dev = torch.device(device)
        self.model, ck = load_world_model(model, self.dev)
        self.frames = self.model.frames
        wm_tasks = list(ck["tasks"])
        self.task = torch.as_tensor([wm_tasks.index(task_names[k]) for k in task_of_env], device=self.dev)
        engines = ck.get("engines", [None])
        if len(engines) > 1:
            self.engine = torch.full_like(self.task, engines.index(engine or engines[0]))
        else:
            self.engine = None
        self.N = len(task_of_env)
        self.mjm = mujoco.MjModel.from_xml_string(build_xml(task_names[0]))
        f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.dev)
        self.kp, self.kd, self.tlim = f(ARM_KP), f(ARM_KD), f(TORQUE_LIMIT)
        self.vlim = f(VELOCITY_LIMIT) * ecfg.velocity_limit_scale
        self.s = torch.zeros(self.N, 37, device=self.dev)
        self.action = None
        self._sync()

    def _sync(self):
        self.qpos, self.qvel = to_mujoco(self.frames, self.s, self.task)

    def write(self, ids, qpos, qvel):
        x = torch.cat([qpos, qvel, torch.zeros(len(ids), 1, device=self.dev)], 1)
        self.s[ids] = self.frames.from_mujoco(x, self.task[ids])
        self._sync()

    def begin(self, action, grip_cmd):
        """The step's action and the gripper command before it (WMEnv.step)."""
        self.action = action
        self.s[:, S["g"]] = grip_cmd[:, None]

    @torch.no_grad()
    def simulate(self, ctrl):
        new = self.model(self.s, self.action, self.task, self.engine)
        # a blown-up prediction (the policy drove the model off its data) stays put and counts as unsafe
        bad = ~torch.isfinite(new).all(1) | (new[:, S["p"]].abs() > 2).any(1)
        self.s = torch.where(bad[:, None], self.s, new)
        self._sync()
        q, qd = self.s[:, S["q"]], self.s[:, S["qd"]]
        viol = (qd.abs() > self.vlim).any(1) | bad
        tau = torch.minimum(torch.maximum(self.kp * (ctrl[:, :7] - q) - self.kd * qd, -self.tlim), self.tlim)
        return viol, tau

    def close(self):
        pass


class WMEnv(GpuAssemblyEnv):
    def step(self, action):
        a = torch.as_tensor(np.asarray(action) if not torch.is_tensor(action) else action, dtype=torch.float32,
                            device=self.dev)
        self.backend.begin(a, self.grip_cmd)
        return super().step(a)


def make_wm_env(task_names, task_of_env, ecfg, device="cuda:0", model=None, engine=None):
    backend = WMBackend(task_names, task_of_env, ecfg, device, model, engine)
    return WMEnv(backend, task_names, task_of_env, ecfg, backend.mjm, device)

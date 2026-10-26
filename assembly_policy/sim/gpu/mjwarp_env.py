"""MuJoCo Warp backend: the CPU task models (sim/scene.build_xml) on the GPU, one MuJoCo Warp model per task with a
world per environment of that task.

The models are the CPU ones, compiled by MuJoCo from the same MJCF; the SDF plugins (nut and bolt threads,
connector grids) run as MuJoCo Warp user SDFs (warp_sdf.py). One control step of a task is a CUDA graph of its
substeps, each followed by a kernel that records joint-speed violations and the clamped PD torques (as the CPU
environment computes them from every substep's state). The tasks' graphs are launched on separate streams.
"""
from __future__ import annotations

import mujoco
import numpy as np
import torch
import warp as wp

from ..env import EnvConfig
from ..robot import Indices
from ..scene import ARM_KD, ARM_KP, TORQUE_LIMIT, VELOCITY_LIMIT, PhysicsConfig, build_xml
from ..tasks import TASKS
from . import warp_sdf
from .base import NQ, NV, GpuAssemblyEnv


@wp.kernel
def _accumulate(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), ctrl: wp.array2d(dtype=float),
                qadr: wp.array(dtype=int), dadr: wp.array(dtype=int), kp: wp.array(dtype=float), kd: wp.array(dtype=float),
                tlim: wp.array(dtype=float), vlim: wp.array(dtype=float), viol: wp.array(dtype=int),
                tau: wp.array2d(dtype=float)):
    w, j = wp.tid()
    qd = qvel[w, dadr[j]]
    if wp.abs(qd) > vlim[j]:
        viol[w] = 1
    tau[w, j] += wp.clamp(kp[j] * (ctrl[w, j] - qpos[w, qadr[j]]) - kd[j] * qd, -tlim[j], tlim[j])


@wp.kernel
def _zero(viol: wp.array(dtype=int), tau: wp.array2d(dtype=float)):
    w, j = wp.tid()
    if j == 0:
        viol[w] = 0
    tau[w, j] = 0.0


class _TaskSim:
    def __init__(self, name, pcfg, ids, ecfg, device, nconmax, njmax, grid_store, grid_type):
        import mujoco_warp as mjw
        self.name, self.ids = name, ids
        self.mjm = mujoco.MjModel.from_xml_string(build_xml(name, pcfg))
        mjd = mujoco.MjData(self.mjm)
        mujoco.mj_forward(self.mjm, mjd)
        n = len(ids)
        with wp.ScopedDevice(device):
            self.m = mjw.put_model(self.mjm)
            self.m.opt.warn_overflow = 0      # (the CPU models also stop their line search at ls_iterations)
            warp_sdf.patch_model(self.m, self.mjm, grid_type, grid_store)
            self.d = mjw.put_data(self.mjm, mjd, nworld=n, nconmax=nconmax, njmax=njmax)
            self.substeps = int(round(ecfg.control_dt / self.mjm.opt.timestep))
            ix = Indices(self.mjm)
            f = lambda x: wp.array(np.asarray(x, np.float32), dtype=float)
            self.qadr, self.dadr = wp.array(ix.arm_qpos.astype(np.int32), dtype=int), wp.array(ix.arm_dof.astype(np.int32), dtype=int)
            self.kp, self.kd, self.tlim = f(ARM_KP), f(ARM_KD), f(TORQUE_LIMIT)
            self.vlim = f(np.asarray(VELOCITY_LIMIT) * ecfg.velocity_limit_scale)
            self.viol = wp.zeros(n, dtype=int)
            self.tau = wp.zeros((n, 7), dtype=float)
            self.stream = wp.Stream(device)
        # the qpos / qvel layout of every task model is the base's: arm, fingers, part
        assert self.mjm.nq == NQ and self.mjm.nv == NV and list(ix.arm_qpos) == list(range(7))
        assert ix.part_qpos == 9 and ix.part_dof == 9
        t = lambda a: wp.to_torch(a)
        self.qpos, self.qvel, self.ctrl = t(self.d.qpos), t(self.d.qvel), t(self.d.ctrl)
        self.warm, self.time = t(self.d.qacc_warmstart), t(self.d.time)
        self.viol_t, self.tau_t = t(self.viol), t(self.tau)
        self.graph = None

    def _substeps(self):
        import mujoco_warp as mjw
        n = len(self.ids)
        wp.launch(_zero, (n, 7), inputs=[self.viol, self.tau])
        for _ in range(self.substeps):
            mjw.step(self.m, self.d)
            wp.launch(_accumulate, (n, 7), inputs=[self.d.qpos, self.d.qvel, self.d.ctrl, self.qadr, self.dadr, self.kp,
                                                   self.kd, self.tlim, self.vlim, self.viol, self.tau])

    def capture(self):
        with wp.ScopedStream(self.stream):
            self._substeps()                      # build the kernels before capturing
            with wp.ScopedCapture() as cap:
                self._substeps()
        self.graph = cap.graph

    def launch(self):
        wp.capture_launch(self.graph, stream=self.stream)


class MJWarpBackend:
    """All environments' worlds, grouped by task; qpos / qvel are (N, 16) / (N, 15) torch copies of the state,
    refreshed after each simulate() and write()."""

    def __init__(self, task_names, task_of_env, ecfg: EnvConfig, pcfg: PhysicsConfig, device="cuda:0",
                 nconmax=None, njmax=None):
        self.device = device
        self.task = np.asarray(task_of_env)
        self.N = len(self.task)
        self.dev = torch.device(device)
        bolt, nut, grid = warp_sdf.plugin_types()
        warp_sdf.install(bolt, nut, grid)
        store = warp_sdf.GridStore(device)
        self.sims = []
        self.local = np.zeros(self.N, dtype=np.int64)        # index of each environment in its task's worlds
        for k, name in enumerate(task_names):
            ids = np.nonzero(self.task == k)[0]
            if len(ids) == 0:
                continue
            fam = TASKS[name].family
            nc = nconmax or {"rod": 48, "peg": 48, "gear": 96, "nut": 96, "connector": 128}[fam]
            nj = njmax or 6 * nc
            self.local[ids] = np.arange(len(ids))
            self.sims.append(_TaskSim(name, pcfg, ids, ecfg, device, nc, nj, store, grid))
        self.substeps = np.zeros(self.N, dtype=np.int64)
        for s in self.sims:
            s.ids_t = torch.as_tensor(s.ids, device=self.dev)
            self.substeps[s.ids] = s.substeps
            s.capture()
        wp.synchronize_device(device)
        self.qpos = torch.zeros(self.N, NQ, device=self.dev)
        self.qvel = torch.zeros(self.N, NV, device=self.dev)
        self.local_t = torch.as_tensor(self.local, device=self.dev)
        self.mjm = self.sims[0].mjm

    def close(self):
        pass

    def _gather(self):
        for s in self.sims:
            self.qpos[s.ids_t] = s.qpos
            self.qvel[s.ids_t] = s.qvel

    def write(self, env_ids, qpos, qvel):
        """Set the state of some environments (torch: ids (n,), qpos (n, 16), qvel (n, 15))."""
        task = torch.as_tensor(self.task, device=self.dev)[env_ids]
        for k, s in enumerate(self.sims):
            sel = torch.nonzero(task == int(self.task[s.ids[0]]))[:, 0]
            if len(sel) == 0:
                continue
            w = self.local_t[env_ids[sel]]
            s.qpos[w] = qpos[sel]
            s.qvel[w] = qvel[sel]
            s.warm[w] = 0.0
            s.time[w] = 0.0
        wp.synchronize_device(self.device)
        self._gather()

    def simulate(self, ctrl):
        """One control step with the PD targets ctrl (N, 8); returns (velocity violation (N,), mean torques (N, 7))."""
        viol = torch.zeros(self.N, dtype=torch.bool, device=self.dev)
        tau = torch.zeros(self.N, 7, device=self.dev)
        for s in self.sims:
            s.ctrl[:] = ctrl[s.ids_t]
        torch.cuda.current_stream(self.dev).synchronize()
        for s in self.sims:
            s.launch()
        wp.synchronize_device(self.device)
        for s in self.sims:
            viol[s.ids_t] = s.viol_t > 0
            tau[s.ids_t] = s.tau_t / s.substeps
        self._gather()
        return viol, tau


def make_mjwarp_env(task_names, task_of_env, ecfg: EnvConfig, pcfg: PhysicsConfig | None = None, device="cuda:0", **kw):
    pcfg = pcfg or PhysicsConfig(fine_families=ecfg.fine_families, fine_sdf_initpoints=ecfg.fine_sdf_initpoints,
                                 fine_sdf_iterations=ecfg.fine_sdf_iterations)
    backend = MJWarpBackend(task_names, task_of_env, ecfg, pcfg, device, **kw)
    return GpuAssemblyEnv(backend, task_names, task_of_env, ecfg, backend.mjm, device)

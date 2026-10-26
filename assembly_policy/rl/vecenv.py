"""Auto-resetting vector environments with a learner-side reset provider (Algorithm 1).

Both implementations expose:
  observe() -> compact obs dict (post-reset)
  step(action) -> reward, done, info, final_obs (pre-reset obs rows of done envs), finished_entry
where finished_entry[i] is the bank entry of the episode that just ended in env i (for the
sampler update). Each env always holds a pre-sampled `next` entry, drawn by the provider after
the previous step's outcomes were recorded; on termination the env resets to it immediately.

LocalVecEnv runs BatchedAssemblyEnv in-process. RemoteVecEnv runs slices on worker processes or
jobs (one TCP round trip per control step), see remote.py. GpuVecEnv runs a GPU backend (sim/gpu), whose
environments keep their task: the provider draws each one's next entry from its own task's.
"""
from __future__ import annotations

import numpy as np

from ..sim.env import BatchedAssemblyEnv, EnvConfig

OBS_KEYS = ("proprio", "task", "tcp_p", "tcp_R", "part_p", "part_q", "finger")
INFO_KEYS = ("success", "safety", "timeout", "d_asm", "cost", "unstable")


class LocalVecEnv:
    def __init__(self, task_names, n_envs, ecfg: EnvConfig, provider, t0=None):
        # provider(k, finished_entries=None, outcomes=None) -> (entries, states, tasks); it records the
        # finished episodes' outcomes in the sampler before drawing k new entries.
        self.provider = provider
        self.env = BatchedAssemblyEnv(task_names, np.zeros(n_envs, dtype=np.int64), ecfg)
        self.N, self.horizon = n_envs, self.env.horizon
        e, s, t = provider(n_envs)
        self.entry = e
        self.env.set_states(np.arange(n_envs), s, t)
        self.next_entry, self.next_state, self.next_task = provider(n_envs)
        if t0 is not None:
            self.env.t[:] = t0

    def observe(self):
        return self.env.observe()

    def step(self, action):
        ep_len = self.env.t + 1
        r, done, info = self.env.step(action)
        ids = np.nonzero(done)[0]
        final_obs = None
        finished = self.entry[ids].copy()
        if len(ids):
            o = self.env.observe()
            final_obs = {k: o[k][ids] for k in OBS_KEYS}
            self.env.set_states(ids, self.next_state[ids], self.next_task[ids])
            self.entry[ids] = self.next_entry[ids]
            e, s, t = self.provider(len(ids), finished, info["success"][ids])
            self.next_entry[ids], self.next_state[ids], self.next_task[ids] = e, s, t
        info["ep_len"] = ep_len
        info["valid"] = np.ones(self.N, bool)
        return r, done, info, final_obs, finished

    def alive_mask(self):
        return np.ones(self.N, bool)

    def close(self):
        self.env.close()


def split_tasks(n_envs, n_tasks):
    """The task of each of n_envs environments: an equal share of every task (the first ones one more)."""
    return np.arange(n_envs) % n_tasks


class GpuVecEnv(LocalVecEnv):
    """LocalVecEnv on a GPU backend (sim/gpu.make_env): task_of_env fixes each environment's task, and
    provider(k, finished, outcomes, tasks=...) draws the next entries within those tasks."""

    def __init__(self, backend, task_names, task_of_env, ecfg: EnvConfig, provider, t0=None, **backend_kw):
        from ..sim.gpu import make_env
        self.provider = provider
        self.task_of_env = np.asarray(task_of_env, dtype=np.int64)
        n_envs = len(self.task_of_env)
        self.env = make_env(backend, task_names, self.task_of_env, ecfg, **backend_kw)
        self.N, self.horizon = n_envs, self.env.horizon
        e, s, t = provider(n_envs, tasks=self.task_of_env)
        self.entry = e
        self.env.set_states(np.arange(n_envs), s, t)
        self.next_entry, self.next_state, self.next_task = provider(n_envs, tasks=self.task_of_env)
        if t0 is not None:
            self.env.t[:] = t0

    def observe(self):
        return self.env.observe_torch()

    def step(self, action):
        return self.finish(*self.simulate(action))

    def simulate(self, action):
        """The physics of a step (no provider calls: safe in a thread of its own, see MultiGpuVecEnv)."""
        ep_len = self.env.t + 1
        return (ep_len,) + tuple(self.env.step(action))

    def finish(self, ep_len, r, done, info):
        """The resets after a step: the finished environments' final observations, their next starts."""
        ids = np.nonzero(done)[0]
        final_obs = None
        finished = self.entry[ids].copy()
        if len(ids):
            o = self.env.observe_torch()
            final_obs = {k: o[k][ids] for k in OBS_KEYS}
            self.env.set_states(ids, self.next_state[ids], self.next_task[ids])
            self.entry[ids] = self.next_entry[ids]
            e, s, t = self.provider(len(ids), finished, info["success"][ids], tasks=self.task_of_env[ids])
            self.next_entry[ids], self.next_state[ids], self.next_task[ids] = e, s, t
        info["ep_len"] = ep_len
        info["valid"] = np.ones(self.N, bool)
        return r, done, info, final_obs, finished


class MultiGpuVecEnv:
    """GpuVecEnv over several GPUs: the environments in contiguous blocks, one GpuVecEnv per device, stepped at once
    (a thread each; the simulators release the GIL), the resets after, in turn (the provider is not thread-safe).
    Observations and final observations are gathered on the first device's (the learner's, by default)."""

    def __init__(self, backend, task_names, task_of_env, ecfg: EnvConfig, provider, t0=None, devices=("cuda:0",),
                 gather="cuda:0", **backend_kw):
        import torch
        from concurrent.futures import ThreadPoolExecutor
        task_of_env = np.asarray(task_of_env, dtype=np.int64)
        self.N = len(task_of_env)
        self.blocks = np.array_split(np.arange(self.N), len(devices))
        self.parts = [GpuVecEnv(backend, task_names, task_of_env[b], ecfg, provider,
                                None if t0 is None else np.asarray(t0)[b], device=d, **backend_kw)
                      for b, d in zip(self.blocks, devices)]
        self.horizon = self.parts[0].horizon
        self.gather = torch.device(gather)
        self.pool = ThreadPoolExecutor(len(devices))
        self._torch = torch

    def _cat(self, dicts):
        return {k: self._torch.cat([d[k].to(self.gather, non_blocking=True) for d in dicts]) for k in dicts[0]}

    def observe(self):
        return self._cat([p.observe() for p in self.parts])

    def step(self, action):
        action = np.asarray(action)
        sims = list(self.pool.map(lambda pb: pb[0].simulate(action[pb[1]]), zip(self.parts, self.blocks)))
        outs = [p.finish(*sim) for p, sim in zip(self.parts, sims)]
        r = np.concatenate([o[0] for o in outs])
        done = np.concatenate([o[1] for o in outs])
        info = {k: np.concatenate([o[2][k] for o in outs]) for k in outs[0][2]}
        finals = [o[3] for o in outs if o[3] is not None]
        final_obs = self._cat(finals) if finals else None
        finished = np.concatenate([o[4] for o in outs])
        return r, done, info, final_obs, finished

    def alive_mask(self):
        return np.ones(self.N, bool)

    def close(self):
        for p in self.parts:
            p.close()
        self.pool.shutdown()

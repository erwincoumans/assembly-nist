"""Auto-resetting vector environments with a learner-side reset provider (Algorithm 1).

Both implementations expose:
  observe() -> compact obs dict (post-reset)
  step(action) -> reward, done, info, final_obs (pre-reset obs rows of done envs), finished_entry
where finished_entry[i] is the bank entry of the episode that just ended in env i (for the
sampler update). Each env always holds a pre-sampled `next` entry, drawn by the provider after
the previous step's outcomes were recorded; on termination the env resets to it immediately.

LocalVecEnv runs BatchedAssemblyEnv in-process. RemoteVecEnv runs slices on worker processes or
jobs (one TCP round trip per control step), see remote.py.
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

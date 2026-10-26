"""GPU simulation backends with the interface of sim/env.BatchedAssemblyEnv (see base.py).

  mujoco   CPU MuJoCo (sim/env.py), the reference
  mjwarp   MuJoCo Warp: the same task models on the GPU (mjwarp_env.py)
  mmp      Metamorphysics (mmp_env.py)
"""
from __future__ import annotations

BACKENDS = ("mujoco", "mjwarp", "mmp")


def make_env(backend, task_names, task_of_env, ecfg, **kw):
    """An environment of the backend for these tasks (the task index of each environment)."""
    if backend == "mujoco":
        from ..env import BatchedAssemblyEnv
        return BatchedAssemblyEnv(task_names, task_of_env, ecfg)
    if backend == "mjwarp":
        from .mjwarp_env import make_mjwarp_env
        return make_mjwarp_env(task_names, task_of_env, ecfg, **kw)
    if backend == "mmp":
        from .mmp_env import make_mmp_env
        return make_mmp_env(task_names, task_of_env, ecfg, **kw)
    raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")

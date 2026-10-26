"""Logged transitions of a GPU backend for fitting a world model: the policy's (sampled) actions, with extra action
noise on a share of the environments, from reset-bank starts; one environment in `test_every` starts from the
evaluation banks and is held out.

python -m assembly_policy.wm.collect --policy checkpoints/e8192_it12500.pt --out data/wm/mmp.npz --envs 4096 --steps 1500

Saved: state (T + 1, N, 32) float32 [qpos 16, qvel 15, gripper command], action (T, N, 8), valid (T, N) (the transition
t -> t + 1 is a step of one episode, not across a reset), task (N,), test (N,), tasks (names); the episodes' last
transitions (valid False) end in term_state (M, 32) at (term_t, term_n): the states the resets replaced, ended by
term_kind (0 success, 1 safety, 2 timeout).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..evaluate import TorchPolicy, env_config
from ..sim.gpu import make_env
from ..sim.tasks import BANK_DIR, resolve_tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="mmp")
    ap.add_argument("--backend-kw", default="{}")
    ap.add_argument("--tasks", default="all20")
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--noise-share", type=float, default=0.25, help="environments with extra action noise")
    ap.add_argument("--noise", type=float, default=0.5, help="std of the extra action noise")
    ap.add_argument("--test-every", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nthread", type=int, default=16, help="CPU threads of --backend mujoco")
    a = ap.parse_args()
    tasks = resolve_tasks(a.tasks)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    pol = TorchPolicy(a.policy, tasks, a.device)
    cfg = env_config(pol.args)
    N = a.envs
    task = np.arange(N) % len(tasks)
    test = (np.arange(N) // len(tasks)) % a.test_every == 0          # (every task has test environments)
    noisy = rng.random(N) < a.noise_share
    banks = {(k, split): np.load(Path(BANK_DIR) / f"{t}_{split}.npz")["states"]
             for k, t in enumerate(tasks) for split in ("train", "eval")}

    def starts(ids):
        return np.stack([banks[(int(task[i]), "eval" if test[i] else "train")][
            rng.integers(len(banks[(int(task[i]), "eval" if test[i] else "train")]))] for i in ids])

    if a.backend == "mujoco":
        cfg.nthread = a.nthread
        env = make_env("mujoco", tasks, task, cfg)
    else:
        env = make_env(a.backend, tasks, task, cfg, device=a.device, **json.loads(a.backend_kw))
    env.set_states(np.arange(N), starts(np.arange(N)), task)
    T = a.steps
    state = np.zeros((T + 1, N, 32), np.float32)
    action = np.zeros((T, N, 8), np.float32)
    valid = np.zeros((T, N), bool)
    noise_scale = torch.as_tensor(np.where(noisy, a.noise, 0.0)[:, None], dtype=torch.float32, device=a.device)
    term_t, term_n, term_state, term_kind = [], [], [], []
    t0 = time.time()

    def record(t):
        if a.backend == "mujoco":       # (BatchedAssemblyEnv: [time, qpos, qvel])
            state[t, :, :31] = env.state[:, 1:32]
            state[t, :, 31] = env.grip_cmd
            return
        b = env.backend
        state[t, :, :16] = b.qpos.cpu().numpy()
        state[t, :, 16:31] = b.qvel.cpu().numpy()
        state[t, :, 31] = env.grip_cmd.cpu().numpy()

    record(0)
    for t in range(T):
        with torch.no_grad():
            o = env.observe_torch() if hasattr(env, "observe_torch") else env.observe()
            f = lambda k, dt=torch.float32: torch.as_tensor(o[k], dtype=dt, device=pol.dev)
            pts = pol.pts(pol.remap[f("task", torch.long)], f("tcp_p"), f("tcp_R"), f("part_p"), f("part_q"), f("finger"))
            act = pol.model.dist(pts, f("proprio")).sample()
            act = act + noise_scale * torch.randn_like(act)
        action[t] = act.cpu().numpy()
        _, done, info = env.step(act.cpu().numpy() if a.backend == "mujoco" else act)
        valid[t] = ~done
        ids = np.nonzero(done)[0]
        if len(ids):
            record(t + 1)
            term_t.append(np.full(len(ids), t)); term_n.append(ids); term_state.append(state[t + 1, ids])
            term_kind.append(np.where(np.asarray(info["success"])[ids], 0, np.where(np.asarray(info["safety"])[ids], 1, 2)))
            env.set_states(ids, starts(ids), task[ids])
        record(t + 1)
        if t % 100 == 0:
            print(f"step {t}/{T}: {time.time() - t0:.0f}s, episodes ended {int((~valid[:t + 1]).sum())}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cat = lambda xs, shape, dt: np.concatenate(xs) if xs else np.zeros(shape, dt)
    np.savez(a.out, state=state, action=action, valid=valid, task=task, test=test, noisy=noisy, tasks=np.array(tasks),
             policy=a.policy, backend=a.backend, term_t=cat(term_t, 0, np.int64), term_n=cat(term_n, 0, np.int64),
             term_state=cat(term_state, (0, 32), np.float32), term_kind=cat(term_kind, 0, np.int64))
    print(f"saved {a.out}: {T} steps x {N} environments in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

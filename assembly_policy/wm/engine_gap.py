"""How far apart the engines are, in the world model's terms (fit.py): the held-out windows of one engine's log
(collect.py) are replayed open loop in an engine, from the logged states with the logged actions, and their errors
after 1, 8 and 32 steps reported as fit.py reports a world model's (on the same windows). Replayed in the engine that
logged them, the errors are the floor: what setting a state loses (solver warm starts, contact caches) and the
engine's own nondeterminism.

python -m assembly_policy.wm.engine_gap --data data/wm/mmp.npz --backend mjwarp --out runs/wm/gap/mmp_in_mjwarp.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..sim.gpu import make_env
from .fit import evaluate, prepare


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="a collect.py log")
    ap.add_argument("--backend", required=True, help="the engine to replay it in")
    ap.add_argument("--backend-kw", default="{}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-windows", type=int, default=40000)
    ap.add_argument("--nthread", type=int, default=16, help="CPU threads of --backend mujoco")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    data, frames, ms, run = prepare([a.data], dev)
    test = torch.as_tensor(np.nonzero(data["test"])[0], device=dev)
    slot = torch.full((len(data["test"]),), -1, dtype=torch.long, device=dev)
    slot[test] = torch.arange(len(test), device=dev)            # (the replaying environment of each test one)
    task_of = data["task"][test].cpu().numpy()
    cfg = data["ecfg"]
    if a.backend == "mujoco":
        cfg.nthread = a.nthread
        env = make_env("mujoco", data["tasks"], task_of, cfg)
    else:
        env = make_env(a.backend, data["tasks"], task_of, cfg, device=a.device, **json.loads(a.backend_kw))
    cpu = a.backend == "mujoco"

    def state():
        if cpu:
            x = np.concatenate([env.state[:, 1:32], np.asarray(env.grip_cmd)[:, None]], 1)
            return torch.as_tensor(x, dtype=torch.float32, device=dev)
        return torch.cat([env.backend.qpos, env.backend.qvel, env.grip_cmd[:, None]], 1).float()

    def replay(t, e, s, act, tk):
        ids = slot[e].cpu().numpy()
        x = data["state"][t, e]
        env.set_states(ids, np.concatenate([np.zeros((len(ids), 1), np.float32), x[:, :31].cpu().numpy()], 1),
                       task_of[ids])
        g = x[:, 31]
        if cpu:
            env.grip_cmd[ids] = g.cpu().numpy()
        else:
            env.grip_cmd[torch.as_tensor(ids, device=env.grip_cmd.device)] = g.to(env.grip_cmd.device)
        full = np.zeros((len(task_of), 8), np.float32) if cpu else torch.zeros(len(task_of), 8, device=dev)
        out = [s[0]]
        for j in range(len(act)):
            full[ids] = act[j].cpu().numpy() if cpu else act[j]
            env.step(full)
            out.append(frames.from_mujoco(state()[torch.as_tensor(ids, device=dev)], tk))
        return torch.stack(out)

    print(f"{data['engines'][0]} log replayed in {a.backend}: {len(test)} test environments", flush=True)
    report = evaluate(replay, frames, data, ms, run, test, a.eval_windows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(data=a.data, backend=a.backend, report=report), indent=1))


if __name__ == "__main__":
    main()

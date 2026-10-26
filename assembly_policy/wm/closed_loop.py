"""The policy's episodes in a world model next to an engine's, from the same evaluation-bank starts: success by
family and the assembly distance over time (where imagined episodes part from simulated ones).

python -m assembly_policy.wm.closed_loop --model runs/wm/all/world_model.pt --engine mmp --out runs/wm/all/closed_loop.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..evaluate import TorchPolicy, env_config
from ..sim.gpu import make_env
from ..sim.tasks import BANK_DIR, resolve_tasks

FAMILIES = ("loose", "pickup", "inhand", "partial", "near")


def episodes(env, policy, horizon):
    """d_asm (T, N) until each episode ends (then nan), success (N,), safety (N,), end step (N,)."""
    N = env.N
    d = np.full((horizon, N), np.nan)
    done_at = np.full(N, -1); success = np.zeros(N, bool); safety = np.zeros(N, bool)
    torch.manual_seed(0)
    for t in range(horizon):
        _, done, info = env.step(policy(env))
        live = done_at < 0
        d[t, live] = info["d_asm"][live]
        newly = live & done
        done_at[newly] = t; success[newly] = info["success"][newly]; safety[newly] = info["safety"][newly]
        if (done_at >= 0).all():
            break
    return d, success, safety, done_at


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="checkpoints/e8192_it12500.pt")
    ap.add_argument("--model", required=True, help="fit.py world model")
    ap.add_argument("--engine", default="mmp", help="the engine to compare with (and to imagine, of a multi-engine model)")
    ap.add_argument("--engine-kw", default="{}")
    ap.add_argument("--trials", type=int, default=64, help="starts per task")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    tasks = resolve_tasks("all20")
    pol = TorchPolicy(a.policy, tasks, a.device)
    cfg = env_config(pol.args)
    rng = np.random.default_rng(10_000)
    banks = [np.load(Path(BANK_DIR) / f"{t}_eval.npz") for t in tasks]
    E = np.array([(k, i) for k, b in enumerate(banks) for i in rng.integers(0, len(b["states"]), a.trials)])
    starts = np.stack([banks[k]["states"][i] for k, i in E])
    fam = np.array([banks[k]["family"][i] for k, i in E])
    res = {}
    for name, backend, kw in (("model", "wm", dict(model=a.model, engine=a.engine)),
                              (a.engine, a.engine, json.loads(a.engine_kw))):
        env = make_env(backend, tasks, E[:, 0], cfg, device=a.device, **kw)
        env.set_states(np.arange(len(E)), starts, E[:, 0])
        d, success, safety, done_at = episodes(env, pol, env.horizon)
        env.close()
        res[name] = dict(d=d, success=success, safety=safety, done_at=done_at)
    out = {}
    print(f"{'family':8s} {'n':>4s}  {'success model/engine':>20s}  {'safety':>11s}  median d_asm mm at steps 0/5/10/25/50/100 "
          "(model | engine)")
    for f, fname in list(enumerate(FAMILIES)) + [(None, "all")]:
        m = fam == f if f is not None else np.ones(len(E), bool)
        if not m.any():
            continue
        row = {}
        for name in ("model", a.engine):
            r = res[name]
            dd = r["d"][:, m]
            row[name] = dict(success=float(r["success"][m].mean()), safety=float(r["safety"][m].mean()),
                             d_asm_mm=[float(np.nanmedian(dd[t]) * 1e3) if np.isfinite(dd[t]).any() else None
                                       for t in (0, 5, 10, 25, 50, 100)])
        out[fname] = row
        fmt = lambda v: " ".join(f"{x:5.1f}" if x is not None else "    -" for x in v)
        print(f"{fname:8s} {m.sum():4d}  {row['model']['success']:9.2f}/{row[a.engine]['success']:<9.2f}  "
              f"{row['model']['safety']:5.2f}/{row[a.engine]['safety']:<5.2f}  {fmt(row['model']['d_asm_mm'])} | "
              f"{fmt(row[a.engine]['d_asm_mm'])}", flush=True)
    by_task = {t: dict(model=float(res["model"]["success"][E[:, 0] == k].mean()),
                       engine=float(res[a.engine]["success"][E[:, 0] == k].mean())) for k, t in enumerate(tasks)}
    print("per task (model/engine): " + ", ".join(f"{t} {v['model']:.2f}/{v['engine']:.2f}" for t, v in by_task.items()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(model=a.model, engine=a.engine, families=out, tasks=by_task), indent=1))


if __name__ == "__main__":
    main()

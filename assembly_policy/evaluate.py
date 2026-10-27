"""Evaluation protocol (as described in the AlphaNIST paper, Fig. 7): frozen mixed-reset evaluation bank, uniform within each task,
5 seeds x 32 trials per task, success = full assembly within 15 s without a simultaneous safety failure.
Actions are sampled from the policy (as in the paper) or, with --deterministic, the policy mean.

python -m assembly_policy.evaluate --policy runs/<run>/model_010000.pt --out runs/<run>/eval/model_010000.json
python -m assembly_policy.evaluate --policy checkpoints/e8192_it12500.pt --backend mjwarp --out eval_mjwarp.json   # on the GPU
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .rl.nets import ActorCritic
from .rl.obs import PointTables
from .sim.resets import FAMILIES
from .sim.env import EnvConfig, parse_scales
from .sim.gpu import BACKENDS, gpu_devices, make_env
from .sim.tasks import BANK_DIR, resolve_tasks


def env_config(args, nthread=4):
    """The EnvConfig a checkpoint was trained with. Checkpoints store it as args["env"]; checkpoints of the original
    reproduction runs stored a few individual settings instead (everything else had today's defaults)."""
    if "env" in args:
        env = dict(args["env"])
        env["action_scale_joints"] = tuple(env.get("action_scale_joints") or ())
        return EnvConfig(**{**env, "nthread": nthread})
    legacy = {k: args[k] for k in ("fine_families", "fine_sdf_initpoints", "fine_sdf_iterations", "grip_close",
                                   "action_scale", "cost_coef") if k in args}
    legacy["action_scale_joints"] = parse_scales(args.get("action_scale_joints", ""))
    return EnvConfig(nthread=nthread, **legacy)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def evaluate(policy, tasks, bank_dir, cfg, seeds=(0, 1, 2, 3, 4), trials=32, backend="mujoco", backend_kw=None):
    """Roll out `trials` evaluation-bank starts per task and seed; returns overall / per-task / per-family success.
    backend: the simulator (sim/gpu.BACKENDS), with its keyword arguments backend_kw."""
    banks = [np.load(Path(bank_dir) / f"{t}_eval.npz") for t in tasks]
    entries = []  # (seed, task, idx)
    for s in seeds:
        rng = np.random.default_rng(10_000 + s)
        for k, b in enumerate(banks):
            for i in rng.integers(0, len(b["states"]), trials):
                entries.append((s, k, i))
    E = np.array(entries)
    N = len(E)
    env = make_env(backend, tasks, E[:, 1], cfg, **(backend_kw or {}))
    env.set_states(np.arange(N), np.stack([banks[k]["states"][i] for _, k, i in E]), E[:, 1])
    fam = np.array([banks[k]["family"][i] for _, k, i in E])
    done_at = np.full(N, -1); success = np.zeros(N, bool); safety = np.zeros(N, bool)
    torch.manual_seed(0)
    for t in range(env.horizon):
        _, done, info = env.step(policy(env))
        newly = (done_at < 0) & done          # finished episodes keep stepping but their outcome is frozen
        done_at[newly] = t; success[newly] = info["success"][newly]; safety[newly] = info["safety"][newly]
        if (done_at >= 0).all():
            break
    env.close()
    per_seed = [success[E[:, 0] == s].mean() for s in seeds]
    res = dict(overall=dict(mean=float(np.mean(per_seed)), std=float(np.std(per_seed)), per_seed=[float(x) for x in per_seed],
                            wilson95=wilson(int(success.sum()), N), n=N, safety=float(safety.mean())),
               per_task={}, per_family={})
    for k, t in enumerate(tasks):
        m = E[:, 1] == k
        ps = [success[m & (E[:, 0] == s)].mean() for s in seeds]
        res["per_task"][t] = dict(mean=float(np.mean(ps)), std=float(np.std(ps)), per_seed=[float(x) for x in ps], n=int(m.sum()),
                                  by_family={f: float(success[m & (fam == i)].mean()) if (m & (fam == i)).any() else None
                                             for i, f in enumerate(FAMILIES)})
    for i, f in enumerate(FAMILIES):
        m = fam == i
        res["per_family"][f] = dict(mean=float(success[m].mean()) if m.any() else None, n=int(m.sum()))
    return res


class TorchPolicy:
    """A trained checkpoint as a callable env -> actions (actor only)."""

    def __init__(self, ckpt, tasks, device="cpu", deterministic=False):
        d = torch.load(ckpt, map_location=device, weights_only=False)
        a = d["args"]
        # point tables are sampled per training-task index: build them for the training task list and map the
        # environment's task indices (any subset, any order) onto it
        train_tasks = resolve_tasks(a.get("tasks", "all20"))
        self.pts = PointTables(train_tasks, a["n_part"], a["n_fix"], a["n_grip"], device=device)
        self.remap = torch.as_tensor([train_tasks.index(t) for t in tasks], device=device)
        self.model = ActorCritic(self.pts.n_pts, max_std=a.get("max_std", 2.0), min_std=a.get("min_std", 1e-3)).to(device)
        actor = {k: v for k, v in d["model"].items() if not k.startswith(("critic.", "priv_norm."))}
        self.model.load_state_dict(actor, strict=False); self.model.eval()
        self.dev, self.args, self.deterministic = device, a, deterministic
        self.it, self.transitions = d.get("it"), d.get("transitions")

    @torch.no_grad()
    def __call__(self, env):
        o = env.observe_torch() if hasattr(env, "observe_torch") else env.observe()
        f = lambda k, dt=torch.float32: torch.as_tensor(o[k], dtype=dt, device=self.dev)
        pts = self.pts(self.remap[f("task", torch.long)], f("tcp_p"), f("tcp_R"), f("part_p"), f("part_q"), f("finger"))
        dist = self.model.dist(pts, f("proprio"))
        return (dist.mean if self.deterministic else dist.sample()).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="checkpoint (model_*.pt)")
    ap.add_argument("--tasks", default="all20")
    ap.add_argument("--bank-dir", default=str(BANK_DIR))
    ap.add_argument("--nthread", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--device", default=None, help="policy device (default: cpu for mujoco, else cuda)")
    ap.add_argument("--backend", default="mujoco", choices=BACKENDS, help="simulator (GPU backends: sim/gpu)")
    ap.add_argument("--gpus", default="0", help="the GPU of a GPU backend (one: 0, 1, ...)")
    ap.add_argument("--backend-kw", default="{}", help='JSON options of the GPU backend, e.g. \'{"geometry": "mjfit"}\'')
    ap.add_argument("--seeds", type=int, default=5, help="evaluation seeds (paper: 5)")
    ap.add_argument("--trials", type=int, default=32, help="trials per task and seed (paper: 32)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--deterministic", action="store_true", help="mean actions (default: sampled, as in the paper)")
    a = ap.parse_args()
    tasks = resolve_tasks(a.tasks)
    sim_device = gpu_devices(a.gpus)
    if len(sim_device) != 1:
        ap.error("--gpus: an evaluation simulates on one GPU")
    sim_device = sim_device[0]
    device = a.device or ("cpu" if a.backend == "mujoco" else sim_device)
    pol = TorchPolicy(a.policy, tasks, device, deterministic=a.deterministic)
    cfg = env_config(pol.args, a.nthread)
    kw = {} if a.backend == "mujoco" else dict(device=sim_device, **json.loads(a.backend_kw))
    t0 = time.time()
    res = evaluate(pol, tasks, a.bank_dir, cfg, seeds=tuple(range(a.seeds)), trials=a.trials, backend=a.backend, backend_kw=kw)
    res["meta"] = dict(policy=str(a.policy), tasks=tasks, bank_dir=str(a.bank_dir), deterministic=a.deterministic,
                       backend=a.backend, backend_kw=json.loads(a.backend_kw), wall_s=round(time.time() - t0, 1),
                       iteration=pol.it, transitions=pol.transitions,
                       env={k: v for k, v in dataclasses.asdict(cfg).items() if k != "nthread"})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(dict(overall=res["overall"], per_family=res["per_family"])))


if __name__ == "__main__":
    main()

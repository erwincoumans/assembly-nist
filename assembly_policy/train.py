"""PPO training of one shared policy on all 20 NIST assemblies (following Algorithm 1 of the AlphaNIST paper) on CPU MuJoCo.

Environments are simulated either in this process (thread pool, --nthread) or by remote CPU worker processes
(--remote-workers N; see scripts/workers.sh), which is how the 8,192-environment run is done. Every finished
episode updates the beta reset sampler; new episodes start from reset-bank entries drawn from it. Checkpoints
are written atomically every --save-every iterations and the run resumes automatically from run_dir.

python -m assembly_policy.train --run-dir runs/demo --envs 256 --iters 200          # small local run
python -m assembly_policy.train --run-dir runs/e8192 --remote-workers 960           # the reported setting
python -m assembly_policy.train --run-dir runs/mmp --backend mmp --envs 4096           # on the GPU (sim/gpu)

On a GPU backend (--backend mjwarp or mmp) every environment keeps one task (an equal share of environments per
task) and the success-guided sampler draws each one's starts from its task's bank entries.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .rl.nets import ActorCritic
from .rl.obs import PointTables
from .sim.resets import FAMILIES
from .rl.sampler import ResetSampler
from .sim.env import HISTORY, PROPRIO_DIM, EnvConfig, parse_scales
from .sim.tasks import BANK_DIR, FINE_STEP_COST, ROOT, STEP_COST, TASKS, resolve_tasks
from .rl.vecenv import LocalVecEnv

OBS = ("prop", "task", "tcp_p", "tcp_R", "part_p", "part_q", "finger")


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tasks", default="all20", help="all20 (default), or a task set / '+'-separated list")
    ap.add_argument("--bank-dir", default=str(BANK_DIR))
    ap.add_argument("--envs", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=10000)
    # PPO
    ap.add_argument("--steps", type=int, default=32, help="rollout steps per environment and iteration")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4, help="initial learning rate (adaptive-KL schedule)")
    ap.add_argument("--desired-kl", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.005)
    ap.add_argument("--vf-coef", type=float, default=1.0)
    ap.add_argument("--max-grad", type=float, default=1.0)
    ap.add_argument("--init-std", type=float, default=1.0)
    ap.add_argument("--min-std", type=float, default=1e-3)
    ap.add_argument("--max-std", type=float, default=2.0)
    # reset bank and sampler
    ap.add_argument("--sampler", default="beta", choices=["beta", "uniform"])
    ap.add_argument("--eps-q", type=float, default=1e-4, help="beta sampler floor")
    ap.add_argument("--history", type=int, default=50, help="outcomes kept per bank entry (L)")
    ap.add_argument("--bank-per-family", type=int, default=640,
                    help="training entries per (task, reset family); 640 x 5 x 20 = 64,000 entries (0 = all)")
    # environment
    ap.add_argument("--action-scale-joints", default="0.05+0.05+0.05+0.06+0.07+0.1+0.17",
                    help="rad per unit action for each arm joint ('+'-separated)")
    ap.add_argument("--cost-coef", type=float, default=2e-5, help="control cost c (actions, joint speeds, torques)")
    ap.add_argument("--action-rate-coef", type=float, default=1e-4,
                    help="smoothness: reward -= coef * sum_j (a_t,j - a_t-1,j)^2 over the arm actions")
    ap.add_argument("--action-rate-start", type=int, default=5000, help="iteration from which the action-rate term is on")
    ap.add_argument("--n-part", type=int, default=128)
    ap.add_argument("--n-fix", type=int, default=128)
    ap.add_argument("--n-grip", type=int, default=64)
    # infrastructure
    ap.add_argument("--backend", default="mujoco", choices=("mujoco", "mjwarp", "mmp"),
                    help="simulator: CPU MuJoCo (local threads or remote workers) or a GPU backend (sim/gpu)")
    ap.add_argument("--backend-kw", default="{}", help="JSON options of the GPU backend")
    ap.add_argument("--eval-backend", default=None, help="the background evaluations' simulator (default: --backend)")
    ap.add_argument("--eval-backend-kw", default=None, help="its JSON options (default: --backend-kw)")
    ap.add_argument("--sim-device", default="cuda:0",
                    help="GPU(s) of a GPU backend's simulation: a comma-separated list splits the environments")
    ap.add_argument("--remote-workers", type=int, default=0, help="number of remote worker processes (0 = simulate here)")
    ap.add_argument("--nthread", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)),
                    help="simulation threads when simulating locally")
    ap.add_argument("--accept-timeout", type=float, default=300.0, help="start with partial remote workers after this (s)")
    ap.add_argument("--min-conn-frac", type=float, default=0.5)
    ap.add_argument("--micro-batch", type=int, default=65536, help="samples per forward/backward chunk (GPU memory)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--keep-every", type=int, default=500, help="also keep model_<it>.pt every this many iterations")
    ap.add_argument("--no-eval", action="store_true", help="do not evaluate kept checkpoints in the background")
    ap.add_argument("--eval-threads", type=int, default=4)
    ap.add_argument("--max-hours", type=float, default=1e9, help="save and exit after this wall time")
    return ap.parse_args(argv)


def load_banks(tasks, bank_dir, per_family=0):
    """Pooled training bank: states, task ids, family ids (first per_family entries of each (task, family))."""
    S, T, F = [], [], []
    for k, t in enumerate(tasks):
        z = np.load(Path(bank_dir) / f"{t}_train.npz")
        fam = z["family"]
        keep = np.ones(len(fam), bool)
        if per_family > 0:
            keep[:] = False
            for f in np.unique(fam):
                keep[np.nonzero(fam == f)[0][:per_family]] = True
        S.append(z["states"][keep]); F.append(fam[keep]); T.append(np.full(keep.sum(), k))
    return np.concatenate(S), np.concatenate(T), np.concatenate(F)


class Trainer:
    def __init__(self, a):
        self.a = a
        self.dev = torch.device(a.device)
        if self.dev.type == "cuda" and self.dev.index is None:      # (not whichever device a simulator makes current)
            self.dev = torch.device("cuda", torch.cuda.current_device())
        self.tasks = resolve_tasks(a.tasks)
        self.run = Path(a.run_dir)
        self.run.mkdir(parents=True, exist_ok=True)
        self.bank_S, self.bank_T, self.bank_F = load_banks(self.tasks, a.bank_dir, a.bank_per_family)
        print(f"[train] {len(self.tasks)} tasks, {len(self.bank_S)} reset-bank entries", flush=True)
        np.random.seed(a.seed); torch.manual_seed(a.seed)
        self.sampler = ResetSampler(len(self.bank_S), a.sampler, a.eps_q, a.history, seed=a.seed)
        self.pts = PointTables(self.tasks, a.n_part, a.n_fix, a.n_grip, device=self.dev)
        self.model = ActorCritic(self.pts.n_pts, init_std=a.init_std, max_std=a.max_std, min_std=a.min_std).to(self.dev)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=a.lr)    # one optimizer for actor and critic
        self.lr, self.it, self.transitions = a.lr, 0, 0
        self._children = []                                             # background evaluations
        self.ecfg = EnvConfig(nthread=a.nthread, cost_coef=a.cost_coef, action_scale_joints=parse_scales(a.action_scale_joints))
        self.args = dict(vars(a), env={k: v for k, v in dataclasses.asdict(self.ecfg).items() if k != "nthread"})
        (self.run / "config.json").write_text(json.dumps(self.args, indent=1))
        self._maybe_resume()
        horizon = int(round(self.ecfg.horizon_s / self.ecfg.control_dt))
        t0 = np.random.default_rng(a.seed + 7).integers(0, horizon, a.envs)   # desynchronized episode clocks
        if a.backend != "mujoco":
            from .rl.vecenv import GpuVecEnv, MultiGpuVecEnv, split_tasks
            devices = a.sim_device.split(",")
            tasks = split_tasks(a.envs, len(self.tasks))
            if len(devices) > 1:
                self.env = MultiGpuVecEnv(a.backend, self.tasks, tasks, self.ecfg, self._provider, t0, devices=devices,
                                          gather=self.dev, **json.loads(a.backend_kw))
            else:
                self.env = GpuVecEnv(a.backend, self.tasks, tasks, self.ecfg, self._provider, t0, device=devices[0],
                                     **json.loads(a.backend_kw))
        elif a.remote_workers > 0:
            from .rl.remote import RemoteVecEnv
            fine = set(self.ecfg.fine_families.split("+"))
            cost = [(FINE_STEP_COST if TASKS[t].family in fine and TASKS[t].family in FINE_STEP_COST else STEP_COST)
                    [TASKS[t].family] for t in self.tasks]
            self.env = RemoteVecEnv(self.run, self.tasks, a.envs, self.ecfg, self._provider, a.remote_workers, t0,
                                    accept_timeout=a.accept_timeout, min_frac=a.min_conn_frac, task_cost=cost,
                                    entry_key=self.bank_T * len(FAMILIES) + self.bank_F,
                                    n_keys=len(self.tasks) * len(FAMILIES))
        else:
            self.env = LocalVecEnv(self.tasks, a.envs, self.ecfg, self._provider, t0)

    # ------------------------------------------------------------- helpers
    def _provider(self, k, finished=None, outcomes=None, tasks=None):
        """Record finished episodes in the sampler, then draw k reset-bank entries (of the given tasks, one each)."""
        if finished is not None and len(finished):
            self.sampler.update(finished, outcomes)
        e = self.sampler.sample(k) if tasks is None else self.sampler.sample_within(tasks, self.bank_T)
        return e, self.bank_S[e], self.bank_T[e]

    def _obs(self, o, ids=None):
        sel = (lambda x: x) if ids is None else (lambda x: x[ids])
        f = lambda k, dt=torch.float32: torch.as_tensor(sel(o[k]), dtype=dt, device=self.dev)
        return dict(prop=f("proprio"), task=f("task", torch.long), tcp_p=f("tcp_p"), tcp_R=f("tcp_R"),
                    part_p=f("part_p"), part_q=f("part_q"), finger=f("finger"))

    def _points(self, ob):
        return self.pts(ob["task"], ob["tcp_p"], ob["tcp_R"], ob["part_p"], ob["part_q"], ob["finger"])

    # ------------------------------------------------------------- checkpoints
    def save(self, keep=False):
        torch.save(self.sampler.state_dict(), self.run / "sampler.pt.tmp"); os.replace(self.run / "sampler.pt.tmp", self.run / "sampler.pt")
        d = dict(model=self.model.state_dict(), opt=self.opt.state_dict(), lr=self.lr, it=self.it,
                 transitions=self.transitions, args=self.args)
        torch.save(d, self.run / "checkpoint.pt.tmp"); os.replace(self.run / "checkpoint.pt.tmp", self.run / "checkpoint.pt")
        if keep:
            ck = self.run / f"model_{self.it:06d}.pt"
            torch.save(dict(model=self.model.state_dict(), it=self.it, transitions=self.transitions, args=self.args), ck)
            if not self.a.no_eval:
                self._evaluate(ck)

    def _evaluate(self, ck):
        """Background evaluation of a kept checkpoint (paper protocol) -> eval/<name>.json."""
        out = self.run / "eval" / f"{ck.stem}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        log = open(out.with_suffix(".log"), "w")
        self._children.append(subprocess.Popen(
            [sys.executable, "-m", "assembly_policy.evaluate", "--policy", str(ck), "--tasks", "+".join(self.tasks),
             "--bank-dir", str(self.a.bank_dir), "--out", str(out), "--nthread", str(self.a.eval_threads),
             "--backend", self.a.eval_backend or self.a.backend,
             "--backend-kw", self.a.eval_backend_kw or (self.a.backend_kw if not self.a.eval_backend else "{}"),
             "--sim-device", self.a.sim_device.split(",")[0]],
            stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT), env={**os.environ, "OMP_NUM_THREADS": "2"}))

    def _maybe_resume(self):
        p = self.run / "checkpoint.pt"
        if not p.exists():
            return
        d = torch.load(p, map_location=self.dev, weights_only=False)
        self.model.load_state_dict(d["model"])
        self.opt.load_state_dict(d["opt"])
        self.model.project_()
        self.lr, self.it, self.transitions = d["lr"], d["it"], d["transitions"]
        sp = self.run / "sampler.pt"
        if sp.exists():
            st = torch.load(sp, weights_only=False)
            if len(st["cnt"]) == self.sampler.M:
                self.sampler.load_state_dict(st)
        self._set_lr()
        print(f"[train] resumed at iteration {self.it}", flush=True)

    def _set_lr(self):
        for g in self.opt.param_groups:
            g["lr"] = self.lr

    # ------------------------------------------------------------- rollout
    def collect(self):
        a, N, T = self.a, self.a.envs, self.a.steps
        rate = a.action_rate_coef if self.it >= a.action_rate_start else 0.0
        buf = {k: [] for k in OBS}
        acts, logps, vals, rews, dones, mus, valids = [], [], [], [], [], [], []
        st = dict(done=0, success=0, safety=0, ep_len=[], fam_n=np.zeros(len(FAMILIES)), fam_s=np.zeros(len(FAMILIES)),
                  task_n=np.zeros(len(self.tasks)), task_s=np.zeros(len(self.tasks)), sim=0.0)
        self.model.eval()
        for t in range(T):
            ob = self._obs(self.env.observe())
            alive = torch.as_tensor(self.env.alive_mask(), device=self.dev)
            with torch.no_grad():
                self.model.norm.update(ob["prop"][alive])
                pts = self._points(ob)
                dist = self.model.dist(pts, ob["prop"])
                act = dist.sample()
                logp = dist.log_prob(act).sum(-1)
                v = self.model.value(pts, ob["prop"])
                # smoothness term; the previous arm action is the newest proprio frame's columns 22..28
                prev = ob["prop"].view(N, HISTORY, PROPRIO_DIM)[:, -1, 22:29]
                rate_pen = (rate * ((act[:, :7] - prev) ** 2).sum(1)).cpu().numpy()
            t1 = time.perf_counter()
            r, done, info, final_obs, finished = self.env.step(act.cpu().numpy())
            st["sim"] += time.perf_counter() - t1
            valid = info["valid"]
            r = r.astype(np.float32) - np.where(valid, rate_pen, 0.0).astype(np.float32)
            ids = np.nonzero(done)[0]
            if len(ids) and info["timeout"][ids].any():   # time-outs bootstrap from the final state's value
                sel = np.nonzero(info["timeout"][ids])[0]
                obf = self._obs(final_obs, sel)
                with torch.no_grad():
                    r[ids[sel]] += a.gamma * self.model.value(self._points(obf), obf["prop"]).cpu().numpy()
            for k in OBS:
                buf[k].append(ob[k])
            acts.append(act); logps.append(logp); vals.append(v); mus.append(dist.mean)
            rews.append(torch.as_tensor(r, device=self.dev))
            # a transition whose worker was lost (or that precedes a worker joining) ends the GAE trace
            dones.append(torch.as_tensor(done | ~valid, device=self.dev)); valids.append(torch.as_tensor(valid, device=self.dev))
            if len(ids):
                succ = info["success"][ids]
                st["done"] += len(ids); st["success"] += int(succ.sum()); st["safety"] += int(info["safety"][ids].sum())
                st["ep_len"] += info["ep_len"][ids].tolist()
                np.add.at(st["fam_n"], self.bank_F[finished], 1); np.add.at(st["fam_s"], self.bank_F[finished], succ)
                np.add.at(st["task_n"], self.bank_T[finished], 1); np.add.at(st["task_s"], self.bank_T[finished], succ)
        ob = self._obs(self.env.observe())
        with torch.no_grad():
            last_v = self.model.value(self._points(ob), ob["prop"])
        batch = {k: torch.stack(v) for k, v in buf.items()}
        batch.update(act=torch.stack(acts), logp=torch.stack(logps), val=torch.stack(vals), mu=torch.stack(mus),
                     rew=torch.stack(rews), done=torch.stack(dones).float(), valid=torch.stack(valids))
        adv = torch.zeros_like(batch["rew"]); gae = torch.zeros(N, device=self.dev)
        for t in reversed(range(T)):                         # GAE
            nv = last_v if t == T - 1 else batch["val"][t + 1]
            nd = 1.0 - batch["done"][t]
            delta = batch["rew"][t] + a.gamma * nv * nd - batch["val"][t]
            gae = delta + a.gamma * a.lam * nd * gae
            gae = torch.where(batch["valid"][t], gae, torch.zeros_like(gae))   # lost transitions do not leak -V
            adv[t] = gae
        batch["ret"], batch["adv"] = adv + batch["val"], adv
        self.transitions += int(batch["valid"].sum())
        return batch, st

    # ------------------------------------------------------------- PPO update
    def update(self, batch):
        a = self.a
        self.model.train()
        n_all = batch["rew"].numel()
        flat = {k: v.reshape(n_all, *v.shape[2:]) for k, v in batch.items()}
        keep = torch.nonzero(flat.pop("valid")).squeeze(1)  # drop transitions of lost/joining workers
        flat = {k: v[keep] for k, v in flat.items()}
        B = len(keep)
        flat["adv"] = (flat["adv"] - flat["adv"].mean()) / (flat["adv"].std() + 1e-8)
        mb = B // a.minibatches
        sigma_old = self.model.std().detach().clone()
        logs = dict(pi=0.0, vf=0.0, ent=0.0, kl=0.0)
        for _ in range(a.epochs):
            perm = torch.randperm(B, device=self.dev)
            for i in range(a.minibatches):
                idx = perm[i * mb:(i + 1) * mb]
                self.opt.zero_grad(set_to_none=True)
                kl = torch.zeros((), device=self.dev)
                for c0 in range(0, len(idx), a.micro_batch):     # gradient accumulation over micro-batches
                    cidx = idx[c0:c0 + a.micro_batch]
                    w = len(cidx) / len(idx)
                    ob = {k: flat[k][cidx] for k in OBS}
                    pts = self._points(ob)
                    d = self.model.dist(pts, ob["prop"])
                    logp = d.log_prob(flat["act"][cidx]).sum(-1)
                    ent = d.entropy().sum(-1).mean()
                    v = self.model.value(pts, ob["prop"])
                    with torch.no_grad():   # analytic KL(old || new) between diagonal Gaussians
                        mu_o, mu_n, sd_n = flat["mu"][cidx], d.mean, d.stddev
                        kl += w * (torch.log(sd_n / sigma_old) + (sigma_old ** 2 + (mu_o - mu_n) ** 2) / (2 * sd_n ** 2) - 0.5).sum(-1).mean()
                    ratio = torch.exp(logp - flat["logp"][cidx])
                    A = flat["adv"][cidx]
                    pi_loss = -torch.min(ratio * A, ratio.clamp(1 - a.clip, 1 + a.clip) * A).mean()
                    v_old, ret = flat["val"][cidx], flat["ret"][cidx]
                    v_clip = v_old + (v - v_old).clamp(-a.clip, a.clip)
                    vf_loss = torch.max((v - ret) ** 2, (v_clip - ret) ** 2).mean()
                    (w * (pi_loss + a.vf_coef * vf_loss - a.ent * ent)).backward()
                    logs["pi"] += w * pi_loss.item(); logs["vf"] += w * vf_loss.item(); logs["ent"] += w * ent.item()
                if kl.item() > 2.0 * a.desired_kl:                # adaptive-KL learning rate (rsl_rl schedule)
                    self.lr = max(1e-5, self.lr / 1.5)
                elif 0.0 < kl.item() < a.desired_kl / 2.0:
                    self.lr = min(1e-3, self.lr * 1.5)
                self._set_lr()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), a.max_grad)
                self.opt.step()
                self.model.project_()
                logs["kl"] += kl.item()
        n = a.epochs * a.minibatches
        return {k: v / n for k, v in logs.items()}

    def train(self):
        a = self.a
        log_f = open(self.run / "log.jsonl", "a")
        t_start = time.time()
        while self.it < a.iters:
            t0 = time.time()
            try:
                batch, st = self.collect()
            except (EOFError, ConnectionError, BrokenPipeError) as e:
                print(f"[train] simulation worker lost ({e!r}); exiting for restart", flush=True)
                raise SystemExit(3)
            t1 = time.time()
            losses = self.update(batch)
            self.it += 1
            n_ep = max(st["done"], 1)
            rec = dict(it=self.it, transitions=self.transitions, fps=a.envs * a.steps / (time.time() - t0),
                       t_sim=st["sim"], t_update=time.time() - t1, lr=self.lr, std=float(self.model.std().mean()),
                       episodes=st["done"], success=st["success"] / n_ep, safety=st["safety"] / n_ep,
                       ep_len=float(np.mean(st["ep_len"])) if st["ep_len"] else 0.0,
                       fam_success={f: (st["fam_s"][i] / st["fam_n"][i] if st["fam_n"][i] else None) for i, f in enumerate(FAMILIES)},
                       task_success={t: (st["task_s"][i] / st["task_n"][i] if st["task_n"][i] else None) for i, t in enumerate(self.tasks)},
                       visited=float((self.sampler.cnt > 0).mean()), **losses)
            if torch.cuda.is_available():     # (per device: torch's reserved memory and the device's used memory, GB)
                rec["gpu_mem"] = [(round(torch.cuda.memory_reserved(d) / 2**30, 2),
                                   round((lambda f: (f[1] - f[0]) / 2**30)(torch.cuda.mem_get_info(d)), 2))
                                  for d in range(torch.cuda.device_count())]
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            if self.it % 10 == 0 or self.it <= 3:
                fam = " ".join(f"{f[:4]}={v:.2f}" for f, v in rec["fam_success"].items() if v is not None)
                print(f"it {self.it} T={self.transitions / 1e6:.1f}M fps={rec['fps']:.0f} train_success={rec['success']:.3f} "
                      f"[{fam}] safety={rec['safety']:.3f} len={rec['ep_len']:.0f} kl={losses['kl']:.4f} lr={self.lr:.1e} "
                      f"std={rec['std']:.3f} sim={st['sim']:.1f}s update={rec['t_update']:.1f}s", flush=True)
            if self.it % a.save_every == 0 or self.it == a.iters:
                self.save(keep=self.it % a.keep_every == 0 or self.it == a.iters)
            if (time.time() - t_start) / 3600 > a.max_hours:
                self.save()
                break


def main():
    tr = Trainer(parse())
    tr.train()
    for p in tr._children:    # let the background evaluations of the last kept checkpoints finish
        p.wait()


if __name__ == "__main__":
    main()

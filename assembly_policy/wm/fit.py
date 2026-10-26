"""Fit the world model (model.py) to logged transitions (collect.py) on k-step autoregressive windows, and measure its
open-loop accuracy on the held-out environments: errors after 1, 8 and 32 control steps by task family and by phase
of the window's start (free, held, near the fixture, inserting), against copying the start state, and whether it
predicts the assembly events (d_asm < 1 mm) the simulator has.

python -m assembly_policy.wm.fit --data data/wm/mmp_4096x1500.npz --out runs/wm/mmp
python -m assembly_policy.wm.fit --data data/wm/mmp.npz data/wm/mjwarp.npz data/wm/mujoco.npz --out runs/wm/all

Several data files (engines): the engine is an input of the model, and the held-out environments of each engine are
evaluated with each engine's context (the diagonal: the model's accuracy; off it: how far the engines' dynamics are
apart, as the model sees them).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from ..evaluate import env_config
from ..sim.scene import build_xml
from ..sim.tasks import TASKS
from .model import D, DELTA_DIM, S, TaskFrames, WorldModel

PHASES = ("free", "held", "near", "insert")


def load(paths, device):
    """The files' environments side by side (shorter logs padded with invalid steps), each with its file's engine;
    term_id[t, n] >= 0: the episode ends with the transition t, in term_state[term_id] (not in state[t + 1]).
    Transitions from or to blown-up states (parts beyond 2 m, 5 m/s or 200 rad/s) are left out."""
    zs = [np.load(p, allow_pickle=True) for p in paths]
    tasks = [str(t) for t in zs[0]["tasks"]]
    assert all([str(t) for t in z["tasks"]] == tasks for z in zs), "the files must have the same tasks"
    T = max(z["action"].shape[0] for z in zs)

    def pad(x, n):
        return np.concatenate([x, np.repeat(x[-1:], n - len(x), 0)]) if len(x) < n else x
    state = np.concatenate([pad(z["state"], T + 1) for z in zs], 1)
    action = np.concatenate([pad(z["action"], T) for z in zs], 1)
    valid = np.concatenate([np.concatenate([z["valid"], np.zeros((T - len(z["valid"]), z["valid"].shape[1]), bool)])
                            for z in zs], 1)
    engine = np.concatenate([np.full(z["task"].shape, k) for k, z in enumerate(zs)])
    sane = lambda x: (np.isfinite(x).all(-1) & (np.abs(x[..., 9:12]) < 2).all(-1)      # (parts flung off by the
                      & (np.linalg.norm(x[..., 25:28], axis=-1) < 5)                    # SDF contacts of MJWarp and
                      & (np.linalg.norm(x[..., 28:31], axis=-1) < 200))                 # MuJoCo; MMP: 0.01%)
    ok = sane(state)
    valid &= ok[:-1] & ok[1:]
    term_id = np.full(valid.shape, -1, np.int64)
    term_state, term_env, n0, m0 = [], [], 0, 0
    for z in zs:
        if "term_t" in z:           # (older logs lack the episodes' last states)
            keep = sane(z["term_state"]) & ok[z["term_t"], z["term_n"] + n0]
            term_id[z["term_t"][keep], z["term_n"][keep] + n0] = m0 + np.nonzero(keep)[0]
            term_state.append(z["term_state"])
            term_env.append(z["term_n"] + n0)
            m0 += len(z["term_t"])
        n0 += z["task"].shape[0]
    term_state = np.concatenate(term_state) if term_state else np.zeros((0, 32), np.float32)
    term_env = np.concatenate(term_env) if term_env else np.zeros(0, np.int64)
    return dict(state=torch.as_tensor(state, device=device), action=torch.as_tensor(action, device=device),
                valid=torch.as_tensor(valid, device=device),
                task=torch.as_tensor(np.concatenate([z["task"] for z in zs]), device=device),
                engine=torch.as_tensor(engine, device=device), test=np.concatenate([z["test"] for z in zs]),
                term_id=torch.as_tensor(term_id, device=device), term_state=torch.as_tensor(np.nan_to_num(term_state), device=device),
                term_env=torch.as_tensor(term_env, device=device),
                tasks=tasks, policy=str(zs[0]["policy"]), engines=[str(z["backend"]) for z in zs])


def runs(valid, ended):
    """run[t, n]: the number of transitions from t on within the episode (its last, `ended`, included)."""
    T = valid.shape[0]
    out = torch.zeros(valid.shape, dtype=torch.int32, device=valid.device)
    nxt = torch.zeros(valid.shape[1], dtype=torch.int32, device=valid.device)
    for t in range(T - 1, -1, -1):
        nxt = torch.where(valid[t], nxt + 1, ended[t].int())
        out[t] = nxt
    return out


def windows(run, envs, k, n, gen):
    """n random (t, env) starts of k-step windows within episodes, among the given environments."""
    ok = (run[:, envs] >= k).nonzero()
    pick = ok[torch.randint(len(ok), (n,), generator=gen, device=ok.device)]
    return pick[:, 0], envs[pick[:, 1]]


def windows_per_env(run, envs, k, gen):
    """One random start of a k-step window within an episode per environment (that has one): (t, env)."""
    ok = run[:, envs] >= k
    c = ok.sum(0)
    has = c > 0
    r = (torch.rand(len(envs), generator=gen, device=run.device) * c).long()
    t = (ok.cumsum(0) > r[None]).int().argmax(0)
    return t[has], envs[has]


def gather(data, ms, t, e, k):
    idx = t[None] + torch.arange(k + 1, device=t.device)[:, None]
    s = ms[idx, e[None]]
    tid = data["term_id"][idx[:-1], e[None]]           # (an episode's last transition ends in its terminal state)
    s[1:] = torch.where(tid[..., None] >= 0, data["term_ms"][tid.clamp(min=0)], s[1:])
    return s, data["action"][idx[:-1], e[None]], data["task"][e], data["engine"][e]


def prepare(paths, dev, features="base"):
    """The logs (load) with their model states: data, frames, ms (T + 1, N, 37) and run (runs)."""
    data = load(paths, dev)
    tasks = data["tasks"]
    ck = torch.load(data["policy"], map_location="cpu", weights_only=False)
    ecfg = env_config(ck["args"])
    frames = TaskFrames(tasks, mujoco.MjModel.from_xml_string(build_xml(tasks[0])), ecfg, dev, len(data["engines"]),
                        features)
    T1 = data["state"].shape[0]
    task = data["task"]
    ms = torch.stack([frames.from_mujoco(data["state"][t], task) for t in range(T1)])     # (T + 1, N, 37)
    ended = data["term_id"] >= 0
    data["term_ms"] = (frames.from_mujoco(data["term_state"], task[data["term_env"]]) if len(data["term_env"])
                       else torch.zeros(1, ms.shape[-1], device=dev))
    data["ecfg"] = ecfg
    return data, frames, ms, runs(data["valid"], ended)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, nargs="+", help="collect.py logs (one per engine)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8, help="autoregressive window of the fit")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", default="1024,1024,1024")
    ap.add_argument("--eval-windows", type=int, default=40000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cross", action="store_true", help="also evaluate each engine's data in the others' contexts")
    ap.add_argument("--features", default="geo", choices=("base", "geo", "geo2", "geo3"))
    ap.add_argument("--prior", type=int, default=1, help="predict the positions' change beyond constant velocity")
    ap.add_argument("--loss", default="mse", choices=("mse", "huber"))
    a = ap.parse_args()
    dev = torch.device(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator(device=dev).manual_seed(0)
    data, frames, ms, run = prepare(a.data, dev, a.features)
    tasks = data["tasks"]
    T1, N = data["state"].shape[:2]
    test = torch.as_tensor(np.nonzero(data["test"])[0], device=dev)
    train = torch.as_tensor(np.nonzero(~data["test"])[0], device=dev)
    print(f"{T1 - 1} steps x {N} envs ({', '.join(data['engines'])}); train {len(train)}, test {len(test)} envs; "
          f"{int(data['valid'].sum()) + int((data['term_id'] >= 0).sum())} transitions "
          f"({int((data['term_id'] >= 0).sum())} ending episodes)", flush=True)

    # normalizers from one-step transitions of the training environments
    t, e = windows(run, train, 1, 200000, gen)
    s, act, tk, en = gather(data, ms, t, e, 1)
    model = WorldModel(frames, frames.features(s[0], act[0], tk, en).shape[-1],
                       [int(h) for h in a.hidden.split(",")], a.prior).to(dev)
    model.inp.fit(frames.features(s[0], act[0], tk, en))
    model.out.fit(frames.delta(s[0], s[1]) - model.prior(s[0]))
    opt = torch.optim.Adam(model.net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=a.lr * 0.05)
    t0 = time.time()
    for step in range(a.steps + 1):
        t, e = windows(run, train, a.k, a.batch, gen)
        s, act, tk, en = gather(data, ms, t, e, a.k)
        pred = [s[0]]
        loss = 0.0
        for j in range(a.k):
            pred.append(model(pred[-1], act[j], tk, en))
            err = frames.delta(s[j + 1], pred[-1]) / model.out.std       # (in units of one step's typical change)
            loss = loss + (torch.nn.functional.huber_loss(err, torch.zeros_like(err), delta=1.0) if a.loss == "huber"
                           else (err ** 2).mean())
        loss = loss / a.k
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0:
            print(f"step {step}: loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(dict(model=model.state_dict(), tasks=tasks, hidden=a.hidden, data=a.data, engines=data["engines"],
                    features=a.features, prior=a.prior, policy=data["policy"]), out / "world_model.pt")
    report = {}
    engines = data["engines"]
    for i, name in enumerate(engines):
        own = test[data["engine"][test] == i]
        for j, ctx in enumerate(engines):
            if j != i and len(engines) > 1 and not a.cross:
                continue
            print(f"\n##### {name} data, {ctx} context", flush=True)
            c = j if len(engines) > 1 else None
            report[f"{name}|{ctx}"] = evaluate(lambda t, e, s, act, tk, c=c: model.rollout(
                s[0], act, tk, None if c is None else torch.full_like(e, c)), frames, data, ms, run, own, a.eval_windows)
    (out / "accuracy.json").write_text(json.dumps(report, indent=1))


def phase_of(frames, s0, tk):
    """The phase of states (B, 37): 3 inserting (d_asm < 5 mm), 2 near the fixture (< 30 mm), 1 held, 0 free."""
    fk = frames.fk.forward(s0[:, S["q"]])
    G, gp = frames.goal_R[tk], frames.goal_p[tk]
    tcp = (G.transpose(-1, -2) @ (fk["tcp_p"] - gp)[..., None])[..., 0]
    held = ((s0[:, S["p"]] - tcp).norm(dim=-1) < 0.06) & (s0[:, S["f"]].mean(-1) < 0.035)
    d0 = frames.d_asm(s0, tk)
    return torch.where(d0 < 0.005, 3, torch.where(d0 < 0.03, 2, torch.where(held, 1, 0)))


@torch.no_grad()
def evaluate(predict, frames, data, ms, run, test, n, seed=0):
    """Open-loop errors on held-out windows, by family and phase, after 1, 8 and 32 steps: predict(t, e, s, act, tk)
    -> (k + 1, B, 37) the predicted states of the windows (t, e). Rounds of one window per test environment, the same
    for the same test environments and seed (whatever predicts)."""
    tasks = data["tasks"]
    fams = sorted({TASKS[t].family for t in tasks})
    fam_of = torch.as_tensor([fams.index(TASKS[t].family) for t in tasks], device=ms.device)
    gen = torch.Generator(device=ms.device).manual_seed(seed)
    report = {}
    for H in (1, 8, 32):
        rows = []
        for _ in range(-(-n // len(test))):
            t, e = windows_per_env(run, test, H, gen)
            s, act, tk, _ = gather(data, ms, t, e, H)
            pred = predict(t, e, s, act, tk)
            true_d = frames.d_asm(s, tk)
            pred_d = frames.d_asm(pred, tk)
            phase = phase_of(frames, s[0], tk)
            dp = (pred[-1][:, S["p"]] - s[-1][:, S["p"]]).norm(dim=-1) * 1e3
            dp0 = (s[0][:, S["p"]] - s[-1][:, S["p"]]).norm(dim=-1) * 1e3
            Rp = pred[-1][:, S["R"]].reshape(-1, 3, 3)
            Rt = s[-1][:, S["R"]].reshape(-1, 3, 3)
            R0 = s[0][:, S["R"]].reshape(-1, 3, 3)
            ang = lambda A, B: torch.rad2deg(torch.acos(((A.transpose(-1, -2) @ B).diagonal(dim1=-2, dim2=-1).sum(-1) - 1)
                                                        .div(2).clamp(-1, 1)))
            rows.append(torch.stack([fam_of[tk].float(), phase.float(), dp, dp0, ang(Rp, Rt), ang(R0, Rt),
                                     (pred[-1][:, S["q"]] - s[-1][:, S["q"]]).abs().amax(-1) * 1e3,
                                     (pred[-1][:, S["f"]] - s[-1][:, S["f"]]).abs().amax(-1) * 1e3,
                                     (true_d.amin(0) < 0.001).float(), (pred_d.amin(0) < 0.001).float(),
                                     (true_d[-1] - pred_d[-1]).abs() * 1e3], -1))
        r = torch.cat(rows).cpu().numpy()
        out = {}
        print(f"\n=== horizon {H} steps ({H * 0.04:.2f} s): median / 90th percentile; copy = the start state's error",
              flush=True)
        print(f"{'family':10s} {'phase':7s} {'n':>6s}  {'part mm':>13s} {'copy mm':>9s}  {'part deg':>13s} "
              f"{'copy deg':>9s}  {'arm mrad':>13s} {'finger mm':>13s} {'|d_asm| mm':>12s}  {'events (recall/prec)':>20s}")
        for fi, fam in enumerate(fams + ["all"]):
            for ph in range(5):
                m = (r[:, 0] == fi if fam != "all" else np.ones(len(r), bool)) & (r[:, 1] == ph if ph < 4 else True)
                if m.sum() < 20:
                    continue
                q = lambda col: (np.nanmedian(r[m, col]), np.nanpercentile(r[m, col], 90))   # (blown-up replays: nan)
                ev_t, ev_p = r[m, 8] > 0, r[m, 9] > 0
                rec = (ev_t & ev_p).sum() / max(ev_t.sum(), 1)
                prec = (ev_t & ev_p).sum() / max(ev_p.sum(), 1)
                name = PHASES[ph] if ph < 4 else "any"
                vals = dict(n=int(m.sum()), part_mm=q(2), copy_mm=q(3), part_deg=q(4), copy_deg=q(5), arm_mrad=q(6),
                            finger_mm=q(7), d_asm_mm=q(10), events=int(ev_t.sum()), recall=float(rec), precision=float(prec))
                out[f"{fam}/{name}"] = {k: (list(map(float, v)) if isinstance(v, tuple) else v) for k, v in vals.items()}
                print(f"{fam:10s} {name:7s} {m.sum():6d}  {q(2)[0]:6.3f}/{q(2)[1]:6.2f} {q(3)[0]:9.2f}  "
                      f"{q(4)[0]:6.2f}/{q(4)[1]:6.2f} {q(5)[0]:9.2f}  {q(6)[0]:6.1f}/{q(6)[1]:6.1f} "
                      f"{q(7)[0]:6.3f}/{q(7)[1]:6.2f} {q(10)[0]:5.2f}/{q(10)[1]:6.2f}  "
                      f"{ev_t.sum():5d} {rec:5.2f}/{prec:5.2f}", flush=True)
        report[H] = out
    return report


if __name__ == "__main__":
    main()

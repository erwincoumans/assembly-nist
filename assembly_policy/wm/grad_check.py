"""Whether the world model's gradients are the engine's, for SHAC-style training that steps the engine forward and
differentiates through the model at the engine's states (so that the model's errors do not compound): directional
derivatives of the states after h = 1, 2, 4, 8 steps along random perturbations of the arm actions, by central
differences of a deterministic engine at several step sizes (small: the engine's own derivative, large: smoothed
over contact switches) against the model's (its own central differences, at the same and a tiny step).

python -m assembly_policy.wm.grad_check --data data/wm/mjwarp.npz --model runs/wm/abl/C_geo_prior/world_model.pt \\
    --out runs/wm/abl/C_geo_prior/grad_check.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..sim.gpu import make_env
from .backend import load_world_model
from .fit import PHASES, gather, phase_of, prepare, windows_per_env
from .model import S, so3_log

HORIZONS = (1, 2, 4, 8)


def derivatives(traj, eps):
    """traj (H + 1, V, W, 37) with variants [base, +eps0, -eps0, +eps1, ...] -> per eps: dp (H, W, 3), drot, dq."""
    out = []
    for k, e in enumerate(eps):
        sp, sm = traj[:, 1 + 2 * k], traj[:, 2 + 2 * k]
        Rp = sp[..., S["R"]].reshape(sp.shape[:-1] + (3, 3))
        Rm = sm[..., S["R"]].reshape(sm.shape[:-1] + (3, 3))
        out.append(dict(p=(sp[..., S["p"]] - sm[..., S["p"]]) / (2 * e),
                        rot=so3_log(Rp @ Rm.transpose(-1, -2)) / (2 * e),
                        q=(sp[..., S["q"]] - sm[..., S["q"]]) / (2 * e)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/wm/mjwarp.npz", help="a collect.py log (its held-out windows)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", default="mjwarp", help="the (deterministic) engine")
    ap.add_argument("--engine", default=None, help="the engine context of a multi-engine model")
    ap.add_argument("--eps", default="0.01,0.05,0.2", help="engine step sizes (action units)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device(a.device)
    eps = [float(x) for x in a.eps.split(",")]
    data, _, ms, run = prepare([a.data], dev)
    model, ck = load_world_model(a.model, dev)
    frames = model.frames
    H = max(HORIZONS)
    test = torch.as_tensor(np.nonzero(data["test"])[0], device=dev)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    t, e = windows_per_env(run, test, H, gen)
    s, act, tk, _ = gather(data, ms, t, e, H)            # (H + 1, W, 37), (H, W, 8)
    W = len(e)
    delta = torch.randn(H, W, 8, generator=gen, device=dev)
    delta[..., 7] = 0                                    # (the gripper's latch is not differentiable)
    phase = phase_of(frames, s[0], tk).cpu().numpy()

    def variants(eps_list):
        A = [act] + [act + sg * x * delta for x in eps_list for sg in (1.0, -1.0)]
        return torch.stack(A, 1)                         # (H, V, W, 8)

    # the engine: every variant of every window in its own environment
    A = variants(eps)
    V = A.shape[1]
    task_of = data["task"][e].repeat(V).cpu().numpy()
    env = make_env(a.backend, data["tasks"], task_of, data["ecfg"], device=a.device)
    x0 = data["state"][t, e]
    env.set_states(np.arange(V * W), np.concatenate([np.zeros((V * W, 1), np.float32),
                                                     x0[:, :31].repeat(V, 1).cpu().numpy()], 1), task_of)
    env.grip_cmd[:] = x0[:, 31].repeat(V).to(env.grip_cmd.device)
    tk_all = tk.repeat(V)
    traj = [s[0][None].expand(V, W, -1)]
    for j in range(H):
        env.step(A[j].reshape(V * W, 8))
        x = torch.cat([env.backend.qpos, env.backend.qvel, env.grip_cmd[:, None]], 1).float()
        traj.append(frames.from_mujoco(x, tk_all).reshape(V, W, -1))
    env.close()
    g_engine = derivatives(torch.stack(traj), eps)

    # the model: the same variants and a tiny step
    eps_m = eps + [1e-3]
    A = variants(eps_m)
    Vm = A.shape[1]
    en = None
    if frames.n_engines > 1:
        en = torch.full((Vm * W,), list(ck["engines"]).index(a.engine or a.backend), device=dev)
    with torch.no_grad():
        pred = model.rollout(s[0].repeat(Vm, 1), A.reshape(H, Vm * W, 8), tk.repeat(Vm), en)
    g_model = derivatives(pred.reshape(H + 1, Vm, W, -1), eps_m)

    cos = lambda x, y: (x * y).sum(-1) / (x.norm(dim=-1) * y.norm(dim=-1)).clamp(min=1e-12)
    report = {}
    print(f"{W} windows; cosine similarity of the directional derivatives (median), |model| / |engine| (median)")
    for h in HORIZONS:
        for ke, xe in enumerate(eps):
            ge = g_engine[ke]
            rows = {}
            for name, gm in (("same step", g_model[ke]), ("tiny step", g_model[-1])):
                for qty in ("p", "rot", "q"):
                    c = cos(gm[qty][h], ge[qty][h]).cpu().numpy()
                    r = (gm[qty][h].norm(dim=-1) / ge[qty][h].norm(dim=-1).clamp(min=1e-9)).cpu().numpy()
                    for ph in range(5):
                        m = phase == ph if ph < 4 else np.ones(W, bool)
                        if m.sum() < 10:
                            continue
                        rows[f"{name}/{qty}/{PHASES[ph] if ph < 4 else 'any'}"] = dict(
                            cos=float(np.median(c[m])), ratio=float(np.median(r[m])), n=int(m.sum()))
            # the engine against itself: this step size against the smallest
            for qty in ("p", "rot", "q"):
                c = cos(ge[qty][h], g_engine[0][qty][h]).cpu().numpy()
                rows[f"engine vs smallest/{qty}/any"] = dict(cos=float(np.median(c)), n=W)
            report[f"h{h}/eps{xe}"] = rows
            line = "  ".join(f"{qty} {rows[f'same step/{qty}/any']['cos']:5.2f} ({rows[f'same step/{qty}/any']['ratio']:4.2f})"
                             f" tiny {rows[f'tiny step/{qty}/any']['cos']:5.2f}" for qty in ("p", "rot", "q"))
            print(f"h {h} eps {xe:5.2f}: {line}   engine~smallest p {rows['engine vs smallest/p/any']['cos']:5.2f}",
                  flush=True)
    print("by phase (h 1 / h 8, part position, same step):")
    for ke, xe in enumerate(eps):
        print(f"  eps {xe:5.2f}: " + "  ".join(
            f"{PHASES[ph]} {report[f'h1/eps{xe}'].get(f'same step/p/{PHASES[ph]}', {}).get('cos', float('nan')):5.2f}/"
            f"{report[f'h8/eps{xe}'].get(f'same step/p/{PHASES[ph]}', {}).get('cos', float('nan')):5.2f}"
            for ph in range(4)))
    print("median |d p / d action| (mm per action unit), h 1 / h 8, engine at eps 0.05 | model:")
    k = min(range(len(eps)), key=lambda i: abs(eps[i] - 0.05))
    for ph in range(4):
        m = torch.as_tensor(phase == ph, device=dev)
        if m.sum() < 10:
            continue
        n = lambda g, h: float(g["p"][h][m].norm(dim=-1).median() * 1e3)
        print(f"  {PHASES[ph]:7s} {n(g_engine[k], 1):7.3f} / {n(g_engine[k], 8):7.3f} | {n(g_model[k], 1):7.3f} / "
              f"{n(g_model[k], 8):7.3f}")
        report[f"norm/{PHASES[ph]}"] = dict(engine=[n(g_engine[k], 1), n(g_engine[k], 8)],
                                            model=[n(g_model[k], 1), n(g_model[k], 8)])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(model=a.model, data=a.data, backend=a.backend, eps=eps, report=report), indent=1))


if __name__ == "__main__":
    main()

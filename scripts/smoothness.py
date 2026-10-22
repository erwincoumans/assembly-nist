"""How jerky is the arm? Rolls out loose starts of a few tasks with sampled (evaluation) and mean (deterministic)
actions and reports success plus smoothness: mean |delta a| between control steps (arm dims), mean joint speed,
mean |joint acceleration| (finite difference of joint velocity across 40 ms steps) and the mean number of
joint-velocity sign reversals per joint per second (back-and-forth chatter).
python scripts/smoothness.py <ckpt> <bank dir> [tasks R8,P8,Gm,BNC] [n per task] [family loose]"""
import sys as _s; from pathlib import Path as _P; _s.path.insert(0, str(_P(__file__).resolve().parents[1]))
import sys
import numpy as np
import torch
from assembly_policy.evaluate import TorchPolicy, env_config
from assembly_policy.sim.resets import FAMILIES
from assembly_policy.sim.env import BatchedAssemblyEnv
from assembly_policy.sim.tasks import ALL_TASKS

ckpt, bank = sys.argv[1:3]
run_tasks = (sys.argv[3] if len(sys.argv) > 3 else "R8,P8,Gm,BNC").split(",")
n = int(sys.argv[4]) if len(sys.argv) > 4 else 64
fam = sys.argv[5] if len(sys.argv) > 5 else "loose"
tasks = list(ALL_TASKS)
S, ids = [], []
for t in run_tasks:
    z = np.load(f"{bank}/{t}_eval.npz")
    s = z["states"][z["family"] == FAMILIES.index(fam)][:n]
    S.append(s); ids += [tasks.index(t)] * len(s)
S = np.concatenate(S); ids = np.asarray(ids)
for det in (False, True):
    torch.manual_seed(0)
    pol = TorchPolicy(ckpt, tasks, "cpu", deterministic=det)
    env = BatchedAssemblyEnv(tasks, ids, env_config(pol.args, 8))
    env.set_states(np.arange(len(S)), S, ids)
    N = len(S); done_at = np.full(N, -1); succ = np.zeros(N, bool)
    da, sp, acc, rev, cnt = 0.0, 0.0, 0.0, 0.0, 0
    prev_a = None; prev_qd = env.state[:, env.sqd].copy()
    for t in range(env.horizon):
        a = np.asarray(pol(env))
        _, done, info = env.step(a)
        live = done_at < 0
        qd = env.state[:, env.sqd]
        if prev_a is not None and live.any():
            da += np.abs(a[live, :7] - prev_a[live, :7]).mean() * live.sum()
            sp += np.abs(qd[live]).mean() * live.sum()
            acc += (np.abs(qd[live] - prev_qd[live]) / env.ecfg.control_dt).mean() * live.sum()
            rev += ((np.sign(qd[live]) != np.sign(prev_qd[live])) & (np.abs(qd[live]) > 0.02)).mean() * live.sum()
            cnt += live.sum()
        prev_a, prev_qd = a, qd.copy()
        nd = live & done
        succ[nd & info["success"]] = True
        done_at[nd] = t
        if (done_at >= 0).all():
            break
    print(f"{'mean' if det else 'sampled'} actions ({fam}, {','.join(run_tasks)}; {N} episodes): success {succ.mean():.2f} | "
          f"|delta a| {da / cnt:.3f}, |qd| {sp / cnt:.3f} rad/s, |qdd| {acc / cnt:.2f} rad/s^2, "
          f"velocity reversals {rev / cnt / env.ecfg.control_dt:.2f} per joint per s", flush=True)
    for t in run_tasks:
        m = ids == tasks.index(t)
        print(f"   {t}: success {succ[m].mean():.2f}")

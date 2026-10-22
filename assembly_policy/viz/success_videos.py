"""Success showcase video for a checkpoint: for every task, roll out the stochastic policy from loose starts
(part resting on the table, gripper away; the hardest reset family) until one episode succeeds, up to
--trials starts per task, and show that episode from its first frame. If no start succeeds, the last trial
is shown instead and labelled as such. Played at 30 fps in real time by default (--speed 1; every control
step is recorded and output frames repeat the latest step), or sped up with --speed k.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw

from ..evaluate import TorchPolicy, env_config
from ..sim.resets import FAMILIES
from ..sim.scene import build_xml
from ..sim.env import STATE_SPEC, BatchedAssemblyEnv, EnvConfig
from ..sim.tasks import BANK_DIR, TASKS, resolve_tasks, scene_pose

PLAY_FPS, HOLD_S = 30, 1.0


def loose_starts(task, bank_dir, n, rng, generate=32):
    """n loose-family start states (eval bank first, then training bank, without repetition). Without reset banks,
    `generate` fresh loose states are drawn with the reset generator instead (a few seconds per task)."""
    pool = []
    for split in ("eval", "train"):
        p = Path(bank_dir) / f"{task}_{split}.npz"
        if p.exists():
            z = np.load(p)
            pool.append(z["states"][z["family"] == FAMILIES.index("loose")])
    if not pool:
        from ..sim.resets import ResetGenerator
        from ..sim.scene import PhysicsConfig
        g = ResetGenerator(task, PhysicsConfig(fine_families=EnvConfig.fine_families), seed=int(rng.integers(2 ** 31)))
        pool.append(g.generate("loose", min(n, generate))[0])
    pool = np.concatenate(pool)
    return pool[rng.permutation(len(pool))[:n]]


def search(pol, tasks, bank_dir, trials, batch, nthread, seed, speed=1.0):
    """Per task: (frames, trial index, success, success time) of the first successful loose episode."""
    rng = np.random.default_rng(seed)
    starts = {t: loose_starts(t, bank_dir, trials, rng) for t in tasks}
    result, tried = {}, {t: 0 for t in tasks}
    every = max(1, int(1.0 / (PLAY_FPS / speed * EnvConfig().control_dt)))   # control steps per recorded frame
    while True:
        active = [t for t in tasks if t not in result and tried[t] < len(starts[t])]
        if not active:
            break
        ids, S, slot = [], [], []
        for t in active:
            k = min(batch, len(starts[t]) - tried[t])
            S.append(starts[t][tried[t]:tried[t] + k]); ids += [tasks.index(t)] * k
            slot += [(t, tried[t] + j) for j in range(k)]
        S, ids = np.concatenate(S), np.asarray(ids)
        env = BatchedAssemblyEnv(tasks, ids, env_config(pol.args, nthread))
        env.set_states(np.arange(len(ids)), S, ids)
        frames = [env.state.copy()]
        done_at = np.full(len(ids), -1); succ = np.zeros(len(ids), bool)
        for step in range(env.horizon):
            _, done, info = env.step(pol(env))
            new = (done_at < 0) & done
            done_at[new] = step; succ[new] = info["success"][new]
            if (step + 1) % every == 0:
                frames.append(env.state.copy())
            if (done_at >= 0).all():
                break
        frames.append(env.state.copy())
        frames = np.stack(frames)                     # (n_frames, n_envs, state)
        for i, (t, trial) in enumerate(slot):
            end = -(-(done_at[i] + 1) // every) if done_at[i] >= 0 else len(frames) - 1   # first frame at/after the end
            ep = frames[:end + 1, i]
            if succ[i] and t not in result:
                result[t] = dict(frames=ep, trial=trial + 1, success=True, t_succ=(done_at[i] + 1) * env.ecfg.control_dt)
        for t in active:
            tried[t] = min(tried[t] + batch, len(starts[t]))
            if t not in result and tried[t] >= len(starts[t]):
                # last trial of this task (the final slot of this task in the batch)
                i = max(j for j, (tt, _) in enumerate(slot) if tt == t)
                end = len(frames) - 1 if done_at[i] < 0 else -(-(done_at[i] + 1) // every)
                result[t] = dict(frames=frames[:end + 1, i], trial=slot[i][1] + 1, success=bool(succ[i]),
                                 t_succ=(done_at[i] + 1) * env.ecfg.control_dt if succ[i] else None)
        env.close() if hasattr(env, "close") else None
        print(f"searched: " + " ".join(f"{t}:{tried[t]}{'*' if t in result and result[t]['success'] else ''}" for t in tasks), flush=True)
    return result, every


def render(result, tasks, it, trials, out, width, height, every=1, speed=1.0):
    vis = {t: mujoco.MjModel.from_xml_string(build_xml(t, visual=True)) for t in tasks}
    vd = {t: mujoco.MjData(m) for t, m in vis.items()}
    rend = {t: mujoco.Renderer(m, height, width) for t, m in vis.items()}
    cols = int(np.ceil(np.sqrt(len(tasks) * 4 / 3))); rows = int(np.ceil(len(tasks) / cols))
    dt_rec = every * EnvConfig().control_dt                       # simulated seconds per recorded frame
    t_end = max(len(r["frames"]) - 1 for r in result.values()) * dt_rec + HOLD_S * speed
    n_frames = int(t_end * PLAY_FPS / speed) + 1
    cams = {}
    for t in tasks:  # frame the part's start position and the target fixture
        m, d = vis[t], vd[t]
        mujoco.mj_setState(m, d, result[t]["frames"][0], STATE_SPEC); mujoco.mj_forward(m, d)
        part = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "part")].copy()
        fix = scene_pose(TASKS[t].fixture_scene)[0]
        c = mujoco.MjvCamera()
        c.lookat[:] = 0.5 * (part + fix) + [0.0, 0.0, 0.02]
        c.distance = float(np.clip(1.3 * np.linalg.norm(part - fix) + 0.3, 0.35, 0.95))
        c.azimuth, c.elevation = 150.0, -35.0
        cams[t] = c
    video = []
    for f in range(n_frames):
        tiles = []
        for t in tasks:
            r = result[t]; k = min(int(f * speed / PLAY_FPS / dt_rec), len(r["frames"]) - 1)
            m, d = vis[t], vd[t]
            mujoco.mj_setState(m, d, r["frames"][k], STATE_SPEC); mujoco.mj_forward(m, d)
            rend[t].update_scene(d, cams[t])
            img = Image.fromarray(rend[t].render()).convert("RGBA")
            over = Image.new("RGBA", img.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(over)
            l1 = f"{t} | loose start | it {it}"
            l2 = (f"trial {r['trial']}/{trials}: success at {r['t_succ']:.1f} s" if r["success"]
                  else f"no success in {r['trial']} trials (showing last)")
            w = int(max(dr.textlength(l1), dr.textlength(l2))) + 10
            dr.rectangle([0, 0, w, 30], fill=(15, 17, 22, 120))
            dr.text((5, 3), l1, fill=(255, 255, 255, 255))
            dr.text((5, 16), l2, fill=(120, 255, 120, 255) if r["success"] else (255, 190, 130, 255))
            if k == len(r["frames"]) - 1:
                msg = "SUCCESS" if r["success"] else "FAILED (last trial)"
                bw = int(dr.textlength(msg)) + 16; y0 = img.size[1] - 26
                dr.rectangle([0, y0, bw, y0 + 22], fill=(20, 140, 60, 200) if r["success"] else (170, 40, 40, 200))
                dr.text((8, y0 + 5), msg, fill=(255, 255, 255, 255))
            tiles.append(np.asarray(Image.alpha_composite(img, over).convert("RGB")))
        while len(tiles) < rows * cols:
            tiles.append(np.zeros_like(tiles[0]))
        video.append(np.concatenate([np.concatenate(tiles[r_ * cols:(r_ + 1) * cols], 1) for r_ in range(rows)], 0))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, video, fps=PLAY_FPS, codec="libx264", quality=6, macro_block_size=8)
    return len(video)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--tasks", default="all20")
    ap.add_argument("--bank-dir", default=str(BANK_DIR)); ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=500); ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--nthread", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--width", type=int, default=480); ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true", help="mean actions (default: stochastic, as in evaluation)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed-up (1 = real time)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    tasks = resolve_tasks(a.tasks)
    torch.manual_seed(a.seed)
    pol = TorchPolicy(a.ckpt, tasks, a.device, deterministic=a.deterministic)
    res, every = search(pol, tasks, a.bank_dir, a.trials, a.batch, a.nthread, a.seed, a.speed)
    n = render(res, tasks, pol.it, a.trials, a.out, a.width, a.height, every, a.speed)
    summary = {t: dict(success=r["success"], trial=r["trial"], t_succ=r["t_succ"]) for t, r in res.items()}
    Path(a.out).with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {a.out}: {n} frames; " + " ".join(f"{t}={'S@' + str(r['trial']) if r['success'] else 'F'}" for t, r in res.items()))


if __name__ == "__main__":
    main()

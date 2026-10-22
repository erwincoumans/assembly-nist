"""Sequential multi-part demo: all 20 parts lie at random settled loose poses
on the board, a random order is drawn, and the policy (mean actions by default) assembles one part at a time with
a longer budget than the 15 s training horizon (--per-part-s, default 30 s) so that a failed attempt can be
recovered. Between parts a scripted move opens the gripper, lifts the hand and brings it above the next part.

Physics: each stage simulates the training scene of the active task (robot, active part, its exact fixture, the
other fixtures as convex obstacles), starting from the robot state left by the previous stage. The other parts are
therefore not physical during a stage; they are drawn at their resting (or final) poses, and their start poses are
chosen so that no two parts and no part and fixture overlap on the board. Rendering (original asset meshes): one
fixed camera plus a close-up inset (top right) that follows the gripper without rotating with it, 25 fps; policy
segments play --slow times slower (interpolated) and the scripted moves --fast times faster.

python -m assembly_policy.viz.demo --ckpt checkpoints/e8192_it12650.pt --out demo.mp4 [--seed 0] [--pin 2:N16+3:connector]
"""
from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull

from ..evaluate import TorchPolicy, env_config
from .success_videos import loose_starts
from ..sim.robot import Indices, axis_angle_quat, mat_to_quat, quat_mul, quat_to_mat, solve_ik
from ..sim.scene import _add, build_xml, hull_vertices
from ..sim.env import BatchedAssemblyEnv
from ..sim.tasks import ALL_TASKS, BANK_DIR, BOARD_FIXTURES, TASKS, load_obj, scene_pose

FPS = 25
NAMES = {"R4": "4 mm round peg", "R8": "8 mm round peg", "R12": "12 mm round peg", "R16": "16 mm round peg",
         "P4": "4 mm rectangular peg", "P8": "8 mm rectangular peg", "P12": "12 mm rectangular peg",
         "P16": "16 mm rectangular peg", "Gs": "small gear", "Gm": "medium gear", "Gl": "large gear",
         "N4": "M4 nut", "N8": "M8 nut", "N12": "M12 nut", "N16": "M16 nut", "BNC": "BNC connector",
         "DSUB": "D-sub connector", "RJ45": "RJ45 connector", "USB": "USB-A connector", "WP": "waterproof connector"}
PART_RGBA = [1.0, 0.5, 0.05, 1]
FIX_RGBA = [0.15, 0.35, 0.85, 1]


# ------------------------------------------------------------------ layout
def _polygon(points_xy):
    p = np.asarray(points_xy)[:, :2]
    return p[ConvexHull(p).vertices]


def _separated(a, b, margin):
    """Separating-axis test for two convex polygons with a clearance margin."""
    for poly in (a, b):
        e = np.roll(poly, -1, 0) - poly
        normals = np.c_[e[:, 1], -e[:, 0]] / np.linalg.norm(e, axis=1, keepdims=True)
        pa, pb = a @ normals.T, b @ normals.T
        if ((pa.max(0) + margin < pb.min(0)) | (pb.max(0) + margin < pa.min(0))).any():
            return True
    return False


def part_footprint(task, pos, quat):
    v = hull_vertices(TASKS[task].part_asset)
    return _polygon(pos + v @ quat_to_mat(quat).T)


def fixture_footprints():
    out = []
    for prim, stem in BOARD_FIXTURES.items():
        pos, quat = scene_pose(prim)
        out.append(_polygon(pos + hull_vertices(stem) @ quat_to_mat(quat).T))
    return out


def choose_layout(tasks, bank_dir, sp, rng, margin=0.006):
    """One settled loose start per task such that no part touches another part or any fixture on the board."""
    fixtures = fixture_footprints()
    chosen, polys = {}, []
    for t in rng.permutation(tasks):
        pool = loose_starts(t, bank_dir, 4096, rng)
        for m in (margin, margin / 2, 0.0):
            for S in pool:
                pos, quat = S[sp[:3]], S[sp[3:7]]
                poly = part_footprint(t, pos, quat)
                if all(_separated(poly, f, m) for f in fixtures) and all(_separated(poly, q, m) for q in polys):
                    chosen[t] = S.copy(); polys.append(poly)
                    break
            if t in chosen:
                break
        if t not in chosen:      # board too crowded: accept the start with the fewest overlaps
            chosen[t] = pool[0].copy(); polys.append(part_footprint(t, pool[0][sp[:3]], pool[0][sp[3:7]]))
            print(f"[layout] {t}: no overlap-free start found", flush=True)
    return chosen


def make_order(tasks, pin, rng):
    """Random assembly order with some positions pinned ('2:N16+3:connector', 1-based; a family name picks a
    random task of that family)."""
    order = [None] * len(tasks)
    left = [str(t) for t in rng.permutation(tasks)]
    for item in filter(None, pin.replace(",", "+").split("+")):
        pos, what = item.split(":")
        cands = [t for t in left if t == what or TASKS[t].family == what]
        assert cands, f"nothing left to pin for {item!r}"
        order[int(pos) - 1] = cands[0]; left.remove(cands[0])
    return [t if t is not None else left.pop(0) for t in order]


# ------------------------------------------------------------------ visual model
def build_demo_xml(tasks):
    """Robot, board, every fixture and every part (mocap bodies), drawn with the original asset meshes (OBJ,
    in the part / fixture frame): the simulation's collision models are convex proxies plus thread / connector
    SDFs, and a nut's convex hex proxy would hide its bore. No collisions in this model (rendering only)."""
    root = ET.fromstring(build_xml("R4", visual=True))
    world, asset = root.find("worldbody"), root.find("asset")
    for el in list(world):
        if el.get("name", "").startswith(("obst_", "vis_obst_")) or el.get("name") in ("fixture", "part"):
            world.remove(el)
    off = dict(contype="0", conaffinity="0", group="2")

    def mesh(stem):
        name = f"vis_{stem}"
        if asset.find(f"mesh[@name='{name}']") is None:
            v, f = load_obj(stem)
            _add(asset, "mesh", name=name, vertex=v.ravel(), face=f.ravel())
        return name

    for prim, stem in sorted(BOARD_FIXTURES.items()):
        pos, quat = scene_pose(prim)
        body = _add(world, "body", name=f"fixture_{prim}", pos=pos, quat=quat)
        _add(body, "geom", name=f"fixg_{prim}", type="mesh", mesh=mesh(stem), rgba=FIX_RGBA, **off)
    for t in tasks:
        body = _add(world, "body", name=f"part_{t}", mocap="true", pos=[0, 0, -1])
        _add(body, "geom", name=f"partg_{t}", type="mesh", mesh=mesh(TASKS[t].part_asset), rgba=PART_RGBA, **off)
    root.find("visual").find("global").attrib.update(offwidth="1920", offheight="1080")
    return ET.tostring(root, encoding="unicode")


# ------------------------------------------------------------------ simulation
class Demo:
    def __init__(self, a):
        self.a = a
        self.tasks = list(ALL_TASKS)
        self.pol = TorchPolicy(a.ckpt, self.tasks, a.device, deterministic=not a.stochastic)
        self.env = BatchedAssemblyEnv(self.tasks, np.zeros(1, np.int64), env_config(self.pol.args, 1))
        e = self.env
        m0 = e.models[0].model
        nq, nv = m0.nq, m0.nv
        self.robot_q = np.setdiff1d(np.arange(1, 1 + nq), e.sp)
        self.robot_v = np.setdiff1d(np.arange(1 + nq, 1 + nq + nv), e.spd)
        sj = e.ecfg.action_scale_joints
        self.scale = np.asarray(sj, float) if len(sj) else np.full(7, e.ecfg.action_scale)
        self.m_ik = mujoco.MjModel.from_xml_string(build_xml("R4")); self.d_ik = mujoco.MjData(self.m_ik)
        self.ix = Indices(self.m_ik)
        self.frames = []          # (stage, state copy, phase: "script" | "policy")
        self.stage, self.phase = -1, "script"

    def snap(self):
        self.frames.append((self.stage, self.env.state[0].copy(), self.phase))

    def tcp(self):
        f = self.env.models[0].fk.forward(self.env.state[:1, self.env.sq])
        return f["tcp_p"][0], f["tcp_R"][0]

    def step(self, act):
        prev = self.env.state.copy()
        _, done, info = self.env.step(np.asarray(act, float).reshape(1, 8))
        if bool(info["unstable"][0]) or not np.isfinite(self.env.state).all():
            self.env.state[:] = prev       # keep the last good physics state
            return None
        self.snap()
        return info

    def move_to(self, q_goal, steps, grip, hold=0):
        q0 = self.env.state[0, self.env.sq].copy()
        for s in range(steps + hold):
            w = min(1.0, (s + 1) / steps); w = w * w * (3 - 2 * w)
            q_des = q0 + w * (q_goal - q0)
            act = np.zeros(8); act[7] = grip
            act[:7] = np.clip((q_des - self.env.state[0, self.env.sq]) / self.scale, -4.0, 4.0)
            if self.step(act) is None:
                return

    def ik(self, pos, quat):
        q, ok = solve_ik(self.m_ik, self.d_ik, self.ix, pos, quat, q_init=self.env.state[0, self.env.sq].copy(), restarts=3)
        return q, ok

    def run(self, layout, order):
        e = self.env
        # start: home posture, gripper open, first part's state
        cur = layout[order[0]].copy()
        cur[e.sq] = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.785]); cur[e.sf] = 0.04; cur[self.robot_v] = 0.0
        final_pose, results = {t: layout[t][e.sp].copy() for t in order}, []
        for i, t in enumerate(order):
            self.stage, self.phase = i, "script"
            S = layout[t].copy()
            S[self.robot_q] = cur[self.robot_q]; S[self.robot_v] = cur[self.robot_v]
            e.set_states([0], S[None], [self.tasks.index(t)])
            self.snap()
            # scripted approach: gripper open, straight down 12 cm above the part, keeping the current yaw
            p_part = e.state[0, e.sp[:3]]
            _, R = self.tcp()
            yaw = math.atan2(R[1, 0], R[0, 0])
            down = quat_mul(axis_angle_quat([0, 0, 1], yaw), np.array([0.0, 1.0, 0.0, 0.0]))
            q_goal, _ = self.ik(p_part + np.array([0.0, 0.0, 0.12]), down)
            self.move_to(q_goal, steps=int(1.6 * FPS), grip=1.0, hold=3)
            # policy
            self.phase = "policy"
            t0 = len(self.frames); ok = False; steps = int(self.a.per_part_s * FPS)
            for s in range(steps):
                info = self.step(self.pol(e))
                if info is None:
                    print(f"[{t}] simulation became unstable; stage stopped", flush=True)
                    break
                if info["success"][0]:
                    ok = True
                    break
            dur = (len(self.frames) - t0) / FPS
            results.append(dict(task=t, success=ok, seconds=round(dur, 2)))
            print(f"part {i + 1}/{len(order)} {t}: {'assembled' if ok else 'not assembled'} after {dur:.1f} s", flush=True)
            final_pose[t] = e.state[0, e.sp].copy()
            # release: open, then lift 8 cm keeping the orientation
            self.phase = "script"
            self.move_to(e.state[0, e.sq].copy(), steps=6, grip=1.0)
            p, R = self.tcp()
            q_up, _ = self.ik(p + np.array([0.0, 0.0, 0.08]), mat_to_quat(R[None])[0])
            self.move_to(q_up, steps=int(1.0 * FPS), grip=1.0, hold=2)
            final_pose[t] = e.state[0, e.sp].copy()
            cur = e.state[0].copy()
        return results, final_pose


# ------------------------------------------------------------------ rendering
def _slerp(q0, q1, w):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(q0 @ q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + w * (q1 - q0)
    else:
        th = math.acos(d)
        q = (math.sin((1 - w) * th) * q0 + math.sin(w * th) * q1) / math.sin(th)
    return q / np.linalg.norm(q)


def playback(frames, slow, fast, sp):
    """Rendered frames as (sim step k, stage, state, speed label): policy segments slowed down by `slow` (poses
    interpolated between control steps: joints linearly, the active part's orientation by slerp), scripted
    moves sped up by `fast` (every fast-th control step)."""
    for k, (st, state, ph) in enumerate(frames):
        if ph == "script":
            last = k + 1 >= len(frames) or frames[k + 1][2] != "script" or frames[k + 1][0] != st
            if k % fast == 0 or last:
                yield k, st, state, f"{fast}x"
            continue
        prev = frames[k - 1] if k > 0 and frames[k - 1][0] == st else None
        if prev is not None:
            for j in range(1, slow):
                w = j / slow
                s = prev[1] + w * (state - prev[1])
                s[sp[3:7]] = _slerp(prev[1][sp[3:7]], state[sp[3:7]], w)
                yield k - 1 + w, st, s, f"{1 / slow:g}x"
        yield k, st, state, f"{1 / slow:g}x"


def render(demo, layout, order, results, final_pose, out, width, height, label, slow=2, fast=2, banner_s=3.0):
    tasks = demo.tasks
    vm = mujoco.MjModel.from_xml_string(build_demo_xml(tasks)); vd = mujoco.MjData(vm)
    jid = lambda n: mujoco.mj_name2id(vm, mujoco.mjtObj.mjOBJ_JOINT, n)
    vq = [vm.jnt_qposadr[jid(f"joint{i + 1}")] for i in range(7)] + [vm.jnt_qposadr[jid(f"finger_joint{i + 1}")] for i in range(2)]
    e = demo.env
    sq = list(e.sq) + list(e.sf)
    mocap = {t: vm.body_mocapid[mujoco.mj_name2id(vm, mujoco.mjtObj.mjOBJ_BODY, f"part_{t}")] for t in tasks}
    tcp_site = mujoco.mj_name2id(vm, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    main = mujoco.Renderer(vm, height, width)
    iw, ih = int(width * 0.38), int(height * 0.38)          # close-up inset (top right)
    inset = mujoco.Renderer(vm, ih, iw)
    # main view from the robot's side (the arm does not hide the board); inset from the opposite side, lower
    # main view from the robot's side, aimed a little right of the board so the board sits left of the inset
    cam = mujoco.MjvCamera(); cam.distance = 0.8; cam.azimuth = 100.0; cam.elevation = -40.0
    right = np.array([math.sin(math.radians(cam.azimuth)), -math.cos(math.radians(cam.azimuth)), 0.0])
    cam.lookat[:] = np.array([0.47, 0.0, 0.05]) + 0.09 * right
    # inset: from the far side of the board, low and far enough that the hand never wraps around the camera
    close = mujoco.MjvCamera(); close.distance = 0.2; close.azimuth = 215.0; close.elevation = -20.0
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22); font_s = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        try:
            font, font_s = ImageFont.load_default(size=22), ImageFont.load_default(size=18)
        except TypeError:
            font = font_s = ImageFont.load_default()
    res = {i: r for i, r in enumerate(results)}
    policy_end = {}                   # last policy frame of each stage (the result banner shows after it)
    for k, (st, _, ph) in enumerate(demo.frames):
        if ph == "policy":
            policy_end[st] = k
    lookat_s = None
    video = []
    banner, clean = {}, None
    for k, st, state, speed in playback(demo.frames, slow, fast, e.sp):
        for t in tasks:     # parts: finished stages at their final pose, later ones at their start pose
            i_t = order.index(t)
            pose = final_pose[t] if i_t < st else layout[t][e.sp]
            if i_t == st:
                pose = state[e.sp]
            vd.mocap_pos[mocap[t]] = pose[:3]; vd.mocap_quat[mocap[t]] = pose[3:7]
        vd.qpos[vq] = state[sq]
        mujoco.mj_forward(vm, vd)
        main.update_scene(vd, cam)
        img = Image.fromarray(main.render()).convert("RGBA")
        clean = clean or {}
        clean.setdefault("start", img.convert("RGB")); clean["done"] = img.convert("RGB")   # main view, no overlays
        tcp = vd.site_xpos[tcp_site].copy()
        tcp = tcp - np.array([0.0, 0.0, 0.01])                                   # aim at the fingertips
        lookat_s = tcp if lookat_s is None else 0.8 * lookat_s + 0.2 * tcp     # follow the gripper, smoothed
        close.lookat[:] = lookat_s
        inset.update_scene(vd, close)
        sub = Image.fromarray(inset.render()).convert("RGBA")
        img.paste(sub, (width - iw - 12, 12))
        dr = ImageDraw.Draw(img)
        dr.rectangle([width - iw - 14, 10, width - 10, 12 + ih + 2], outline=(255, 255, 255, 255), width=2)
        # elapsed time, bottom right, plain white text
        clock = f"{int(k / FPS) // 60}:{(k / FPS) % 60:04.1f}  ({speed})"
        dr.text((width - 14 - dr.textlength(clock, font=font), height - 38), clock, fill=(255, 255, 255, 255), font=font)
        # result banner: from the end of a part's policy segment for banner_s seconds of video (into the next
        # part's approach if needed)
        if k > policy_end.get(st, -1) and banner.get("stage") != st:
            banner = dict(stage=st, start=len(video))
        if banner.get("stage") is not None and len(video) - banner["start"] < banner_s * FPS:
            t = order[banner["stage"]]; r = res[banner["stage"]]
            msg = f"{NAMES[t]} assembled in {r['seconds']:.1f} s" if r["success"] else f"{NAMES[t]}: not assembled in {r['seconds']:.0f} s"
            bw = dr.textlength(msg, font=font) + 24; y0 = height - 52
            dr.rectangle([12, y0, 12 + bw, y0 + 38], fill=(20, 140, 60, 220) if r["success"] else (170, 40, 40, 220))
            dr.text((24, y0 + 7), msg, fill=(255, 255, 255, 255), font=font)
        video.append(np.asarray(img.convert("RGB")))
    for name, im in (clean or {}).items():                     # first / last main-view frames without overlays
        im.save(str(Path(out).with_suffix("")) + f"_{name}.png")
    if out.endswith(".png"):
        Image.fromarray(video[0]).save(out)
        return 1
    for _ in range(2 * FPS):          # hold the last frame
        video.append(video[-1])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, video, fps=FPS, codec="libx264", quality=7, macro_block_size=8)
    return len(video)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--bank-dir", default=None, help="bank for the loose start poses (default: the checkpoint's)")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--per-part-s", type=float, default=30.0)
    ap.add_argument("--stochastic", action="store_true", help="sampled actions (default: mean actions)")
    ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--still", action="store_true", help="render only the initial layout to <out>.png and exit")
    ap.add_argument("--slow", type=int, default=2, help="policy segments play this many times slower (interpolated)")
    ap.add_argument("--fast", type=int, default=2, help="scripted moves between parts play this many times faster")
    ap.add_argument("--pin", default="", help="fixed positions in the order, e.g. '2:N16+3:connector' (task name or "
                                              "family: rod, peg, gear, nut, connector); the rest is random")
    ap.add_argument("--banner-s", type=float, default=3.0, help="seconds of video the result banner stays up")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    demo = Demo(a)
    bank = a.bank_dir or BANK_DIR
    layout = choose_layout(demo.tasks, bank, demo.env.sp, rng)
    order = make_order(demo.tasks, a.pin, rng)
    print("order:", " ".join(order), flush=True)
    it = demo.pol.it
    label = f"AssembleNIST, 8K-env policy, iteration {it}, {'sampled' if a.stochastic else 'mean'} actions"
    if a.still:
        demo.stage = 0
        S = layout[order[0]].copy()
        demo.env.set_states([0], S[None], [demo.tasks.index(order[0])]); demo.snap()
        render(demo, layout, order, [dict(task=order[0], success=False, seconds=0)], {t: layout[t][demo.env.sp] for t in order},
               a.out.replace(".mp4", "_still.png"), a.width, a.height, label)
        return
    results, final_pose = demo.run(layout, order)
    n = render(demo, layout, order, results, final_pose, a.out, a.width, a.height, label, slow=a.slow, fast=a.fast,
               banner_s=a.banner_s)
    summary = dict(ckpt=a.ckpt, iteration=it, seed=a.seed, per_part_s=a.per_part_s, mean_actions=not a.stochastic,
                   order=order, results=results, assembled=sum(r["success"] for r in results), frames=n)
    Path(a.out).with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

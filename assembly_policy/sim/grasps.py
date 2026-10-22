"""Generic antipodal grasp libraries for arbitrary parts (OmniReset-style grasp sampling).

Candidates: random closing axis d and offset; rays from both sides along d hit the part at
p1, p2 (MuJoCo mj_ray on the part's collision geoms). TCP = midpoint (optionally retracted along
the approach a, a random direction perpendicular to d), TCP y = d (finger closing axis), z = a.
Validation: a floating Franka hand (mocap hand body + the real finger joints/actuator) closes on
the free part under gravity, then shakes; the grasp is kept if the part stays within tolerance
in the hand frame and both fingers touch it. Libraries are stored per part asset.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from .robot import mat_to_quat, quat_to_mat
from .scene import PANDA_XML, TCP_OFFSET, PhysicsConfig, _add, _panda_meshdir, part_geoms_with_mass
from .tasks import ROOT, TASKS

PAD_OFFSET = 0.0015
MAX_HALF = 0.04 + PAD_OFFSET - 0.001     # widest graspable half-width


def floating_hand_xml(task_name, cfg: PhysicsConfig | None = None):
    """Mocap Franka hand with fingers + the task's free part, no gravity on the hand."""
    cfg = cfg or PhysicsConfig()
    panda = ET.parse(PANDA_XML).getroot()
    root = ET.Element("mujoco", model=f"grasp_{task_name}")
    ET.SubElement(root, "compiler", angle="radian", meshdir=str(_panda_meshdir()), autolimits="true")
    ET.SubElement(root, "option", timestep=str(cfg.timestep), integrator=cfg.integrator, cone=cfg.cone,
                  iterations=str(cfg.iterations), ls_iterations=str(cfg.ls_iterations))
    root.append(panda.find("default"))
    asset = ET.SubElement(root, "asset")
    for m in panda.find("asset").findall("mesh"):
        if m.get("name") in ("hand_c",) or m.get("file") == "finger_0.obj":
            asset.append(m)
    world = ET.SubElement(root, "worldbody")
    hand_src = panda.find(".//body[@name='hand']")
    hand = ET.SubElement(world, "body", name="hand", mocap="true", pos="0 0 0.5")
    for g in hand_src.findall("geom"):
        if g.get("class") == "collision":
            g.set("contype", "2"); g.set("conaffinity", "1")
            hand.append(g)
    for fname in ("left_finger", "right_finger"):
        fb = hand_src.find(f"body[@name='{fname}']")
        nb = ET.SubElement(hand, "body", {k: v for k, v in fb.attrib.items()})
        nb.append(fb.find("inertial"))
        j = fb.find("joint"); j2 = ET.SubElement(nb, "joint", {k: v for k, v in j.attrib.items()})
        j2.set("damping", str(cfg.finger_damping)); j2.set("class", "finger")
        j2.set("solreflimit", "0.005 1"); j2.set("solimplimit", "0.99 0.9999 0.001")
        for g in fb.findall("geom"):
            if g.get("class") in ("collision", "fingertip_pad_collision_1"):
                g2 = ET.SubElement(nb, "geom", {k: v for k, v in g.attrib.items()})
                g2.set("friction", cfg.finger_friction); g2.set("condim", "4"); g2.set("priority", "2")
                g2.set("solref", cfg.solref); g2.set("solimp", cfg.solimp)
                g2.set("contype", "2"); g2.set("conaffinity", "1")
    _add(hand, "site", name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.002])
    part = _add(world, "body", name="part", pos=[0, 0, 0.3])
    _add(part, "freejoint", name="part_free")
    for k, g in enumerate(part_geoms_with_mass(TASKS[task_name], cfg)):
        if g["type"] == "sdf":
            continue  # grasps only use the convex outer body
        if g["type"] == "mesh":
            _add(asset, "mesh", name=f"part_mesh{k}", vertex=np.asarray(g.pop("vertex")).ravel())
            g["mesh"] = f"part_mesh{k}"
        g.pop("contype", None); g.pop("conaffinity", None)
        _add(part, "geom", name=f"part{k}", condim="4", friction=cfg.part_friction, solref=cfg.solref,
             solimp=cfg.solimp, priority="1", group="1", contype="1", conaffinity="2", **g)
    ten = ET.SubElement(root, "tendon"); fx = ET.SubElement(ten, "fixed", name="split")
    ET.SubElement(fx, "joint", joint="finger_joint1", coef="0.5"); ET.SubElement(fx, "joint", joint="finger_joint2", coef="0.5")
    eq = ET.SubElement(root, "equality")
    ET.SubElement(eq, "joint", joint1="finger_joint1", joint2="finger_joint2", solimp="0.95 0.99 0.001", solref="0.005 1")
    act = ET.SubElement(root, "actuator")
    ET.SubElement(act, "general", name="grip", tendon="split", ctrlrange="-0.01 0.04", gainprm=f"{cfg.gripper_kp}",
                  biasprm=f"0 {-cfg.gripper_kp} {-cfg.gripper_kv}", forcerange=f"{-cfg.gripper_force} {cfg.gripper_force}",
                  biastype="affine")
    return ET.tostring(root, encoding="unicode")


def _frame(y, z):
    y = y / np.linalg.norm(y); z = z - y * (z @ y); z /= np.linalg.norm(z)
    return np.stack([np.cross(y, z), y, z], 1)


class GraspSampler:
    def __init__(self, task_name, seed=0, cfg: PhysicsConfig | None = None):
        self.m = mujoco.MjModel.from_xml_string(floating_hand_xml(task_name, cfg))
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(seed)
        self.part = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "part")
        self.hand = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.fingers = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("left_finger", "right_finger")]
        self.tcp = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "part_free")
        self.pq, self.pd = self.m.jnt_qposadr[jid], self.m.jnt_dofadr[jid]
        self.fq = [self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"finger_joint{i}")] for i in (1, 2)]
        self.fdof = [self.m.jnt_dofadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"finger_joint{i}")] for i in (1, 2)]
        mujoco.mj_forward(self.m, self.d)
        self.ray_d = mujoco.MjData(self.m)          # part kept at its rest pose for ray casting
        mujoco.mj_forward(self.m, self.ray_d)
        self.gravity = self.m.opt.gravity.copy()
        pg = np.nonzero(self.m.geom_bodyid == self.part)[0]
        self.center = self.d.xipos[self.part].copy()
        # bounding radius of the part about its COM, from geom AABBs
        self.radius = max(np.linalg.norm(self.d.geom_xpos[g] - self.center) + self.m.geom_rbound[g] for g in pg)

    def _ray(self, p, v):
        gid = np.zeros(1, np.int32)
        geomgroup = np.array([0, 1, 0, 0, 0, 0], np.uint8)
        dist = mujoco.mj_ray(self.m, self.ray_d, p, v, geomgroup, 1, -1, gid)
        return dist

    def _normal(self, origin, dvec, R, eps=0.0005):
        """Surface normal (pointing against the ray) at the hit of a ray from origin along dvec."""
        u = np.cross(dvec, [1.0, 0, 0] if abs(dvec[0]) < 0.9 else [0, 1.0, 0]); u /= np.linalg.norm(u)
        v = np.cross(dvec, u)
        pts = []
        for o in (np.zeros(3), u * eps, v * eps):
            t = self._ray(origin + o, dvec)
            if t < 0:
                return None
            pts.append(origin + o + dvec * t)
        n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            return None
        n /= nn
        return n if n @ dvec < 0 else -n

    def candidate(self, max_angle=math.radians(20)):
        R = self.radius + 0.01
        for _ in range(200):
            dvec = self.rng.normal(size=3); dvec /= np.linalg.norm(dvec)
            off = self.rng.normal(size=3); off -= dvec * (off @ dvec)
            off *= self.radius * 0.7 * self.rng.random() / max(np.linalg.norm(off), 1e-9)
            c = self.center + off
            t1 = self._ray(c + dvec * R, -dvec)
            t2 = self._ray(c - dvec * R, dvec)
            if t1 < 0 or t2 < 0:
                continue
            p1, p2 = c + dvec * (R - t1), c - dvec * (R - t2)
            half = np.linalg.norm(p1 - p2) / 2
            if not 0.001 < half < MAX_HALF:
                continue
            n1 = self._normal(c + dvec * R, -dvec, R)    # normal at p1 should point along +dvec
            n2 = self._normal(c - dvec * R, dvec, R)     # normal at p2 should point along -dvec
            if n1 is None or n2 is None or n1 @ dvec < math.cos(max_angle) or -(n2 @ dvec) < math.cos(max_angle):
                continue
            a = self.rng.normal(size=3); a -= dvec * (a @ dvec); a /= np.linalg.norm(a)
            back = self.rng.uniform(0.0, 0.006)
            mid = (p1 + p2) / 2 - a * back
            return mid, _frame(dvec, a), half
        return None

    def validate(self, mid, Rt, half, settle_s=0.4, shake_s=0.6):
        """Close on the part (gravity off), then gravity + shake. Returns the settled grasp
        (tcp pos/quat in the part frame, half-width) or None."""
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        hand_pos = mid - Rt @ np.array([0, 0, TCP_OFFSET])
        d.mocap_pos[0] = hand_pos; d.mocap_quat[0] = mat_to_quat(Rt[None])[0]
        d.qpos[self.fq] = 0.04          # the fully open gripper must not overlap the part
        d.ctrl[0] = -0.01
        mujoco.mj_forward(m, d)
        if any(c.dist < 0 for c in d.contact[:d.ncon]):
            return None
        d.qpos[self.fq] = np.clip(half - PAD_OFFSET + 0.003, 0, 0.04)   # start 3 mm outside the part
        mujoco.mj_forward(m, d)
        if any(c.dist < 0 for c in d.contact[:d.ncon]):
            return None
        m.opt.gravity[:] = 0.0          # close the (slow) fingers without the part falling away
        still = 0
        for i in range(int(1.0 / m.opt.timestep)):
            mujoco.mj_step(m, d)
            still = still + 1 if abs(d.qvel[self.fdof]).max() < 1e-3 else 0
            if still >= 20:
                break
        m.opt.gravity[:] = self.gravity
        for i in range(int(0.1 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        p0 = self._rel()
        # robustness: 2 g along each of +-x, +-y, +-z (mocap hand stays still; a teleported mocap
        # hand cannot exert friction, so shaking it is not a meaningful test)
        g = float(np.linalg.norm(self.gravity))
        for axis in np.vstack([np.eye(3), -np.eye(3)]):
            m.opt.gravity[:] = 2.0 * g * axis
            for i in range(int(0.15 / m.opt.timestep)):
                mujoco.mj_step(m, d)
            if not np.isfinite(d.qpos).all():
                m.opt.gravity[:] = self.gravity
                return None
        m.opt.gravity[:] = self.gravity
        for i in range(int(0.1 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        touch = {m.geom_bodyid[c.geom1] for c in d.contact[:d.ncon]} | {m.geom_bodyid[c.geom2] for c in d.contact[:d.ncon]}
        if not all(f in touch for f in self.fingers) or np.linalg.norm(self._rel() - p0) > 0.004:
            return None
        # settled grasp: TCP pose in the part frame, finger half-width
        Rp = d.xmat[self.part].reshape(3, 3); pp = d.xpos[self.part]
        tcp = d.site_xpos[self.tcp]; Rtcp = d.site_xmat[self.tcp].reshape(3, 3)
        return Rp.T @ (tcp - pp), mat_to_quat((Rp.T @ Rtcp)[None])[0], float(d.qpos[self.fq].mean() + PAD_OFFSET)

    def _rel(self):
        Rh = self.d.xmat[self.hand].reshape(3, 3)
        return Rh.T @ (self.d.xpos[self.part] - self.d.xpos[self.hand])

    def build(self, n, max_tries=None):
        """Return grasps as (tcp_pos_in_part, tcp_quat_in_part, half_width) arrays."""
        out_p, out_q, out_h, tries = [], [], [], 0
        while len(out_p) < n and (max_tries is None or tries < max_tries):
            tries += 1
            c = self.candidate()
            if c is None:
                continue
            g = self.validate(*c)
            if g is not None:
                out_p.append(g[0]); out_q.append(g[1]); out_h.append(g[2])
        return np.array(out_p), np.array(out_q), np.array(out_h), tries


def library_path(part_asset):
    return ROOT / "assets/grasps" / f"{part_asset}.npz"


def load_library(part_asset):
    z = np.load(library_path(part_asset))
    return z["pos"], z["quat"], z["half"]

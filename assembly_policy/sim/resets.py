"""OmniReset-style reset-state generation for the NIST board tasks.

Families (as described in OmniReset and the AlphaNIST paper, Fig. 4), equal mix:
  0 loose      ObjectAnywhereEEAnywhere: part dropped from a random pose and settled; random EE pose.
  1 pickup     ObjectRestingEEGrasped: settled part + gripper at a (noisy) grasp, closing.
  2 inhand     transport / ObjectAnywhereEEGrasped: part held in a grasp; continuum (level r) from upright
               above its fixture (r = 0) to a random pose in the workspace (r = 1).
  3 partial    ObjectPartiallyAssembledEEGrasped: part along its assembly axis at insertion fraction
               alpha in [0, 1.1] (0 = fully seated) + noisy grasp.
  4 near       near-mating: part grasped above its fixture; continuum (level r) of offsets from the entry.
Every stored state is settled (low velocities), has the gripper approach within 60 deg of
down (except loose, as in OmniReset where the EE pose is independent), and contains no
contact penetration beyond tolerance. States are full MuJoCo physics states.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from .robot import FINGER_OPEN, HOME_Q, Indices, axis_angle_quat, mat_to_quat, quat_mul, quat_to_mat, random_quat, solve_ik
from .scene import PhysicsConfig, build_xml, ENV, GRIP, PART, ARM
from .env import STATE_SPEC
from .tasks import TASKS, scene_pose

FAMILIES = ("loose", "pickup", "inhand", "partial", "near")
PAD_OFFSET = 0.0015  # main finger-pad inner face sits 1.5 mm inboard of the finger joint origin


def quat_slerp(q0, q1, t):
    """Spherical interpolation between unit quaternions (w, x, y, z), shortest path."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - t) * th) * q0 + math.sin(t * th) * q1) / math.sin(th)


def finger_q_for(half_width, clearance=0.0002):
    """Finger joint opening that brings the pad face to `clearance` from a part of half-width."""
    return float(np.clip(half_width - PAD_OFFSET + clearance, 0.0, FINGER_OPEN))


@dataclass
class ResetConfig:
    part_xy_lo: tuple = (0.32, -0.20)
    part_xy_hi: tuple = (0.68, 0.20)
    part_drop_z: tuple = (0.03, 0.20)
    ee_lo: tuple = (0.30, -0.30, 0.04)
    ee_hi: tuple = (0.70, 0.30, 0.40)
    inhand_lo: tuple = (0.32, -0.22, 0.06)
    inhand_hi: tuple = (0.68, 0.22, 0.30)
    max_down_angle: float = math.pi / 3     # gripper approach within 60 deg of world -z
    pickup_noise_pos: float = 0.01
    # pickup position noise is capped at this fraction of the grasp half-width (a fixed 10 mm missed the 4-8 mm
    # parts), and this share of pickup states must hold the part between both fingers after closing; the rest
    # stay noisy approaches (retrieval practice)
    pickup_noise_rel: float = 0.6
    pickup_held_frac: float = 0.8
    grip_close: float = -0.01               # closed-gripper target, as EnvConfig.grip_close
    pickup_noise_rot: float = math.pi / 16
    partial_noise_pos: float = 0.005
    partial_noise_rot: float = math.pi / 32
    partial_alpha: tuple = (0.0, 1.1)
    partial_min_dasm: float = 0.002         # reject partial states that already count as assembled (< 1 mm)
    # nuts: a share of partial starts within one pitch above 1.2 mm from the seated pose (< 1 turn of
    # screwing; most partial nut starts need 2-5 turns, which a policy that cannot yet screw never completes)
    nut_close_frac: float = 0.3
    nut_close_min: float = 0.0012
    # small random part pose at the drawn assembly progress (alpha <= 1): lateral offset up to this many
    # metres and tilt up to this many radians about a random axis normal to the insertion axis (plus the same
    # yaw range for non-nut parts; a nut's yaw is fixed by the helix). A draw whose part penetrates the fixture
    # by more than partial_spawn_pen before settling is halved twice, then the exact pose is used.
    partial_pose_lat: float = 0.001
    partial_pose_tilt: float = math.radians(2.0)
    partial_spawn_pen: float = 0.0002
    loose_standoff: float = 0.03            # loose level 0: open gripper this far back from a feasible grasp
    transport_max_cone: float = math.radians(85)  # transport level 1: approach up to near-horizontal
    # a share of transport starts of elongated parts (rods, pegs, connectors) is a part-tilt continuum: the held
    # part's axis is tilt = level * transport_tilt_max from upright (any grasp, random workspace pose), so the
    # policy can learn to reorient a part picked up lying down step by step.
    transport_tilt_frac: float = 0.5
    transport_tilt_max: float = math.pi / 2
    transport_tilt_families: tuple = ("rod", "peg", "connector")
    # near-mating and partial starts of those families also hold the part with a side grasp near its top end
    # (gripper approach up to horizontal) with this probability: where a lying part picked up from the table ends
    # up after reorientation.
    side_grasp_frac: float = 0.35
    side_grasp_cone: float = math.radians(92)
    # near-mating is a difficulty continuum: one level r ~ U(0,1) per state scales every offset, so r -> 0
    # meets the disengaged end of the partial family and r = 1 spans the full box below
    near_height: float = 0.05               # max height of the part base above the assembly entry
    near_lateral: float = 0.01
    near_tilt: float = 0.25
    near_yaw: float = math.pi / 6
    penetration_tol: float = 0.001
    finger_squeeze_tol: float = 0.0025   # finger pads squeezing a held part (soft-pad compression)
    settle_max_steps: int = 300
    stable_steps: int = 10


# ------------------------------------------------------------------ part geometry for grasps
PRISM = {  # name -> (half-width x, half-width y, length) in part frame (axis = +z, base at z=0)
    "round_peg_4mm": (0.002, 0.002, 0.05), "round_peg_8mm": (0.004, 0.004, 0.05),
    "round_peg_12mm": (0.006, 0.006, 0.05), "round_peg_16mm": (0.008, 0.008, 0.05),
    "rectangular_peg_4mm_tight": (0.002, 0.002, 0.05), "rectangular_peg_8mm_tight": (0.004, 0.0035, 0.05),
    "rectangular_peg_12mm_tight": (0.006, 0.004, 0.05), "rectangular_peg_16mm_tight": (0.008, 0.005, 0.05),
}
INSERT_DEPTH = {"hole_8mm": 0.024, "gear_base": 0.020}  # default 0.009 (NIST 9 mm blocks)


def _frame_from_axes(x, y, z):
    R = np.stack([x, y, z], 1)
    return mat_to_quat(R[None])[0]


def sample_grasp(part_asset, rng, kind=None):
    """Returns (tcp_pos_in_part, tcp_quat_in_part, half_width) for prismatic parts.

    TCP frame: z = approach direction (palm -> fingertips), y = finger closing axis.
    axial: approach along the part axis from either end; side: approach perpendicular to it.
    """
    hx, hy, L = PRISM[part_asset]
    rod = part_asset.startswith("round")
    kind = kind or ("axial" if rng.random() < 0.5 else "side")
    if kind in ("axial", "axial_top"):
        top = kind == "axial_top" or rng.random() < 0.5
        phi = rng.uniform(0, 2 * np.pi) if rod else rng.integers(4) * np.pi / 2
        z = np.array([0.0, 0.0, -1.0 if top else 1.0])
        y = np.array([math.cos(phi), math.sin(phi), 0.0])
        x = np.cross(y, z)
        depth = rng.uniform(0.006, 0.040)          # how far the part reaches into the gripper
        h = L - depth if top else depth
        half = hx if rod else (hx if abs(math.cos(phi)) > 0.5 else hy)
        return np.array([0.0, 0.0, h]), _frame_from_axes(x, y, z), half
    psi = rng.uniform(0, 2 * np.pi) if rod else rng.integers(4) * np.pi / 2
    z = np.array([math.cos(psi), math.sin(psi), 0.0])        # approach perpendicular to axis
    y = np.cross(np.array([0.0, 0.0, 1.0]), z)                # closing across the part
    if rng.random() < 0.5:
        y = -y
    x = np.cross(y, z)
    # "side_top": held near the top end, leaving the lower part free for insertion (paper: "picks it up near one end")
    s = rng.uniform(0.6 * L, L - 0.008) if kind == "side_top" else rng.uniform(0.008, L - 0.008)
    reach = hx if rod else (hx if abs(math.cos(psi)) > 0.5 else hy)   # part extent along approach
    back = rng.uniform(0.0, max(0.0, 0.012 - reach))                   # keep fingertips off the support
    half = hy if (not rod and abs(math.cos(psi)) > 0.5) else hx
    return np.array([0.0, 0.0, s]) - z * back, _frame_from_axes(x, y, z), half


_LIBS = {}


def library_grasp(part_asset, rng):
    """Random grasp from the part's validated antipodal library (non-prismatic parts)."""
    if part_asset not in _LIBS:
        from .grasps import load_library
        _LIBS[part_asset] = load_library(part_asset)
    P, Q, H = _LIBS[part_asset]
    i = rng.integers(len(P))
    return P[i].copy(), Q[i].copy(), float(H[i])


def any_grasp(part_asset, rng, kind=None):
    if part_asset in PRISM:
        return sample_grasp(part_asset, rng, kind)
    return library_grasp(part_asset, rng)


class ResetGenerator:
    def __init__(self, task_name: str, pcfg: PhysicsConfig | None = None, rcfg: ResetConfig | None = None, seed=0):
        self.task = TASKS[task_name]
        self.pcfg = pcfg or PhysicsConfig()
        self.rcfg = rcfg or ResetConfig()
        self.m = mujoco.MjModel.from_xml_string(build_xml(task_name, self.pcfg))
        self.d = mujoco.MjData(self.m)
        self.ik_d = mujoco.MjData(self.m)
        self.ix = Indices(self.m)
        self.rng = np.random.default_rng(seed)
        self.fix_pos, self.fix_quat = scene_pose(self.task.fixture_scene)
        self.goal_pos, self.goal_quat = scene_pose(self.task.part_scene)
        self.depth = INSERT_DEPTH.get(self.task.fixture_asset, 0.009)
        self.pitch = None
        if self.task.family == "connector":
            from .tasks import load_obj
            sock_top = load_obj(self.task.fixture_asset)[0][:, 2].max()
            plug_bottom = load_obj(self.task.part_asset)[0][:, 2].min()
            self.depth = max(0.003, float(sock_top - plug_bottom))   # plug travel from entry to seated
        if self.task.family == "nut":
            from .threads import THREADS
            P, crest, root, htop, top, minor, z0, z1, flat = THREADS[task_name]
            dz = self.goal_pos[2] - self.fix_pos[2]
            self.depth = top - (dz + z0)   # travel from the bolt top to the seated nut
            self.pitch = P
        g = self.m.geom_bodyid
        self.part_geoms = set(np.nonzero(g == self.ix.part_body)[0])
        self.finger_geoms = set(np.nonzero(np.isin(g, self.ix.finger_bodies))[0])
        self.grip_geoms = set(np.nonzero(np.isin(g, self.ix.finger_bodies + [self.ix.hand_body]))[0])
        self.nstate = mujoco.mj_stateSize(self.m, STATE_SPEC)
        self.kp = np.asarray(self.task.keypoints)
        self.goal_kp = self.goal_pos + self.kp @ quat_to_mat(self.goal_quat).T

    def _assembly_distance(self):
        """Success metric of sim.py: max distance between part keypoints and their assembled positions."""
        p = self.d.qpos[self.ix.part_qpos:self.ix.part_qpos + 3]
        R = quat_to_mat(self.d.qpos[self.ix.part_qpos + 3:self.ix.part_qpos + 7])
        return float(np.linalg.norm(p + self.kp @ R.T - self.goal_kp, axis=1).max())

    # -------------------------------------------------------------- low-level
    def _set(self, q_arm, finger, part_pos, part_quat, ctrl_finger):
        d = self.d
        mujoco.mj_resetData(self.m, d)
        d.qpos[self.ix.arm_qpos] = q_arm
        d.qpos[self.ix.finger_qpos] = finger
        d.qpos[self.ix.part_qpos:self.ix.part_qpos + 3] = part_pos
        d.qpos[self.ix.part_qpos + 3:self.ix.part_qpos + 7] = part_quat / np.linalg.norm(part_quat)
        d.ctrl[:7] = q_arm
        d.ctrl[7] = ctrl_finger
        mujoco.mj_forward(self.m, d)

    def penetrating_pairs(self, tol=0.0):
        out = []
        for c in self.d.contact[:self.d.ncon]:
            if -c.dist > tol:
                n1 = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, self.m.geom_bodyid[c.geom1])
                n2 = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, self.m.geom_bodyid[c.geom2])
                out.append((n1 or "world", n2 or "world", round(-c.dist * 1e3, 2)))
        return out

    def _valid_contacts(self):
        """Non-finger contacts within penetration_tol and finger-part squeeze within finger_squeeze_tol."""
        fp = 0.0
        for c in self.d.contact[:self.d.ncon]:
            if (c.geom1 in self.part_geoms and c.geom2 in self.finger_geoms) or \
                    (c.geom2 in self.part_geoms and c.geom1 in self.finger_geoms):
                fp = max(fp, -c.dist)
        return self._max_penetration(ignore_finger_part=True) < self.rcfg.penetration_tol and fp < self.rcfg.finger_squeeze_tol

    def _max_penetration(self, ignore_finger_part=False):
        worst = 0.0
        for c in self.d.contact[:self.d.ncon]:
            if ignore_finger_part and ((c.geom1 in self.part_geoms and c.geom2 in self.finger_geoms) or
                                       (c.geom2 in self.part_geoms and c.geom1 in self.finger_geoms)):
                continue
            worst = max(worst, -c.dist)
        return worst

    def _robot_contact(self):
        """True if any robot geom (arm, hand, fingers) touches anything."""
        for c in self.d.contact[:self.d.ncon]:
            if c.dist < 0 and ((self.m.geom_contype[c.geom1] | self.m.geom_contype[c.geom2]) & (ARM | GRIP)):
                return True
        return False

    def _contacts(self):
        robot_part = robot_env = fingers_on_part = 0
        left, right = self.ix.finger_bodies
        touch = set()
        for c in self.d.contact[:self.d.ncon]:
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            if self.ix.part_body in (b1, b2):
                other = b2 if b1 == self.ix.part_body else b1
                if other in (left, right):
                    touch.add(other)
        return touch

    def _settle(self, max_steps=None, part_free=True):
        rc = self.rcfg
        stable = 0
        for i in range(max_steps or rc.settle_max_steps):
            mujoco.mj_step(self.m, self.d)
            v = self.d.qvel
            pv = np.abs(v[self.ix.part_dof:self.ix.part_dof + 3]).sum()
            pw = np.abs(v[self.ix.part_dof + 3:self.ix.part_dof + 6]).sum()
            rv = np.abs(v[self.ix.arm_dof]).sum() + np.abs(v[self.ix.finger_dof]).sum()
            if not np.isfinite(v).all():
                return False
            ok = pv < 0.01 and pw < 0.2 and rv < 0.05
            stable = stable + 1 if ok else 0
            if stable >= rc.stable_steps and i > 20:
                return True
        return False

    def _state(self):
        s = np.empty(self.nstate)
        mujoco.mj_getState(self.m, self.d, s, STATE_SPEC)
        s[0] = 0.0
        return s

    def _down_ok(self, tcp_quat_world, cone=None):
        z = quat_to_mat(tcp_quat_world)[:, 2]
        return z[2] < -math.cos(self.rcfg.max_down_angle if cone is None else cone)

    def _part_in_bounds(self):
        p = self.d.qpos[self.ix.part_qpos:self.ix.part_qpos + 3]
        return 0.15 < p[0] < 0.85 and abs(p[1]) < 0.4 and p[2] > -0.005

    def _ik(self, pos, quat):
        q, ok = solve_ik(self.m, self.ik_d, self.ix, pos, quat, rng=self.rng, restarts=3)
        return q, ok

    def _random_ee_quat(self, cone=None):
        rc = self.rcfg
        # tilt from straight down within max angle, random yaw
        down = np.array([0.0, 1.0, 0.0, 0.0])  # tcp z -> world -z
        yaw = axis_angle_quat([0, 0, 1], self.rng.uniform(-np.pi, np.pi))
        tilt_axis = np.r_[self.rng.normal(size=2), 0.0]
        cone = rc.max_down_angle if cone is None else cone
        tilt = axis_angle_quat(tilt_axis, self.rng.uniform(0, cone) * math.sqrt(self.rng.random()))
        return quat_mul(tilt, quat_mul(yaw, down))

    def _down_grasp(self, part_pos, part_quat, kind=None, tries=200, cone=None):
        """Rejection-sample a grasp whose approach is within the down cone (no IK yet)."""
        for _ in range(tries):
            g = any_grasp(self.task.part_asset, self.rng, kind)
            tp, tq, half = self._world_grasp(part_pos, part_quat, g)
            if self._down_ok(tq, cone):
                return g
        return None

    def _place_with_backoff(self, tp, tq, width, part_pos, part_quat, max_backoff=0.02):
        """IK to (tp,tq); if only robot-environment penetration, retreat along -approach and retry."""
        approach = quat_to_mat(tq)[:, 2]
        back = 0.0
        for _ in range(4):
            q, ok = self._ik(tp - approach * back, tq)
            if not ok:
                return None
            self._set(q, width, part_pos, part_quat, self.rcfg.grip_close)
            pen_env = 0.0
            pen_part = 0.0
            for c in self.d.contact[:self.d.ncon]:
                if c.dist >= 0:
                    continue
                robot = (self.m.geom_contype[c.geom1] | self.m.geom_contype[c.geom2]) & (ARM | GRIP)
                part = c.geom1 in self.part_geoms or c.geom2 in self.part_geoms
                if robot and part:
                    pen_part = max(pen_part, -c.dist)
                elif robot:
                    pen_env = max(pen_env, -c.dist)
            if pen_part > self.rcfg.penetration_tol:
                return None
            if pen_env <= 0.0:
                return q
            back += pen_env + 0.001
            if back > max_backoff:
                return None
        return None

    def _world_grasp(self, part_pos, part_quat, g):
        gp, gq, half = g
        Rp = quat_to_mat(part_quat)
        return part_pos + Rp @ gp, quat_mul(part_quat, gq), half

    # -------------------------------------------------------------- families
    def loose(self):
        rc = self.rcfg
        for _ in range(50):
            xy = self.rng.uniform(rc.part_xy_lo, rc.part_xy_hi)
            pos = np.r_[xy, self.rng.uniform(*rc.part_drop_z)]
            quat = random_quat(self.rng)
            self._set(HOME_Q, FINGER_OPEN, pos, quat, FINGER_OPEN)
            if self._max_penetration() > 0:
                continue
            if not self._settle(600) or not self._part_in_bounds():
                continue
            part_pos = self.d.qpos[self.ix.part_qpos:self.ix.part_qpos + 3].copy()
            part_quat = self.d.qpos[self.ix.part_qpos + 3:self.ix.part_qpos + 7].copy()
            # difficulty level r: the gripper pose interpolates from open at a pre-grasp standoff above a
            # feasible grasp of the settled part (r = 0) to OmniReset's random EE pose (r = 1)
            r = self.rng.uniform()
            g = self._down_grasp(part_pos, part_quat)
            if g is None:
                r = 1.0
            else:
                gp, gq, _ = self._world_grasp(part_pos, part_quat, g)
                pre_p = gp - quat_to_mat(gq)[:, 2] * rc.loose_standoff
            for _ in range(20):
                ee, eq = self.rng.uniform(rc.ee_lo, rc.ee_hi), self._random_ee_quat()
                width = self.rng.uniform(0.0, FINGER_OPEN)
                if r < 1.0:
                    ee, eq = pre_p + r * (ee - pre_p), quat_slerp(gq, eq, r)
                    width = FINGER_OPEN + r * (width - FINGER_OPEN)
                q, ok = self._ik(ee, eq)
                if not ok:
                    continue
                self._set(q, width, part_pos, part_quat, width)
                if self._robot_contact():
                    continue
                if self._settle(200) and self._max_penetration() < rc.penetration_tol and self._part_in_bounds():
                    self.level = r
                    return self._state()
        return None

    def pickup(self, loose_state):
        rc = self.rcfg
        mujoco.mj_setState(self.m, self.d, loose_state, STATE_SPEC)
        mujoco.mj_forward(self.m, self.d)
        part_pos = self.d.qpos[self.ix.part_qpos:self.ix.part_qpos + 3].copy()
        part_quat = self.d.qpos[self.ix.part_qpos + 3:self.ix.part_qpos + 7].copy()
        held = self.rng.random() < rc.pickup_held_frac
        for _ in range(40):
            g = self._down_grasp(part_pos, part_quat)
            if g is None:
                return None
            tp, tq, half = self._world_grasp(part_pos, part_quat, g)
            scale = min(rc.pickup_noise_pos, rc.pickup_noise_rel * half) if held else rc.pickup_noise_pos
            noise = self.rng.uniform(-1, 1, 3) * scale
            noise[2] = abs(noise[2])
            tp = tp + noise
            tq = quat_mul(axis_angle_quat(self.rng.normal(size=3), self.rng.uniform(-1, 1) * rc.pickup_noise_rot), tq)
            if not self._down_ok(tq):
                continue
            if self._place_with_backoff(tp, tq, finger_q_for(half), part_pos, part_quat) is None:
                continue
            if self._settle(150) and self._valid_contacts() and self._part_in_bounds() \
                    and (not held or len(self._contacts()) == 2):
                return self._state()
        return None

    def _tilted_inhand(self):
        """Transport start whose held part is tilted by exactly theta = level * transport_tilt_max from its
        assembled axis (random azimuth, any grasp, gripper approach within the transport cone, random workspace
        position): a reorientation curriculum between upright-held parts and parts held sideways."""
        rc = self.rcfg
        up = quat_to_mat(self.goal_quat)[:, 2]
        lvl = self.rng.uniform()
        theta = lvl * rc.transport_tilt_max
        ref = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(up, ref); e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
        for _ in range(60):
            psi = self.rng.uniform(0, 2 * np.pi)
            w = math.cos(theta) * up + math.sin(theta) * (math.cos(psi) * e1 + math.sin(psi) * e2)
            g = any_grasp(self.task.part_asset, self.rng)
            gp, gq, half = g
            u = quat_to_mat(gq).T @ np.array([0.0, 0.0, 1.0])       # part axis in the TCP frame
            v, c = np.cross(u, w), float(u @ w)
            sv = np.linalg.norm(v)
            if sv < 1e-9:   # parallel / antiparallel
                R1 = np.eye(3) if c > 0 else quat_to_mat(axis_angle_quat(np.cross(u, ref) / max(np.linalg.norm(np.cross(u, ref)), 1e-9), np.pi))
            else:
                K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                R1 = np.eye(3) + K + K @ K * ((1 - c) / sv ** 2)          # rotation taking u onto w
            tq = None
            for _ in range(12):   # free rotation about the part axis: pick one with a down-facing approach
                R = quat_to_mat(axis_angle_quat(w, self.rng.uniform(0, 2 * np.pi))) @ R1
                q_try = mat_to_quat(R[None])[0]
                if self._down_ok(q_try, rc.transport_max_cone):
                    tq = q_try
                    break
            if tq is None:
                continue
            tp = self.rng.uniform(rc.inhand_lo, rc.inhand_hi)
            q, ok = self._ik(tp, tq)
            if not ok:
                continue
            Rp = quat_to_mat(tq) @ quat_to_mat(gq).T
            part_quat = mat_to_quat(Rp[None])[0]
            part_pos = tp - Rp @ gp
            self._set(q, finger_q_for(half), part_pos, part_quat, rc.grip_close)
            if self._max_penetration(ignore_finger_part=True) > 0 or self._max_penetration() > rc.penetration_tol:
                continue
            rel0 = self._part_in_tcp()
            if not self._settle(150):
                continue
            if len(self._contacts()) == 2 and np.linalg.norm(self._part_in_tcp() - rel0) < 0.01 \
                    and self._valid_contacts() and self._part_in_bounds():
                self.level = lvl
                return self._state()
        return None

    def inhand(self):
        """Transport (AlphaNIST 'transport'; OmniReset ObjectAnywhereEEGrasped) as a difficulty continuum:
        level r ~ U(0,1) interpolates the gripper pose from holding the part upright above its fixture
        (r = 0, the far end of near-mating) to a uniformly random down-facing pose in the workspace
        (r = 1, OmniReset's distribution). The grasp is chosen so the part would be upright at r = 0 with
        probability 1 - r, otherwise any grasp (e.g. a peg held sideways, which needs a regrasp)."""
        rc = self.rcfg
        if self.task.family in rc.transport_tilt_families and self.rng.random() < rc.transport_tilt_frac:
            return self._tilted_inhand()
        axis = quat_to_mat(self.fix_quat)[:, 2]
        goal_axis = quat_to_mat(self.goal_quat)[:, 2]
        r = self.rng.uniform()
        cone = rc.max_down_angle + r * (rc.transport_max_cone - rc.max_down_angle)  # approach cone widens with r
        max_tilt = rc.near_tilt + r * (np.pi - rc.near_tilt)                       # part tilt ladder
        for _ in range(60):
            upright = self.rng.random() >= r
            p0 = self.goal_pos + axis * (self.depth + self.rng.uniform(0, rc.near_height))
            yaw = self.rng.uniform(-1, 1) * (rc.near_yaw + r * (np.pi - rc.near_yaw))  # meets near-mating at r=0
            q0 = quat_mul(axis_angle_quat(axis, yaw), self.goal_quat)
            if upright:
                g = self._down_grasp(p0, q0, kind="axial_top" if self.rng.random() < 0.7 else None, cone=cone)
                if g is None:
                    continue
            else:
                g = any_grasp(self.task.part_asset, self.rng)
            gp, gq, half = g
            t0p, t0q, _ = self._world_grasp(p0, q0, g)
            t1p = self.rng.uniform(rc.inhand_lo, rc.inhand_hi)
            t1q = self._random_ee_quat(cone)
            tp = t0p + r * (t1p - t0p)
            tq = quat_slerp(t0q, t1q, r) if upright else t1q
            if not self._down_ok(tq, cone):
                continue
            q, ok = self._ik(tp, tq)
            if not ok:
                continue
            Rp = quat_to_mat(tq) @ quat_to_mat(gq).T
            if math.acos(np.clip(Rp[:, 2] @ goal_axis, -1, 1)) > max_tilt:
                continue
            part_quat = mat_to_quat(Rp[None])[0]
            part_pos = tp - Rp @ gp
            self._set(q, finger_q_for(half), part_pos, part_quat, 0.0)
            if self._max_penetration(ignore_finger_part=True) > 0 or self._max_penetration() > rc.penetration_tol:
                continue
            rel0 = self._part_in_tcp()
            if not self._settle(150):
                continue
            if len(self._contacts()) == 2 and np.linalg.norm(self._part_in_tcp() - rel0) < 0.01 \
                    and self._valid_contacts() and self._part_in_bounds():
                self.level = r
                return self._state()
        return None

    def partial(self):
        rc = self.rcfg
        R_fix = quat_to_mat(self.fix_quat)
        axis = R_fix[:, 2]
        side = self.task.family in rc.transport_tilt_families and self.rng.random() < rc.side_grasp_frac
        for _ in range(60):
            alpha = self.rng.uniform(*rc.partial_alpha)
            if self.task.family == "nut" and self.rng.random() < rc.nut_close_frac:
                alpha = self.rng.uniform(rc.nut_close_min, rc.nut_close_min + self.pitch) / self.depth
            part_pos = self.goal_pos + axis * alpha * self.depth
            part_quat = self.goal_quat.copy()
            if self.task.family == "nut":
                # stay on the helix: raising by h turns a right-hand-threaded nut by +2 pi h / P
                h = alpha * self.depth
                part_quat = quat_mul(axis_angle_quat(axis, 2 * np.pi * h / self.pitch), part_quat)
            if alpha > 1.0:  # slightly disengaged: small lateral / tilt perturbation above the entry
                part_pos = part_pos + np.r_[self.rng.uniform(-1, 1, 2) * 0.002, 0.0]
                part_quat = quat_mul(axis_angle_quat(np.r_[self.rng.normal(size=2), 0.0], self.rng.uniform(0, 0.05)), part_quat)
                pose_noise = None
            else:   # random pose at this progress (unit draws, scaled below)
                pose_noise = (self.rng.uniform(-1, 1, 2), self.rng.normal(size=2), self.rng.uniform(0, 1), self.rng.uniform(-1, 1))
            if side:
                g = self._down_grasp(part_pos, part_quat, kind="side_top", cone=rc.side_grasp_cone)
            else:
                g = self._down_grasp(part_pos, part_quat, kind="axial_top" if self.rng.random() < 0.8 else None)
            if g is None:
                continue
            gnoise = self.rng.uniform(-1, 1, 3) * rc.partial_noise_pos
            gnoise[2] = abs(gnoise[2])
            grot = (self.rng.normal(size=3), self.rng.uniform(-1, 1) * rc.partial_noise_rot)
            use_noise = pose_noise is not None and (rc.partial_pose_lat > 0 or rc.partial_pose_tilt > 0)
            placed = None
            for k in (range(4) if use_noise else (3,)):
                sc = (1.0, 0.5, 0.25, 0.0)[k]
                pp, pq = part_pos, part_quat
                if sc > 0:
                    lat, tax, tmag, yaw = pose_noise
                    pp = part_pos + R_fix[:, :2] @ (lat * rc.partial_pose_lat * sc)
                    if self.task.family != "nut":
                        pq = quat_mul(axis_angle_quat(axis, yaw * rc.partial_pose_tilt * sc), pq)
                    pq = quat_mul(axis_angle_quat(R_fix[:, :2] @ tax, tmag * rc.partial_pose_tilt * sc), pq)
                tp, tq, half = self._world_grasp(pp, pq, g)
                tp = tp + gnoise
                tq = quat_mul(axis_angle_quat(grot[0], grot[1]), tq)
                if not self._down_ok(tq, rc.side_grasp_cone if side else None):
                    break
                if self._place_with_backoff(tp, tq, finger_q_for(half), pp, pq) is None:
                    continue
                if sc > 0 and self._max_penetration(ignore_finger_part=True) > rc.partial_spawn_pen:
                    continue
                placed = (pp, pq)
                break
            if placed is None:
                continue
            if self._settle(150) and self._valid_contacts() and self._part_in_bounds() \
                    and self._assembly_distance() >= (min(rc.partial_min_dasm, rc.nut_close_min - 1e-4)
                                                      if self.task.family == "nut" else rc.partial_min_dasm):
                return self._state()
        return None

    def near(self):
        """Near-mating (AlphaNIST 'near-mating'; SGS 'Stable Grasp'): part grasped above its fixture."""
        rc = self.rcfg
        R_fix = quat_to_mat(self.fix_quat)
        axis = R_fix[:, 2]
        r = self.rng.uniform()
        side = self.task.family in rc.transport_tilt_families and self.rng.random() < rc.side_grasp_frac
        for _ in range(60):
            h = self.depth + r * self.rng.uniform(0, rc.near_height)
            lat = R_fix[:, :2] @ self.rng.uniform(-1, 1, 2) * r * rc.near_lateral
            part_pos = self.goal_pos + axis * h + lat
            tilt_axis = R_fix[:, :2] @ self.rng.normal(size=2)
            q = quat_mul(axis_angle_quat(axis, r * self.rng.uniform(-rc.near_yaw, rc.near_yaw)), self.goal_quat)
            part_quat = quat_mul(axis_angle_quat(tilt_axis, r * self.rng.uniform(0, rc.near_tilt)), q)
            if side:
                g = self._down_grasp(part_pos, part_quat, kind="side_top", cone=rc.side_grasp_cone)
            else:
                g = self._down_grasp(part_pos, part_quat, kind="axial_top" if self.rng.random() < 0.7 else None)
            if g is None:
                continue
            tp, tq, half = self._world_grasp(part_pos, part_quat, g)
            if self._place_with_backoff(tp, tq, finger_q_for(half), part_pos, part_quat) is None:
                continue
            rel0 = self._part_in_tcp()
            if not self._settle(150):
                continue
            if len(self._contacts()) == 2 and np.linalg.norm(self._part_in_tcp() - rel0) < 0.01 \
                    and self._valid_contacts() and self._part_in_bounds():
                self.level = r
                return self._state()
        return None

    def _part_in_tcp(self):
        tcp = self.d.site_xpos[self.ix.tcp_site]
        R = self.d.site_xmat[self.ix.tcp_site].reshape(3, 3)
        return R.T @ (self.d.xpos[self.ix.part_body] - tcp)

    def generate(self, family: str, n: int, log_every=0):
        """n states of one family; self.levels holds each state's difficulty level (near r, else -1)."""
        out, tries, self.levels = [], 0, []
        pool = []
        while len(out) < n:
            tries += 1
            self.level = -1.0
            if family == "loose":
                s = self.loose()
            elif family == "pickup":
                if not pool:
                    pool = [x for x in (self.loose() for _ in range(8)) if x is not None]
                    if not pool:
                        continue
                s = self.pickup(pool.pop())
                self.level = -1.0
            elif family == "inhand":
                s = self.inhand()
            elif family == "partial":
                s = self.partial()
            elif family == "near":
                s = self.near()
            else:
                raise ValueError(family)
            if s is not None:
                out.append(s); self.levels.append(self.level)
                if log_every and len(out) % log_every == 0:
                    print(f"{self.task.name} {family}: {len(out)}/{n} (attempts {tries})", flush=True)
        return np.stack(out), tries

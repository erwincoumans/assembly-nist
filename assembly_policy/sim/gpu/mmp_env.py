"""Metamorphysics (MMP) backend: the task scenes as MMP bodies, the Panda as an MMP robot in reduced coordinates.

Each environment is a world of MMP's pair filter, on a grid (MMP filters worlds after the broadphase), holding:
  the Panda: an Articulation made from the compiled MuJoCo model (Menagerie kinematics, inertias, armature, joint
    damping and collision geoms; the libfranka PD gains as implicit PD drives, the finger actuator's on each finger,
    the fingers coupled as MuJoCo's joint equality couples them, the arm weightless as gravcomp makes it),
  the part: a dynamic body (MuJoCo's mass, centre of mass and principal inertia) with the part's convex geoms,
  the fixture, the board and the other fixtures: static bodies of the task model's geoms,
with the task models' collision bitmasks (contype, conaffinity) as MMP's collision filters. The SDF geoms of nuts
and connectors collide as MMPConfig.geometry says: MuJoCo Warp's SDF contacts (mmp_sdf.py, the default), MMP's SDF
collider, or convex pieces (convex.py). Cylinders become 32-sided prisms, and hulls of many vertices (robot links,
obstacles, proxies) are simplified within a tolerance. Friction is MMP's product of body coefficients, chosen so
that pairs get MuJoCo's (priority) friction, 0.75, but the fingers' (MMPConfig.finger_friction).

The robots' arrays (q, qd, control) and the bodies' records are shared with torch; the state is converted to and
from MuJoCo's (part frame pose, free-joint velocities) on the GPU. One control step is `collisions` world steps of
`substeps` PGS sub-steps; joint-speed violations and the PD torques are taken after each world step.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import torch

from ..env import EnvConfig
from ..scene import (ARM, ARM_KD, ARM_KP, BOLT_PROXY, ENV, GRIP, PART, THREAD, TORQUE_LIMIT, VELOCITY_LIMIT, PhysicsConfig,
                     build_xml)
from ..tasks import TASKS
from .base import NQ, NV, GpuAssemblyEnv, quat_to_mat_t
from .hulls import simplify_hull

# friction per body: products give MuJoCo's pair friction, 2.0 with a finger (priority 2), else 0.75
MU_FINGER = math.sqrt(2.0 / 0.375)
MU_OTHER = 0.375 * MU_FINGER
ROBOT_BODIES = ["link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand", "left_finger",
                "right_finger"]
FINGERS = ("left_finger", "right_finger")


@dataclass
class MMPConfig:
    collisions: int = 4               # world steps (collision detections) per 40 ms control step
    substeps: int = 16                # PGS sub-steps per world step: 0.625 ms, so that MMP's contact stiffness, at
                                      # most 0.125 / sub-step, can be 200 Hz (at 100 Hz the 35 N grip sinks 0.5 mm in)
    pgs: dict = field(default_factory=lambda: dict(contact_hertz=200.0, twist_friction=0.1, linear_slop=0.0002))
    speculative: float | bool = 0.002  # speculative contact distance of convex shapes [m] (True: MMP's default)
    response_every_substep: bool = True
    spacing: float = 2.0              # between the environments' worlds on the grid [m]
    # how the SDF geoms of nuts and connectors collide: "mjwarp" (MuJoCo Warp's collision of the CPU models' SDFs,
    # mmp_sdf.py), "mjfit" (convex pieces, convex.py) or "sdfdcd" (MMP's SdfCollider)
    geometry: str = "mjwarp"
    sdfdcd: dict = field(default_factory=dict)   # SdfDcdContacts' keyword arguments (mmp_sdf.py)
    arm_damping: float = 1.0          # MuJoCo's passive joint damping (Menagerie: 1, fingers: PhysicsConfig)
    # the fingers' coupling (MuJoCo's finger joint equality): spring and damper between the two finger joints
    finger_coupling: tuple = (5e4, 100.0)
    # hulls simplified to within these distances (hulls.simplify_hull): MMP's SAT is serial per pair of compound
    # children, and the Menagerie links' and the fixtures' hulls have 50 to 2800 vertices
    robot_hull_tol: float = 0.002     # the arm and hand
    finger_hull_tol: float = 0.0005
    obstacles: bool = True            # the other fixtures on the board, as static hulls
    proxy_hull_tol: float = 2e-4      # the nuts' and connectors' proxies (hulls of up to 830 vertices)
    merge_links: tuple = ("link5",)   # links whose hulls become one (no compound: SAT per child pair is serial)
    finger_pads_only: bool = False    # (test) fingers collide with their pads only
    obstacle_hull_tol: float = 0.001
    obstacle_bodies: bool = True      # each obstacle a static body of its own (compound children's SAT is serial)
    mpr: bool | None = None           # MMP's MPR pass and its point as the manifold's first (global; None: keep)
    mpr_contact_point: bool | None = None
    part_recycling: bool = False      # MMP's contact recycling (manifolds reused while a pair barely moves) for parts
    # pair friction of a finger with anything: MuJoCo has 2.0 (priority 2), but its soft friction creeps under a
    # sustained push, which lets a held part slide into the palm where it aligns (P4, P8); 1.25 is closest overall
    finger_friction: float = 1.25
    finger_force: float | None = None # per finger [N] (default: PhysicsConfig.gripper_force / 2)


def _import_mmp():
    """metamorphysics and its robots.py (examples/Metamorphysics/python: $MMP_PYTHON_DIR, or a metamorphysics
    checkout beside this repository's parents)."""
    import metamorphysics as mmp
    try:
        import robots
    except ImportError:
        guess = [os.environ.get("MMP_PYTHON_DIR", "")] + [str(p / "metamorphysics/examples/Metamorphysics/python")
                                                           for p in Path(__file__).resolve().parents]
        for d in guess:
            if d and (Path(d) / "robots.py").exists():
                sys.path.insert(0, d)
                break
        import robots
    if not getattr(robots, "_assemblenist_patched", False):
        original = robots.link_shapes

        def link_shapes(info, link, *a, **k):
            return info.shapes[link] if isinstance(info, MjPanda) else original(info, link, *a, **k)
        robots.link_shapes = link_shapes
        robots._assemblenist_patched = True
    return mmp, robots


def xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def _geom_shape(m, g):
    """A MuJoCo geom as a robots.link_shapes shape in its body's frame: ("box", half sizes, pos, quat xyzw) or
    ("hull", points, pos, quat xyzw)."""
    t = int(m.geom_type[g])
    pos, quat = m.geom_pos[g].copy(), xyzw(m.geom_quat[g])
    if t == mujoco.mjtGeom.mjGEOM_BOX.value:
        return ("box", m.geom_size[g].copy(), pos, quat)
    if t == mujoco.mjtGeom.mjGEOM_MESH.value:
        k = m.geom_dataid[g]
        return ("hull", m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]].astype(np.float64), pos, quat)
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER.value:
        r, h = m.geom_size[g][:2]
        a = 2 * np.pi * np.arange(32) / 32
        ring = np.c_[r * np.cos(a), r * np.sin(a)]
        return ("hull", np.vstack([np.c_[ring, np.full(32, -h)], np.c_[ring, np.full(32, h)]]), pos, quat)
    raise NotImplementedError(f"geom type {t} of {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)}")


def panda_urdf():
    """A URDF of the Menagerie Panda with its visual meshes (no collision geometry) in the links' order of MjPanda,
    for drawing it (tds_robots.Scene); written once to the cache directory."""
    from ..scene import PANDA_XML
    out = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "assemblenist" / "menagerie_panda.urdf"
    if out.exists() and out.stat().st_mtime > PANDA_XML.stat().st_mtime:
        return out
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    m = spec.compile()
    meshdir = PANDA_XML.parent / (spec.meshdir or "assets")
    rgba = {mat.name: list(mat.rgba) for mat in spec.materials}

    def rpy(q):
        w, x, y, z = q
        return (math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)), math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
                math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    lines = ['<?xml version="1.0"?>', '<robot name="menagerie_panda">']
    for name in ROBOT_BODIES:
        body = spec.body(name)
        lines.append(f'  <link name="{name}">')
        for g in body.geoms:
            if g.type != mujoco.mjtGeom.mjGEOM_MESH or g.classname.name != "visual":
                continue
            c = rgba.get(g.material, [0.8, 0.8, 0.8, 1.0])
            r = rpy(g.quat)
            lines += [f'    <visual><origin xyz="{g.pos[0]:.6g} {g.pos[1]:.6g} {g.pos[2]:.6g}" rpy="{r[0]:.6g} {r[1]:.6g} {r[2]:.6g}"/>',
                      f'      <geometry><mesh filename="{os.path.relpath(meshdir / spec.mesh(g.meshname).file, out.parent)}"/></geometry>',
                      f'      <material name="{g.material or "grey"}"><color rgba="{c[0]:.4g} {c[1]:.4g} {c[2]:.4g} {c[3]:.4g}"/></material></visual>']
        lines.append('  </link>')
    for name in ROBOT_BODIES[1:]:
        b = m.body(name)
        j = b.jntadr[0]
        kind = "fixed" if j < 0 else ("revolute" if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE else "prismatic")
        jname = m.jnt(j).name if j >= 0 else f"{name}_fixed"
        r = rpy(b.quat)
        lines.append(f'  <joint name="{jname}" type="{kind}"><parent link="{m.body(b.parentid[0]).name}"/>'
                     f'<child link="{name}"/><origin xyz="{b.pos[0]:.6g} {b.pos[1]:.6g} {b.pos[2]:.6g}" '
                     f'rpy="{r[0]:.9g} {r[1]:.9g} {r[2]:.9g}"/>' +
                     (f'<axis xyz="{m.jnt_axis[j][0]:g} {m.jnt_axis[j][1]:g} {m.jnt_axis[j][2]:g}"/>'
                      f'<limit lower="{m.jnt_range[j][0]:g}" upper="{m.jnt_range[j][1]:g}" effort="100" velocity="3"/>'
                      if j >= 0 else "") + '</joint>')
    lines.append('</robot>')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out


class _Link:
    def __init__(self, name):
        self.name, self.mass, self.com, self.inertia = name, 0.0, np.zeros(3), np.zeros((3, 3))
        self.collision_shapes, self.visual_shapes = [], []


class _Joint:
    def __init__(self, name, kind, parent, child):
        self.name, self.type, self.parent, self.child = name, kind, parent, child
        self.position, self.rotation, self.axis = np.zeros(3), np.array([0., 0., 0., 1.]), np.array([1., 0., 0.])
        self.lower = self.upper = self.effort = self.velocity = None

    @property
    def movable(self):
        return self.type != "fixed"


class MjPanda:
    """robots.RobotModel's interface for the Panda of a compiled task model; its collision geoms are in `shapes`
    (robots.link_shapes is patched to take them)."""

    def __init__(self, m, hull_tol=0.0, finger_tol=0.0, merge=(), pads_only=False):
        # (the path: a URDF with the Menagerie visuals, for viewers; also the key of the links' shapes in a world)
        self.path, self.name, self.directory = str(panda_urdf()), "panda", ""
        ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ROBOT_BODIES]
        self.links, self.joints, self.shapes = [], [], {}
        for b, name in zip(ids, ROBOT_BODIES):
            L = _Link(name)
            L.mass, L.com = float(m.body_mass[b]), m.body_ipos[b].copy()
            R = np.empty(9); mujoco.mju_quat2Mat(R, m.body_iquat[b]); R = R.reshape(3, 3)
            L.inertia = R @ np.diag(m.body_inertia[b]) @ R.T
            self.links.append(L)
            self.shapes[name] = [_geom_shape(m, g) for g in range(m.ngeom)
                                 if m.geom_bodyid[g] == b and (m.geom_contype[g] or m.geom_conaffinity[g])]
            if name in merge and len(self.shapes[name]) > 1:
                from robots import shape_points
                self.shapes[name] = [("hull", np.concatenate([shape_points(x) for x in self.shapes[name]]),
                                      np.zeros(3), np.array([0., 0., 0., 1.]))]
            if pads_only and name in FINGERS:
                self.shapes[name] = [x for x in self.shapes[name] if x[0] == "box"]
            tol = finger_tol if name in FINGERS else hull_tol
            if tol > 0:
                self.shapes[name] = [(k, simplify_hull(v, tol) if k == "hull" else v, p, q)
                                     for k, v, p, q in self.shapes[name]]
            if name == "link0":
                continue
            j = m.body_jntadr[b]
            kind = "fixed" if j < 0 else {int(mujoco.mjtJoint.mjJNT_HINGE): "revolute", int(mujoco.mjtJoint.mjJNT_SLIDE): "prismatic"}[int(m.jnt_type[j])]
            J = _Joint(m.jnt(j).name if j >= 0 else f"{name}_fixed", kind, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                                                                        m.body_parentid[b]), name)
            J.position, J.rotation = m.body_pos[b].copy(), xyzw(m.body_quat[b])
            if j >= 0:
                assert np.allclose(m.jnt_pos[j], 0)
                J.axis = m.jnt_axis[j].copy()
                J.lower, J.upper = float(m.jnt_range[j][0]), float(m.jnt_range[j][1])
            self.joints.append(J)
        self.root_link = "link0"
        self.link_index = {l.name: k for k, l in enumerate(self.links)}
        self.collision_shapes = {l.name: [] for l in self.links}
        self.visual_shapes = {l.name: [] for l in self.links}

    @property
    def movable_joints(self):
        return [j for j in self.joints if j.movable]


# ----------------------------------------------------------------------------------------- torch quaternions (wxyz)
def qmul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def qconj(q):
    return torch.cat([q[..., :1], -q[..., 1:]], -1)


class TaskScene:
    """The MMP shapes of one task: the part (compound in its principal frame, mass, inertia, centre of mass and
    principal axes in the part frame), the fixture (compound, pose) and the board with the other fixtures."""

    def __init__(self, name, pcfg, world, register, geometry, obstacles=True, obstacle_tol=0.0, obstacle_bodies=False,
                 proxy_tol=0.0):
        mjm = mujoco.MjModel.from_xml_string(build_xml(name, pcfg))
        self.mjm = mjm
        body = lambda n: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, n)
        part, fixture = body("part"), body("fixture")
        keep = lambda g: (mjm.geom_contype[g] or mjm.geom_conaffinity[g]) and not (mjm.geom_contype[g] & (THREAD | BOLT_PROXY))
        # the part: geoms in the principal (inertial) frame
        self.mass = float(mjm.body_mass[part])
        self.inertia = mjm.body_inertia[part].copy()
        self.ipos, self.iquat = mjm.body_ipos[part].copy(), mjm.body_iquat[part].copy()
        Ri = np.empty(9); mujoco.mju_quat2Mat(Ri, self.iquat); Ri = Ri.reshape(3, 3)
        children = []
        # with convex pieces for the SDFs (mjfit) the part is those alone: its proxies for the board and the robot
        # (a nut's hull fills its hole) would collide with the fixture's pieces
        pieces = geometry not in ("mjwarp", "sdfdcd", "none") and TASKS[name].family in ("nut", "connector")
        proxies = TASKS[name].family in ("nut", "connector") and proxy_tol > 0
        simplify = lambda kind, size: simplify_hull(size, proxy_tol) if proxies and kind == "hull" else size
        for g in range(mjm.ngeom):
            if mjm.geom_bodyid[g] == part and keep(g) and not pieces:
                kind, size, pos, quat = _geom_shape(mjm, g)
                size = simplify(kind, size)
                q = np.empty(4); mujoco.mju_mulQuat(q, np.array([self.iquat[0], *-self.iquat[1:]]), mjm.geom_quat[g])
                children.append((kind, size, Ri.T @ (pos - self.ipos), xyzw(q)))
        from . import convex
        # geometry "mjwarp": the SDF geoms collide in MuJoCo Warp (mmp_sdf.py), the fixture body carries the proxies
        # that only the robot may touch (MMP's contacts of the part with it are dropped)
        self.sdf = geometry in ("mjwarp", "sdfdcd") and TASKS[name].family in ("nut", "connector")
        extra_part, extra_fix = convex.pieces(name, geometry) if TASKS[name].family in ("nut", "connector") \
            and not self.sdf else ([], [])
        for v, pos, q in extra_part:      # (in the part frame)
            qq = np.empty(4); mujoco.mju_mulQuat(qq, np.array([self.iquat[0], *-self.iquat[1:]]), q)
            children.append(("hull", v, Ri.T @ (pos - self.ipos), xyzw(qq)))
        self.part_shape = register(children)
        # the fixture: geoms in the fixture body's frame
        self.fix_pos, self.fix_quat = mjm.body_pos[fixture].copy(), mjm.body_quat[fixture].copy()
        children = [(k, simplify(k, v), p, q) for k, v, p, q in
                    (_geom_shape(mjm, g) for g in range(mjm.ngeom) if mjm.geom_bodyid[g] == fixture
                     and (keep(g) or (self.sdf and mjm.geom_contype[g] & BOLT_PROXY)))]
        children += [("hull", v, pos, xyzw(q)) for v, pos, q in extra_fix]
        self.fix_shape = register(children)
        # the board and the other fixtures (world geoms but the table plane), in the world frame
        children, self.obstacles = [], []
        for g in range(mjm.ngeom):
            if mjm.geom_bodyid[g] != 0 or not keep(g) or mjm.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            kind, size, pos, quat = _geom_shape(mjm, g)
            if mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, g).startswith("obst_"):
                if not obstacles:
                    continue
                if kind == "hull" and obstacle_tol > 0:
                    size = simplify_hull(size, obstacle_tol)
                if obstacle_bodies:
                    self.obstacles.append((register([(kind, size, np.zeros(3), np.array([0., 0., 0., 1.]))]), pos, quat,
                                           mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, g)))
                    continue
            children.append((kind, size, pos, quat))
        # one child (the board, without obstacles in it) is a body of its own shape at its pose: as a compound's
        # child its pairs would go through the serial compound SAT
        if len(children) == 1:
            kind, size, pos, quat = children[0]
            self.scene_shape = register([(kind, size, np.zeros(3), np.array([0., 0., 0., 1.]))])
            self.scene_pos, self.scene_quat = np.asarray(pos, float), np.asarray(quat, float)
        else:
            self.scene_shape = register(children)
            self.scene_pos, self.scene_quat = np.zeros(3), np.array([0., 0., 0., 1.])


class MMPBackend:
    """All environments in one MMP world; qpos / qvel are (N, 16) / (N, 15) torch copies of the state in MuJoCo's
    layout, refreshed after each simulate() and write()."""

    def __init__(self, task_names, task_of_env, ecfg: EnvConfig, pcfg: PhysicsConfig, device="cuda:0",
                 cfg: MMPConfig | None = None):
        import warp as wp
        mmp, robots = _import_mmp()
        self.cfg = cfg = cfg or MMPConfig()
        if cfg.mpr is not None:
            mmp.set_use_mpr(cfg.mpr)
        if cfg.mpr_contact_point is not None:
            mmp.set_use_mpr_contact_point(cfg.mpr_contact_point)
        self.task = np.asarray(task_of_env)
        self.N = N = len(self.task)
        self.dev = torch.device(device)
        self.substeps = np.full(N, cfg.collisions * cfg.substeps)
        side = int(math.ceil(math.sqrt(N)))
        origins = np.zeros((N, 3))
        origins[:, 0] = (np.arange(N) % side) * cfg.spacing
        origins[:, 1] = (np.arange(N) // side) * cfg.spacing
        self.origins_np = origins
        extent = side * cfg.spacing
        world = mmp.World(max_bodies=N * (16 + (19 if cfg.obstacle_bodies else 0)) + 4096, max_pairs_per_body=64, max_mesh_vertices=400000,
                          max_mesh_triangles=800000, device=self.dev.index or 0)
        self.world = world
        world.set_gravity([0.0, 0.0, -9.81])
        cells = int(min(64, max(8, 2 ** math.ceil(math.log2(max(1.0, extent / 6.0))))))
        world.set_solver_grid_dimensions([cells, cells, 2])
        world.set_batch_cell_size(max(6.0, extent / cells))
        world.add_body(0.0, [0, 0, 0], [0, 0, 0, 1], world.register_plane_shape(0, 0, 1, 0), friction=MU_OTHER)
        dt = ecfg.control_dt / cfg.collisions
        self.frame_dt = dt
        self.coupling = coupling = robots.Coupling(world, frame_dt=dt, substeps=cfg.substeps, self_collision=False,
                                                   response_every_substep=cfg.response_every_substep,
                                                   speculative_spheres=False)
        coupling.shapes.merge = False
        world.set_speculative_contacts(convex=cfg.speculative if isinstance(cfg.speculative, bool) else float(cfg.speculative))

        # the robot
        m0 = mujoco.MjModel.from_xml_string(build_xml(task_names[int(self.task[0])], pcfg))
        self.mjm = m0
        self.panda = MjPanda(m0, cfg.robot_hull_tol, cfg.finger_hull_tol, cfg.merge_links, cfg.finger_pads_only)
        joints = [j.name for j in self.panda.movable_joints]
        # the finger actuator acts on the tendon (q1 + q2) / 2: half its gains and force on each finger
        gains = {j: (ARM_KP[i], ARM_KD[i]) for i, j in enumerate(joints[:7])}
        gains.update({j: (0.5 * pcfg.gripper_kp, 0.5 * pcfg.gripper_kv) for j in joints[7:]})
        effort = {j: TORQUE_LIMIT[i] for i, j in enumerate(joints[:7])}
        effort.update({j: cfg.finger_force or 0.5 * pcfg.gripper_force for j in joints[7:]})
        art = robots.Articulation(self.panda, floating=False, gains={f"^{j}$": g for j, g in gains.items()},
                                  effort_limits=effort, joint_ranges="urdf", collision={"links": "all"})
        dof_armature = [m0.dof_armature[m0.jnt_dofadr[mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_JOINT, j)]] for j in joints]
        art.arrays["armature"] = np.array([0.0] + dof_armature, dtype=np.float32)
        art.arrays["gravity_scale"] = 0.0
        nb = art.num_bodies
        assert nb == 10 and art.joint_names[7:] == ["finger_joint1", "finger_joint2"]
        art.arrays["damping"] = np.array([0.0] + [cfg.arm_damping] * 7 + [pcfg.finger_damping] * 2, np.float32)
        art.arrays["coupled"] = np.array([0] * 8 + [9, 8], np.int32)
        art.arrays["coupling_stiffness"] = np.array([0.0] * 8 + [cfg.finger_coupling[0]] * 2, np.float32)
        art.arrays["coupling_damping"] = np.array([0.0] * 8 + [cfg.finger_coupling[1]] * 2, np.float32)
        self.art = art
        self.group = coupling.add_robots(art, num_copies=N, origins=origins, worlds=np.arange(N), friction=MU_OTHER,
                                         link_mass=1.0, visual_hulls=False)

        # the task scenes' shapes, once per task
        hulls = {}

        def register_hull(v):
            v = np.asarray(v, np.float64)
            key = v.round(9).tobytes()
            if key not in hulls:
                try:
                    hulls[key] = world.register_convex_hull_shape(v.astype(np.float32).reshape(-1).tolist(), [1.0, 1.0, 1.0])
                except ValueError:
                    hulls[key] = coupling.shapes._register_hull(v)
            return hulls[key]

        def register(children):
            out = []
            for kind, size, pos, quat in children:
                s = world.register_box_shape(*[float(x) for x in size]) if kind == "box" else register_hull(size)
                out.append((s, [float(x) for x in pos], [float(x) for x in quat]))
            if len(out) == 1 and np.allclose(out[0][1], 0) and np.allclose(out[0][2], [0, 0, 0, 1]):
                return out[0][0]
            idx = world.register_compound_shape(out)
            if idx < 0:
                raise RuntimeError("no room for more shapes in the MMP world")
            return idx

        self.scenes = {}
        for k in np.unique(self.task):
            self.scenes[int(k)] = TaskScene(task_names[int(k)], pcfg, world, register, cfg.geometry, cfg.obstacles,
                                            cfg.obstacle_hull_tol, cfg.obstacle_bodies, cfg.proxy_hull_tol)

        # the bodies of the environments: part, fixture, board (consecutive per kind)
        sc = [self.scenes[int(k)] for k in self.task]
        R0 = lambda q: xyzw(q)
        f32 = lambda x: np.asarray(x, np.float32)
        self.part_body = world.add_bodies(
            f32([s.mass for s in sc]), f32(origins + np.array([0.0, 0.0, 1.0]) + np.arange(N)[:, None] * 0),
            f32(np.tile([0, 0, 0, 1], (N, 1))), np.asarray([s.part_shape for s in sc], np.int32),
            f32(np.full(N, MU_OTHER)), local_inertia=f32([s.inertia for s in sc])) + np.arange(N)
        fix_first = world.add_bodies(f32(np.zeros(N)), f32(origins + np.array([s.fix_pos for s in sc])),
                                     f32([R0(s.fix_quat) for s in sc]), np.asarray([s.fix_shape for s in sc], np.int32),
                                     f32(np.full(N, MU_OTHER)))
        scene_first = world.add_bodies(f32(np.zeros(N)), f32(origins + np.array([s.scene_pos for s in sc])),
                                       f32([s.scene_quat for s in sc]),
                                       np.asarray([s.scene_shape for s in sc], np.int32), f32(np.full(N, MU_OTHER)))
        for e in range(N):
            for b in (self.part_body[e], fix_first + e, scene_first + e):
                coupling._body_world(int(b), e)
        self.scene_body = scene_first + np.arange(N)
        self.obstacle_bodies = [[] for _ in range(N)]       # per environment: (body, geom name of the task model)
        no = np.array([len(s.obstacles) for s in sc])
        if no.sum():
            env = np.repeat(np.arange(N), no)
            obs = [o for s in sc for o in s.obstacles]
            first = world.add_bodies(f32(np.zeros(len(obs))), f32(origins[env] + np.array([o[1] for o in obs])),
                                     f32([o[2] for o in obs]), np.asarray([o[0] for o in obs], np.int32),
                                     f32(np.full(len(obs), MU_OTHER)))
            for i, e in enumerate(env):
                coupling._body_world(first + i, int(e))
                self.obstacle_bodies[e].append((first + i, obs[i][3]))
        if not cfg.part_recycling:
            for k in self.scenes.values():
                world.set_shape_contact_recycling(k.part_shape, False)
        world.finalize()
        coupling.start()
        self.fix_body = fix_first + np.arange(N)
        # the CPU models' collision bitmasks (scene.py) as MMP's pair filter: the arm and the gripper touch the
        # environment and the part, the part the environment; a fixture of robot-only proxies only the robot
        contype = np.full(world.num_bodies, ENV, np.int32)
        conaffinity = np.full(world.num_bodies, PART | GRIP | ARM, np.int32)
        links = self.group.link_bodies()
        for l, name in enumerate(art.link_names):
            b = links[:, l][links[:, l] >= 0]
            contype[b] = GRIP if name in ("hand", *FINGERS) else ARM
            conaffinity[b] = ENV | PART
        contype[self.part_body], conaffinity[self.part_body] = PART, ENV | GRIP | ARM
        for e, k in enumerate(self.task):
            if self.scenes[int(k)].sdf:
                contype[self.fix_body[e]], conaffinity[self.fix_body[e]] = BOLT_PROXY, GRIP | ARM
        world.set_body_collision_filters(contype, conaffinity)
        world.set_pgs(**cfg.pgs)

        # shared views of the robots' arrays and the bodies' records
        wpdev = wp.get_device(str(self.dev))
        view = lambda ptr, shape: wp.to_torch(wp.array(ptr=ptr, dtype=wp.float32, shape=shape, device=wpdev, copy=False))
        g = self.group
        self.q, self.qd = view(g.pointer("q"), (N, 9)), view(g.pointer("qd"), (N, 9))
        self.control = view(g.pointer("control"), (N, 9, 4))
        layout = mmp.struct_layout["MMP_RigidBodyData"]
        assert layout["size"] == 80 and layout["pos"] == 0 and layout["quat"] == 16
        self.bodies = view(world.bodies_ptr, (world.num_bodies, 20))
        self.stream = torch.cuda.ExternalStream(world.stream_ptr, device=self.dev)
        t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=self.dev)
        self.part_ids = t(self.part_body, torch.long)
        self.origins = t(origins)
        self.ipos = t([s.ipos for s in sc])
        self.iquat = t([s.iquat for s in sc])
        self.kp, self.kd, self.tlim = t(ARM_KP), t(ARM_KD), t(TORQUE_LIMIT)
        self.vlim = t(np.asarray(VELOCITY_LIMIT) * ecfg.velocity_limit_scale)
        # finger friction (MMP_RigidBodyData.m_frictionCoeff, float 19 of the record)
        links = g.link_bodies()
        fingers = t(links[:, [art.link_names.index(f) for f in FINGERS]].reshape(-1), torch.long)
        with torch.cuda.stream(self.stream):
            self.bodies[fingers, 19] = cfg.finger_friction / MU_OTHER
        self.qpos = torch.zeros(N, NQ, device=self.dev)
        self.qvel = torch.zeros(N, NV, device=self.dev)
        # SDF contacts (MuJoCo Warp's) of the environments whose tasks collide SDFs, and the part-fixture contacts of
        # MMP's own that they replace (the fixture body is then the robot-only proxies)
        self.sdf = None
        sdf_tasks = [k for k, s in self.scenes.items() if s.sdf]
        if sdf_tasks:
            from .mmp_sdf import SdfContacts, SdfDcdContacts
            groups = []
            for k in sdf_tasks:
                ids = np.nonzero(self.task == k)[0]
                groups.append((task_names[k], ids, self.part_body[ids], self.fix_body[ids],
                               origins[ids] if cfg.geometry == "mjwarp" else self.scenes[k].ipos,
                               None if cfg.geometry == "mjwarp" else self.scenes[k].iquat))
            if cfg.geometry == "mjwarp":
                self.sdf = SdfContacts([g[:5] for g in groups], pcfg, str(self.dev), world.stream_ptr)
            else:
                self.sdf = SdfDcdContacts(mmp, world, groups, str(self.dev), world.stream_ptr, **cfg.sdfdcd)
            self.combined = torch.zeros(0, 28, device=self.dev)
            # (the SDF collision's kernels built and its graph captured now, not in a simulation thread)
            with torch.cuda.stream(self.stream):
                self._read()
                self.sdf.compute(self.qpos)
        self.world.synchronize()

    def close(self):
        pass

    def _read(self):
        """Robot and part state into qpos / qvel (MuJoCo layout), in the world's stream."""
        self.qpos[:, :9] = self.q
        self.qvel[:, :9] = self.qd
        rec = self.bodies[self.part_ids]
        qb = torch.cat([rec[:, 7:8], rec[:, 4:7]], 1)                  # xyzw -> wxyz
        qf = qmul(qb, qconj(self.iquat))
        qf = qf / qf.norm(dim=1, keepdim=True)
        R = quat_to_mat_t(qf)
        r = (R @ self.ipos[:, :, None])[:, :, 0]
        w = rec[:, 12:15]
        self.qpos[:, 9:12] = rec[:, 0:3] - self.origins - r
        self.qpos[:, 12:16] = qf
        self.qvel[:, 9:12] = rec[:, 8:11] - torch.cross(w, r, dim=1)
        self.qvel[:, 12:15] = (R.transpose(1, 2) @ w[:, :, None])[:, :, 0]

    def write(self, env_ids, qpos, qvel):
        self.stream.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(self.stream):
            self.q[env_ids] = qpos[:, :9]
            self.qd[env_ids] = qvel[:, :9]
            self.control[env_ids, :, 0] = qpos[:, :9]
            qf = qpos[:, 12:16] / qpos[:, 12:16].norm(dim=1, keepdim=True)
            R = quat_to_mat_t(qf)
            r = (R @ self.ipos[env_ids][:, :, None])[:, :, 0]
            w = (R @ qvel[:, 12:15, None])[:, :, 0]
            qb = qmul(qf, self.iquat[env_ids])
            rec = self.bodies[self.part_ids[env_ids]]
            rec[:, 0:3] = qpos[:, 9:12] + self.origins[env_ids] + r
            rec[:, 4:8] = torch.cat([qb[:, 1:4], qb[:, 0:1]], 1)
            rec[:, 8:11] = qvel[:, 9:12] + torch.cross(w, r, dim=1)
            rec[:, 12:15] = w
            self.bodies[self.part_ids[env_ids]] = rec
        self.world.sync_articulations()
        with torch.cuda.stream(self.stream):
            self._read()
        torch.cuda.current_stream(self.dev).wait_stream(self.stream)

    def _world_step(self, dt):
        """world.step(dt), or with SDF contacts its stages, the SDF contacts appended to MMP's own."""
        w = self.world
        if self.sdf is None:
            w.step(dt)
            return
        w.compute_aabbs()
        pairs = w.compute_overlapping_pairs()
        nc = w.compute_contacts() if pairs else 0
        with torch.cuda.stream(self.stream):
            self._read()
            ns = self.sdf.compute(self.qpos)
            total = nc + ns
            if len(self.combined) < total:
                self.combined = torch.zeros(int(total * 1.5) + 1024, 28, device=self.dev)
            if nc:
                import warp as wp
                own = wp.to_torch(wp.array(ptr=w.contacts_ptr, dtype=wp.float32, shape=(nc, 28),
                                           device=wp.get_device(str(self.dev)), copy=False))
                self.combined[:nc] = own
            if ns:
                self.combined[nc:total] = self.sdf.records[:ns]
        w.solve_joints(dt)
        w.solve_and_integrate(int(total), self.combined.data_ptr(), dt)

    def simulate(self, ctrl):
        self.stream.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(self.stream):
            self.control[:, :7, 0] = ctrl[:, :7]
            self.control[:, 7:9, 0] = ctrl[:, 7:8]
            viol = torch.zeros(self.N, dtype=torch.bool, device=self.dev)
            tau = torch.zeros(self.N, 7, device=self.dev)
        for _ in range(self.cfg.collisions):
            self._world_step(self.frame_dt)
            with torch.cuda.stream(self.stream):
                qd = self.qd[:, :7]
                viol |= (qd.abs() > self.vlim).any(1)
                tau += torch.minimum(torch.maximum(self.kp * (ctrl[:, :7] - self.q[:, :7]) - self.kd * qd, -self.tlim), self.tlim)
        with torch.cuda.stream(self.stream):
            tau /= self.cfg.collisions
            self._read()
        torch.cuda.current_stream(self.dev).wait_stream(self.stream)
        return viol, tau


def make_mmp_env(task_names, task_of_env, ecfg: EnvConfig, pcfg: PhysicsConfig | None = None, device="cuda:0",
                 cfg: MMPConfig | None = None, **kw):
    pcfg = pcfg or PhysicsConfig(fine_families=ecfg.fine_families, fine_sdf_initpoints=ecfg.fine_sdf_initpoints,
                                 fine_sdf_iterations=ecfg.fine_sdf_iterations)
    cfg = cfg or MMPConfig(**kw)
    with torch.cuda.device(torch.device(device)):
        backend = MMPBackend(task_names, task_of_env, ecfg, pcfg, device, cfg)
        return GpuAssemblyEnv(backend, task_names, task_of_env, ecfg, backend.mjm, device)

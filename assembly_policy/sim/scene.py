"""MJCF scene construction: Franka Panda + NIST board + one assembly pair.

All task models share one layout (9 robot dof + 1 free part, identical actuators
and sensors), so a single batched rollout can mix tasks.

Collision filtering (contype / conaffinity bits):
  1 static environment (table, board, fixture)
  2 movable part
  4 gripper (hand, fingers)
  8 arm links
Pairs tested: env-part, env-gripper, env-arm, part-gripper, part-arm.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from .tasks import (BOARD_CENTER_WORLD_X, BOARD_FIXTURES, BOARD_HALF, BOARD_TOP_WORLD_Z, MESH_DIR, ROOT, TASKS,
                    Task, load_obj, scene_pose)

PANDA_XML = ROOT / "assets/franka/panda.xml"     # MuJoCo Menagerie franka_emika_panda

# libfranka joint impedance example gains (paper: "manufacturer's libfranka stiffness and damping").
ARM_KP = (600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0)
ARM_KD = (50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0)
TORQUE_LIMIT = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
VELOCITY_LIMIT = (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)
TCP_OFFSET = 0.1034  # hand frame -> fingertip centre

ENV, PART, GRIP, ARM = 1, 2, 4, 8
THREAD, BOLT_PROXY = 16, 32   # nut<->bolt thread SDF pair; bolt proxy collides with the robot only


@dataclass
class PhysicsConfig:
    # 2.5 ms with contact time constant 5 ms: at 5 ms / 10 ms, 2.4 m/s drop impacts gained up
    # to 3x kinetic energy (R4 diverged); 2.5 ms / 5 ms dissipates (KE ratio <= 0.96).
    timestep: float = 0.0025
    integrator: str = "implicitfast"
    cone: str = "elliptic"
    iterations: int = 50
    ls_iterations: int = 20
    tolerance: float = 1e-8
    multiccd: bool = False
    # contact parameters (MuJoCo soft-contact)
    solref: str = "0.005 1"
    solimp: str = "0.95 0.99 0.0005 0.5 2"
    part_friction: str = "0.75 0.005 0.0001"
    finger_friction: str = "2.0 0.005 0.0001"
    env_friction: str = "0.75 0.005 0.0001"
    part_mass: float = 0.05         # UW Lab USD MassAPI value for the NIST parts
    # Factory-style gripper: 7500 N/m, 173 N s/m, 40 N per finger. The actuator acts on the
    # finger-split tendon L=(q1+q2)/2, so tendon values are twice the per-finger values.
    gripper_kp: float = 15000.0
    gripper_kv: float = 346.0
    gripper_force: float = 70.0     # 35 N per finger (Franka Hand continuous force rating 70 N)
    # Implicit finger-joint damping caps finger speed near gripper_force/2/finger_damping
    # (~0.13 m/s; real Franka Hand <= 0.1 m/s). Without it the force-saturated servo snaps
    # the 15 g fingers at > 2 m/s and launches parts.
    finger_damping: float = 300.0
    bore_segments: int = 32
    rod_segments: int = 24
    sdf_initpoints: int = 40      # MuJoCo SDF collision (nut/bolt threads); M4 uses 80
    sdf_iterations: int = 10
    # per-family stiff contact physics ("gear+nut" etc.; empty = same physics for every task). At 2.5 ms
    # with a 5 ms contact time constant, impacts push gear teeth through each other and make nuts skip
    # threads; 0.5 ms with a 1 ms time constant blocks both.
    fine_families: str = ""
    fine_timestep: float = 0.0005
    fine_solref: str = "0.001 1"
    fine_solimp: str = "0.95 0.99 0.0001 0.5 2"
    fine_sdf_initpoints: int = 0      # 0 = same as the coarse rule (40, M4 80)
    fine_sdf_iterations: int = 10


def task_physics(task_name: str, cfg: "PhysicsConfig") -> "PhysicsConfig":
    """The physics configuration actually used for one task (stiff contacts for fine-feature families)."""
    import dataclasses
    fams = {f for f in cfg.fine_families.replace(",", "+").split("+") if f}
    if TASKS[task_name].family in fams:
        return dataclasses.replace(cfg, timestep=cfg.fine_timestep, solref=cfg.fine_solref, solimp=cfg.fine_solimp)
    return cfg


def _add(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: (" ".join(f"{x:.9g}" for x in v) if isinstance(v, (list, tuple, np.ndarray)) else str(v))
                                        for k, v in attrs.items()})


def _add_geom(body, asset, name, g, defaults):
    """Add a geom dict (mesh vertices become an asset; type='sdf' uses a plugin instance)."""
    g = dict(g)
    if g["type"] == "mesh":
        _add(asset, "mesh", name=f"{name}_mesh", vertex=np.asarray(g.pop("vertex")).ravel())
        g["mesh"] = f"{name}_mesh"
    inst = g.pop("plugin_instance", None)
    if inst is not None:
        if asset.find(f"mesh[@name='sdf_{inst}']") is None:
            m = _add(asset, "mesh", name=f"sdf_{inst}")
            _add(m, "plugin", instance=inst)
        g["mesh"] = f"sdf_{inst}"
    el = _add(body, "geom", name=name, **{**defaults, **g})
    if inst is not None:
        _add(el, "plugin", instance=inst)
    return el


def _quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.r_[math.cos(angle / 2), math.sin(angle / 2) * axis]


# ---------------------------------------------------------------- fixtures
def bore_round_boxes(radius, height, outer_half, n, z0=0.0):
    """Tangent boxes whose inner faces circumscribe a circle of `radius`, plus a square frame."""
    geoms = []
    tang = radius * math.tan(math.pi / n) * 1.15
    thick = radius * (math.sqrt(2) - 1) + 0.0015
    for i in range(n):
        th = 2 * math.pi * i / n
        c = (radius + thick / 2) * np.array([math.cos(th), math.sin(th)])
        geoms.append(dict(type="box", size=[thick / 2, tang, height / 2], pos=[c[0], c[1], z0 + height / 2],
                          quat=_quat_from_axis_angle([0, 0, 1], th)))
    # square frame touching the circle (opening 2r x 2r)
    for sx in (-1, 1):
        w = (outer_half - radius) / 2
        geoms.append(dict(type="box", size=[w, outer_half, height / 2], pos=[sx * (radius + w), 0, z0 + height / 2]))
        geoms.append(dict(type="box", size=[radius, w, height / 2], pos=[0, sx * (radius + w), z0 + height / 2]))
    return geoms


def bore_rect_boxes(hx, hy, height, outer_half, z0=0.0):
    geoms = []
    for s in (-1, 1):
        w = (outer_half - hx) / 2
        geoms.append(dict(type="box", size=[w, outer_half, height / 2], pos=[s * (hx + w), 0, z0 + height / 2]))
        w = (outer_half - hy) / 2
        geoms.append(dict(type="box", size=[hx, w, height / 2], pos=[0, s * (hy + w), z0 + height / 2]))
    return geoms


def fixture_geoms(task: Task, cfg: PhysicsConfig):
    """Returns (collision geoms in fixture frame, sample-surface mesh stem)."""
    f = task.fixture_asset
    if f.startswith("round_hole"):
        r = {"round_hole_4mm": 0.00205, "round_hole_8mm": 0.004049, "round_hole_12mm": 0.006099,
             "round_hole_16mm": 0.00825}[f]
        return bore_round_boxes(r, 0.009, 0.0125, cfg.bore_segments)
    if f == "hole_8mm":  # Forge-style deep 8 mm hole: r=4.5 mm, depth 24 mm, closed bottom
        g = bore_round_boxes(0.0045, 0.025, 0.0122, cfg.bore_segments)
        g.append(dict(type="box", size=[0.0122, 0.0122, 0.0015], pos=[0, 0, -0.0015]))
        return g
    if f.startswith("rectangular_hole"):
        hx, hy = {"rectangular_hole_4mm": (0.002055, 0.002055), "rectangular_hole_8mm": (0.004072, 0.003567),
                  "rectangular_hole_12mm": (0.006089, 0.004069), "rectangular_hole_16mm": (0.008109, 0.005079)}[f]
        return bore_rect_boxes(hx, hy, 0.009, 0.0125)
    if task.family == "connector":
        return [dict(type="sdf", plugin_instance="socket", contype=THREAD, conaffinity=THREAD)] + \
               [dict(type="mesh", vertex=v, contype=BOLT_PROXY, conaffinity=GRIP | ARM) for v in proxy_pieces(f)]
    if f.startswith("bolt_"):
        from .threads import THREADS
        P, crest, root, htop, top, *_ = THREADS[task.name]
        return [dict(type="sdf", plugin_instance="bolt", contype=THREAD, conaffinity=THREAD),
                dict(type="cylinder", size=[crest, (top - htop) / 2], pos=[0, 0, (top + htop) / 2],
                     contype=BOLT_PROXY, conaffinity=GRIP | ARM)]
    if f == "gear_base":
        from .gears import GEARS, gear_base_geoms, gear_convex_pieces
        from .robot import quat_to_mat
        geoms = gear_base_geoms()
        fpos, fquat = scene_pose(task.fixture_scene)
        Rf = quat_to_mat(fquat)
        for other in GEARS:
            if other == task.part_asset:
                continue
            gpos, gquat = scene_pose(other)            # assembled pose of the static, meshed neighbour
            R = Rf.T @ quat_to_mat(gquat); t = Rf.T @ (gpos - fpos)
            for piece in gear_convex_pieces(other):
                geoms.append(dict(type="mesh", vertex=piece @ R.T + t))
        return geoms
    raise NotImplementedError(f"fixture collision model for {f}")


def chamfered_prism(radius, length, chamfer, n):
    """Convex polygonal rod (vertices on the circle), 45-degree chamfers at both ends."""
    ang = 2 * math.pi * np.arange(n) / n
    ring = lambda r, z: np.c_[r * np.cos(ang), r * np.sin(ang), np.full(n, z)]
    return np.vstack([ring(radius - chamfer, 0), ring(radius, chamfer), ring(radius, length - chamfer),
                      ring(radius - chamfer, length)])


def part_geoms(task: Task, cfg: PhysicsConfig):
    p = task.part_asset
    if p.startswith("round_peg"):
        r = {"round_peg_4mm": 0.0019985, "round_peg_8mm": 0.003998, "round_peg_12mm": 0.005998,
             "round_peg_16mm": 0.0079975}[p]
        return [dict(type="mesh", vertex=chamfered_prism(r, 0.05, 0.0005, cfg.rod_segments))]
    if p.startswith("rectangular_peg"):
        hx, hy = {"rectangular_peg_4mm_tight": (0.002, 0.002), "rectangular_peg_8mm_tight": (0.004, 0.0035),
                  "rectangular_peg_12mm_tight": (0.006, 0.004), "rectangular_peg_16mm_tight": (0.008, 0.005)}[p]
        return [dict(type="box", size=[hx, hy, 0.025], pos=[0, 0, 0.025])]
    if p.startswith("gear_"):
        from .gears import gear_convex_pieces
        return [dict(type="mesh", vertex=v) for v in gear_convex_pieces(p)]
    if task.family == "connector":
        # coarse convex proxies for board / fixtures / robot; the grid SDF only meets the socket SDF
        return [dict(type="mesh", vertex=v) for v in proxy_pieces(p)] + \
               [dict(type="sdf", plugin_instance="plug", contype=THREAD, conaffinity=THREAD, density=0)]
    if p.startswith("nut_"):
        # convex hex body for contacts with board / fixtures / robot; the thread SDF only meets the bolt
        return [dict(type="mesh", vertex=hull_vertices(p)),
                dict(type="sdf", plugin_instance="nut", contype=THREAD, conaffinity=THREAD, density=0)]
    raise NotImplementedError(f"part collision model for {p}")


# ---------------------------------------------------------------- robot
def _panda_meshdir():
    return PANDA_XML.parent / "assets"


def build_xml(task_name: str, cfg: PhysicsConfig | None = None, visual: bool = False) -> str:
    cfg = task_physics(task_name, cfg or PhysicsConfig())
    task = TASKS[task_name]
    root = ET.parse(PANDA_XML).getroot()
    root.set("model", f"assemblenist_{task_name}")
    root.find("compiler").set("meshdir", str(_panda_meshdir()))
    opt = root.find("option")
    opt.attrib.clear()
    opt.attrib.update(timestep=str(cfg.timestep), integrator=cfg.integrator, cone=cfg.cone,
                      iterations=str(cfg.iterations), ls_iterations=str(cfg.ls_iterations),
                      tolerance=str(cfg.tolerance), impratio="1")
    if cfg.multiccd:
        _add(opt, "flag", multiccd="enable")
    if task.family == "connector":
        from .threads import load_plugin
        load_plugin()
        pid, sid = CONNECTOR_GRIDS[task_name]
        ext = _add(root, "extension")
        pe = _add(ext, "plugin", plugin="assemblenist.sdf.grid")
        for inst, gid in (("plug", pid), ("socket", sid)):
            ie = _add(pe, "instance", name=inst)
            _add(ie, "config", key="id", value=str(gid))
        opt.set("sdf_initpoints", str(cfg.sdf_initpoints))
        opt.set("sdf_iterations", str(cfg.sdf_iterations))
    if task.family == "nut":
        from .threads import load_plugin, thread_attributes
        load_plugin()
        bolt_a, nut_a = thread_attributes(task_name)
        ext = _add(root, "extension")
        for plug, inst, attrs in (("assemblenist.sdf.bolt", "bolt", bolt_a), ("assemblenist.sdf.nut", "nut", nut_a)):
            pe = _add(ext, "plugin", plugin=plug)
            ie = _add(pe, "instance", name=inst)
            for k2, v2 in attrs.items():
                _add(ie, "config", key=k2, value=f"{v2:.9g}")
        # M4 needs denser sampling at the coarse 2.5 ms step; at the fine step contacts move little per step
        fine = task.family in {f for f in cfg.fine_families.replace(",", "+").split("+") if f}
        if fine and cfg.fine_sdf_initpoints > 0:
            opt.set("sdf_initpoints", str(cfg.fine_sdf_initpoints)); opt.set("sdf_iterations", str(cfg.fine_sdf_iterations))
        else:
            opt.set("sdf_initpoints", str(max(80, cfg.sdf_initpoints) if task_name == "N4" else cfg.sdf_initpoints))
            opt.set("sdf_iterations", str(cfg.sdf_iterations))
    root.remove(root.find("keyframe"))
    if visual:
        # brighter rendering for progress videos: strong ambient (no global illumination in MuJoCo),
        # a soft key light as the only shadow caster, two fill lights, low specular on the arm shell
        vis = _add(root, "visual")
        _add(vis, "headlight", ambient=[0.3, 0.3, 0.3], diffuse=[0.3, 0.3, 0.3], specular=[0.03, 0.03, 0.03])
        _add(vis, "quality", shadowsize="4096", offsamples="4")
        _add(vis, "global", offwidth="1280", offheight="960")
        for mat in root.find("asset").findall("material"):
            mat.set("specular", "0.15"); mat.set("shininess", "0.3")
            if mat.get("name") == "black":
                mat.set("rgba", "0.35 0.35 0.37 1")
        for d in root.find("default").iter("default"):
            if d.get("class") == "panda" and d.find("material") is not None:
                d.find("material").set("specular", "0.15")
        for li in list(root.find("worldbody").findall("light")):
            root.find("worldbody").remove(li)
    # drop visual meshes unless rendering
    asset = root.find("asset")
    if not visual:
        for body in root.iter("body"):
            for g in list(body.findall("geom")):
                if g.get("class") == "visual":
                    body.remove(g)
        for m in list(asset.findall("mesh")):
            if not m.get("name", "").endswith(("_c", "_c0", "_c1", "_c2")) and m.get("file") != "finger_0.obj":
                asset.remove(m)
    # collision classes -> bitmasks
    defaults = root.find("default")
    for d in defaults.iter("default"):
        if d.get("class") == "collision":
            d.find("geom").attrib.update(contype=str(ARM), conaffinity=str(ENV | PART), condim="3",
                                         friction=cfg.env_friction)
    for body in root.iter("body"):
        name = body.get("name", "")
        if name.startswith("link") or name == "hand":
            body.set("gravcomp", "1")
        if name in ("hand", "left_finger", "right_finger"):
            # Grip only through the flat 17 x 8 mm main pad: the four small Menagerie fingertip
            # boxes protrude 1.5 mm past it and produce tilted edge contacts that squirt parts out.
            for g in list(body.findall("geom")):
                if g.get("class", "") in ("fingertip_pad_collision_2", "fingertip_pad_collision_3",
                                          "fingertip_pad_collision_4", "fingertip_pad_collision_5"):
                    body.remove(g)
            for g in body.findall("geom"):
                if g.get("class", "").startswith(("collision", "fingertip")):
                    g.set("contype", str(GRIP)); g.set("conaffinity", str(ENV | PART))
                    if name != "hand":
                        g.set("friction", cfg.finger_friction); g.set("condim", "4"); g.set("priority", "2")
                        g.set("solref", cfg.solref); g.set("solimp", cfg.solimp)
    # arm actuators: libfranka impedance as implicit PD position servos, torque-limited
    acts = root.find("actuator")
    for i, a in enumerate(list(acts)):
        if i < 7:
            a.set("gainprm", f"{ARM_KP[i]}"); a.set("biasprm", f"0 {-ARM_KP[i]} {-ARM_KD[i]}")
            a.set("forcerange", f"{-TORQUE_LIMIT[i]} {TORQUE_LIMIT[i]}")
        else:  # finger tendon: ctrl is target opening per finger in metres
            a.set("ctrlrange", "-0.01 0.04"); a.set("gainprm", f"{cfg.gripper_kp}")
            a.set("biasprm", f"0 {-cfg.gripper_kp} {-cfg.gripper_kv}")
            a.set("forcerange", f"{-cfg.gripper_force} {cfg.gripper_force}")
    for j in root.iter("joint"):
        if j.get("name", "").startswith("finger_joint"):
            j.set("damping", f"{cfg.finger_damping}")
            # stiff travel limits: a default soft limit lets the 35 N finger force push ~0.1 m past it
            j.set("solreflimit", "0.005 1"); j.set("solimplimit", "0.99 0.9999 0.001")
    hand = root.find(".//body[@name='hand']")
    _add(hand, "site", name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.002], group="4")
    world = root.find("worldbody")
    # table plane and board plate
    _add(world, "geom", name="table", type="plane", size=[1.5, 1.5, 0.1], pos=[0, 0, 0], contype=ENV,
         conaffinity=PART | GRIP | ARM, friction=cfg.env_friction, rgba=[0.3, 0.32, 0.36, 1])
    if visual:
        _add(world, "light", name="key", pos=[0.9, -0.6, 1.6], dir=[-0.35, 0.3, -1], diffuse=[0.45, 0.45, 0.45],
             specular=[0.1, 0.1, 0.1], castshadow="true", directional="true")
        _add(world, "light", name="fill1", pos=[0.2, 0.8, 1.2], dir=[0.1, -0.5, -1], diffuse=[0.2, 0.2, 0.2],
             specular=[0, 0, 0], castshadow="false", directional="true")
        _add(world, "light", name="fill2", pos=[1.2, 0.3, 0.8], dir=[-1, -0.2, -0.6], diffuse=[0.12, 0.12, 0.12],
             specular=[0, 0, 0], castshadow="false", directional="true")
    _add(world, "geom", name="board", type="box", size=[BOARD_HALF, BOARD_HALF, BOARD_TOP_WORLD_Z / 2],
         pos=[BOARD_CENTER_WORLD_X, 0, BOARD_TOP_WORLD_Z / 2], contype=ENV, conaffinity=PART | GRIP | ARM,
         friction=cfg.env_friction, rgba=[0.78, 0.79, 0.81, 1])
    # active fixture (exact collision model)
    fpos, fquat = scene_pose(task.fixture_scene)
    fix = _add(world, "body", name="fixture", pos=fpos, quat=fquat)
    for k, g in enumerate(fixture_geoms(task, cfg)):
        _add_geom(fix, asset, f"fix{k}", g, dict(contype=ENV, conaffinity=PART | GRIP | ARM,
                  friction=cfg.env_friction, solref=cfg.solref, solimp=cfg.solimp, rgba=[0.15, 0.35, 0.85, 1]))
    _add(fix, "site", name="fixture_frame", size=[0.002], group="4")
    # all other board fixtures: static convex-hull obstacles (their parts are absent)
    for prim, stem in sorted(BOARD_FIXTURES.items()):
        if prim == task.fixture_scene:
            continue
        mname = f"hull_{stem}"
        if asset.find(f"mesh[@name='{mname}']") is None:
            _add(asset, "mesh", name=mname, vertex=hull_vertices(stem).ravel())
        pos, quat = scene_pose(prim)
        _add(world, "geom", name=f"obst_{prim}", type="mesh", mesh=mname, pos=pos, quat=quat, contype=ENV,
             conaffinity=PART | GRIP | ARM, friction=cfg.env_friction, rgba=[0.5, 0.52, 0.56, 1])
    # movable part (free body); initial pose = assembled pose (overwritten by resets)
    # Default part pose is parked 1 m above the board, far from the assembled pose, so that an
    # automatic mj_resetData after a numerical instability can never look like an assembly.
    part = _add(world, "body", name="part", pos=[BOARD_CENTER_WORLD_X, 0.0, 1.0], quat=[1, 0, 0, 0])
    _add(part, "freejoint", name="part_free")
    for k, g in enumerate(part_geoms_with_mass(task, cfg)):
        _add_geom(part, asset, f"part{k}", g, dict(contype=PART, conaffinity=ENV | GRIP | ARM, condim="4",
                  friction=cfg.part_friction, solref=cfg.solref, solimp=cfg.solimp, priority="1",
                  rgba=[1.0, 0.5, 0.05, 1]))
    _add(part, "site", name="part_frame", size=[0.002], group="4")
    if visual:   # draw the original asset meshes; the collision proxies (convex hulls, SDF hosts) go to hidden group 3
        _visual_meshes(root, task)
    return ET.tostring(root, encoding="unicode")


def _visual_meshes(root, task):
    from .tasks import load_obj
    asset, world = root.find("asset"), root.find("worldbody")
    for g in world.iter("geom"):
        if g.get("name", "").startswith(("fix", "part", "obst_")):
            g.set("group", "3")

    def mesh(stem):
        name = f"vis_{stem}"
        if asset.find(f"mesh[@name='{name}']") is None:
            v, f = load_obj(stem)
            _add(asset, "mesh", name=name, vertex=v.ravel(), face=f.ravel())
        return name

    off = dict(contype="0", conaffinity="0", group="2", density="0")
    _add(root.find(".//body[@name='fixture']"), "geom", type="mesh", mesh=mesh(task.fixture_asset), rgba=[0.15, 0.35, 0.85, 1], **off)
    _add(root.find(".//body[@name='part']"), "geom", type="mesh", mesh=mesh(task.part_asset), rgba=[1.0, 0.5, 0.05, 1], **off)
    for prim, stem in sorted(BOARD_FIXTURES.items()):
        if prim != task.fixture_scene:
            pos, quat = scene_pose(prim)
            _add(world, "geom", name=f"vis_obst_{prim}", type="mesh", mesh=mesh(stem), pos=pos, quat=quat,
                 rgba=[0.5, 0.52, 0.56, 1], **off)


def _geom_volume(g):
    t = g["type"]
    if t == "box":
        return 8.0 * float(np.prod(g["size"]))
    if t == "cylinder":
        return math.pi * g["size"][0] ** 2 * 2.0 * g["size"][1]
    if t == "mesh":
        from scipy.spatial import ConvexHull
        return float(ConvexHull(np.asarray(g["vertex"]).reshape(-1, 3)).volume)
    return 0.0  # sdf geoms carry no mass


def part_geoms_with_mass(task, cfg):
    """Part geoms with a common density so that the whole part weighs cfg.part_mass."""
    geoms = part_geoms(task, cfg)
    vol = sum(_geom_volume(g) for g in geoms)
    mass = part_mass(task, cfg)
    for g in geoms:
        g.setdefault("density", mass / vol if g["type"] != "sdf" else 0.0)
    return geoms


CONNECTOR_GRIDS = {"BNC": (1, 2), "DSUB": (3, 4), "RJ45": (5, 6), "USB": (7, 8), "WP": (9, 10)}


def proxy_pieces(stem):
    return [np.asarray(v, dtype=float) for v in np.load(ROOT / "assets/proxies" / f"{stem}.npy", allow_pickle=True)]


def part_mass(task, cfg):
    """USD MassAPI value of the part when present (D-sub 40 g, others 50 g)."""
    from .tasks import manifest
    m = manifest()["assets"].get(task.part_asset, {}).get("mass")
    return float(m) if m else cfg.part_mass


def hull_vertices(stem: str) -> np.ndarray:
    """Convex-hull vertices of an asset's collision mesh (cached)."""
    cache = ROOT / "assets/hulls" / f"{stem}.npy"
    if cache.exists():
        return np.load(cache)
    from scipy.spatial import ConvexHull
    v, _ = load_obj(stem)
    hv = v[ConvexHull(v).vertices]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, hv)
    return hv

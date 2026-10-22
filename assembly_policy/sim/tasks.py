"""NIST Assembly Task Board 1 (ATB1) task registry.

Assembled poses come from UW Lab's NIST scene (nist_scene.usd; assets/meshes/manifest.json,
key "scene"). For every pair the assembled part frame equals the placed part frame
in that scene; the fixture is placed at its own scene pose. Board/scene frame is
mapped to the robot world frame by SCENE_TO_WORLD (a local choice: board centre
0.5 m in front of the Panda base, board resting on the table plane z=0).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "assets/meshes"

# Board mesh spans x,y in [-0.384, 0] and z in [0.0116, 0.0206] in the scene frame.
BOARD_HALF = 0.192
BOARD_CENTER_WORLD_X = 0.50
SCENE_TO_WORLD = np.array([BOARD_CENTER_WORLD_X + BOARD_HALF, BOARD_HALF, -0.0116])
BOARD_TOP_WORLD_Z = 0.0206 - 0.0116


@dataclass(frozen=True)
class Task:
    name: str            # paper notation, e.g. "R16"
    family: str          # rod | peg | gear | connector | nut
    part_asset: str      # mesh stem in assets/meshes
    fixture_asset: str
    part_scene: str      # prim name in nist_scene.usd
    fixture_scene: str
    # collision builder parameters
    params: dict = field(default_factory=dict)
    # success keypoints in part frame (metres); assembled = coincident with fixture-frame copies
    keypoints: tuple = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.02))


def _rod(n, mm, hole_asset="round_hole", hole_scene=None):
    return Task(n, "rod", f"round_peg_{mm}mm", f"{hole_asset}_{mm}mm" if hole_asset != "hole" else f"hole_{mm}mm",
                f"rod_{mm}mm", hole_scene or f"hole_{mm}mm")


TASKS: dict[str, Task] = {t.name: t for t in [
    _rod("R4", 4), _rod("R8", 8, hole_asset="hole"), _rod("R12", 12), _rod("R16", 16),
    Task("P4", "peg", "rectangular_peg_4mm_tight", "rectangular_hole_4mm", "rectangle_peg_4mm", "rectangle_hole_4mm"),
    Task("P8", "peg", "rectangular_peg_8mm_tight", "rectangular_hole_8mm", "rectangle_peg_8mm", "rectangle_hole_8mm"),
    Task("P12", "peg", "rectangular_peg_12mm_tight", "rectangular_hole_12mm", "rectangle_peg_12mm", "rectangle_hole_12mm"),
    Task("P16", "peg", "rectangular_peg_16mm_tight", "rectangular_hole_16mm", "rectangle_peg_16mm", "rectangle_hole_16mm"),
]}

TASKS.update({t.name: t for t in [
    # gear keypoints lie on each gear's own axis (gear-set frame has the centres on the x axis)
    Task("Gs", "gear", "gear_small", "gear_base", "gear_small", "gear_base",
         keypoints=((0.05075, 0.0, 0.005), (0.05075, 0.0, 0.025))),
    Task("Gm", "gear", "gear_medium", "gear_base", "gear_medium", "gear_base",
         keypoints=((0.02025, 0.0, 0.005), (0.02025, 0.0, 0.025))),
    Task("Gl", "gear", "gear_large", "gear_base", "gear_large", "gear_base",
         keypoints=((-0.03025, 0.0, 0.005), (-0.03025, 0.0, 0.025))),
    Task("N4", "nut", "nut_m4", "bolt_m4", "nut_m4", "bolt_m4"),
    Task("N8", "nut", "nut_m8", "bolt_m8", "nut_m8", "bolt_m8"),
    Task("N12", "nut", "nut_m12", "bolt_m12", "nut_m12", "bolt_m12"),
    Task("N16", "nut", "nut_m16_high_poly", "bolt_m16_high_poly", "nut", "bolt_m_16"),
    Task("BNC", "connector", "bnc_plug", "bnc_socket", "bnc_plug", "bnc_socket"),
    Task("DSUB", "connector", "dsub_plug", "dsub_socket", "DSUB_Male_Connector", "DSUB_Female_Connector"),
    Task("RJ45", "connector", "rj45_plug", "rj45_socket", "rj45_plug", "rj45_socket"),
    Task("USB", "connector", "usb_a_plug", "usb_a_socket", "usb_a_plug", "usb_a_socket"),
    Task("WP", "connector", "waterproof_plug", "waterproof_socket", "waterproof_plug", "waterproof_socket"),
]})

# relative CPU cost of one control step by task family (measured: 0.78 / 1.9 / 6.1 / 9.6 ms per env-step)
STEP_COST = {"rod": 1.0, "peg": 1.0, "gear": 2.4, "connector": 7.8, "nut": 12.3}
# families simulated with stiff contacts (--fine-families; 0.5 ms timestep, SDF search 20 x 5):
# scripts/bench_families.py, bank starts, relative to the rods above. Learners refine both tables online.
FINE_STEP_COST = {"gear": 8.6, "nut": 15.0}

INSERTION_TASKS = ("R4", "R8", "R12", "R16", "P4", "P8", "P12", "P16")
GEAR_TASKS = ("Gs", "Gm", "Gl")
NUT_TASKS = ("N4", "N8", "N12", "N16")
CONNECTOR_TASKS = ("BNC", "DSUB", "RJ45", "USB", "WP")
ALL_TASKS = tuple(TASKS)

# named task sets
TASK_SETS = {"all20": ALL_TASKS, "ins8": INSERTION_TASKS, "gears": GEAR_TASKS, "nuts": NUT_TASKS,
             "connectors": CONNECTOR_TASKS}


def resolve_tasks(spec: str):
    """'all20' / 'ins8' / ... or a comma / '+' list -> list of task names (validated)."""
    names = []
    for part in spec.replace("+", ",").split(","):
        names += list(TASK_SETS.get(part, (part,)))
    for n in names:
        if n not in TASKS:
            raise ValueError(f"unknown task {n!r} in {spec!r}")
    return names

# Static fixtures present on the board in every single-pair episode: (mesh stem, scene prim).
BOARD_FIXTURES = {t.fixture_scene: t.fixture_asset for t in TASKS.values()}


def manifest():
    return json.loads((MESH_DIR / "manifest.json").read_text())


def scene_pose(prim: str):
    """(pos_world, quat_wxyz) of a scene prim mapped to the robot world frame."""
    entry = manifest()["scene"][prim]
    return np.asarray(entry["pos"]) + SCENE_TO_WORLD, np.asarray(entry["quat_wxyz"], dtype=float)


def load_obj(stem: str):
    verts, faces = [], []
    for line in open(MESH_DIR / f"{stem}.obj"):
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(x) - 1 for x in line.split()[1:4]])
    return np.asarray(verts), np.asarray(faces, dtype=np.int64)


# where reset banks and runs are written (override with $ASSEMBLENIST_DATA)
DATA_DIR = Path(os.environ.get("ASSEMBLENIST_DATA", ROOT / "data"))
BANK_DIR = DATA_DIR / "banks"

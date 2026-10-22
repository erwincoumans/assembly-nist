"""ISO-metric nut/bolt parameters for the custom SDF plugin (plugin/assemblenist_sdf.cc), and the plugin loader.

The plugin provides analytic thread SDFs (bolts, nuts) and grid SDFs for the connectors (assets/sdf/grid_*.bin).
Values are measured from the UW Lab meshes (assets/meshes/{bolt,nut}_*.obj): pitch from the helical
crest phase, crest/root radii from radial extents. Nut phase is chosen so the nist_scene assembled
pose is exactly threaded (bolt phase 0).
"""
from __future__ import annotations

import os
from pathlib import Path

import mujoco

from .tasks import ROOT, scene_pose

PLUGIN = Path(os.environ.get("ASSEMBLENIST_SDF_PLUGIN", ROOT / "plugin" / "build" / "libassemblenist_sdf.so"))
SDF_GRID_DIR = ROOT / "assets" / "sdf"
# radial nut-bolt clearance (m). 0.1 mm wedged the M4-M12 threads under the SDF contact search; 0.2 mm threads at
# the right pitch and still blocks 5-50 N axial pushes.
CLEARANCE = float(os.environ.get("ASSEMBLENIST_THREAD_CLEARANCE", 0.0002))
# truncation (fraction of the pitch) of the V tips of nut teeth and bolt grooves (ISO-like flat crests/roots).
# Sharp tips wedged under the SDF contact search; with 0.2 every nut spins freely along the helix at 0.02 N m
# and 5-50 N axial pushes do not skip threads.
TRUNC = float(os.environ.get("ASSEMBLENIST_THREAD_TRUNC", 0.2))

# name -> (pitch, bolt crest r, bolt root r, bolt head top z, bolt top z, nut minor r, nut z0, nut z1, nut hex flat)
THREADS = {
    "N4": (0.0007, 0.001986, 0.001282, 0.004, 0.020, 0.001499, 0.0048, 0.0080, 0.0035),
    "N8": (0.00125, 0.003963, 0.002857, 0.008, 0.026, 0.003100, 0.0093, 0.0158, 0.0065),
    "N12": (0.00175, 0.005934, 0.004464, 0.012, 0.032, 0.004741, 0.0130, 0.0230, 0.0095),
    "N16": (0.002, 0.007842, 0.006385, 0.010, 0.035, 0.006905, 0.0100, 0.0230, 0.0120),
}
_loaded = False


def load_plugin():
    global _loaded
    if not _loaded:
        if not PLUGIN.exists():
            raise FileNotFoundError(f"{PLUGIN} not found: build it with `bash plugin/build.sh`")
        os.environ.setdefault("ASSEMBLENIST_SDF_DIR", str(SDF_GRID_DIR))    # connector SDF grids
        mujoco.mj_loadPluginLibrary(str(PLUGIN))
        _loaded = True


def thread_attributes(task):
    from .tasks import TASKS
    P, crest, root, htop, top, minor, z0, z1, flat = THREADS[task]
    rp = crest - 0.3248 * P
    t = TASKS[task]
    dz = scene_pose(t.part_scene)[0][2] - scene_pose(t.fixture_scene)[0][2]
    phase = (dz / P) % 1.0
    if TRUNC > 0:   # flat crests/roots instead of sharp V tips (the M16 profile is already truncated this way)
        root = max(root, rp - 0.433 * P + TRUNC * P)                    # bolt groove bottom
        minor = max(minor, rp + CLEARANCE - 0.433 * P + TRUNC * P)      # nut tooth crest
    bolt = dict(pitch=P, rp=rp, root=root, crest=crest, thread_z0=htop, thread_z1=top, head_flat=flat * 1.5,
                head_top=htop, phase=0.0)
    nut = dict(pitch=P, rp=rp, clearance=CLEARANCE, minor=minor, major=crest + 1.5 * CLEARANCE, flat=flat,
               z0=z0, z1=z1, phase=phase)
    return bolt, nut

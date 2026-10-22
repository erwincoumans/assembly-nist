"""Convex decomposition of the NIST ATB1 gear set (UW-Lab / Factory meshes).

Each gear (module 1 mm involute spur gear; 20/40/60 teeth; 5.0 mm bore) is decomposed into:
  * one convex prism per tooth (2D convex hull of the source mesh points of that tooth,
    extruded over the 10 mm face width, z = 5..15 mm in the gear frame),
  * convex annular sectors for the web (bore radius -> root radius) and the hub (z = 15..30 mm).
The gear base is a 150 x 75 x 5 mm plate with three 4.75 mm shafts (top at z = 25 mm, 0.5 mm chamfer).
All coordinates are in the shared gear-set frame of the source meshes (gear centres on the x axis).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import ConvexHull

from .tasks import load_obj

GEARS = {  # name -> (centre x in gear-set frame, teeth, root radius, hub radius)
    "gear_small": (0.05075, 20, 0.00875, 0.00875),
    "gear_medium": (0.02025, 40, 0.01875, 0.015),
    "gear_large": (-0.03025, 60, 0.02875, 0.01875),
}
BORE = 0.005
FACE_Z = (0.005, 0.015)
HUB_Z = (0.015, 0.030)
SHAFT_R, SHAFT_TOP, SHAFT_CHAMFER = 0.00475, 0.025, 0.0005


def _prism(poly2d, z0, z1):
    return np.vstack([np.c_[poly2d, np.full(len(poly2d), z0)], np.c_[poly2d, np.full(len(poly2d), z1)]])


def _sectors(r_in, r_out, z0, z1, n, cx):
    out = []
    for i in range(n):
        a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        pts = np.array([[r_in * math.cos(a0), r_in * math.sin(a0)], [r_out * math.cos(a0), r_out * math.sin(a0)],
                        [r_out * math.cos(a1), r_out * math.sin(a1)], [r_in * math.cos(a1), r_in * math.sin(a1)]])
        # push the inner vertices out so each bore chord is tangent to the true 5.0 mm radius
        # (shaft clearance stays 0.25 mm); the outer edge stays inside r_out (teeth overlap it)
        pts[[0, 3]] /= math.cos(math.pi / n)
        out.append(_prism(pts + [cx, 0.0], z0, z1))
    return out


TOOTH_INSET = 0.0002  # teeth stop 0.2 mm short of both faces: a gear resting on a face touches only web/hub


def gear_convex_pieces(name, web_sectors=12, hub_sectors=12):
    """List of (V,3) vertex arrays (convex pieces) in the gear-set frame."""
    cx, nteeth, r_root, r_hub = GEARS[name]
    V, _ = load_obj(name)
    x, y = V[:, 0] - cx, V[:, 1]
    r, th = np.hypot(x, y), np.arctan2(y, x)
    ring = (V[:, 2] > FACE_Z[0] - 1e-4) & (V[:, 2] < FACE_Z[1] + 1e-4) & (r > r_root + 2e-4)
    pieces = []
    # tooth centres from tip clusters
    r_tip = r[ring].max()
    tip_ang = np.sort(np.mod(th[ring & (r > r_tip - 3e-4)], 2 * np.pi))
    gaps = np.diff(np.r_[tip_ang, tip_ang[0] + 2 * np.pi])
    starts = np.r_[0, np.nonzero(gaps[:-1] > 0.05)[0] + 1]
    centres = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(tip_ang)
        seg = tip_ang[s:e]
        centres.append(np.angle(np.exp(1j * seg).mean()))
    assert len(centres) == nteeth, (name, len(centres))
    half = math.pi / nteeth
    for c in centres:
        d = np.angle(np.exp(1j * (th - c)))
        sel = ring & (np.abs(d) < half)
        P = np.c_[x[sel], y[sel]]
        # add two points slightly inside the root circle at the tooth flanks so teeth overlap the web
        flank = np.abs(d[sel]).max()
        rr = r_root - 2e-4
        P = np.vstack([P, [[rr * math.cos(c - 0.6 * half), rr * math.sin(c - 0.6 * half)],
                           [rr * math.cos(c + 0.6 * half), rr * math.sin(c + 0.6 * half)]]])
        hull = P[ConvexHull(P).vertices]
        pieces.append(_prism(hull + [cx, 0.0], FACE_Z[0] + TOOTH_INSET, FACE_Z[1] - TOOTH_INSET))
    pieces += _sectors(BORE, r_root, *FACE_Z, web_sectors, cx)
    if r_hub > BORE + 1e-4:
        pieces += _sectors(BORE, r_hub, *HUB_Z, hub_sectors, cx)
    return pieces


def gear_base_geoms():
    """Primitive geoms (MJCF dicts) for the gear base in its own frame."""
    g = [dict(type="box", size=[0.075, 0.0375, 0.0025], pos=[0, 0, 0.0025])]
    for name, (cx, *_rest) in GEARS.items():
        g.append(dict(type="cylinder", size=[SHAFT_R, (SHAFT_TOP - SHAFT_CHAMFER - 0.005) / 2],
                      pos=[cx, 0, 0.005 + (SHAFT_TOP - SHAFT_CHAMFER - 0.005) / 2]))
        g.append(dict(type="cylinder", size=[SHAFT_R - SHAFT_CHAMFER, SHAFT_CHAMFER / 2],
                      pos=[cx, 0, SHAFT_TOP - SHAFT_CHAMFER / 2]))
    return g

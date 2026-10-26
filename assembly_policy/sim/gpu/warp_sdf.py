"""plugin/assemblenist_sdf.cc in Warp: the bolt and nut thread SDFs and the connector grid SDFs, as MuJoCo Warp's
user SDF functions (collision_sdf.user_sdf, user_sdf_grad), so that the task models load into MuJoCo Warp unchanged.

MuJoCo Warp passes a geom's plugin attributes (the plugin's declared ones: 9 for bolts and nuts, the grid id for
grids) and the plugin's type index. The grids' samples are uploaded once (float16, as in the files) and
`patch_model` adds each grid instance's dimensions, origin, spacing and device address to its attributes.
"""
from __future__ import annotations

import math

import numpy as np
import warp as wp
from mujoco_warp._src.types import vec_pluginattr

TAN60 = wp.constant(1.7320508075688772)
TWO_PI = wp.constant(2.0 * math.pi)
SQRT_HALF = wp.constant(0.7071067811865476)
# grid attributes after patch_model: id, nx, ny, nz, ox, oy, oz, h, then the address in four 16-bit pieces
GRID_ATTR = 12


@wp.func
def _seg(r: float, s: float, ar: float, as_: float, br: float, bs: float) -> wp.vec3:
    """Distance from (r, s) to the segment a-b, and the offset from its closest point."""
    vr = br - ar
    vs = bs - as_
    t = wp.clamp(((r - ar) * vr + (s - as_) * vs) / (vr * vr + vs * vs), 0.0, 1.0)
    dr = r - (ar + t * vr)
    ds = s - (as_ + t * vs)
    return wp.vec3(wp.sqrt(dr * dr + ds * ds), dr, ds)


@wp.func
def _thread2d(r: float, s: float, P: float, top: float, lo: float, hi: float, below: bool) -> wp.vec3:
    """Signed distance to the periodic trapezoid profile clamp(top - tan60 |s - P/2|, lo, hi) and its (r, s)
    gradient; material where r < profile (below) or r > profile."""
    s1 = 0.5 * P - (top - hi) / TAN60
    s0 = 0.5 * P - (top - lo) / TAN60
    s0 = wp.max(s0, 0.0)
    s1 = wp.max(s1, s0)
    s2 = P - s1
    s3 = P - s0
    best = _seg(r, s, hi, s2 - P, lo, s3 - P)
    c = _seg(r, s, lo, s3 - P, lo, s0)
    if c[0] < best[0]:
        best = c
    c = _seg(r, s, lo, s0, hi, s1)
    if c[0] < best[0]:
        best = c
    c = _seg(r, s, hi, s1, hi, s2)
    if c[0] < best[0]:
        best = c
    c = _seg(r, s, hi, s2, lo, s3)
    if c[0] < best[0]:
        best = c
    c = _seg(r, s, lo, s3, lo, P + s0)
    if c[0] < best[0]:
        best = c
    c = _seg(r, s, lo, P + s0, hi, P + s1)
    if c[0] < best[0]:
        best = c
    rs = wp.clamp(top - TAN60 * wp.abs(s - 0.5 * P), lo, hi)
    inside = r < rs
    if not below:
        inside = r > rs
    sgn = 1.0
    if inside:
        sgn = -1.0
    if best[0] > 1e-12:
        return wp.vec3(sgn * best[0], sgn * best[1] / best[0], sgn * best[2] / best[0])
    gr = -1.0
    if below:
        gr = 1.0
    return wp.vec3(sgn * best[0], gr, 0.0)


@wp.func
def _frac(x: float) -> float:
    return x - wp.floor(x)


@wp.func
def _thread_grad(p: wp.vec3, r: float, P: float, d: wp.vec3) -> wp.vec3:
    """(dist/dr, dist/ds) of a thread profile to xyz through the helix coordinates."""
    k = -P / TWO_PI / wp.max(r * r, 1e-18)
    ux = 0.0
    uy = 0.0
    if r > 1e-12:
        ux = p[0] / r
        uy = p[1] / r
    return wp.vec3(d[1] * ux + d[2] * (-k * p[1]), d[1] * uy + d[2] * (k * p[0]), d[2])


@wp.func
def bolt_sdf(p: wp.vec3, a: vec_pluginattr, grad: bool) -> wp.vec4:
    """Externally threaded shank (pitch, rp, root, crest, thread_z0, thread_z1, head_flat, head_top, phase):
    (distance, gradient)."""
    P = a[0]
    rp = a[1]
    root = a[2]
    crest = a[3]
    tz0 = a[4]
    tz1 = a[5]
    ph = a[8]
    r = wp.sqrt(p[0] * p[0] + p[1] * p[1])
    s = P * _frac(p[2] / P - wp.atan2(p[1], p[0]) / TWO_PI + ph)
    d = _thread2d(r, s, P, rp + 0.25 * P * TAN60, root, crest, True)
    dist = d[0]
    g = wp.vec3(0.0)
    if grad:
        g = _thread_grad(p, r, P, d)
    cap0 = tz0 - p[2]
    cap1 = p[2] - tz1
    cham = (p[2] - tz1 + (r - root)) * SQRT_HALF
    if cap0 > dist:
        dist = cap0
        g = wp.vec3(0.0, 0.0, -1.0)
    if cap1 > dist:
        dist = cap1
        g = wp.vec3(0.0, 0.0, 1.0)
    if cham > dist:
        dist = cham
        ux = 0.0
        uy = 0.0
        if r > 1e-12:
            ux = p[0] / r
            uy = p[1] / r
        g = wp.vec3(SQRT_HALF * ux, SQRT_HALF * uy, SQRT_HALF)
    return wp.vec4(dist, g[0], g[1], g[2])


@wp.func
def _hexagon2d(x: float, y: float, a: float) -> float:
    kx = -0.8660254037844386
    ky = 0.5
    kz = 0.5773502691896258
    px = wp.abs(y)
    py = wp.abs(x)
    d = 2.0 * wp.min(kx * px + ky * py, 0.0)
    px -= d * kx
    py -= d * ky
    cx = wp.clamp(px, -kz * a, kz * a)
    qx = px - cx
    qy = py - a
    ln = wp.sqrt(qx * qx + qy * qy)
    if qy < 0.0:
        return -ln
    return ln


@wp.func
def _hex_body(p: wp.vec3, flat: float, z0: float, z1: float) -> float:
    wx = _hexagon2d(p[0], p[1], flat)
    wy = wp.abs(p[2] - 0.5 * (z1 + z0)) - 0.5 * (z1 - z0)
    out = wp.sqrt(wp.max(wx, 0.0) * wp.max(wx, 0.0) + wp.max(wy, 0.0) * wp.max(wy, 0.0))
    return wp.min(wp.max(wx, wy), 0.0) + out


@wp.func
def nut_sdf(p: wp.vec3, a: vec_pluginattr, grad: bool) -> wp.vec4:
    """Hex prism with an internal thread (pitch, rp, clearance, minor, major, flat, z0, z1, phase)."""
    P = a[0]
    rp = a[1]
    c = a[2]
    minor = a[3]
    major = a[4]
    flat = a[5]
    z0 = a[6]
    z1 = a[7]
    ph = a[8]
    r = wp.sqrt(p[0] * p[0] + p[1] * p[1])
    s = P * _frac(p[2] / P - wp.atan2(p[1], p[0]) / TWO_PI + ph)
    d = _thread2d(r, s, P, rp + c + 0.25 * P * TAN60, minor, major, False)
    body = _hex_body(p, flat, z0, z1)
    if body > d[0]:
        g = wp.vec3(0.0)
        if grad:   # the (rarely active) outer hex body: central differences
            e = 1e-6
            g = wp.vec3(_hex_body(p + wp.vec3(e, 0.0, 0.0), flat, z0, z1) - _hex_body(p - wp.vec3(e, 0.0, 0.0), flat, z0, z1),
                        _hex_body(p + wp.vec3(0.0, e, 0.0), flat, z0, z1) - _hex_body(p - wp.vec3(0.0, e, 0.0), flat, z0, z1),
                        _hex_body(p + wp.vec3(0.0, 0.0, e), flat, z0, z1) - _hex_body(p - wp.vec3(0.0, 0.0, e), flat, z0, z1)) / (2.0 * e)
        return wp.vec4(body, g[0], g[1], g[2])
    g = wp.vec3(0.0)
    if grad:
        g = _thread_grad(p, r, P, d)
    return wp.vec4(d[0], g[0], g[1], g[2])


@wp.func
def grid_sdf(p: wp.vec3, a: vec_pluginattr, grad: bool) -> wp.vec4:
    """Trilinear float16 grid (attributes of patch_model); outside it, plus the distance to its box."""
    nx = int(a[1])
    ny = int(a[2])
    nz = int(a[3])
    h = a[7]
    ptr = wp.uint64(int(a[8])) | (wp.uint64(int(a[9])) << wp.uint64(16)) | (wp.uint64(int(a[10])) << wp.uint64(32)) \
        | (wp.uint64(int(a[11])) << wp.uint64(48))
    v = wp.array(ptr=ptr, shape=(nx * ny * nz,), dtype=wp.float16)
    fx = (p[0] - a[4]) / h
    fy = (p[1] - a[5]) / h
    fz = (p[2] - a[6]) / h
    cx = wp.clamp(fx, 0.0, float(nx) - 1.000001)
    cy = wp.clamp(fy, 0.0, float(ny) - 1.000001)
    cz = wp.clamp(fz, 0.0, float(nz) - 1.000001)
    e = wp.vec3((fx - cx) * h, (fy - cy) * h, (fz - cz) * h)
    ix = int(wp.floor(cx))
    iy = int(wp.floor(cy))
    iz = int(wp.floor(cz))
    tx = cx - float(ix)
    ty = cy - float(iy)
    tz = cz - float(iz)
    b = (iz * ny + iy) * nx + ix
    sy = nx
    sz = nx * ny
    c000 = float(v[b])
    c100 = float(v[b + 1])
    c010 = float(v[b + sy])
    c110 = float(v[b + sy + 1])
    c001 = float(v[b + sz])
    c101 = float(v[b + sz + 1])
    c011 = float(v[b + sz + sy])
    c111 = float(v[b + sz + sy + 1])
    c00 = c000 + (c100 - c000) * tx
    c10 = c010 + (c110 - c010) * tx
    c01 = c001 + (c101 - c001) * tx
    c11 = c011 + (c111 - c011) * tx
    c0 = c00 + (c10 - c00) * ty
    c1 = c01 + (c11 - c01) * ty
    out = wp.length(e)
    g = wp.vec3(0.0)
    if grad:
        if out > 1e-12:
            g = e / out
        else:
            gx = ((c100 - c000) * (1.0 - ty) + (c110 - c010) * ty) * (1.0 - tz) + ((c101 - c001) * (1.0 - ty) + (c111 - c011) * ty) * tz
            gy = (c10 - c00) * (1.0 - tz) + (c11 - c01) * tz
            gz = c1 - c0
            g = wp.vec3(gx, gy, gz) / h
    return wp.vec4(c0 + (c1 - c0) * tz + out, g[0], g[1], g[2])


_installed = None


def install(bolt_type, nut_type, grid_type):
    """Make MuJoCo Warp evaluate these SDFs for plugin geoms of the given plugin type indices (before its kernels are
    first built)."""
    global _installed
    key = (int(bolt_type), int(nut_type), int(grid_type))
    if _installed == key:
        return
    if _installed is not None:
        raise RuntimeError(f"user SDFs already installed for plugin types {_installed}, not {key}")
    import mujoco_warp._src.collision_sdf as cs
    from mujoco_warp._src.types import vec_pluginattr
    BOLT, NUT, GRID = key

    @wp.func
    def user_sdf(p: wp.vec3, attr: vec_pluginattr, sdf_type: int) -> float:
        if sdf_type == BOLT:
            return bolt_sdf(p, attr, False)[0]
        if sdf_type == NUT:
            return nut_sdf(p, attr, False)[0]
        if sdf_type == GRID:
            return grid_sdf(p, attr, False)[0]
        return 1.0e10

    @wp.func
    def user_sdf_grad(p: wp.vec3, attr: vec_pluginattr, sdf_type: int) -> wp.vec3:
        r = wp.vec4(0.0)
        if sdf_type == BOLT:
            r = bolt_sdf(p, attr, True)
        elif sdf_type == NUT:
            r = nut_sdf(p, attr, True)
        elif sdf_type == GRID:
            r = grid_sdf(p, attr, True)
        return wp.vec3(r[1], r[2], r[3])

    cs.user_sdf = user_sdf
    cs.user_sdf_grad = user_sdf_grad
    _installed = key


def plugin_types():
    """Global plugin type indices of the bolt, nut and grid SDF plugins (the library must be loaded)."""
    import mujoco
    from ..threads import load_plugin
    load_plugin()
    out = []
    for name in ("assemblenist.sdf.bolt", "assemblenist.sdf.nut", "assemblenist.sdf.grid"):
        out.append(mujoco.mj_getPluginIndex(name) if hasattr(mujoco, "mj_getPluginIndex") else _plugin_index(name))
    return out


def _plugin_index(name):
    """The plugin's type index, from a one-geom model that uses it."""
    import mujoco
    attrs = {"assemblenist.sdf.grid": '<config key="id" value="1"/>'}.get(name, "")
    xml = (f'<mujoco><extension><plugin plugin="{name}"><instance name="i">{attrs}</instance></plugin></extension>'
           f'<asset><mesh name="m"><plugin instance="i"/></mesh></asset><worldbody><body><freejoint/>'
           f'<geom type="sdf" mesh="m"><plugin instance="i"/></geom></body></worldbody></mujoco>')
    return int(mujoco.MjModel.from_xml_string(xml).plugin[0])


class GridStore:
    """The connector grids (assets/sdf/grid_<id>.bin) on the device, float16, uploaded once per id."""

    def __init__(self, device):
        self.device = device
        self.grids = {}

    def get(self, gid):
        if gid not in self.grids:
            from ..threads import SDF_GRID_DIR
            raw = (SDF_GRID_DIR / f"grid_{gid}.bin").read_bytes()
            n = np.frombuffer(raw, np.int32, 3, 0)
            o = np.frombuffer(raw, np.float64, 4, 12)
            v = np.frombuffer(raw, np.float16, int(np.prod(n)), 44)
            self.grids[gid] = (n, o, wp.array(v, dtype=wp.float16, device=self.device))
        return self.grids[gid]

    def attributes(self, gid):
        n, o, arr = self.get(gid)
        ptr = arr.ptr
        return [float(gid), float(n[0]), float(n[1]), float(n[2]), float(o[0]), float(o[1]), float(o[2]), float(o[3])] + \
               [float((ptr >> (16 * i)) & 0xFFFF) for i in range(4)]


def patch_model(m, mjm, grid_type, store):
    """Add the grid data to the grid instances' attributes of a MuJoCo Warp model (put_model of mjm)."""
    if not len(m.plugin_attr):
        return
    plugin = m.plugin.numpy()
    attr = m.plugin_attr.numpy()
    changed = False
    for i, t in enumerate(plugin):
        if t == grid_type:
            gid = int(round(attr[i][0]))
            attr[i][:GRID_ATTR] = store.attributes(gid)
            changed = True
    if changed:
        m.plugin_attr.assign(attr)

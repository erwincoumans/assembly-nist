"""Signed-distance grids of connector plugs/sockets for the grid-SDF plugin (assemblenist.sdf.grid).

Distances are computed from the UW-Lab collision meshes with libigl's generalized winding number
(robust to the small holes in some CAD exports), sampled on a regular grid covering the mesh bounding
box plus padding, clamped to +-CLAMP and stored as float16 with a small header:
  int32 nx, ny, nz; float64 ox, oy, oz, h; float16 values[nx*ny*nz] (x fastest)
"""
import argparse, json, struct, sys, time
from pathlib import Path

import igl
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assembly_policy.sim.tasks import ROOT, load_obj  # noqa: E402

OUT = ROOT / "assets/sdf"
CLAMP = 0.004
PAD = 0.003

# grid id -> mesh stem (ids are referenced from the MJCF plugin config)
GRIDS = {1: "bnc_plug", 2: "bnc_socket", 3: "dsub_plug", 4: "dsub_socket", 5: "rj45_plug", 6: "rj45_socket",
         7: "usb_a_plug", 8: "usb_a_socket", 9: "waterproof_plug", 10: "waterproof_socket"}


def build(stem, h):
    V, F = load_obj(stem)
    lo, hi = V.min(0) - PAD, V.max(0) + PAD
    n = np.ceil((hi - lo) / h).astype(int) + 1
    xs, ys, zs = (lo[i] + h * np.arange(n[i]) for i in range(3))
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")          # x fastest in memory
    P = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    out = np.empty(len(P))
    chunk = 2_000_000
    for i in range(0, len(P), chunk):
        res = igl.signed_distance(P[i:i + chunk], V, F.astype(np.int64), sign_type=igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER)
        out[i:i + chunk] = res[0]
    out = np.clip(out, -CLAMP, CLAMP).astype(np.float16)
    return n, lo, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=",".join(map(str, GRIDS)))
    ap.add_argument("--h", type=float, default=0.0002)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for gid in map(int, a.ids.split(",")):
        stem = GRIDS[gid]
        t0 = time.time()
        h = a.h if "waterproof" not in stem else max(a.h, 0.00025)
        n, lo, vals = build(stem, h)
        with open(OUT / f"grid_{gid}.bin", "wb") as f:
            f.write(struct.pack("<3i4d", *map(int, n), *map(float, lo), float(h)))
            f.write(vals.tobytes())
        print(json.dumps(dict(id=gid, stem=stem, shape=n.tolist(), h=h, mb=vals.nbytes / 1e6,
                              inside_frac=float((vals < 0).mean()), seconds=time.time() - t0)), flush=True)
    (OUT / "grids.json").write_text(json.dumps({str(k): v for k, v in GRIDS.items()}, indent=1))


if __name__ == "__main__":
    main()

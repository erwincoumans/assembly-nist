"""Extract collision meshes of all NIST ATB1 parts/fixtures from UW-Lab USD assets.

Requires usd-core (pip install usd-core) and a local copy of the UW Lab NIST assets (Props/NIST of
uwlab-assets, commit f9486a6), pointed to by $NIST_USD_DIR. Writes one OBJ per asset (metres,
asset-root frame, triangulated) plus a manifest with the assembled board layout
parsed from nist_scene.usd. Unit/up-axis metadata of a few source files is
inconsistent (vertices are already metres, Z-up), so it is deliberately ignored.
"""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Sdf

SRC = Path(os.environ["NIST_USD_DIR"])
OUT = Path(__file__).resolve().parents[1] / "assets/meshes"


def mesh_arrays(prim, root_inv):
    m = UsdGeom.Mesh(prim)
    pts = np.asarray(m.GetPointsAttr().Get(), dtype=np.float64)
    T = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)) @ root_inv
    pts = (np.c_[pts, np.ones(len(pts))] @ T)[:, :3]
    idx = np.asarray(m.GetFaceVertexIndicesAttr().Get())
    cnt = np.asarray(m.GetFaceVertexCountsAttr().Get())
    faces, c = [], 0
    for n in cnt:
        ids = idx[c:c + n]; c += n
        faces += [[ids[0], ids[k], ids[k + 1]] for k in range(1, n - 1)]
    return pts, np.asarray(faces, dtype=np.int64)


def extract(path):
    st = Usd.Stage.Open(str(path))
    default = st.GetDefaultPrim()
    root = default if default else next(iter(st.GetPseudoRoot().GetChildren()))
    root_inv = np.linalg.inv(np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(root)))
    verts, faces, nv = [], [], 0
    rigid = mass = None
    for p in st.Traverse():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid = str(p.GetPath())
            if rigid is not None:
                rb_inv = np.linalg.inv(np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(p)))
        if p.HasAPI(UsdPhysics.MassAPI):
            mass = UsdPhysics.MassAPI(p).GetMassAttr().Get()
    frame_inv = rb_inv if rigid else root_inv
    for p in st.Traverse():
        if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI):
            v, f = mesh_arrays(p, frame_inv)
            verts.append(v); faces.append(f + nv); nv += len(v)
    v = np.concatenate(verts); f = np.concatenate(faces)
    return v, f, dict(rigid_prim=rigid, mass=mass)


def scene_layout():
    st = Usd.Stage.Open(str(SRC / "nist_scene.usd"), load=Usd.Stage.LoadNone)
    layer, out = st.GetRootLayer(), {}
    for p in st.GetPrimAtPath("/World").GetChildren():
        spec = layer.GetPrimAtPath(p.GetPath())
        refs = list(spec.referenceList.prependedItems) + list(spec.referenceList.explicitItems) if spec else []
        t, q = p.GetAttribute("xformOp:translate").Get(), p.GetAttribute("xformOp:orient").Get()
        if t is None or not refs:
            continue
        out[p.GetName()] = dict(usd=str(refs[0].assetPath).split("/")[-1], pos=list(t),
                                quat_wxyz=[q.GetReal(), *q.GetImaginary()])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = dict(source="UW-Lab/uwlab-assets Props/NIST", assets={}, scene=scene_layout())
    for path in sorted(SRC.glob("*.usd")):
        if path.stem in ("nist_scene", "nut_thread_scene", "kit_tray"):
            continue
        v, f, meta = extract(path)
        obj = OUT / f"{path.stem}.obj"
        with open(obj, "w") as fh:
            fh.write(f"# {path.name} collision mesh (UW-Lab/uwlab-assets), metres, asset frame\n")
            fh.writelines(f"v {a:.9g} {b:.9g} {c:.9g}\n" for a, b, c in v)
            fh.writelines(f"f {a+1} {b+1} {c+1}\n" for a, b, c in f)
        manifest["assets"][path.stem] = dict(
            nverts=len(v), nfaces=len(f), bounds=[v.min(0).tolist(), v.max(0).tolist()],
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **meta)
        print(f"{path.stem:32s} v={len(v):7d} f={len(f):7d} lo={np.round(v.min(0),4)} hi={np.round(v.max(0),4)} mass={meta['mass']}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()

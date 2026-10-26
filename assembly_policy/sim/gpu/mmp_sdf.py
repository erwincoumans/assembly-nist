"""Signed-distance contacts for the Metamorphysics backend from MuJoCo Warp's collision detection: the nut and bolt
threads and the connector grids of the CPU models (the plugin's SDFs, warp_sdf.py), with MuJoCo's SDF contact search.

For each task with SDF geoms, a MuJoCo Warp model holds only the fixture (static) and the part (free) with their SDF
geoms, a world per environment of the task. Before each MMP solve the parts' poses are copied into it, MuJoCo Warp
finds the contacts, and they are written as one-point MMP manifolds (MMP_Contact4: the point on the fixture with
the signed distance, the normal on the fixture pointing to the part) after MMP's own contacts.

MuJoCo Warp writes its contacts to slots taken with atomics, in a different order every run, and MMP's determinism
sort is stable by (bodyA, bodyB, childA, childB), which all of a part's contacts with its fixture share but childA:
so the contacts are put in an order of their contents (canonical), with childA the geom pair and the contact's rank
among the pair's - also the key that the manifold model's warm start matches a contact to the previous step's by.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch
import warp as wp

from ..scene import THREAD, PhysicsConfig, build_xml
from . import warp_sdf

CONTACT_WORDS = 28        # MMP_Contact4: 112 bytes


def sdf_xml(task, pcfg: PhysicsConfig):
    """The task model with only the fixture's and the part's SDF geoms."""
    root = ET.fromstring(build_xml(task, pcfg))
    world = root.find("worldbody")
    for el in list(world):
        if el.tag == "body" and el.get("name") in ("fixture", "part"):
            for g in list(el.findall("geom")):
                if int(g.get("contype", "1")) != THREAD:
                    el.remove(g)
            for s in list(el.findall("site")):
                el.remove(s)
        else:
            world.remove(el)
    ET.SubElement(world.find("body[@name='part']"), "inertial", pos="0 0 0", mass="0.05", diaginertia="1e-6 1e-6 1e-6")
    for tag in ("actuator", "tendon", "equality", "sensor", "contact", "keyframe"):
        for el in root.findall(tag):
            root.remove(el)
    asset = root.find("asset")
    for m in list(asset.findall("mesh")):
        if m.get("file"):
            asset.remove(m)
    return ET.tostring(root, encoding="unicode")


@wp.kernel
def _gather(src: wp.array2d(dtype=float), env: wp.array(dtype=int), qpos: wp.array2d(dtype=float)):
    w, k = wp.tid()
    qpos[w, k] = src[env[w], 9 + k]


@wp.kernel
def _to_mmp(nacon: wp.array(dtype=int), dist: wp.array(dtype=float), pos: wp.array(dtype=wp.vec3),
            frame: wp.array(dtype=wp.mat33), geom: wp.array(dtype=wp.vec2i), worldid: wp.array(dtype=int),
            is_part: wp.array(dtype=int), part_body: wp.array(dtype=int), fixture_body: wp.array(dtype=int),
            origin: wp.array(dtype=wp.vec3), offset: int, out_f: wp.array2d(dtype=float), out_i: wp.array2d(dtype=int)):
    i = wp.tid()
    if i >= nacon[0]:
        return
    w = worldid[i]
    g = geom[i]
    f = frame[i]
    n = wp.vec3(f[0, 0], f[0, 1], f[0, 2])        # from geom1 to geom2
    d = dist[i]
    p = pos[i] + origin[w]
    nb = n
    pb = p - n * (0.5 * d)                         # the point on geom1
    if is_part[g[0]] == 1:                         # the part is geom1: B (the fixture) is geom2
        nb = -n
        pb = p + n * (0.5 * d)
    r = offset + i
    for k in range(16):
        out_f[r, k] = 0.0
    out_f[r, 0] = pb[0]
    out_f[r, 1] = pb[1]
    out_f[r, 2] = pb[2]
    out_f[r, 3] = d
    out_f[r, 16] = nb[0]
    out_f[r, 17] = nb[1]
    out_f[r, 18] = nb[2]
    out_f[r, 19] = 1.0
    out_i[r, 20] = 45874 << 16                     # (compressed friction 0.7; the manifold model uses the bodies')
    out_i[r, 21] = 0
    out_i[r, 22] = part_body[w]
    out_i[r, 23] = -fixture_body[w]
    out_i[r, 24] = g[0] * 64 + g[1]                  # (the geom pair; canonical() adds the contact's rank)
    out_i[r, 25] = -2
    out_i[r, 26] = 0
    out_i[r, 27] = 0


def canonical(records):
    """Contact records (n, 28) put in an order of their contents in place: by the part's body and the geom pair (childA
    as _to_mmp writes it), then the point, the distance and the normal; childA becomes geom pair * 1024 + the rank among
    the pair's contacts."""
    n = len(records)
    if n < 2:
        if n:
            records.view(torch.int32)[:, 24] *= 1024
        return
    b = records.view(torch.int32)
    pack = lambda hi, lo: (b[:, hi].long() << 32) | (b[:, lo].long() & 0xFFFFFFFF)
    order = torch.arange(n, device=records.device)
    for key in (pack(18, 17), pack(16, 3), pack(2, 1), pack(0, 24), pack(22, 24)):     # least significant first
        order = order[torch.sort(key[order], stable=True).indices]
    records[:] = records[order]
    group = pack(22, 24)
    start = torch.ones(n, dtype=torch.bool, device=records.device)
    start[1:] = group[1:] != group[:-1]
    index = torch.arange(n, device=records.device)
    first = torch.cummax(torch.where(start, index, 0), 0).values
    b[:, 24] = (b[:, 24] * 1024 + (index - first)).int()


class _Task:
    def __init__(self, name, pcfg, env_ids, part_body, fixture_body, origins, device, grid_store, grid_type, nconmax):
        import mujoco_warp as mjw
        self.mjm = mujoco.MjModel.from_xml_string(sdf_xml(name, pcfg))
        mjd = mujoco.MjData(self.mjm)
        mujoco.mj_forward(self.mjm, mjd)
        n = len(env_ids)
        self.env_ids = torch.as_tensor(env_ids, device=device)
        with wp.ScopedDevice(device):
            self.env = wp.array(np.asarray(env_ids, np.int32), dtype=int)
            self.m = mjw.put_model(self.mjm)
            self.m.opt.warn_overflow = 0
            warp_sdf.patch_model(self.m, self.mjm, grid_type, grid_store)
            self.d = mjw.put_data(self.mjm, mjd, nworld=n, nconmax=nconmax, njmax=4 * nconmax)
            part = self.mjm.body("part").id
            self.is_part = wp.array((self.mjm.geom_bodyid == part).astype(np.int32), dtype=int)
            self.part_body = wp.array(np.asarray(part_body, np.int32), dtype=int)
            self.fixture_body = wp.array(np.asarray(fixture_body, np.int32), dtype=int)
            self.origin = wp.array(np.asarray(origins, np.float32), dtype=wp.vec3)
        self.qpos = wp.to_torch(self.d.qpos)
        self.capacity = self.d.naconmax


class SdfContacts:
    """MuJoCo Warp SDF contacts of the environments of the tasks with SDF geoms (task index -> environments)."""

    def __init__(self, tasks, pcfg, device, stream_ptr, nconmax=96):
        """tasks: [(task name, env ids, part MMP bodies, fixture MMP bodies, env origins)]."""
        self.device = device
        self.stream = wp.Stream(wp.get_device(device), cuda_stream=stream_ptr)
        bolt, nut, grid = warp_sdf.plugin_types()
        warp_sdf.install(bolt, nut, grid)
        store = self.store = warp_sdf.GridStore(device)     # (kept: the models hold raw pointers to its grids)
        self.tasks = [_Task(t[0], pcfg, *t[1:], device, store, grid, nconmax) for t in tasks]
        self.capacity = sum(t.capacity for t in self.tasks)
        self.records = torch.zeros(self.capacity, CONTACT_WORDS, device=device)
        self._f = wp.from_torch(self.records)
        self._i = wp.from_torch(self.records.view(torch.int32), dtype=wp.int32)
        self.graph = None
        self.qpos_src = None

    def _record(self):
        import mujoco_warp as mjw
        from mujoco_warp._src.smooth import kinematics
        src = wp.from_torch(self.qpos_src)
        offset = 0
        for t in self.tasks:
            wp.launch(_gather, (len(t.env_ids), 7), inputs=[src, t.env, t.d.qpos])
            kinematics(t.m, t.d)
            mjw.collision(t.m, t.d)
            c = t.d.contact
            wp.launch(_to_mmp, t.capacity, inputs=[t.d.nacon, c.dist, c.pos, c.frame, c.geom, c.worldid, t.is_part,
                                                   t.part_body, t.fixture_body, t.origin, offset, self._f, self._i])
            offset += t.capacity

    def compute(self, qpos):
        """Contacts at the parts' poses (qpos (N, 16), MuJoCo's layout, each environment's in its own frame; the same
        tensor every call: the work is a CUDA graph that reads it) into records; returns their count (read back)."""
        if self.graph is None or qpos.data_ptr() != self.qpos_src.data_ptr():
            self.qpos_src = qpos
            with wp.ScopedStream(self.stream):
                self._record()                    # (builds the kernels)
                with wp.ScopedCapture() as cap:
                    self._record()
            self.graph = cap.graph
        wp.capture_launch(self.graph, stream=self.stream)
        counts = [t.d.nacon for t in self.tasks]
        n = [int(c.numpy()[0]) for c in counts]
        # (records are packed per task: move each task's block after the previous one's contacts)
        out, offset = 0, 0
        for t, k in zip(self.tasks, n):
            k = min(k, t.capacity)
            if out != offset and k:
                self.records[out:out + k] = self.records[offset:offset + k].clone()
            out += k
            offset += t.capacity
        canonical(self.records[:out])
        return out


# SdfDcdContacts' collider settings by family, and by task where one differs (pretrained policy, 64 to 160 episodes
# per task): the threads are exact distance fields, whose overlaps are the sums of their depths, and a few contacts
# hold them (more jam the nuts: N16 98% with 2, 88% with 4, 17% with 32); the connectors' grids of thin walls are not
# distances away from their surfaces, so an overlap is twice its shallower depth, and the plugs need many contacts
# around their shells (USB 9% with 4, 40% with 8, 71% with 32; MuJoCo Warp's contacts 82%) - but BNC's round
# bayonet fewer (99% with 8, 92% with 32)
FAMILY_SETTINGS = {
    "nut": dict(seeds=8, contacts=2, shallow_overlap=False),
    "connector": dict(seeds=32, contacts=32, shallow_overlap=True),
}
TASK_SETTINGS = {"BNC": dict(seeds=16, contacts=8)}


class SdfDcdContacts:
    """The same contacts from Metamorphysics' own SDF collider (sdf_dcd: Giles and Andrews' search with a temporal
    cache of points per pair), with the plugin's SDFs: the thread SDFs as its bolt and nut shapes, the connectors'
    grids as its grids, a collider per setting: FAMILY_SETTINGS and TASK_SETTINGS (`nut`, `connector`: dicts that
    override a family's, `tasks`: {task: dict}). The other settings are the sdf_dcd demo's (for 1 m bodies) scaled
    by `scale`."""

    def __init__(self, mmp, world, tasks, device, stream_ptr, scale=0.002, iterations=50, margin=5e-4, nut=None,
                 connector=None, tasks_settings=None):
        """tasks: [(task name, env ids, part MMP bodies, fixture MMP bodies, part ipos, part iquat (wxyz))]."""
        from ..tasks import TASKS
        from ..threads import SDF_GRID_DIR, THREADS, thread_attributes
        from ..scene import CONNECTOR_GRIDS
        self.world, self.stream_ptr = world, stream_ptr
        self.dev = torch.device(device)
        overrides = dict(nut=nut or {}, connector=connector or {})
        per_task = {**TASK_SETTINGS, **(tasks_settings or {})}
        groups = {}                     # settings -> tasks
        for t in tasks:
            fam = "nut" if TASKS[t[0]].family == "nut" else "connector"
            settings = {**FAMILY_SETTINGS[fam], **per_task.get(t[0], {}), **overrides[fam]}
            groups.setdefault((fam, tuple(sorted(settings.items()))), []).append(t)
        self.colliders = []             # (collider, pairs)
        self.body_shape = {}            # (the collider and its shape of each body, for evaluating its field)
        for (fam, settings), group in groups.items():
            settings = dict(settings)
            n = sum(len(t[1]) for t in group)
            c = mmp.SdfCollider(max_bodies=world.num_bodies, max_pairs=max(n, 1), device=self.dev.index or 0)
            c.configure(iterations=iterations, margin=margin, contact_offset=0.0, tolerance=0.01 * scale,
                        epsilon=0.1 * scale ** 2, alpha=0.01, cache_replacement_threshold=0.01 * scale, **settings)
            grids = {}

            def grid(gid):
                if gid not in grids:
                    raw = (SDF_GRID_DIR / f"grid_{gid}.bin").read_bytes()
                    dims = np.frombuffer(raw, np.int32, 3, 0)
                    o = np.frombuffer(raw, np.float64, 4, 12)
                    v = np.frombuffer(raw, np.float16, int(np.prod(dims)), 44).astype(np.float32)
                    v = np.ascontiguousarray(v.reshape(dims[2], dims[1], dims[0]).transpose(2, 1, 0))
                    grids[gid] = c.register_grid(v, o[:3].tolist(), [float(o[3])] * 3)
                return grids[gid]

            pairs = []
            for name, ids, parts, fixtures, ipos, iquat in group:
                if fam == "nut":
                    b, nut_ = thread_attributes(name)
                    b, nut_ = list(b.values()), list(nut_.values())
                    R = nut_[5] * 1.1547005383792515 + 0.0005
                    part = c.register_shape("nut", nut_, [-R, -R, nut_[6] - 5e-4, R, R, nut_[7] + 5e-4])
                    R = b[3] + 0.0003
                    fixture = c.register_shape("bolt", b, [-R, -R, b[4] - 3e-4, R, R, b[5] + 3e-4])
                else:
                    pid, sid = CONNECTOR_GRIDS[name]
                    part, fixture = grid(pid), grid(sid)
                # the part's SDF in its body's frame (the centre of mass and principal axes)
                Ri = np.empty(9); mujoco.mju_quat2Mat(Ri, np.asarray(iquat, float)); Ri = Ri.reshape(3, 3)
                conj = np.array([-iquat[1], -iquat[2], -iquat[3], iquat[0]])          # xyzw of the inverse
                part = c.register_compound([(part, tuple(float(x) for x in Ri.T @ -np.asarray(ipos)),
                                             tuple(float(x) for x in conj))])
                for pb, fb in zip(parts, fixtures):
                    c.set_body_shape(int(pb), part)
                    c.set_body_shape(int(fb), fixture)
                    self.body_shape[int(pb)], self.body_shape[int(fb)] = (c, part), (c, fixture)
                    pairs.append((int(pb), int(fb), 0, 0))
            self.colliders.append((c, torch.as_tensor(np.asarray(pairs, np.int32), device=self.dev)))
        self.aabbs = torch.zeros(world.num_bodies, 8, device=self.dev)
        self.records = None

    def compute(self, part_qpos=None):
        """The colliders' contacts into records (in the world's stream); returns their count."""
        w = self.world
        out = []
        for c, pairs in self.colliders:
            c.update_aabbs(w.num_bodies, w.bodies_ptr, self.aabbs.data_ptr(), self.stream_ptr)
            n = c.compute_pairs(len(pairs), pairs.data_ptr(), w.bodies_ptr, self.aabbs.data_ptr(), self.stream_ptr)
            if n:
                out.append(wp.to_torch(wp.array(ptr=c.contacts_ptr, dtype=wp.float32, shape=(n, CONTACT_WORDS),
                                                device=wp.get_device(str(self.dev)), copy=False)))
        if out:
            self.records = out[0] if len(out) == 1 else torch.cat(out)
        return sum(len(r) for r in out)

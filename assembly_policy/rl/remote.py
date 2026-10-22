"""Remote simulation workers for one learner: one TCP round trip per control step.

The learner (RemoteVecEnv) listens on a TCP port and writes its address to RUN_DIR/server.json.
Each worker process connects, receives a slice of environments (current and next reset states),
and then serves STEP messages: apply replacement `next` states, step, auto-reset finished envs
to their `next` state, and return rewards, terminations, pre-reset observations of finished envs
and post-reset observations. The bank, sampler, policy and learner stay on the learner, so a
remote run is the same algorithm as a local one (vecenv.LocalVecEnv). Messages are pickled numpy
arrays over multiprocessing.connection, authenticated with a per-run key.
"""
from __future__ import annotations

import json
import os
import secrets
import selectors
import socket
import time
from multiprocessing.connection import Client, Listener, wait
from pathlib import Path

import numpy as np

from ..sim.env import HISTORY, PROPRIO_DIM, BatchedAssemblyEnv, EnvConfig
from .vecenv import INFO_KEYS, OBS_KEYS

PROTO = 2   # 2: workers send only the newest proprio frame; the learner keeps the 5-frame history


def _pack(o, proto, ids=None):
    """Observation dict for the wire: protocol 2 replaces the 150-d proprio history by its newest
    30-d frame (the learner rebuilds the history exactly: shift, or repeat after a reset)."""
    sel = (lambda x: x) if ids is None else (lambda x: x[ids])
    d = {k: sel(o[k]) for k in OBS_KEYS if k != "proprio"}
    if proto >= 2:
        d["frame"] = np.ascontiguousarray(sel(o["proprio"])[:, -PROPRIO_DIM:])
    else:
        d["proprio"] = sel(o["proprio"])
    return d


def _nodelay(conn):
    """Disable Nagle: Connection.send writes header and payload separately for >16 KB messages,
    which with delayed ACKs can stall each message by ~40 ms."""
    try:
        sock = socket.socket(fileno=os.dup(conn.fileno()))
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.close()  # closes only the duplicate descriptor; the option stays on the socket
    except OSError:
        pass
    return conn


class RemoteVecEnv:
    """Elastic set of remote simulation workers. The N environments are split into n_conns fixed slices,
    one per worker connection. Training starts once accept_timeout has passed with >= min_frac of the
    slices connected (or all are); unconnected slices are inactive. A worker that disconnects (e.g. its
    job was preempted) makes its slice inactive; a worker that connects later (listener stays open) is
    given an inactive slice with fresh reset states. Inactive environments return zero reward,
    done=False and info["valid"]=False, so the trainer masks their transitions out of PPO."""

    def __init__(self, run_dir, task_names, n_envs, ecfg: EnvConfig, provider, n_conns, t0=None, port=0,
                 accept_timeout=600.0, min_frac=0.5, task_cost=None, init_wait=60.0, entry_key=None, n_keys=None):
        self.N, self.provider, self.task_names = n_envs, provider, list(task_names)
        self.ecfg = ecfg
        # load balancing works on cost keys: the task, or (task, reset family) when entry_key maps every reset
        # entry to a key (an engaged nut costs ~15x a free one, so the start family predicts episode cost)
        tc = np.ones(len(task_names)) if task_cost is None else np.asarray(task_cost, dtype=float)
        self.entry_key = None if entry_key is None else np.asarray(entry_key, dtype=np.int64)
        T = len(task_names) if entry_key is None else int(n_keys)
        self.cost = tc[np.arange(T) * len(task_names) // T]      # prior; keys are task-major
        self.cost_prior = self.cost.copy()
        # online cost model (_learn_cost): decayed normal equations, fitted flag
        self._cA, self._cb, self._cy, self._cp, self._csteps, self._cfit = np.zeros((T, T)), np.zeros(T), 0.0, 0.0, 0, False
        self.horizon = int(round(ecfg.horizon_s / ecfg.control_dt))
        key = secrets.token_bytes(16)
        self.listener = Listener((socket.gethostbyname(socket.gethostname()), port), backlog=n_conns + 64,
                                 authkey=key)
        addr = self.listener.address
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        info = json.dumps(dict(host=addr[0], port=addr[1], key=key.hex(), n_conns=n_conns, time=time.time()))
        tmp = Path(run_dir) / "server.json.tmp"
        tmp.write_text(info)
        os.replace(tmp, Path(run_dir) / "server.json")
        print(f"[remote] listening on {addr}; waiting for {n_conns} worker connections", flush=True)
        conns, start = [], time.time()
        self._sock = self.listener._listener._socket
        self._sock.settimeout(5.0)
        while len(conns) < n_conns:
            try:
                conns.append(_nodelay(self.listener.accept()))
            except (socket.timeout, TimeoutError):
                if time.time() - start > accept_timeout and len(conns) >= max(1, int(min_frac * n_conns)):
                    break
                continue
            except (OSError, EOFError):  # handshake failed: ignore this client
                continue
            if len(conns) % 16 == 0:
                print(f"[remote] {len(conns)}/{n_conns} connected after {time.time()-start:.0f}s", flush=True)
        # slices are fixed by the target connection count; the listener stays open for late/replacement workers
        bounds = np.linspace(0, n_envs, n_conns + 1).astype(int)
        self.slices = [slice(bounds[i], bounds[i + 1]) for i in range(n_conns)]
        self.worker_of = np.repeat(np.arange(n_conns), np.diff(bounds))
        # per slice: summed cost of the pending next episodes, i.e. the slice's composition about one episode
        # ahead (balancing current + next made the two layers anti-correlated: a slice running expensive
        # episodes was given cheap next ones, then expensive ones again)
        self.load = np.zeros(n_conns)
        self.speed = np.ones(n_conns)   # per-worker time per unit cost relative to the median worker
        slots = np.arange(n_envs)
        self.entry, cur_s, cur_t = self._balanced(slots, *provider(n_envs))
        self.cur_task, self.cur_key = cur_t.copy(), self._key(self.entry, cur_t)
        self.next_entry, self.next_state, self.next_task = self._balanced(slots, *provider(n_envs))
        self.next_key = self._key(self.next_entry, self.next_task)
        self.load[:] = 0.0
        np.add.at(self.load, self.worker_of, self.cost[self.next_key])
        self.t0 = np.zeros(n_envs, dtype=np.int64) if t0 is None else np.asarray(t0)
        self.conn_of = [None] * n_conns           # slice -> live connection
        self.slice_of = {}                         # connection -> slice
        self.hosts = [""] * n_conns
        self.alive = np.zeros(n_envs, bool)
        self.hist = np.zeros((n_envs, HISTORY, PROPRIO_DIM), np.float32)   # protocol-2 proprio histories
        self.pending = {}                          # connection -> slice, waiting for its hello/obs
        self.sel = selectors.DefaultSelector()     # persistent poller over active connections
        self._obs = None
        for i, c in enumerate(conns):
            self._send_init(i, c, cur_s[self.slices[i]], cur_t[self.slices[i]], fresh=False)
        # start as soon as one worker is ready (gather others for up to init_wait s); the rest join between steps
        t_init = time.time()
        while self.pending and (not self.alive.any() or time.time() - t_init < init_wait):
            ready = wait(list(self.pending), timeout=5.0)
            for c in ready:
                self._finish_join(c, self.pending[c], block=True)
        hosts = {}
        for h in self.hosts:
            if h:
                hosts[h] = hosts.get(h, 0) + 1
        print(f"[remote] {int(sum(c is not None for c in self.conn_of))}/{n_conns} workers ready in {time.time()-start:.0f}s "
              f"({len(self.pending)} still initializing, join later) from {len(hosts)} hosts; "
              f"{int(self.alive.sum())}/{n_envs} envs active", flush=True)
        self._sock.settimeout(1.0)
        self._pending_replace = [None] * n_conns
        self.reset_profile()

    # ------------------------------------------------------------------ worker membership
    def _send_init(self, i, c, cur_s, cur_t, fresh):
        sl = self.slices[i]
        if fresh:  # replacement worker: new episodes for the whole slice
            n = sl.stop - sl.start
            self.load[i] = 0.0
            self.speed[i] = 1.0
            e, s_, t_ = self.provider(n)
            self.entry[sl], cur_s, cur_t = e, s_, t_
            self.cur_task[sl], self.cur_key[sl] = t_, self._key(e, t_)
            e2, s2, t2 = self.provider(n)
            self.next_entry[sl], self.next_state[sl], self.next_task[sl] = e2, s2, t2
            self.next_key[sl] = self._key(e2, t2)
            self.load[i] = self.cost[self.next_key[sl]].sum()
            t0 = np.zeros(n, dtype=np.int64)
        else:
            t0 = self.t0[sl]
        try:
            # only non-default settings: workers staged with older code (fewer EnvConfig fields) stay compatible
            ecfg = {k: v for k, v in vars(self.ecfg).items() if v != getattr(EnvConfig, k, None)}
            c.send(dict(cmd="init", tasks=self.task_names, ecfg=ecfg, state=cur_s, task=cur_t,
                        next_state=self.next_state[sl], next_task=self.next_task[sl], t=t0, proto=PROTO))
        except (OSError, EOFError):
            c.close()
            return
        self.pending[c] = i

    def _finish_join(self, c, i, block=False):
        """Receive a joining worker's hello and first observation; activate its slice."""
        try:
            if not block and not c.poll():
                return False
            info = c.recv()
            obs = c.recv()
        except (OSError, EOFError):
            self.pending.pop(c, None); c.close()
            return False
        self.pending.pop(c, None)
        sl = self.slices[i]
        if "frame" in obs:  # protocol 2: a fresh episode's history is its first frame repeated
            self.hist[sl] = obs["frame"][:, None, :]
            obs = {**obs, "proprio": self.hist[sl].reshape(sl.stop - sl.start, -1)}
        if self._obs is None:  # inactive envs carry a copy of a real observation (finite, never trained on)
            self._obs = {k: np.repeat(np.asarray(obs[k])[:1], self.N, axis=0) for k in OBS_KEYS}
        for k in OBS_KEYS:
            self._obs[k][sl] = obs[k]
        self.conn_of[i], self.slice_of[c], self.hosts[i] = c, i, info["host"]
        self.sel.register(c, selectors.EVENT_READ, i)
        self.alive[sl] = True
        return True

    def _drop(self, i, why=""):
        c = self.conn_of[i]
        if c is not None:
            try:
                self.sel.unregister(c)
            except (KeyError, ValueError):
                pass
            try:
                c.close()
            except OSError:
                pass
            self.slice_of.pop(c, None)
        self.conn_of[i] = None
        self.alive[self.slices[i]] = False
        print(f"[remote] worker for slice {i} ({self.hosts[i]}) lost{': ' + why if why else ''}; "
              f"{int(self.alive.sum())}/{self.N} envs active", flush=True)

    def _accept_new(self):
        """Non-blocking: accept late or replacement workers and give each an inactive slice."""
        import select
        while True:
            r, _, _ = select.select([self._sock], [], [], 0)
            if not r:
                return
            try:
                c = _nodelay(self.listener.accept())
            except (OSError, EOFError, socket.timeout):
                return
            busy = set(self.pending.values())
            free = [i for i in range(len(self.slices)) if self.conn_of[i] is None and i not in busy]
            if not free:
                c.close()          # surplus worker: it retries this learner every few minutes (worker_main)
                continue
            self._send_init(free[0], c, None, None, fresh=True)
            print(f"[remote] worker joining slice {free[0]}", flush=True)

    def alive_mask(self):
        return self.alive.copy()

    # ------------------------------------------------------------------ load balancing
    def _key(self, e, t):
        return t.copy() if self.entry_key is None else self.entry_key[e]

    def _balanced(self, slots, e, s, t):
        """Assign k drawn reset entries to k free env slots so that per-worker simulation cost stays even
        (greedy: most expensive entry to the least-loaded worker). Only the slot assignment changes, not
        the sampled entries, so the algorithm is unchanged; it removes stragglers in mixed-task runs."""
        k, cost = len(slots), self.cost[self._key(e, t)]
        if k <= 1 or np.all(self.cost == self.cost[0]):
            np.add.at(self.load, self.worker_of[slots], cost)
            return e, s, t
        if k > 4096:  # initial fill of all slots: deal entries in decreasing cost, snake order over workers
            W = len(self.load)
            order = np.argsort(-cost, kind="stable")
            ranks = np.arange(k); rnd, pos = ranks // W, ranks % W
            wk = np.where(rnd % 2 == 0, pos, W - 1 - pos)            # worker for the j-th most expensive draw
            free = {w: list(slots[self.worker_of[slots] == w]) for w in range(W)}
            out = np.empty(k, dtype=np.int64)
            spill = []
            for j, w in zip(order, wk):
                if free[w]:
                    out[j] = free[w].pop()
                else:
                    spill.append(j)
            rest = [x for w in range(W) for x in free[w]]
            for j, x in zip(spill, rest):
                out[j] = x
            np.add.at(self.load, self.worker_of[out], cost)
            inv = np.empty(k, dtype=np.int64)
            inv[np.searchsorted(slots, out)] = np.arange(k)
            return e[inv], s[inv], t[inv]
        order = np.argsort(-cost, kind="stable")
        free = list(slots)
        out = np.empty(k, dtype=np.int64)
        for j in order:  # k is a few per step (one per finished episode)
            w = self.worker_of[free]
            pick = int(np.argmin((self.load[w] + cost[j]) * self.speed[w]))   # earliest finish on uneven nodes
            out[j] = pick_slot = free.pop(pick)
            self.load[self.worker_of[pick_slot]] += cost[j]
        inv = np.empty(k, dtype=np.int64)  # slot order -> draw index
        inv[np.searchsorted(slots, out)] = np.arange(k)
        return e[inv], s[inv], t[inv]

    def _learn_cost(self, got):
        """Online model of worker compute time, used by _balanced: per-task env-step costs (ms) by decayed
        ridge least squares of each worker's step time on its slice's task counts (the prior table only
        sets the scale and the start), and per-worker speed factors (slower CPU / busier node > 1)."""
        got = [(i, ct) for i, ct in got if np.isfinite(ct) and ct > 0]
        if not got:
            return
        T = len(self.cost)
        idx = np.array([i for i, _ in got]); y = 1000.0 * np.array([ct for _, ct in got])
        X = np.zeros((len(self.slices), T))
        np.add.at(X, (self.worker_of, self.cur_key), 1.0)
        X = X[idx]
        ys, g = y / self.speed[idx], 0.998          # memory ~500 steps
        self._cA = g * self._cA + X.T @ X; self._cb = g * self._cb + X.T @ ys
        self._cy = g * self._cy + ys.sum(); self._cp = g * self._cp + (X @ self.cost_prior).sum()
        self._csteps += 1
        if self._csteps % 32 == 0:
            prior = self.cost_prior * self._cy / max(self._cp, 1e-9)
            lam = 1e-3 * np.trace(self._cA) / T
            c = np.linalg.solve(self._cA + lam * np.eye(T), self._cb + lam * prior)
            self.cost = np.maximum(c, 0.02 * c.max())
            self.load[:] = 0.0
            np.add.at(self.load, self.worker_of, self.cost[self.next_key])
            self._cfit = True
        if self._cfit:
            r = np.clip(y / np.maximum(X @ self.cost, 1e-6), 0.2, 5.0)
            self.speed[idx] = 0.98 * self.speed[idx] + 0.02 * r
            live = np.array([c is not None for c in self.conn_of])
            if live.any():
                self.speed /= np.median(self.speed[live])

    def reset_profile(self):
        self.prof = dict(wall=[], ct_mean=[], ct_max=[], ct_sum=np.zeros(len(self.slices)), ct_n=np.zeros(len(self.slices)),
                         active=[])

    def profile(self):
        """Per-step learner round-trip times vs. worker compute times since the last call (ms)."""
        p = self.prof
        if not p["wall"]:
            return {}
        w = 1000 * np.asarray(p["wall"])
        slow = int(np.argmax(p["ct_sum"]))
        # per-worker mean step time: persistent stragglers (max close to ct_max) vs. per-step spikes (max << ct_max)
        cw = 1000 * p["ct_sum"][p["ct_n"] > 0] / p["ct_n"][p["ct_n"] > 0]
        X = np.zeros(len(self.task_names)); np.add.at(X, self.cur_task[self.slices[slow]], 1.0)
        out = dict(rt_med=float(np.median(w)), rt_p90=float(np.percentile(w, 90)), rt_max=float(w.max()),
                   ct_mean=1000 * float(np.nanmean(p["ct_mean"])), ct_max=1000 * float(np.nanmean(p["ct_max"])),
                   slow_host=self.hosts[slow], active_envs=float(np.mean(p["active"])),
                   task_cost=[round(float(c), 1) for c in self.cost],
                   ct_w=[round(float(np.percentile(cw, q)), 1) for q in (50, 90, 99, 100)] if len(cw) else None,
                   slow_tasks={self.task_names[t]: int(n) for t, n in enumerate(X) if n}, slow_speed=float(self.speed[slow]),
                   speed_max=float(self.speed[[c is not None for c in self.conn_of]].max(initial=0.0)))
        self.reset_profile()
        return out

    # ------------------------------------------------------------------ stepping
    def observe(self):
        return {k: v.copy() for k, v in self._obs.items()}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        alive_before = self.alive.copy()
        live = [i for i, c in enumerate(self.conn_of) if c is not None]
        sent = []
        for i in live:
            try:
                self.conn_of[i].send(dict(cmd="step", action=action[self.slices[i]], replace=self._pending_replace[i]))
                sent.append(i)
            except (OSError, EOFError) as e:
                self._drop(i, repr(e))
        self._pending_replace = [None] * len(self.slices)
        t0 = time.perf_counter()
        N = self.N
        reward = np.zeros(N, np.float32); done = np.zeros(N, bool); ep_len = np.zeros(N, np.int64)
        info = {k: np.zeros(N, dtype=np.float64 if k in ("d_asm", "cost") else bool) for k in INFO_KEYS}
        final_parts, res, cts, got = [], set(), [], []
        expected = set(sent)
        while expected:  # unpickle and place replies as they arrive (overlaps with slower workers)
            events = self.sel.select(timeout=60.0)
            if not events:  # hung workers: drop them rather than stall the whole run
                for i in list(expected):
                    self._drop(i, "no reply in 60 s")
                expected.clear()
            for key, _ in events:
                i = key.data
                if i not in expected:
                    continue
                expected.discard(i)
                try:
                    r = key.fileobj.recv()
                except (OSError, EOFError) as e:
                    self._drop(i, repr(e))
                    continue
                res.add(i)
                sl = self.slices[i]
                o, fin = r["obs"], r["final"]
                if "frame" in o:  # protocol 2: rebuild the proprio histories
                    d = np.asarray(r["done"], bool)
                    prev = self.hist[sl]
                    if fin is not None:
                        dl = np.nonzero(d)[0]
                        fh = np.concatenate([prev[dl, 1:], fin["frame"][:, None, :]], 1)
                        fin = {**fin, "proprio": fh.reshape(len(dl), -1)}
                    new = np.concatenate([prev[:, 1:], o["frame"][:, None, :]], 1)
                    new[d] = o["frame"][d][:, None, :]          # auto-reset envs start a new history
                    self.hist[sl] = new
                    o = {**o, "proprio": new.reshape(len(new), -1)}
                for k in OBS_KEYS:
                    self._obs[k][sl] = o[k]
                reward[sl], done[sl], ep_len[sl] = r["reward"], r["done"], r["ep_len"]
                for k in INFO_KEYS:
                    info[k][sl] = r[k]
                if fin is not None:
                    final_parts.append((sl.start + np.nonzero(r["done"])[0], {k: fin[k] for k in OBS_KEYS}))
                ct = r.get("ct", np.nan); cts.append(ct); got.append((i, ct)); self.prof["ct_sum"][i] += ct
                self.prof["ct_n"][i] += 1
        ct = np.array(cts) if cts else np.array([np.nan])
        self._learn_cost(got)
        self.prof["wall"].append(time.perf_counter() - t0); self.prof["ct_mean"].append(np.nanmean(ct))
        self.prof["ct_max"].append(np.nanmax(ct)); self.prof["active"].append(int(self.alive.sum()))
        info["ep_len"] = ep_len
        info["valid"] = alive_before & self.alive        # transitions of envs whose worker died are dropped
        ids = np.nonzero(done)[0]
        final_obs = None
        finished = self.entry[ids].copy()
        if len(ids):
            final_parts.sort(key=lambda p: p[0][0])
            order = np.concatenate([p[0] for p in final_parts])
            final_obs = {k: np.concatenate([p[1][k] for p in final_parts]) for k in OBS_KEYS}
            assert np.array_equal(order, ids)
            np.subtract.at(self.load, self.worker_of[ids], self.cost[self.next_key[ids]])   # next becomes current
            self.cur_task[ids], self.cur_key[ids] = self.next_task[ids], self.next_key[ids]
            self.entry[ids] = self.next_entry[ids]
            e, s, t = self._balanced(ids, *self.provider(len(ids), finished, info["success"][ids]))
            self.next_entry[ids], self.next_state[ids], self.next_task[ids] = e, s, t
            self.next_key[ids] = self._key(e, t)
            for i in res:
                sl = self.slices[i]
                m = (ids >= sl.start) & (ids < sl.stop)
                if m.any():
                    self._pending_replace[i] = (ids[m] - sl.start, s[m], t[m])
        # membership changes take effect between steps: new workers become active for the next action
        self._accept_new()
        for c, i in list(self.pending.items()):
            self._finish_join(c, i)
        return reward, done, info, final_obs, finished

    def close(self):
        for c in self.conn_of:
            if c is None:
                continue
            try:
                c.send(dict(cmd="close"))
            except Exception:
                pass
        self.listener.close()


class _NoSlice(Exception):
    """The learner closed the connection before sending init: all slices were taken."""


def _serve(conn, nthread):
    try:
        init = conn.recv()
    except (EOFError, OSError) as e:
        raise _NoSlice() from e
    ecfg = EnvConfig(**{**init["ecfg"], "nthread": nthread})
    n = len(init["state"])
    env = BatchedAssemblyEnv(init["tasks"], np.zeros(n, dtype=np.int64), ecfg)
    env.set_states(np.arange(n), init["state"], init["task"])
    env.t[:] = init["t"]
    nxt_s, nxt_t = init["next_state"].copy(), init["next_task"].copy()
    proto = min(int(init.get("proto", 1)), PROTO)
    conn.send(dict(host=socket.gethostname(), nthread=nthread, proto=proto))
    conn.send(_pack(env.observe(), proto))
    while True:
        try:
            msg = conn.recv()
        except (EOFError, ConnectionError, OSError):
            print("[worker] learner disconnected", flush=True)
            break
        if msg["cmd"] == "close":
            break
        t0 = time.perf_counter()
        if msg["replace"] is not None:
            ids, s, t = msg["replace"]
            nxt_s[ids], nxt_t[ids] = s, t
        ep_len = env.t + 1
        r, done, info = env.step(msg["action"])
        ids = np.nonzero(done)[0]
        final = None
        if len(ids):
            final = _pack(env.observe(), proto, ids)
            env.set_states(ids, nxt_s[ids], nxt_t[ids])
        conn.send(dict(reward=r, done=done, obs=_pack(env.observe(), proto), final=final, ep_len=ep_len,
                       ct=time.perf_counter() - t0, **{k: info[k] for k in INFO_KEYS}))
    env.close()


def worker_main(run_dir, nthread, wait_s=7200, poll_s=30.0):
    """Serve learners at run_dir until none appears for wait_s; reconnects after learner restarts."""
    path, last = Path(run_dir) / "server.json", None
    while True:
        t0, conn, fail_since = time.time(), None, None
        while conn is None:
            if path.exists():
                info = json.loads(path.read_text())
                if info["time"] != last:
                    try:
                        conn = _nodelay(Client((info["host"], info["port"]), authkey=bytes.fromhex(info["key"])))
                        last = info["time"]
                        break
                    except Exception as e:  # refused / timed out / handshake interrupted: retry
                        if fail_since is None:
                            fail_since = time.time()
                            print(f"[worker] connect failed ({e!r}); retrying", flush=True)
                        if time.time() - fail_since > 300:  # learner already started without us (or is gone):
                            print(f"[worker] giving up on learner {info['time']}; waiting for a new one", flush=True)
                            last, fail_since = info["time"], None
                            continue
                        time.sleep(1 + 4 * np.random.random())
                        continue
            if time.time() - t0 > wait_s:
                print("[worker] no learner; exiting", flush=True)
                return
            time.sleep(poll_s)
        try:
            _serve(conn, nthread)
        except _NoSlice:  # surplus: retry the same learner later (a slice frees up when another worker's job ends)
            print("[worker] no free slice; retrying in ~2 min", flush=True)
            last = None
            time.sleep(90 + 60 * np.random.random())
        except (EOFError, OSError) as e:  # learner preempted/restarted mid-message: wait for the next one
            print(f"[worker] learner connection lost ({e!r})", flush=True)
        finally:
            conn.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--nthread", type=int, default=1)
    ap.add_argument("--poll", type=float, default=30.0)
    ap.add_argument("--wait", type=float, default=7200.0)
    a = ap.parse_args()
    worker_main(a.run_dir, a.nthread, a.wait, a.poll)

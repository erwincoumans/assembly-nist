"""Batched multi-task assembly simulator on CPU MuJoCo (mujoco.rollout thread pool).

One control step = hold a joint-position target for `substeps` physics steps entirely in
C (rollout), grouped by task model. Observations are compact (poses, joints, history);
point clouds are synthesized on the learner (see obs.py).

MDP (after Sec. IV-C/IV-D of the AlphaNIST paper):
  action a in R^8: 7 relative joint-position increments (q_target = q + scale*a) and a
  binary gripper command (a[7] < 0 closes); reward 1[d_asm < 1 mm] once, minus a control
  cost; termination on success, arm joint velocity limit, or 15 s horizon (time-out).
"""
from __future__ import annotations

from dataclasses import dataclass

import threading
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
from mujoco import rollout

from .kinematics import FrankaFK
from .robot import Indices, quat_to_mat
from .scene import ARM_KD, ARM_KP, TORQUE_LIMIT, VELOCITY_LIMIT, PhysicsConfig, build_xml
from .tasks import TASKS, scene_pose

STATE_SPEC = mujoco.mjtState.mjSTATE_FULLPHYSICS
PROPRIO_DIM = 30   # tcp pose 7, twist 6, joints 9, previous action 8
HISTORY = 5


def parse_scales(s):
    """'0.02+...' (7 per-joint action scales; '+', ',' or ':' separated, '+' survives sbatch --export) -> tuple;
    '' -> () (use the scalar action_scale)."""
    if not s:
        return ()
    if not isinstance(s, str):
        return tuple(float(x) for x in s)
    v = tuple(float(x) for x in s.replace("+", ",").replace(":", ",").split(","))
    assert len(v) == 7, f"--action-scale-joints needs 7 values, got {s!r}"
    return v


@dataclass
class EnvConfig:
    control_dt: float = 0.04          # 25 Hz
    horizon_s: float = 15.0
    action_scale: float = 0.02        # rad per unit action (all joints), unless action_scale_joints is set
    # per-joint scales (7 values). The relative target q + scale * a never catches up under the libfranka gains, so a
    # joint moves at ~(kp/kd) * scale * a rad/s: a uniform 0.02 gives the wrist 0.06-0.14 rad/s per unit action (a
    # 90-degree wrist turn takes 15-27 s at |a| = 1). The default equalizes all joints at ~0.5 rad/s per unit action.
    action_scale_joints: tuple = (0.05, 0.05, 0.05, 0.06, 0.07, 0.1, 0.17)
    success_threshold: float = 0.001
    velocity_limit_scale: float = 1.0
    cost_coef: float = 2e-5           # c in Eq. 6 on normalized terms; ~0.06 per random 15 s episode
    nthread: int = 4
    chunk: int = 4                    # envs per work item for the simulation thread pool
    grip_deadband: float = 0.5        # |a7| below this keeps the previous gripper command
    # closed-gripper opening target (m/finger): -0.01 saturates the servo at the 35 N/finger cap for every part,
    # like a force-controlled Franka grasp (0.0 made the force scale with the part width)
    grip_close: float = -0.01
    fine_families: str = "gear+nut"   # stiff-contact physics (0.5 ms, PhysicsConfig) for these task families
    fine_sdf_initpoints: int = 20     # SDF contact search for those families
    fine_sdf_iterations: int = 5


class TaskModel:
    def __init__(self, name, pcfg: PhysicsConfig, nthread):
        self.name = name
        self.model = mujoco.MjModel.from_xml_string(build_xml(name, pcfg))
        self.datas = [mujoco.MjData(self.model) for _ in range(nthread)]
        self.idx = Indices(self.model)
        self.fk = FrankaFK(self.model)
        t = TASKS[name]
        fpos, fquat = scene_pose(t.fixture_scene)
        self.fixture_pos, self.fixture_R = fpos, quat_to_mat(fquat)
        self.keypoints = np.asarray(t.keypoints)
        # assembled keypoints in world: part frame == part scene pose
        ppos, pquat = scene_pose(t.part_scene)
        self.goal_kp = ppos + self.keypoints @ quat_to_mat(pquat).T
        g = self.goal_kp[1] - self.goal_kp[0]
        self.goal_axis = g / np.linalg.norm(g)


class BatchedAssemblyEnv:
    def __init__(self, task_names, task_of_env, ecfg: EnvConfig | None = None, pcfg: PhysicsConfig | None = None):
        self.ecfg = ecfg or EnvConfig()
        self.pcfg = pcfg or PhysicsConfig(fine_families=self.ecfg.fine_families,
                                          fine_sdf_initpoints=self.ecfg.fine_sdf_initpoints,
                                          fine_sdf_iterations=self.ecfg.fine_sdf_iterations)
        self.task_names = list(task_names)
        self.models = [TaskModel(n, self.pcfg, self.ecfg.nthread) for n in self.task_names]
        m0 = self.models[0].model
        self.nstate = mujoco.mj_stateSize(m0, STATE_SPEC)
        for tm in self.models:
            assert mujoco.mj_stateSize(tm.model, STATE_SPEC) == self.nstate and tm.model.nu == m0.nu
        self.idx = self.models[0].idx
        # physics substeps per control step, per task model (fine-feature tasks may use a smaller timestep)
        self.substeps_k = [int(round(self.ecfg.control_dt / tm.model.opt.timestep)) for tm in self.models]
        self.dt_k = np.array([s * tm.model.opt.timestep for s, tm in zip(self.substeps_k, self.models)])
        self.substeps = self.substeps_k[0]
        self.horizon = int(round(self.ecfg.horizon_s / self.ecfg.control_dt))
        self.N = len(task_of_env)
        self.task = np.asarray(task_of_env, dtype=np.int64)
        self.state = np.zeros((self.N, self.nstate))
        self.prev_action = np.zeros((self.N, 8))
        self.grip_cmd = np.zeros(self.N)
        self.hist = np.zeros((self.N, HISTORY, PROPRIO_DIM), dtype=np.float32)
        self.t = np.zeros(self.N, dtype=np.int64)
        # Work queue over env chunks (mixed tasks) with per-thread, per-task MjData; mujoco.rollout
        # releases the GIL. Avoids per-task barriers and load imbalance of one pool call per task.
        self.pool = ThreadPoolExecutor(self.ecfg.nthread) if self.ecfg.nthread > 1 else None
        self._tls = threading.local()
        self.vlim = np.asarray(VELOCITY_LIMIT) * self.ecfg.velocity_limit_scale
        self.kp, self.kd, self.tlim = np.asarray(ARM_KP), np.asarray(ARM_KD), np.asarray(TORQUE_LIMIT)
        self.jlo, self.jhi = self.idx.jnt_range[:, 0], self.idx.jnt_range[:, 1]
        # state layout: [time, qpos(nq), qvel(nv)]
        nq = m0.nq
        self.sq = 1 + self.idx.arm_qpos
        self.sf = 1 + self.idx.finger_qpos
        self.sp = np.arange(1 + self.idx.part_qpos, 8 + self.idx.part_qpos)
        self.sqd = 1 + nq + self.idx.arm_dof
        self.spd = np.arange(1 + nq + self.idx.part_dof, 7 + nq + self.idx.part_dof)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()

    def _work(self, item):
        k, ids, ctrl, out = item
        tm = self.models[k]
        datas = getattr(self._tls, "datas", None)
        if datas is None:
            datas = self._tls.datas = [mujoco.MjData(m.model) for m in self.models]
        n, S = len(ids), self.substeps_k[k]
        st = np.empty((n, S, self.nstate))
        sd = np.empty((n, S, tm.model.nsensordata))
        rollout.rollout([tm.model] * n, [datas[k]], np.ascontiguousarray(self.state[ids]),
                        np.ascontiguousarray(np.repeat(ctrl[ids, None, :], S, 1)),
                        nstep=S, skip_checks=True, state=st, sensordata=sd)
        # per-env summaries over the substeps (tasks may differ in substep count)
        qd_sub, q_sub = st[:, :, self.sqd], st[:, :, self.sq]
        out["final"][ids] = st[:, -1]
        out["viol"][ids] = (np.abs(qd_sub) > self.vlim).any(axis=(1, 2))
        out["tau"][ids] = np.clip(self.kp * (ctrl[ids, None, :7] - q_sub) - self.kd * qd_sub, -self.tlim, self.tlim).mean(1)

    # ------------------------------------------------------------- state helpers
    def set_states(self, env_ids, states, task_ids=None):
        env_ids = np.asarray(env_ids)
        if task_ids is not None:
            self.task[env_ids] = task_ids
        self.state[env_ids] = states
        self.state[env_ids, 0] = 0.0
        self.t[env_ids] = 0
        self.prev_action[env_ids] = 0.0
        # latch starts consistent with the reset state: closed unless the fingers are (nearly) open
        self.grip_cmd[env_ids] = np.where(self.state[env_ids][:, self.sf].mean(1) > 0.036, 0.04, self.ecfg.grip_close)
        prop = self._proprio(env_ids)
        self.hist[env_ids] = prop[:, None, :]

    def _fk(self, ids):
        q = self.state[ids][:, self.sq]
        return self.models[0].fk.forward(q)

    def _proprio(self, ids):
        s = self.state[ids]
        fk = self.models[0].fk.forward(s[:, self.sq])
        lin, ang = self.models[0].fk.tcp_twist(fk, s[:, self.sqd])
        quat = _mat_to_quat_batch(fk["tcp_R"])
        # sign relative to the gripper-down orientation (0,1,0,0): w >= 0 canonicalization flips sign
        # step to step exactly where the gripper operates (w ~ 0)
        quat *= np.where(quat[:, 1:2] < 0, -1.0, 1.0)
        prop = np.concatenate([fk["tcp_p"], quat, lin, ang, s[:, self.sq], s[:, self.sf], self.prev_action[ids]], 1)
        return prop.astype(np.float32)

    def observe(self):
        """Compact observation dict (numpy). Point clouds are built on the learner."""
        s = self.state
        fk = self.models[0].fk.forward(s[:, self.sq])
        return dict(proprio=self.hist.reshape(self.N, -1).copy(), task=self.task.copy(),
                    tcp_p=fk["tcp_p"].astype(np.float32), tcp_R=fk["tcp_R"].astype(np.float32),
                    part_p=s[:, self.sp[:3]].astype(np.float32),
                    part_q=s[:, self.sp[3:7]].astype(np.float32),
                    finger=s[:, self.sf].astype(np.float32))

    # ------------------------------------------------------------- dynamics
    def step(self, action):
        a = np.array(action, dtype=np.float64, copy=True)
        q = self.state[:, self.sq]
        ctrl = np.empty((self.N, 8))
        scale = np.asarray(self.ecfg.action_scale_joints, float) if len(self.ecfg.action_scale_joints) else self.ecfg.action_scale
        ctrl[:, :7] = np.clip(q + scale * a[:, :7], self.jlo, self.jhi)
        # gripper: deadband latch on the 1-D command (open above +g, close below -g, else hold)
        g = self.ecfg.grip_deadband
        self.grip_cmd = np.where(a[:, 7] > g, 0.04, np.where(a[:, 7] < -g, self.ecfg.grip_close, self.grip_cmd))
        ctrl[:, 7] = self.grip_cmd
        out = dict(final=np.empty((self.N, self.nstate)), viol=np.zeros(self.N, bool), tau=np.empty((self.N, 7)))
        items = []
        chunk = max(self.ecfg.chunk, -(-self.N // (4 * max(self.ecfg.nthread, 1))))
        for k in range(len(self.models)):
            ids = np.nonzero(self.task == k)[0]
            if len(ids) == 0:
                continue
            if self.ecfg.nthread <= 1:
                items.append((k, ids, ctrl, out))
            else:
                items += [(k, c, ctrl, out) for c in np.array_split(ids, -(-len(ids) // chunk)) if len(c)]
        if self.ecfg.nthread <= 1:
            for it in items:
                self._work(it)
        else:
            list(self.pool.map(self._work, items))
        t_prev = self.state[:, 0].copy()
        self.state = out["final"].copy()
        self.t += 1
        vel_violation = out["viol"]
        # MuJoCo resets mjData (time -> 0) on divergence; detect it through time continuity
        reset_detected = np.abs(self.state[:, 0] - (t_prev + self.dt_k[self.task])) > 1e-6
        nonfinite = ~np.isfinite(self.state).all(1) | reset_detected
        # applied joint torques (clamped implicit PD), averaged over substeps
        cost = self.control_cost(a, self.state[:, self.sqd], out["tau"])
        d_asm = self.assembly_distance()
        # a safety failure in the same step vetoes success ("no simultaneous safety failure")
        safety = vel_violation | nonfinite
        success = (d_asm < self.ecfg.success_threshold) & ~safety
        timeout = (self.t >= self.horizon) & ~success & ~safety
        reward = success.astype(np.float64) - cost
        self.prev_action = a
        prop = self._proprio(np.arange(self.N))
        self.hist = np.concatenate([self.hist[:, 1:], prop[:, None]], 1)
        info = dict(success=success, safety=safety, timeout=timeout, d_asm=d_asm, cost=cost, unstable=nonfinite)
        return reward.astype(np.float32), success | safety | timeout, info

    def control_cost(self, a, qd, tau):
        """c * (||a||^2 + ||qd/qd_max||^2 + ||tau/tau_max||^2); normalization is a local choice."""
        return self.ecfg.cost_coef * ((a ** 2).sum(1) + ((qd / self.vlim) ** 2).sum(1) + ((tau / self.tlim) ** 2).sum(1))

    def assembly_distance(self):
        d = np.empty(self.N)
        pp, pq = self.state[:, self.sp[:3]], self.state[:, self.sp[3:7]]
        R = quat_to_mat(pq)
        for k, tm in enumerate(self.models):
            ids = np.nonzero(self.task == k)[0]
            if len(ids) == 0:
                continue
            kp = pp[ids, None, :] + np.einsum("nij,kj->nki", R[ids], tm.keypoints)
            d[ids] = np.linalg.norm(kp - tm.goal_kp[None], axis=2).max(1)
        return d


def _mat_to_quat_batch(R):
    """Rotation matrices (N,3,3) -> quaternions (N,4) wxyz with w >= 0 (Shepperd: largest-component pivot,
    exact also at 180-degree rotations, where the sign-from-antisymmetric-part shortcut fails)."""
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    c = np.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], 1)
    k = np.argmax(c, 1)
    s = np.sqrt(np.maximum(c[np.arange(len(R)), k], 1e-300)) * 2      # 4 * |q_k|
    q = np.empty((R.shape[0], 4))
    a21, a12, a02, a20, a10, a01 = R[:, 2, 1], R[:, 1, 2], R[:, 0, 2], R[:, 2, 0], R[:, 1, 0], R[:, 0, 1]
    cand = {0: (s / 4, (a21 - a12) / s, (a02 - a20) / s, (a10 - a01) / s),
            1: ((a21 - a12) / s, s / 4, (a01 + a10) / s, (a02 + a20) / s),
            2: ((a02 - a20) / s, (a01 + a10) / s, s / 4, (a12 + a21) / s),
            3: ((a10 - a01) / s, (a02 + a20) / s, (a12 + a21) / s, s / 4)}
    for j, comps in cand.items():
        m = k == j
        if m.any():
            q[m] = np.stack([x[m] for x in comps], 1)
    return q * np.where(q[:, :1] < 0, -1.0, 1.0)

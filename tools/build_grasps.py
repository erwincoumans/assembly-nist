"""Build a validated antipodal grasp library for one task's part (multiprocess)."""
import argparse, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assembly_policy.sim.grasps import GraspSampler, library_path  # noqa: E402
from assembly_policy.sim.tasks import TASKS  # noqa: E402


def work(args):
    task, seed, n, max_tries = args
    gs = GraspSampler(task, seed=seed)
    return gs.build(n, max_tries=max_tries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--max-tries-per-worker", type=int, default=40000)
    a = ap.parse_args()
    per = -(-a.n // a.workers)
    t0 = time.time()
    with Pool(a.workers) as pool:
        res = pool.map(work, [(a.task, 1000 + i, per, a.max_tries_per_worker) for i in range(a.workers)])
    P = np.concatenate([r[0] for r in res if len(r[0])]); Q = np.concatenate([r[1] for r in res if len(r[1])])
    H = np.concatenate([r[2] for r in res if len(r[2])]); tries = sum(r[3] for r in res)
    out = library_path(TASKS[a.task].part_asset); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, pos=P, quat=Q, half=H)
    print(json.dumps(dict(task=a.task, part=TASKS[a.task].part_asset, grasps=len(P), candidates=tries,
                          acceptance=len(P) / max(tries, 1), wall_s=time.time() - t0)))


if __name__ == "__main__":
    main()

"""Build the reset banks: for every task, equal numbers of the five reset families (loose, pickup, transport,
partial, near-mating), a training bank and a disjoint evaluation bank.

python scripts/build_banks.py                              # all 20 tasks, 640 train + 128 eval per (task, family)
python scripts/build_banks.py --tasks R8+N12 --workers 32  # a subset (e.g. one Slurm array task per task)

Output: <out>/<task>_{train,eval}.npz with arrays states (full MuJoCo physics state), family, level, and a .json
with generation statistics. Seeds are namespaced by (task, split, family, worker), so train and eval banks never
share generator streams; the eval bank is also checked to share no state with the train bank.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assembly_policy.sim.resets import FAMILIES, ResetGenerator  # noqa: E402
from assembly_policy.sim.scene import PhysicsConfig  # noqa: E402
from assembly_policy.sim.env import EnvConfig  # noqa: E402
from assembly_policy.sim.tasks import BANK_DIR, resolve_tasks  # noqa: E402


def seed_for(*keys):
    return int.from_bytes(hashlib.sha256("/".join(map(str, keys)).encode()).digest()[:8], "little")


def work(args):
    task, split, fam, worker, n, fine = args
    g = ResetGenerator(task, PhysicsConfig(fine_families=fine), seed=seed_for("assemblenist_v1", task, split, fam, worker))
    t0 = time.time()
    S, tries = g.generate(fam, n)
    return fam, worker, S, np.asarray(g.levels), tries, time.time() - t0


def row_hashes(S):
    return {hashlib.sha1(np.round(s[1:], 9).tobytes()).hexdigest() for s in S}


def build(task, split, per_family, pool, workers, out, fine):
    jobs = []
    for fam in FAMILIES:
        per = [per_family // workers + (i < per_family % workers) for i in range(workers)]
        jobs += [(task, split, fam, w, n, fine) for w, n in enumerate(per) if n > 0]
    t0 = time.time()
    res = pool.map(work, jobs, chunksize=1)
    states, fams, levels, stats = [], [], [], {}
    for fam, w, S, lv, tries, el in sorted(res, key=lambda r: (FAMILIES.index(r[0]), r[1])):
        states.append(S); fams.append(np.full(len(S), FAMILIES.index(fam))); levels.append(lv)
        st = stats.setdefault(fam, dict(n=0, attempts=0, cpu_s=0.0))
        st["n"] += len(S); st["attempts"] += tries; st["cpu_s"] += round(el, 1)
    S, F, Lv = np.concatenate(states), np.concatenate(fams), np.concatenate(levels)
    if split == "eval" and (out / f"{task}_train.npz").exists():
        overlap = len(row_hashes(S) & row_hashes(np.load(out / f"{task}_train.npz")["states"]))
        assert overlap == 0, f"{overlap} eval states identical to training states"
    np.savez_compressed(out / f"{task}_{split}.npz", states=S, family=F, level=Lv)
    meta = dict(task=task, split=split, per_family=per_family, families=FAMILIES, fine_families=fine, stats=stats,
                wall_s=round(time.time() - t0, 1), workers=workers)
    (out / f"{task}_{split}.json").write_text(json.dumps(meta, indent=1))
    print(f"{task} {split}: {len(S)} states in {meta['wall_s']:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="all20")
    ap.add_argument("--per-family", type=int, default=640, help="training states per (task, family)")
    ap.add_argument("--eval-per-family", type=int, default=128, help="evaluation states per (task, family)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)))
    ap.add_argument("--out", default=str(BANK_DIR))
    ap.add_argument("--fine-families", default=EnvConfig.fine_families,
                    help="families built with the stiff-contact physics used in training")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as pool:
        for task in resolve_tasks(a.tasks):
            build(task, "train", a.per_family, pool, a.workers, out, a.fine_families)
            build(task, "eval", a.eval_per_family, pool, a.workers, out, a.fine_families)


if __name__ == "__main__":
    main()

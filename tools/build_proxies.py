"""Coarse CoACD convex proxies of connector parts (contacts other than plug<->socket, which use grid SDFs)."""
import sys, json, time
from pathlib import Path
import numpy as np, coacd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assembly_policy.sim.tasks import ROOT, load_obj  # noqa: E402
coacd.set_log_level("error")
stem, thr = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
V, F = load_obj(stem); t0 = time.time()
parts = coacd.run_coacd(coacd.Mesh(V, F), threshold=thr, max_convex_hull=24, preprocess_mode="auto", resolution=2000,
                        mcts_iterations=60, seed=0)
out = ROOT / "assets/proxies"; out.mkdir(parents=True, exist_ok=True)
np.save(out / f"{stem}.npy", np.array([np.asarray(p[0]) for p in parts], dtype=object), allow_pickle=True)
print(json.dumps(dict(stem=stem, parts=len(parts), seconds=time.time() - t0)))

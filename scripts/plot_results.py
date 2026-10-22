"""Evaluation success vs. training iteration, with the numbers reported in the AlphaNIST paper for reference.

python scripts/plot_results.py                                  # the reference run (results/reference_e8192.csv)
python scripts/plot_results.py --runs runs/e8192 runs/e4096     # your runs (reads <run>/eval/*.json)
Writes results/success_vs_iteration.png (or --out).
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "legend.fontsize": 12})

ROOT = Path(__file__).resolve().parents[1]
# AlphaNIST (arXiv 2610.13354) Fig. 7: evaluation success after 25K iterations, beta sampler, sparse reward
PAPER = {1024: 38.6, 4096: 78.9, 8192: 86.9}
COLORS = ["#005f73", "#bb3e03", "#0a9396", "#9b2226", "#3d405b"]


def run_curve(run):
    pts = []
    for f in sorted(Path(run, "eval").glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("meta", {}).get("iteration") is not None:
            pts.append((d["meta"]["iteration"], 100 * d["overall"]["mean"]))
    return sorted(pts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--reference", default=str(ROOT / "results/reference_e8192.csv"))
    ap.add_argument("--out", default=str(ROOT / "results/success_vs_iteration.png"))
    a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=200)
    if a.reference and Path(a.reference).exists():
        rows = list(csv.DictReader(open(a.reference)))
        pts = [(int(r["iteration"]), float(r["success"])) for r in rows if r["success"]]
        ax.plot(*zip(*pts), color=COLORS[0], lw=1.8, marker="o", ms=4.5, label="AssembleNIST, 8K envs (reference run)")
    for k, run in enumerate(a.runs):
        pts = run_curve(run)
        if pts:
            ax.plot(*zip(*pts), color=COLORS[(k + 1) % len(COLORS)], lw=1.2, marker="o", ms=3, label=Path(run).name)
    for envs, val in PAPER.items():
        ax.plot([25000], [val], marker="D", ms=8, mfc="white", mec="black", mew=1.8, ls="none")
        ax.annotate(f"AlphaNIST, {envs // 1024}K envs: {val}%", (25000, val), xytext=(-10, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=12)
    ax.plot([], [], marker="D", ms=8, mfc="white", mec="black", mew=1.8, ls="none", label="AlphaNIST paper (Isaac Lab), 25K iterations")
    ax.set_xlabel("training iteration"); ax.set_ylabel("success rate (%)")
    ax.set_xlim(0, 26000); ax.set_ylim(0, 100)
    ax.grid(True, color="#e6e5df", lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

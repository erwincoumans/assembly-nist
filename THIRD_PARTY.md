# Third-party work and assets

AssembleNIST is an independent implementation that builds on ideas from the AlphaNIST paper (Zhang et al.,
*AlphaNIST: Emergent Assembly Skills from Binary Task Rewards*, arXiv:2610.13354, <https://alpha-nist.github.io/>) and
the SGS paper (Zhang et al., *A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot
Control*, arXiv:2610.12465, <https://sgs-rl.github.io/>). It does not contain code from those projects and is not
affiliated with or endorsed by their authors.

| What | Source | License | Where |
| --- | --- | --- | --- |
| Physics engine, `mujoco.rollout` | [MuJoCo](https://github.com/google-deepmind/mujoco) | Apache-2.0 | dependency |
| Franka Emika Panda model and meshes | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) `franka_emika_panda` | Apache-2.0 (`assets/franka/LICENSE`) | `assets/franka/` (unmodified; the scene builder edits it in memory) |
| NIST Assembly Task Board 1 part / fixture meshes and board layout | [UW Lab](https://github.com/UW-Lab/UWLab) NIST assets (`uwlab-assets`, `Props/NIST`, commit `f9486a6`), modelled after the NIST Assembly Task Board 1 | **to be confirmed before redistribution** (UW Lab's code is BSD-3-Clause) | `assets/meshes/` (OBJ meshes extracted with `tools/extract_nist_meshes.py`), and derived files in `assets/hulls/`, `assets/proxies/`, `assets/sdf/`, `assets/grasps/` |
| Signed-distance grids of connector meshes | computed with [libigl](https://libigl.github.io/) (generalized winding number) | see upstream | offline tool `tools/build_sdf_grids.py` (outputs shipped in `assets/sdf/`) |
| Convex proxies of connectors | computed with [CoACD](https://github.com/SarahWeiii/CoACD) | see upstream | offline tool `tools/build_proxies.py` (outputs shipped in `assets/proxies/`) |
| USD parsing | [usd-core](https://pypi.org/project/usd-core/) | see upstream | offline tool `tools/extract_nist_meshes.py` (outputs shipped in `assets/meshes/`) |
| PPO with adaptive-KL learning rate | design follows [rsl_rl](https://github.com/leggedrobotics/rsl_rl) | see upstream | re-implemented in `assembly_policy/train.py` |
| Actor/critic architecture | SimBa (Lee et al., 2024), arranged as described in the AlphaNIST paper (Fig. 3) | - | `assembly_policy/rl/nets.py` |
| Reset families (loose, pickup, transport, partial, near-mating); Success Guided Sampling | OmniReset-style reset distributions and the SGS sampler as described in the AlphaNIST and SGS papers | - | `assembly_policy/sim/resets.py`, `assembly_policy/rl/sampler.py` |
| Joint impedance gains | [libfranka](https://github.com/frankaemika/libfranka) joint impedance example | see upstream | values in `assembly_policy/sim/scene.py` |

Everything else (thread and connector SDF plugin, scene builder, reset generator, batched simulator, remote
workers, training and evaluation code) was written for this repository.

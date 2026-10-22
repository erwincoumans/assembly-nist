<div align="center">

# AssembleNIST

![MuJoCo](https://img.shields.io/badge/MuJoCo-3.x-1f6feb) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c) ![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab)

[Tao Sun](https://taosun.io) · Stanford University

**A single reinforcement-learning policy for all 20 assemblies of the NIST Assembly Task Board,
learned from binary rewards with a Franka arm in MuJoCo.**

<img src="docs/teaser.gif" width="90%"/>

*The pretrained policy assembles all 20 parts one at a time using mean actions.
Assembly is shown at half speed; moves between parts at double speed.*

</div>

## Disclaimer

This repository is an **independent implementation** of the approach introduced by
[AlphaNIST](https://alpha-nist.github.io/) and [Success Guided Sampling (SGS)](https://sgs-rl.github.io/).
It is built in MuJoCo and makes its own design choices for contact modeling, the action space and the start-state
distributions. It is not affiliated with or endorsed by the AlphaNIST authors.

## Results

With 8,192 parallel environments, the policy reaches 98.4 % evaluation success after 10K PPO iterations
(98.7 % after 12.5K), at most half of the 25K iterations reported for AlphaNIST.

<p align="center"><img src="results/success_vs_iteration.png" width="75%"/></p>

| Setting | Overall | Loose | Pickup | Transport | Partial | Near-mating |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| AssembleNIST, 5K iter | 94.8 % | 91.6 | 93.9 | 95.1 | 97.8 | 95.7 |
| AssembleNIST, 10K iter (default) | **98.4 %** | 96.9 | 98.4 | 99.4 | 98.7 | 98.7 |
| AssembleNIST, 12.5K iter | **98.7 %** | 98.1 | 98.8 | 98.8 | 99.0 | 98.7 |
| *AlphaNIST paper (Isaac Lab), 25K iter* | *86.9 %* | – | – | – | – | – |

Evaluation follows the AlphaNIST protocol: a frozen start-state bank, 5 seeds × 32 trials per task,
and sampled policy actions. Success requires assembly within 15 seconds without a safety violation.
The AlphaNIST result uses a different simulator, asset set, and implementation, so it is not a direct comparison.
At 10K iterations every task reaches at least 95 %. The full learning curve is in
[`results/reference_e8192.csv`](results/reference_e8192.csv).

<table>
<tr>
<td align="center"><img src="docs/board_start.png" width="100%"/><br/><sub>20 parts at random poses</sub></td>
<td align="center"><img src="docs/board_done.png" width="100%"/><br/><sub>after sequential assembly</sub></td>
</tr>
</table>

## Quick start

Clone the repository, install the dependencies and build the contact plugin (requires `g++` on x86-64):

```bash
git clone https://github.com/suniique/assembly-nist.git && cd assembly-nist
pip install -e .
bash plugin/build.sh
```

Generate a demo with the pretrained policy:

```bash
MUJOCO_GL=egl python -m assembly_policy.viz.demo --ckpt checkpoints/e8192_it12500.pt --out demo.mp4
```

Use `MUJOCO_GL=osmesa` if EGL is unavailable. The demo generates settled start poses if reset banks are absent;
`--seed` changes the layout and assembly order, and `--pin 2:N16+3:connector` fixes positions in the order.
Pretrained policies after 2.5K, 5K, 7.5K, 10K and 12.5K iterations are in [`checkpoints/`](checkpoints).

## Train and evaluate

Build the reset banks before training or evaluation:

```bash
python scripts/build_banks.py --workers 16
```

Evaluate the pretrained policy (3,200 episodes; about one hour on 16 CPU cores):

```bash
python -m assembly_policy.evaluate --policy checkpoints/e8192_it12500.pt --out eval.json
```

For a small run on one machine:

```bash
python -m assembly_policy.train --run-dir runs/small --envs 512 --nthread 32
```

The default training configuration uses 8,192 environments and 10K PPO iterations.
Runs resume automatically from their latest checkpoint.

<details>
<summary>Distributed training with Slurm</summary>

The reference run used one GPU learner and 960 CPU worker processes across about 30 nodes,
reaching 10K iterations in roughly 26 hours. Adjust worker counts to your cluster.

```bash
sbatch scripts/slurm/build_banks.sbatch
sbatch --export=ALL,RUN_DIR=runs/e8192,NCONN=960 scripts/slurm/learner.sbatch
for i in $(seq 30); do
  sbatch --export=ALL,RUN_DIR=runs/e8192 scripts/slurm/workers.sbatch
done
```

The first command builds reset banks as a job array; skip it if the banks already exist.
The learner and workers must share the run directory. For clusters without Slurm, use
`assembly_policy.train --remote-workers` and `scripts/workers.sh`.

</details>

## Method

- **Policy:** a SimBa-style actor and critic trained with PPO, using part, fixture, and gripper point clouds
  plus proprioception history.
- **Control:** relative joint-position targets and a binary gripper at 25 Hz with libfranka impedance gains.
  Per-joint action scales account for the joints' different responses; an action-rate penalty encourages smooth motion.
- **Reward:** binary assembly success at 1 mm keypoint tolerance, with small control penalties.
  Episodes end on success, a joint-speed violation, or the 15-second limit.
- **Start states:** loose, pickup, transport, partial, and near-mating poses, sampled with SGS.
- **Contacts:** primitive geometry for rods and pegs, convex gear teeth, helical-thread SDFs for nuts and bolts,
  and mesh SDFs for connectors. Gears and nuts use 0.5 ms simulation steps; other parts use 2.5 ms.

## Acknowledgements

This project builds on *AlphaNIST: Emergent Assembly Skills from Binary Task Rewards* and
*A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control*
(Zhang et al., 2026). Thanks to Octi Zhang for discussions and help with training details.
The board assets come from [UW Lab](https://github.com/UW-Lab/UWLab) and the robot model from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).

See [`THIRD_PARTY.md`](THIRD_PARTY.md) for third-party code, assets, and licenses.
If you use AssembleNIST, please cite the works listed in [`CITATION.bib`](CITATION.bib).

## License

The code is released under the [MIT License](LICENSE). Third-party assets keep their own licenses; see
[`THIRD_PARTY.md`](THIRD_PARTY.md).

## Contact

For questions and feedback, please reach out to taosun@stanford.edu.

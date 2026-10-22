# AssembleNIST

![MuJoCo](https://img.shields.io/badge/MuJoCo-3.x-1f6feb) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c) ![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab)

**[Tao Sun](https://taosun.io) | Stanford University**

<!-- <img src="docs/teaser.gif" width="100%"/> -->
https://github.com/user-attachments/assets/81129b64-3911-485c-b9f9-cc1282c2d831


*A single reinforcement-learning policy for all 20 assemblies of the NIST Assembly Task Board,
learned from binary rewards with a Franka arm in MuJoCo.*

## Disclaimer

This repository is an **independent implementation** of the approach introduced by
[AlphaNIST](https://alpha-nist.github.io/) and [Success Guided Sampling (SGS)](https://sgs-rl.github.io/).
It is built in MuJoCo environment. 

Since the AlphaNIST hasn’t yet released its source code (as of Oct 21, 2026), I made own design choices for contact modeling, the action space and the start-state
distributions. I am not affiliated with or endorsed by the AlphaNIST authors.

## Evaluation results

With 8,192 parallel environments, the policy reaches **98.4%** evaluation success after 10K PPO iterations, outperforming the AlphaNIST’s performance with less than half of the 25K iterations reported.

**Success rate (%) by training iterations.**

<p align="center"><img src="results/success_vs_iteration.png" width="100%"/></p>

The full learning curve can be found in
[`results/reference_e8192.csv`](results/reference_e8192.csv).

<table>
<tr>
<td align="center"><img src="docs/board_start.png" width="100%"/><br/>20 parts at random initial poses</td>
<td align="center"><img src="docs/board_done.png" width="100%"/><br/>After assembly</td>
</tr>
</table>

**Success rate (%) by part type.**

| Part type | Ours<br/>5k iter | Ours<br/>10k iter | Ours<br/>12.5k iter | Original AlphaNIST<br/>25k iter |
| --- | :---: | :---: | :---: | :---: |
| Rods | 96.6 | 98.6 | 98.6 | – |
| Pegs | 96.6 | 98.1 | 98.6 | – |
| Gears | 98.1 | 98.3 | 99.0 | – |
| Nuts | 87.2 | 98.0 | 97.5 | – |
| Connectors | 96.1 | 99.0 | 99.6 | – |
| **Overall** | 94.8 | 98.4 | **98.7** | 86.9 |

**Success rate (%) by starting state.**

| Starting | Ours<br/>5k iter | Ours<br/>10k iter | Ours<br/>12.5k iter | Original AlphaNIST<br/>25k iter |
| --- | :---: | :---: | :---: | :---: |
| Loose | 91.6 | 96.9 | 98.1 | – |
| Pickup | 93.9 | 98.4 | 98.8 | – |
| Transport | 95.1 | 99.4 | 98.8 | – |
| Partial | 97.8 | 98.7 | 99.0 | – |
| Near-mating | 95.7 | 98.7 | 98.7 | – |
| **Overall** | 94.8 | 98.4 | **98.7** | 86.9 |


> [!NOTE]
> Evaluation follows the AlphaNIST protocol: 5 seeds × 32 trials per task, and sampled policy actions. Success requires assembly within 15 seconds without a safety violation.
> Please note that the AlphaNIST result uses a different simulator, asset set, and implementation, so it is not a direct comparison.


**More visuals.** MP4 files can be found in [docs/videos](docs/videos).

<table>
<tr>
<td align="center">https://github.com/user-attachments/assets/60241ac2-89e9-4702-907a-cbec783369f9</td>
<td align="center">https://github.com/user-attachments/assets/8eb57846-5abc-4145-8ebb-3c181249a0fe</td>
<td align="center">https://github.com/user-attachments/assets/d3960eca-87aa-4707-9ef0-f01f1a15aa56</td>
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

## Training and evaluation

First, we need to build the reset banks before training:

```bash
python scripts/build_banks.py --workers 16
```

Evaluate the pretrained policy. It runs 3200 episodes for about one hour on 16 CPU cores:

```bash
python -m assembly_policy.evaluate --policy checkpoints/e8192_it12500.pt --out eval.json
```

For training the policy, you can run with one machine on a smaller environment size (e.g., 512):

```bash
python -m assembly_policy.train --run-dir runs/small --envs 512 --nthread 32
```

To achieve best performance, the default training configuration uses 8,192 environments and 10K PPO iterations. Below is a script to train it on a Slurm cluster. 

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

## What failed first and turned out to matter most?

My first long runs plateaued below 50 % (46 % after 23K iterations). Getting the policy to learn efficiently
took several non-obvious changes; these are the lessons that mattered most. The final run changed several of
them at once, so their individual contributions were not isolated.

1. **Keep exploring.** With entropy 0.001, an initial std of 0.5 and a std floor of 0.1, the arm's std sat at the
   floor and the adaptive-KL learning rate was pinned at its minimum, so success crept up about 1 point per
   1,000 iterations. A learned std starting at 1.0 with no floor and entropy 0.005 reached 49 % after 2K iterations
   and 98 % after 5.5K.
2. **Make the joints fast enough to finish in 15 s.** A relative target `q + s·a` never catches up under the
   libfranka gains, so a joint moves at roughly `(kp/kd)·s·a`. With a uniform 0.02 rad scale the wrist needed
   15–27 s for a 90° turn: the policy lifted a lying peg in 98 % of pickup starts but never turned it upright, and
   transport success fell from 88 % (part held within 30° of upright) to 0 % (beyond 60°). Per-joint scales that give
   every joint about 0.5 rad/s per unit action made reorientation possible.
3. **Spend samples on the frontier.** Success Guided Sampling with ε = 0.05 still put roughly a fifth of the draws on
   starts that were always or never solved; ε = 1e-4 over a smaller 64K-entry bank keeps nearly all draws on starts
   the policy sometimes solves. With ε = 1e-3, a 4K-environment run learned to end hopeless-looking episodes with a
   joint-speed violation to avoid the per-step control cost (67 % safety stops in evaluation).
4. **Fix the physics before tuning RL.** Nut success stayed at 0 % under every training change until the thread
   geometry was fixed: sharp V-shaped thread tips wedged in MuJoCo's SDF contact search (a free M8 nut turned less
   than 80° in 3 s under 0.02 N·m). Flattening the tips and a 0.2 mm radial clearance let every nut spin freely
   without skipping threads. Gear teeth and threads also needed 0.5 ms steps with stiff contacts, and 5 ms steps
   (10 ms contact time constant) added up to 3× kinetic energy on drop impacts, so the default step is 2.5 ms.
5. **Give the sampler intermediate states.** Transport starts hold rods, pegs and connectors at any tilt, and
   near-mating and partial starts include side grasps near the top of the part: the states between "picked up lying
   down" and "inserted from the side". Partial starts are drawn by assembly progress with a small random pose.
6. **Make the gripper robust.** The 1-D gripper command latches with a ±0.5 deadband (a plain sign threshold
   released held parts after any noisy sample), and the closing target saturates the finger force at 35 N for every
   part (a zero-width target scaled the force with part width, so thin parts slipped).
7. **Watch for silent bugs.** Clamping the log-std only in the forward pass froze dimensions at their bounds (zero
   gradient), so the std is projected in place. Point clouds are sampled per training-task index, so evaluating a
   subset of tasks with re-sampled points dropped success from 99 % to 16 %. A safety violation in the same step
   must veto success, and transitions lost with a disconnected worker must be masked out of GAE.
8. **Smooth the motion late.** Sampled actions look jittery. A small action-rate penalty (1e-4 · Σ(aₜ − aₜ₋₁)²)
   added after 5K iterations made the arm about 30 % smoother with no loss in success; videos use mean actions.

**Final design**

- **Policy:** a SimBa-style actor and critic trained with PPO, using part, fixture, and gripper point clouds
  plus proprioception history.
- **Control:** relative joint-position targets and a binary gripper at 25 Hz with libfranka impedance gains,
  per-joint action scales, and an action-rate penalty.
- **Reward:** binary assembly success at 1 mm keypoint tolerance, with small control penalties.
  Episodes end on success, a joint-speed violation, or the 15-second limit.
- **Start states:** loose, pickup, transport, partial, and near-mating poses, sampled with SGS.
- **Contacts:** primitive geometry for rods and pegs, convex gear teeth, helical-thread SDFs for nuts and bolts,
  and mesh SDFs for connectors. Gears and nuts use 0.5 ms simulation steps and other parts use 2.5 ms.

## Acknowledgements

This project builds on 

- *AlphaNIST: Emergent Assembly Skills from Binary Task Rewards*.
- *A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control*
(Zhang et al., 2026).

Thanks to [Octi Zhang](https://zoctipus.github.io) for helpful guidance with training details and [Jing Zhang](https://jingz6676.github.io/) for helpful discussions!
The NIST board assets come from [UW Lab](https://github.com/UW-Lab/UWLab) and the robot model from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).

## Citation

If you use AssembleNIST, please cite this repository together with AlphaNIST and SGS
(all entries are also in [`CITATION.bib`](CITATION.bib)):

```bibtex
@misc{sun2026assemblenist,
  title        = {AssembleNIST: One Policy for All NIST Board Assemblies in MuJoCo},
  author       = {Sun, Tao},
  year         = {2026},
  howpublished = {\url{https://github.com/suniique/assembly-nist}}
}

@article{zhang2026alphanist,
  title   = {AlphaNIST: Emergent Assembly Skills from Binary Task Rewards},
  author  = {Zhang, Octi and Akinola, Iretiayo and Scalise, Rosario and Guaman Castro, Mateo and Yin, Patrick and
             Widmer, Tobias and Macklin, Miles and Amevor, Chris and Turpin, Dylan and Van Gelder, Dirk and
             Carius, Jan and Handa, Ankur and State, Gavriel and Guo, Yunrong and Narang, Yashraj},
  journal = {arXiv preprint arXiv:2610.13354},
  year    = {2026}
}

@article{zhang2026sgs,
  title   = {A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control},
  author  = {Zhang, Octi and Guaman Castro, Mateo and Yin, Patrick and Dagnino, Ignacio and Gupta, Abhishek and
             Scalise, Rosario and Boots, Byron},
  journal = {arXiv preprint arXiv:2610.12465},
  year    = {2026}
}
```

## License

The code is released under the [MIT License](LICENSE). See
[`THIRD_PARTY.md`](THIRD_PARTY.md) for third-party assets, codes, and their licenses. 

## Contact

For questions and feedback, please reach out to taosun@stanford.edu.

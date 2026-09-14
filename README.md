# Walker / Epiplexity

**Latest experiment:** [tuned SB3 Zoo PPO, standard versus standard + epiplexity](ZOO.md), with matched seeds and videos.

A video-first reproduction of the **epiplexity-only Walker2d** experiment in Yanbo Zhang and Michael Levin, [Intelligence from Learnable Novelty](https://arxiv.org/abs/2607.18433), Table 1 and Appendix H.

[Watch the original recordings](https://lcrh.github.io/walker-epiplexity/original.html).

The training reward is only the increment in trajectory epiplexity. No environment task reward, survival bonus, forward-velocity term, imitation data, or pretrained policy enters training. The environment still terminates on falls, as in the original implementation. The policy observes the standard 17-dimensional state, not pixels.

## Reproduce

```sh
uv venv --python 3.13
uv pip install --python .venv/bin/python -r requirements-lock.txt
.venv/bin/python check.py
.venv/bin/python train.py --seed 0
.venv/bin/python render.py --seed 0 --all-stages
# Additional final-policy checks:
.venv/bin/python train.py --seed 1
.venv/bin/python train.py --seed 2
.venv/bin/python render.py --seed 1
.venv/bin/python render.py --seed 2
.venv/bin/python publish_results.py
```

Training uses one CPU thread per process. On Linux, rendering may require an appropriate MuJoCo GL backend (`MUJOCO_GL=egl` with a compatible GPU, or an available display). The recorded runs use macOS arm64. Serve `docs/` for the static viewer; no build step is required.

## Fidelity

`vendor/src/` is an **unmodified** snapshot of the [authors’ code](https://github.com/Zhangyanbo/learnable-novelty) at `22541fa076c8dc11e182c5d611a1beec11662ad1`. `train.py` calls their `train_run` and substitutes a PPO subclass whose only addition is a checkpoint callback. Checkpoint saving does not alter the training random-number state. Intermediate checkpoints are taken after completed PPO updates; no intermediate evaluation feeds back into learning.

- Walker2d-v5, Gymnasium 1.2.3, MuJoCo 3.9.0, SB3 2.8.0, Torch 2.10.0.
- PPO default MLP, 8 environments, 1,024-step rollouts, minibatches of 256, learning rate 0.0003, gamma 0.999, GAE lambda 0.98.
- Frozen 32-feature reservoir, horizon 16, ridge 0.3, fixed normalization from random-policy trajectories. Pure reward mode `epiplexity`, `beta=None`.
- 600,000 requested steps; SB3 completes the final rollout at **606,208 actual steps**.
- Training seeds 0, 1, 2. Seed 0 chosen in advance for the training-stage video. No seed selection based on performance.
- Final policies evaluated deterministically on 100 episodes, reset seeds 7000–7099, matching the authors. Videos contain the first three of those episodes, including falls and resets. No slow motion; rendered at 50 fps from a 125 Hz control simulation.
- The paper reports **327 ± 45 task return over ten training seeds**. This smaller check does not reproduce its ten-seed statistical claim. A similar task return does not by itself establish sustained walking.

Raw evaluation episodes, provenance, stages, and training curves are published alongside the videos. The released policy ZIP files include SB3 model/optimizer state; intermediate ZIPs are playback checkpoints, not bit-exact training-resumption snapshots of MuJoCo and online reward estimator states.

## Checks

`check.py` tests task-reward isolation by changing the environment reward by a million while holding observations/actions fixed, verifies that intrinsic rewards telescope to final epiplexity, checks finite rewards, and verifies checkpoint saves preserve Torch/NumPy RNG state.

## Attribution and license

Reference implementation copyright 2026 Yanbo Zhang. See [vendor/LICENSE](vendor/LICENSE): Apache 2.0 **with the Tufts Academic Use Only rider**, restricting the reference software to academic, non-commercial research. This repository retains that license and attribution; it is a research reproduction. Video capture, evaluation reporting, and the static viewer are additions to the upstream experiment, not original-author results.

## Recorded result

Three-seed mean task return: **278.79 ± 2.84** (population standard deviation across seed means).

| Training seed | Task return, 100 evaluation episodes | Mean episode length | Mean displacement |
|---|---:|---:|---:|
| 0 | 278.04 | 154.4 steps | 1.00 m |
| 1 | 282.58 | 155.2 steps | 1.03 m |
| 2 | 275.75 | 148.1 steps | 1.03 m |

These runs fall below the reported paper mean. They show brief forward motion followed by falling, rather than sustained walking. The cause of the numerical gap is not established; three seeds and a macOS arm64 run are not a ten-seed platform-matched replication.

### Diagnostic baseline

One task-reward-only run (seed 0, otherwise the same reference setup) scored **301.51**, compared with the paper’s **296 ± 45**. This suggests the whole PPO/environment pipeline is not uniformly underperforming. It does not isolate the epiplexity gap: seed sampling, platform-sensitive reward numerics, or differences between the released code and the original experimental runs remain possible explanations. See `measurements/baseline-seed-0/`.

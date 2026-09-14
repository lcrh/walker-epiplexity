# Tuned PPO: standard versus standard + epiplexity

Uses the SB3 Zoo Walker2d PPO recipe at [630883a18f9b77e15dfaefb8a891633d766ef99c](https://github.com/DLR-RM/rl-baselines3-zoo/blob/630883a18f9b77e15dfaefb8a891633d766ef99c/hyperparams/ppo.yml). The configuration is named `Walker2d-v4` upstream; we apply it to **Walker2d-v5** in both conditions. This is a Zoo-recipe comparison on v5, not a claim to reproduce its old v3 benchmark exactly.

[Watch the comparison](https://lcrh.github.io/walker-epiplexity/) · [Original epiplexity-only experiment](https://lcrh.github.io/walker-epiplexity/original.html)

## Configuration

One environment, 512-step rollouts, minibatch 32, 20 epochs, learning rate 0.0000505041, gamma 0.99, GAE lambda 0.95, clip 0.1, entropy coefficient 0.000585045, value coefficient 0.871923, gradient norm limit 1. Default SB3 MLP policy. See `configs/zoo-walker.json` and the retained upstream YAML.

As in Zoo, `VecNormalize` normalizes observations and rewards and uses PPO's gamma. Each checkpoint includes its own normalization statistics. Evaluation freezes those statistics and reports **raw task reward**, never normalized reward. This differs materially from the earlier paper recipe, which did not normalize the policy inputs or PPO rewards (its separate reservoir input normalization is a different operation).

Both conditions start from scratch with matched policy initialization, seeds 0–2, and a one-million-step budget (actual 1,000,448, completing the last rollout). All seeds are reported. No early stopping or best-checkpoint selection.

The mixed condition adds the authors' unchanged epiplexity increment to raw task reward **before** VecNormalize. Its fixed coefficient is calibrated on random trajectories using the paper's 0.1 episode-scale target: `beta = 0.0184949293`. No coefficient search uses the evaluation results. Reward-estimator parameters and random-policy normalization are shared across seeds and stored in `measurements/zoo/calibration.pt`.

The viewer opens seed 1 as a walking example, selected after the 500k validation check. All three seeds enter the aggregate results and remain selectable. This is illustrative video selection, not model selection for the reported metrics. Training-stage videos are available for seeds 0 and 1.

Final evaluation uses 100 deterministic episodes per policy, reset seeds 7000–7099. Intermediate validation uses 10 episodes, seeds 9000–9009, in a separate process and never feeds back into training. Videos show evaluation seeds 7000–7002 for final policies, and seed 7000 for intermediate policies. Each trial occupies eight seconds; if it terminates early, its last frame is held and clearly marked until the next reset. Physics plays at real speed, with no extra actions after a fall.

## Run

Install the existing pinned environment as described in `README.md`, then:

```sh
.venv/bin/python zoo_train.py --calibrate
.venv/bin/python zoo_check.py
for seed in 0 1 2; do
  .venv/bin/python zoo_train.py --mode standard --seed "$seed"
  .venv/bin/python zoo_train.py --mode mixed --seed "$seed"
done
.venv/bin/python zoo_render.py --mode standard --seed 0 --all-stages
.venv/bin/python zoo_render.py --mode mixed --seed 0 --all-stages
for seed in 1 2; do
  .venv/bin/python zoo_render.py --mode standard --seed "$seed" --all-stages
  .venv/bin/python zoo_render.py --mode mixed --seed "$seed" --all-stages
done
.venv/bin/python zoo_publish.py
```

Run each training process with one CPU thread (the script enforces this). Independent seed/condition processes may run concurrently. Existing runs are protected against accidental overwrite. Policy ZIP and VecNormalize PKL files must remain paired for playback. These are evaluation checkpoints, not exact simulator/reward-estimator resume snapshots.

## Checks

`zoo_check.py` verifies matched observation normalization on identical trajectories, correct bonus addition in raw reward units, normalization save/load equivalence, and frozen evaluation statistics. The three paired initial policy hashes are identical; provenance is retained for each run. Reference reward code remains unmodified.

## Attribution

Zoo settings: DLR-RM / RL Baselines3 Zoo; see `configs/ZOO-LICENSE`. Epiplexity reference: Yanbo Zhang and Michael Levin; see `vendor/LICENSE` for its Apache license plus Tufts academic/non-commercial research rider. The comparison and recordings here are new runs, not the authors' recorded results.

## Additional epiplexity-only run

The same Zoo settings also support `--mode epiplexity`: the policy receives only the intrinsic increment, with task reward dropped before reward normalization. A seed-2 run has been launched at the same one-million-step budget. Its results are separate from the standard-versus-mixed comparison.

```sh
.venv/bin/python zoo_train.py --mode epiplexity --seed 2
.venv/bin/python zoo_render.py --mode epiplexity --seed 2 --all-stages
```

## Zoo comparison results

| Condition | Task return (mean ± SD across three seed means) | Mean episode steps | Full-episode rate |
|---|---:|---:|---:|
| Standard | 2824.9 ± 990.1 | 647.1 | 30.3% |
| Standard + epiplexity | 3254.0 ± 1784.2 | 741.3 | 63.7% |

Each final policy is evaluated on the same 100 episodes, seeds 7000–7099. Error bars are across training seeds, not evaluation episodes. Three seeds are a small comparison, not a significance claim. The standard and mixed policies both start from scratch, with matched initial policy hashes.

- Seed 0: 1437.6 standard → 734.0 mixed (-703.6).
- Seed 1: 3353.7 standard → 4403.5 mixed (+1049.8).
- Seed 2: 3683.4 standard → 4624.5 mixed (+941.1).

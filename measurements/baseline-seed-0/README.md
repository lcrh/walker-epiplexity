Diagnostic task-reward-only run, seed 0. Uses the same environment, dependencies, 600,000-step budget, and authors unmodified train_run; mode standard. No checkpoint callback. Evaluation: 100 deterministic episodes, seeds 7000–7099.

Run with PYTHONPATH=vendor/src and one Torch/BLAS thread:

```python
import torch
torch.set_num_threads(1)
from rl.training import train_run
from rl.envs import ENV_BY_NAME
train_run(ENV_BY_NAME["walker2d"], "standard", 0, output_dir="results/baseline")
```

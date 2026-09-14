"""Run the authors' unchanged training function, adding only checkpoint recording."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor' / 'src'))
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from rl import training
from rl.envs import ENV_BY_NAME

torch.set_num_threads(1)

class SaveStages(BaseCallback):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.started = time.monotonic()
        self.thresholds = iter([100_000, 200_000, 400_000])
        self.next_stage = next(self.thresholds, None)
        self.stages = []
    def save(self):
        path = self.directory / f'policy_{self.num_timesteps:07d}.zip'
        self.model.save(path)
        self.stages.append({'steps':self.num_timesteps, 'file':path.name})
        (self.directory/'stages.json').write_text(json.dumps(self.stages, indent=2))
    def _on_training_start(self):
        self.save()
    def _on_rollout_start(self):
        # Here the preceding PPO update is complete (rollout-end runs before it).
        if self.next_stage is not None and self.num_timesteps >= self.next_stage:
            self.save()
            self.next_stage = next(self.thresholds, None)
        if self.num_timesteps:
            elapsed=time.monotonic()-self.started
            print(json.dumps({'steps':self.num_timesteps,'seconds':round(elapsed,1),'fps':round(self.num_timesteps/elapsed)}),flush=True)
    def _on_step(self):
        return True
    def _on_training_end(self):
        self.save()

class RecordingPPO(PPO):
    def learn(self, *args, **kwargs):
        kwargs['callback'] = SaveStages(checkpoint_dir)
        result = super().learn(*args, **kwargs)
        return result

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--steps',type=int,default=600_000)
    args=parser.parse_args()
    checkpoint_dir=ROOT/'results'/f'seed-{args.seed}'
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    import importlib.metadata, platform
    (checkpoint_dir/'provenance.json').write_text(json.dumps({'seed':args.seed,'requested_steps':args.steps,'platform':platform.platform(),'python':sys.version,'packages':{n:importlib.metadata.version(n) for n in ['torch','mujoco','numpy','gymnasium','stable-baselines3']},'upstream_commit':'22541fa076c8dc11e182c5d611a1beec11662ad1','reward':'epiplexity only; beta=None; no task reward','threads':1},indent=2))
    training.PPO=RecordingPPO
    training.train_run(ENV_BY_NAME['walker2d'],'epiplexity',args.seed,total_timesteps=args.steps,output_dir=str(ROOT/'results'/'authors-metrics'))

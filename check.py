"""Focused scientific checks: reward isolation, telescoping, and recorder neutrality."""
import os
os.environ['OMP_NUM_THREADS']='1'
import sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'/'src'))
import numpy as np
import torch
import gymnasium as gym
from rl.calibrate import build_reservoir
from rl.envs import ENV_BY_NAME
from rl.reward import EpiplexityRewardWrapper
from stable_baselines3 import PPO
torch.set_num_threads(1)
spec=ENV_BY_NAME['walker2d']
res=build_reservoir(spec,32,.3,1e-8,'cpu',0)
norm={'x_mu':torch.zeros(17),'x_inv':torch.ones(17),'f_mu':torch.zeros(32),'f_inv':torch.ones(32),'y_mu':torch.zeros(272),'y_inv':torch.ones(272)}
class AlterTaskReward(gym.RewardWrapper):
    def reward(self,reward):return reward+1_000_000
left=EpiplexityRewardWrapper(gym.make(spec.env_id),res,spec,norm,beta=None)
right=EpiplexityRewardWrapper(AlterTaskReward(gym.make(spec.env_id)),res,spec,norm,beta=None)
left.reset(seed=7000);right.reset(seed=7000)
summed=0.
for step in range(60):
    action=np.zeros(6,dtype=np.float32)
    a=left.step(action);b=right.step(action)
    assert a[1]==b[1], 'Task reward leaked into intrinsic reward'
    assert np.isfinite(a[1])
    summed+=a[1]
    if a[2] or a[3]:break
assert abs(summed-left._prev)<1e-6
assert step>=16 and left._prev>0
left.close();right.close()
env=gym.make(spec.env_id);model=PPO('MlpPolicy',env,seed=0,n_steps=8,batch_size=8,device='cpu')
state=torch.random.get_rng_state().clone();npstate=np.random.get_state()
with tempfile.TemporaryDirectory() as tmp:model.save(Path(tmp)/'policy')
assert torch.equal(state,torch.random.get_rng_state())
after=np.random.get_state();assert np.array_equal(npstate[1],after[1]) and npstate[2:]==after[2:]
env.close()
print('PASS: task reward isolation; intrinsic increments telescope; finite rewards; checkpoint saves preserve training RNG.')

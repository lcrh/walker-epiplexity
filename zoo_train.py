"""Pinned SB3 Zoo Walker2d recipe, task-only versus task + calibrated epiplexity."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import argparse,hashlib,json,sys,time,platform
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'/'src'))
from rl.calibrate import build_reservoir,estimate_normalization,calibrate_beta
from rl.envs import ENV_BY_NAME
from rl.reward import EpiplexityRewardWrapper
CONFIG=json.loads((ROOT/'configs'/'zoo-walker.json').read_text())
torch.set_num_threads(1)

class EpisodeMeasures(gym.Wrapper):
    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs)
        self.total=0.;self.x0=float(self.unwrapped.data.qpos[0]);self.steps=0
        return obs,info
    def step(self,action):
        obs,reward,term,trunc,info=self.env.step(action)
        self.total+=float(reward);self.steps+=1
        info=dict(info)
        if term or trunc:info.update(task_return=self.total,displacement=float(self.unwrapped.data.qpos[0])-self.x0,episode_steps=self.steps,full_episode=float(trunc and not term))
        return obs,reward,term,trunc,info

def calibration():
    path=ROOT/'results'/'zoo'/'calibration.pt'
    if path.exists():return path
    path.parent.mkdir(parents=True,exist_ok=True)
    spec=ENV_BY_NAME['walker2d'];res=build_reservoir(spec,32,.3,1e-8,'cpu',0)
    norm=estimate_normalization(res,spec,0,1e-8)
    beta=calibrate_beta(res,spec,norm,0,.1,1e-8)
    torch.save({'reservoir':res.state_dict(),'norm':norm,'beta':beta},path)
    path.with_suffix('.json').write_text(json.dumps({'beta':beta,'target':.1,'random_probe_episodes':64,'reservoir_seed':1,'normalization_base_seed':0},indent=2))
    print(f'Calibrated beta={beta:.9g}',flush=True)
    return path

def make_training_env(mode,seed,directory):
    artifact=None
    if mode!='standard':artifact=torch.load(calibration(),weights_only=True)
    def make():
        env=EpisodeMeasures(gym.make(CONFIG['env']))
        if mode!='standard':
            res=build_reservoir(ENV_BY_NAME['walker2d'],32,.3,1e-8,'cpu',0)
            res.load_state_dict(artifact['reservoir'])
            env=EpiplexityRewardWrapper(env,res,ENV_BY_NAME['walker2d'],artifact['norm'],beta=None if mode=='epiplexity' else artifact['beta'])
        return Monitor(env,str(directory/'train'),info_keywords=('task_return','displacement','episode_steps','full_episode'))
    env=VecNormalize(DummyVecEnv([make]),gamma=CONFIG['ppo']['gamma'])
    env.seed(seed)
    return env

class Checkpoints(BaseCallback):
    def __init__(self,directory):
        super().__init__();self.directory=directory;self.stages=[];self.started=time.monotonic();self.last_report=0;self.next_stage=iter([100000,250000,500000,750000]);self.threshold=next(self.next_stage,None)
    def save(self):
        stem=f'policy_{self.num_timesteps:07d}'
        self.model.save(self.directory/f'{stem}.zip')
        self.model.get_vec_normalize_env().save(self.directory/f'{stem}.pkl')
        self.stages.append({'steps':self.num_timesteps,'model':f'{stem}.zip','normalization':f'{stem}.pkl'})
        (self.directory/'stages.json').write_text(json.dumps(self.stages,indent=2))
    def _on_training_start(self):self.save()
    def _on_rollout_start(self):
        if self.threshold is not None and self.num_timesteps>=self.threshold:self.save();self.threshold=next(self.next_stage,None)
        if self.num_timesteps-self.last_report>=25000:
            elapsed=time.monotonic()-self.started
            print(json.dumps({'steps':self.num_timesteps,'seconds':round(elapsed,1),'fps':round(self.num_timesteps/elapsed)}),flush=True);self.last_report=self.num_timesteps
    def _on_step(self):return True
    def _on_training_end(self):self.save()

def normalized_policy(directory,stage):
    vec=VecNormalize.load(directory/stage['normalization'],DummyVecEnv([lambda:gym.make(CONFIG['env'])]))
    vec.training=False;vec.norm_reward=False
    model=PPO.load(directory/stage['model'],device='cpu')
    return model,vec

def evaluate(directory,stage,episodes=100,start_seed=7000):
    model,vec=normalized_policy(directory,stage)
    env=gym.make(CONFIG['env']);rows=[]
    for ep in range(episodes):
        obs,_=env.reset(seed=start_seed+ep);total=0.;steps=0;x0=float(env.unwrapped.data.qpos[0])
        while True:
            action,_=model.predict(vec.normalize_obs(obs),deterministic=True)
            obs,reward,term,trunc,_=env.step(action);total+=float(reward);steps+=1
            if term or trunc:break
        rows.append({'eval_seed':start_seed+ep,'task_return':total,'steps':steps,'seconds':steps*env.unwrapped.dt,'displacement':float(env.unwrapped.data.qpos[0])-x0,'full_episode':bool(trunc and not term)})
    env.close();vec.close()
    return {**stage,'mean_task_return':float(np.mean([r['task_return'] for r in rows])),'std_task_return':float(np.std([r['task_return'] for r in rows])),'mean_episode_steps':float(np.mean([r['steps'] for r in rows])),'mean_displacement':float(np.mean([r['displacement'] for r in rows])),'full_episode_fraction':float(np.mean([r['full_episode'] for r in rows])),'episodes':rows}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['standard','mixed','epiplexity'],default='standard');p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=CONFIG['steps']);p.add_argument('--calibrate',action='store_true');args=p.parse_args()
    if args.calibrate:calibration();sys.exit()
    directory=ROOT/'results'/'zoo'/f'{args.mode}-{args.seed}'
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'stages.json').exists():raise RuntimeError('Run already exists; preserve it and choose another output location.')
    env=make_training_env(args.mode,args.seed,directory)
    model=PPO(CONFIG['policy'],env,seed=args.seed,device='cpu',**CONFIG['ppo'])
    initial_hash=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.policy.state_dict().values())).hexdigest()
    import importlib.metadata
    provenance={**CONFIG,'mode':args.mode,'seed':args.seed,'requested_steps':args.steps,'initial_policy_sha256':initial_hash,'beta':json.loads(calibration().with_suffix('.json').read_text())['beta'] if args.mode=='mixed' else None if args.mode=='epiplexity' else 0,'platform':platform.platform(),'packages':{k:importlib.metadata.version(k) for k in ['torch','numpy','mujoco','gymnasium','stable-baselines3']}}
    (directory/'provenance.json').write_text(json.dumps(provenance,indent=2))
    callback=Checkpoints(directory)
    model.learn(total_timesteps=args.steps,callback=callback)
    env.close()
    result=evaluate(directory,callback.stages[-1]);(directory/'evaluation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'}),flush=True)

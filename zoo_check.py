"""Checks for the comparison-specific reward and normalization wiring."""
import tempfile
from pathlib import Path
import numpy as np
import torch
import zoo_train as z
from stable_baselines3.common.vec_env import VecNormalize,DummyVecEnv
import gymnasium as gym
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)
    standard=z.make_training_env('standard',123,root/'standard')
    mixed=z.make_training_env('mixed',123,root/'mixed')
    a=standard.reset();b=mixed.reset();assert np.array_equal(a,b)
    beta=torch.load(z.calibration(),weights_only=True)['beta'];previous=0.;seen=0
    for step in range(60):
        action=np.zeros((1,6),dtype=np.float32)
        a,ra,da,_=standard.step(action);b,rb,db,_=mixed.step(action)
        assert np.array_equal(a,b) and np.array_equal(da,db)
        if da[0]:break
        current=mixed.venv.envs[0].env._prev
        expected=float(standard.get_original_reward()[0])+beta*(current-previous)
        assert np.isclose(float(mixed.get_original_reward()[0]),expected,rtol=2e-5,atol=2e-5)
        seen+=current!=previous;previous=current
    assert seen>0
    stats=root/'normalize.pkl';mixed.save(stats)
    restored=VecNormalize.load(stats,DummyVecEnv([lambda:gym.make(z.CONFIG['env'])]))
    restored.training=False;restored.norm_reward=False
    raw=np.ones(17,dtype=np.float64)
    assert np.array_equal(restored.normalize_obs(raw),mixed.normalize_obs(raw))
    count=restored.obs_rms.count
    restored.reset();restored.step(np.zeros((1,6),dtype=np.float32))
    assert restored.obs_rms.count==count
    standard.close();mixed.close();restored.close()
print('PASS: matched observation normalization, bonus added in raw reward units, saved normalization round-trip, frozen evaluation statistics.')

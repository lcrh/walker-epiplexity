"""Evaluate saved policies and render real-time MuJoCo videos, without training."""
import os
os.environ['OMP_NUM_THREADS']='1'
import argparse, json
from pathlib import Path
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parent
FPS=50

def font(size):
    for path in ['/System/Library/Fonts/Helvetica.ttc','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(path).exists():return ImageFont.truetype(path,size)
    return ImageFont.load_default(size=size)

def frame(rgb,steps,seed,episode,sim_time):
    canvas=Image.new('RGB',(1280,720),'#101719')
    canvas.paste(Image.fromarray(rgb),(0,80))
    d=ImageDraw.Draw(canvas)
    d.text((30,18),'WALKER / EPIPLEXITY ONLY',font=font(25),fill='#e8eeeb')
    label='Untrained' if steps==0 else f'{steps:,} training steps'
    d.text((1250,21),label,font=font(23),fill='#ade1c0',anchor='ra')
    d.text((30,688),f'Seed {seed}  ·  Episode {episode+1}  ·  {sim_time:.2f}s',font=font(17),fill='#bac9c5')
    d.text((1250,688),'Real-time physics · No task reward in training',font=font(17),fill='#bac9c5',anchor='ra')
    return np.asarray(canvas)

def evaluate(model, episodes):
    env=gym.make('Walker2d-v5')
    rows=[]
    for episode in range(episodes):
        obs,_=env.reset(seed=7000+episode)
        x0=float(env.unwrapped.data.qpos[0]);total=0.;length=0
        while True:
            action,_=model.predict(obs,deterministic=True)
            obs,reward,term,trunc,_=env.step(action)
            total+=float(reward);length+=1
            if term or trunc:break
        rows.append({'eval_seed':7000+episode,'task_return':total,'length':length,'seconds':length*env.unwrapped.dt,'displacement':float(env.unwrapped.data.qpos[0])-x0})
    env.close()
    return {'mean_task_return':float(np.mean([r['task_return'] for r in rows])),'std_task_return':float(np.std([r['task_return'] for r in rows])),'mean_episode_steps':float(np.mean([r['length'] for r in rows])),'mean_displacement':float(np.mean([r['displacement'] for r in rows])),'episodes':rows}

def record(model,steps,seed,path,episodes=3):
    env=gym.make('Walker2d-v5',render_mode='rgb_array',width=1280,height=600)
    writer=imageio.get_writer(path,fps=FPS,codec='libx264',quality=8,macro_block_size=1,ffmpeg_params=['-movflags','+faststart'])
    frames=0
    for episode in range(episodes):
        obs,_=env.reset(seed=7000+episode)
        t=0.;next_frame=0.
        while True:
            if t+1e-9>=next_frame:
                rendered=frame(env.render(),steps,seed,episode,t)
                writer.append_data(rendered);frames+=1;next_frame+=1/FPS
                if frames==1:imageio.imwrite(path.with_suffix('.jpg'),rendered)
            action,_=model.predict(obs,deterministic=True)
            obs,_,term,trunc,_=env.step(action);t+=env.unwrapped.dt
            if term or trunc:break
    writer.close();env.close()
    return frames/FPS

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--all-stages',action='store_true');args=p.parse_args()
    directory=ROOT/'results'/f'seed-{args.seed}'
    stages=json.loads((directory/'stages.json').read_text())
    if not args.all_stages:stages=stages[-1:]
    out=ROOT/'docs'/'media';out.mkdir(parents=True,exist_ok=True)
    data=[]
    for stage in stages:
        print(f'Rendering seed {args.seed}, step {stage["steps"]}',flush=True)
        model=PPO.load(directory/stage['file'],device='cpu')
        result=evaluate(model,100 if stage==stages[-1] else 10)
        filename=f'seed-{args.seed}-{stage["steps"]:07d}.mp4'
        seconds=record(model,stage['steps'],args.seed,out/filename)
        row={**stage,'seed':args.seed,'video':'media/'+filename,'poster':'media/'+filename.replace('.mp4','.jpg'),'duration':seconds,**result}
        data.append(row)
        (directory/'evaluation.json').write_text(json.dumps(data,indent=2))
        print(json.dumps({k:v for k,v in row.items() if k!='episodes'}),flush=True)

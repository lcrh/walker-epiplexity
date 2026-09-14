"""Matched eight-second evaluation clips. Freeze and label falls until the next reset."""
import argparse,json
from pathlib import Path
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw
import zoo_train as z
from render import font
ROOT=z.ROOT
FPS=50
LABELS={'standard':'Standard','mixed':'Standard + epiplexity','epiplexity':'Epiplexity only'}

def overlay(rgb,mode,seed,steps,episode,time,ended):
    img=Image.new('RGB',(960,640),'#101719');img.paste(Image.fromarray(rgb),(0,60))
    d=ImageDraw.Draw(img)
    d.text((22,17),LABELS[mode],font=font(23),fill='#e3eee7')
    d.text((938,20),f'{steps:,} steps',font=font(18),fill='#b2d8c4',anchor='ra')
    d.text((22,615),f'Seed {seed} · Trial {episode+1} · {time:.2f}s',font=font(16),fill='#b9ccc2')
    d.text((938,615),'Fell · frame held until next trial' if ended else 'Real-time · deterministic policy',font=font(16),fill='#edb58e' if ended else '#b9ccc2',anchor='ra')
    return np.asarray(img)

def record(directory,stage,mode,seed,path,episodes):
    model,vec=z.normalized_policy(directory,stage)
    env=gym.make(z.CONFIG['env'],render_mode='rgb_array',width=960,height=540)
    writer=imageio.get_writer(path,fps=FPS,codec='libx264',quality=7,macro_block_size=1,ffmpeg_params=['-movflags','+faststart'])
    for episode in range(episodes):
        obs,_=env.reset(seed=7000+episode);steps=0;ended=False;endtime=None;rgb=None
        for frame_idx in range(8*FPS):
            desired_time=frame_idx/FPS
            while steps*env.unwrapped.dt+1e-9<desired_time and not ended:
                action,_=model.predict(vec.normalize_obs(obs),deterministic=True)
                obs,_,term,trunc,_=env.step(action);steps+=1
                if term or trunc:ended=True;endtime=steps*env.unwrapped.dt;rgb=env.render()
            if not ended:rgb=env.render()
            output=overlay(rgb,mode,seed,stage['steps'],episode,endtime if ended else steps*env.unwrapped.dt,ended)
            writer.append_data(output)
            if episode==0 and frame_idx==0:imageio.imwrite(path.with_suffix('.jpg'),output)
    writer.close();env.close();vec.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=LABELS,required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--all-stages',action='store_true');args=p.parse_args()
    directory=ROOT/'results'/'zoo'/f'{args.mode}-{args.seed}'
    stages=json.loads((directory/'stages.json').read_text())
    if not args.all_stages:stages=stages[-1:]
    media=ROOT/'docs'/'zoo-media';media.mkdir(parents=True,exist_ok=True)
    saved=directory/'videos.json'
    rows=json.loads(saved.read_text()) if saved.exists() else []
    for stage in stages:
        if any(r['steps']==stage['steps'] and (ROOT/'docs'/r['video']).exists() for r in rows):continue
        print(f'Rendering {args.mode}-{args.seed} at {stage["steps"]}',flush=True)
        name=f'{args.mode}-{args.seed}-{stage["steps"]:07d}.mp4'
        metrics=z.evaluate(directory,stage,episodes=10,start_seed=9000) if stage['steps']<z.CONFIG['steps'] else json.loads((directory/'evaluation.json').read_text())
        episodes=3 if stage['steps']>=z.CONFIG['steps'] else 1
        record(directory,stage,args.mode,args.seed,media/name,episodes)
        row={**metrics,'mode':args.mode,'seed':args.seed,'video':f'zoo-media/{name}','poster':f'zoo-media/{name[:-4]}.jpg','duration':8*episodes,'recorded_episodes':episodes,'padding':'After termination, the final frame is held and labeled until the next trial.'}
        rows.append(row);rows.sort(key=lambda r:r['steps']);(directory/'videos.json').write_text(json.dumps(rows,indent=2))
        print(json.dumps({k:v for k,v in row.items() if k!='episodes'}),flush=True)

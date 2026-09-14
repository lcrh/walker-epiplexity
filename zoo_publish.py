"""Publish the matched Zoo comparison, raw data, checkpoint pairs, and curves."""
from pathlib import Path
import csv,json,shutil,subprocess,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import zoo_train as z
ROOT=z.ROOT
records=[];summary={};pairs=[]
for mode in ['standard','mixed']:
    mode_results=[]
    for seed in range(3):
        source=ROOT/'results'/'zoo'/f'{mode}-{seed}'
        videos=json.loads((source/'videos.json').read_text());records.extend(videos)
        result=json.loads((source/'evaluation.json').read_text());mode_results.append(result)
        target=ROOT/'measurements'/'zoo'/f'{mode}-{seed}';target.mkdir(parents=True,exist_ok=True)
        for name in ['provenance.json','stages.json','evaluation.json','videos.json','train.monitor.csv']:shutil.copy2(source/name,target/name)
        checkpoints=ROOT/'checkpoints'/'zoo'/f'{mode}-{seed}';checkpoints.mkdir(parents=True,exist_ok=True)
        # Each policy must travel with the normalization snapshot from that same stage.
        for stage in json.loads((source/'stages.json').read_text()):
            for field in ['model','normalization']:shutil.copy2(source/stage[field],checkpoints/stage[field])
    summary[mode]={'mean':float(np.mean([r['mean_task_return'] for r in mode_results])),'std':float(np.std([r['mean_task_return'] for r in mode_results])),'mean_episode_steps':float(np.mean([r['mean_episode_steps'] for r in mode_results])),'mean_displacement':float(np.mean([r['mean_displacement'] for r in mode_results])),'full_episode_fraction':float(np.mean([r['full_episode_fraction'] for r in mode_results])),'seeds':[{k:v for k,v in r.items() if k!='episodes'} for r in mode_results]}
for seed in range(3):pairs.append({'seed':seed,'standard':summary['standard']['seeds'][seed]['mean_task_return'],'mixed':summary['mixed']['seeds'][seed]['mean_task_return']})
for row in pairs:row['difference']=row['mixed']-row['standard']
data={'config':z.CONFIG,'beta':json.loads((ROOT/'results/zoo/calibration.json').read_text())['beta'],'summary':summary,'pairs':pairs,'videos':records}
(ROOT/'docs'/'zoo-data.json').write_text(json.dumps(data,indent=2))
(ROOT/'measurements'/'zoo'/'comparison.json').write_text(json.dumps({k:v for k,v in data.items() if k!='videos'},indent=2))
for suffix in ['pt','json']:shutil.copy2(ROOT/f'results/zoo/calibration.{suffix}',ROOT/f'measurements/zoo/calibration.{suffix}')
fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15);ax.set_xlabel('Training steps')
for mode,color,label in [('standard','#366854','Standard'),('mixed','#ad7243','Standard + epiplexity')]:
    for seed in range(3):
        with (ROOT/'results'/'zoo'/f'{mode}-{seed}'/'train.monitor.csv').open() as f:
            next(f);rows=list(csv.DictReader(f))
        steps=np.cumsum([int(r['l']) for r in rows]);window=min(50,len(rows))
        for ax,column in zip(axes,['task_return','full_episode']):
            values=np.array([float(r[column]) for r in rows]);smoothed=np.convolve(values,np.ones(window)/window,mode='valid')
            ax.plot(steps[window-1:],smoothed,color=color,alpha=.7,lw=1.2,label=label if seed==0 else None)
axes[0].set_title('Task return during training');axes[0].set_ylabel('50-episode mean · three seeds');axes[0].legend(frameon=False)
axes[1].set_title('Surviving the full episode');axes[1].set_ylabel('Fraction reaching 1,000 steps');axes[1].set_ylim(-.02,1.02)
fig.savefig(ROOT/'docs'/'zoo-training.png',dpi=180);plt.close(fig)
for video_seed in [0,1]:
    finals=[next(r for r in records if r['seed']==video_seed and r['mode']==m and r['steps']>=z.CONFIG['steps']) for m in ['standard','mixed']]
    subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(ROOT/'docs'/finals[0]['video']),'-i',str(ROOT/'docs'/finals[1]['video']),'-filter_complex','[0:v][1:v]hstack=inputs=2[v]','-map','[v]','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(ROOT/f'docs/zoo-media/comparison-seed-{video_seed}.mp4')],check=True)
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['checkpoints/zoo','docs/zoo-media'] for p in sorted((ROOT/folder).rglob('*')) if p.is_file()}
(ROOT/'measurements/zoo/sha256.json').write_text(json.dumps(manifest,indent=2))
text='\n## Zoo comparison results\n\n| Condition | Task return (mean ± SD across three seed means) | Mean episode steps | Full-episode rate |\n|---|---:|---:|---:|\n'
for mode,label in [('standard','Standard'),('mixed','Standard + epiplexity')]:
    r=summary[mode];text+=f"| {label} | {r['mean']:.1f} ± {r['std']:.1f} | {r['mean_episode_steps']:.1f} | {r['full_episode_fraction']:.1%} |\n"
text+='\nEach final policy is evaluated on the same 100 episodes, seeds 7000–7099. Error bars are across training seeds, not evaluation episodes. Three seeds are a small comparison, not a significance claim. The standard and mixed policies both start from scratch, with matched initial policy hashes.\n\n'
for row in pairs:text+=f"- Seed {row['seed']}: {row['standard']:.1f} standard → {row['mixed']:.1f} mixed ({row['difference']:+.1f}).\n"
p=ROOT/'ZOO.md';p.write_text(p.read_text().split('\n## Zoo comparison results')[0]+text)
print(text,flush=True)

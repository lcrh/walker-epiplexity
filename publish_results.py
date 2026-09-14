"""Collect measurements and checkpoints and create the static report/curves."""
from pathlib import Path
import csv,json,shutil,hashlib,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
runs=[json.loads((ROOT/'results'/f'seed-{seed}'/'evaluation.json').read_text()) for seed in range(3)]
finals=[run[-1] for run in runs]
means=[row['mean_task_return'] for row in finals]
data={'mean':float(np.mean(means)),'std':float(np.std(means)),'stages':runs[0],'finals':finals,'paper':{'mean':327,'std':45,'training_seeds':10},'training_seeds':[0,1,2],'upstream_commit':'22541fa076c8dc11e182c5d611a1beec11662ad1'}
(ROOT/'docs'/'data.json').write_text(json.dumps(data,indent=2))
fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
for ax in axes:ax.spines[['top','right']].set_visible(False);ax.set_xlabel('Training steps');ax.grid(alpha=.15)
for seed,color in zip(range(3),['#245b46','#b47b45','#5574a4']):
    sub='epiplexity' if seed==0 else f'epiplexity_s{seed}'
    source=ROOT/'results'/'authors-metrics'/'walker2d'/sub
    target=ROOT/'measurements'/f'seed-{seed}';target.mkdir(parents=True,exist_ok=True)
    points=[]
    for file in sorted(source.glob('*.monitor.csv')):
        shutil.copy2(file,target/file.name)
        with file.open() as f:
            next(f);rows=list(csv.DictReader(f))
        points.extend((float(r['t']),int(r['l']),float(r['r']),float(r['task_return'])) for r in rows)
    points.sort();steps=np.cumsum([r[1] for r in points])
    for axis,column in zip(axes,[2,3]):
        values=np.array([r[column] for r in points]);window=min(100,len(values))
        smooth=np.convolve(values,np.ones(window)/window,mode='valid')
        axis.plot(steps[window-1:],smooth,label=f'Seed {seed}',color=color,lw=1.5)
    for name in ['provenance.json','stages.json','evaluation.json']:
        shutil.copy2(ROOT/'results'/f'seed-{seed}'/name,target/name)
    shutil.copy2(source/'metrics.json',target/'authors-metrics.json')
    checkpoints=ROOT/'checkpoints'/f'seed-{seed}';checkpoints.mkdir(parents=True,exist_ok=True)
    for path in (ROOT/'results'/f'seed-{seed}').glob('policy_*.zip'):
        if seed==0 or path.name==runs[seed][-1]['file']:shutil.copy2(path,checkpoints/path.name)
axes[0].set_title('Training reward: epiplexity');axes[0].set_ylabel('Episode intrinsic return · 100-episode mean')
axes[1].set_title('Task return (logged only, not optimized)');axes[1].set_ylabel('Episode task return · 100-episode mean');axes[1].legend(frameon=False)
fig.savefig(ROOT/'docs'/'training.png',dpi=180);plt.close(fig)
concat=ROOT/'results'/'montage.txt'
concat.write_text(''.join("file '"+str(ROOT/'docs'/stage['video'])+"'\n" for stage in runs[0]))
subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-f','concat','-safe','0','-i',str(concat),'-c','copy','-movflags','+faststart',str(ROOT/'docs'/'media'/'training-stages.mp4')],check=True)
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['checkpoints','docs/media'] for p in sorted((ROOT/folder).rglob('*')) if p.is_file()}
(ROOT/'measurements'/'sha256.json').write_text(json.dumps(manifest,indent=2))
text=f'''\n## Recorded result\n\nThree-seed mean task return: **{data['mean']:.2f} ± {data['std']:.2f}** (population standard deviation across seed means).\n\n| Training seed | Task return, 100 evaluation episodes | Mean episode length | Mean displacement |\n|---|---:|---:|---:|\n'''
for row in finals:text+=f"| {row['seed']} | {row['mean_task_return']:.2f} | {row['mean_episode_steps']:.1f} steps | {row['mean_displacement']:.2f} m |\n"
text+='\nThese runs fall below the reported paper mean. They show brief forward motion followed by falling, rather than sustained walking. The cause of the numerical gap is not established; three seeds and a macOS arm64 run are not a ten-seed platform-matched replication.\n'
readme=ROOT/'README.md';base=readme.read_text().split('\n## Recorded result')[0];readme.write_text(base+text)
print(text)

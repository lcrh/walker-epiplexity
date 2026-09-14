const modes=['standard','mixed'];const players=modes.map(m=>document.getElementById(m));
let seed=1,selectedSteps=null,data,playRequest=0;
function button(text,selected,fn){const b=document.createElement('button');b.textContent=text;b.setAttribute('aria-pressed',String(selected));b.onclick=fn;return b;}
function show(){
 playRequest++;
 const available=data.videos.filter(v=>v.mode==='standard'&&v.seed===seed).sort((a,b)=>a.steps-b.steps);
 if(!available.some(v=>v.steps===selectedSteps))selectedSteps=available.at(-1).steps;
 const seedBox=document.getElementById('seeds');seedBox.replaceChildren(...[0,1,2].map(s=>button(String(s),s===seed,()=>{seed=s;show();})));
 const stageBox=document.getElementById('stages');stageBox.replaceChildren(...available.map(v=>button(v.steps===0?'Untrained':v.steps>=1000000?'1M':`${Math.round(v.steps/1000)}k`,v.steps===selectedSteps,()=>{selectedSteps=v.steps;show();})));
 for(const mode of modes){const row=data.videos.find(v=>v.mode===mode&&v.seed===seed&&v.steps===selectedSteps);const video=document.getElementById(mode);video.pause();video.src=row.video;video.poster=row.poster;
 document.getElementById(`${mode}-caption`).textContent=`Seed ${seed} · ${row.steps.toLocaleString()} steps`;
 const link=document.getElementById(`${mode}-download`);link.href=row.video;link.download=row.video.split('/').pop();}
 document.getElementById('playback-note').textContent=`${selectedSteps>=1000000?'Three held-out trials':'One held-out trial'} · Real speed. Falls are held and labeled until the next matched trial.`;
 document.getElementById('sync').textContent='▶ Play together';
}
document.getElementById('sync').onclick=async()=>{
 const request=++playRequest;const control=document.getElementById('sync');
 if(players.some(v=>!v.paused)){players.forEach(v=>v.pause());control.textContent='▶ Play together';return;}
 control.textContent='Loading…';
 try{
  await Promise.all(players.map(v=>v.readyState>=3?Promise.resolve():new Promise((resolve,reject)=>{v.addEventListener('canplay',resolve,{once:true});v.addEventListener('error',()=>reject(Error('Video could not load')),{once:true});})));
  if(request!==playRequest)return;
  players.forEach(v=>v.currentTime=0);
  await Promise.all(players.map(v=>v.play()));
  if(request!==playRequest)return;
  players.forEach(v=>v.currentTime=0);control.textContent='Ⅱ Pause both';
 }catch(error){players.forEach(v=>v.pause());control.textContent='▶ Play together';document.getElementById('playback-note').textContent=error.message;}
};
setInterval(()=>{const [a,b]=players;if(!a.paused&&!b.paused&&a.readyState>=3&&b.readyState>=3&&Math.abs(a.currentTime-b.currentTime)>.12)b.currentTime=a.currentTime;},250);
players.forEach(v=>v.addEventListener('ended',()=>{if(players.every(p=>p.ended))document.getElementById('sync').textContent='▶ Play together';}));
fetch('zoo-data.json').then(r=>{if(!r.ok)throw Error('Comparison data unavailable');return r.json();}).then(d=>{data=d;
 for(const mode of modes){const r=data.summary[mode];document.getElementById(`${mode}-score`).textContent=`${Math.round(r.mean).toLocaleString()} ± ${Math.round(r.std).toLocaleString()}`;document.getElementById(`${mode}-survival`).textContent=`${Math.round(100*r.full_episode_fraction)}%`;}
 const delta=data.summary.mixed.mean-data.summary.standard.mean;
 document.getElementById('finding').textContent=`The epiplexity bonus ${delta>=0?'raises':'lowers'} the mean task return by ${Math.abs(Math.round(delta)).toLocaleString()} in these three seeds. See the clips and episode survival alongside the score.`;
 show();
}).catch(e=>document.getElementById('finding').textContent=e.message);

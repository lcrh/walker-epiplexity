const video=document.querySelector('#video');
fetch('data.json').then(r=>{if(!r.ok)throw Error('Recordings unavailable');return r.json()}).then(data=>{
  const buttons=[];
  function select(stage,button){
    buttons.forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
    video.src=stage.video;video.poster=stage.poster;
    document.querySelector('#caption').textContent=`Seed ${stage.seed} · ${stage.steps.toLocaleString()} steps · 3 evaluation episodes`;
    const link=document.querySelector('#download');link.href=stage.video;link.download=stage.video.split('/').pop();
    video.play().catch(()=>{});
  }
  function add(stage,parent,label){const b=document.createElement('button');b.textContent=label;b.setAttribute('aria-pressed','false');b.onclick=()=>select(stage,b);buttons.push(b);document.querySelector(parent).append(b);return b;}
  let primary;
  for(const stage of data.stages){const b=add(stage,'#stages',stage.steps===0?'Untrained':`${Math.round(stage.steps/1000)}k`);primary=[stage,b];}
  for(const stage of data.finals.filter(s=>s.seed!==0))add(stage,'#seeds',`Seed ${stage.seed}`);
  document.querySelector('#result').textContent=`${data.mean.toFixed(0)} ± ${data.std.toFixed(0)}`;
  select(...primary);
}).catch(error=>{document.querySelector('#caption').textContent=error.message;});

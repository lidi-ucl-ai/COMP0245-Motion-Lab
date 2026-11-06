import * as THREE from 'three';
import {OrbitControls} from '/lib/OrbitControls.js';
import {ColladaLoader} from '/lib/ColladaLoader.js';
import {createResourceLoader} from '/resource-loader.js';
import {initMonitoring} from '/monitoring-ui.js';
const $=id=>document.getElementById(id), message=t=>$('message').textContent=t;
function setViewMode(mode,remember=true){
 const detailed=mode==='detailed';document.body.dataset.view=detailed?'detailed':'simple';
 $('view-simple').setAttribute('aria-pressed',String(!detailed));$('view-detailed').setAttribute('aria-pressed',String(detailed));
 $('view-mode-summary').textContent=detailed?'3D view with speed, torque and position monitors':'3D view and run results';
 if(remember){try{localStorage.setItem('sblidi-view-mode',detailed?'detailed':'simple');}catch(e){}}
}
let savedView='simple';try{savedView=localStorage.getItem('sblidi-view-mode')||'simple';}catch(e){}
setViewMode(savedView,false);$('view-simple').onclick=()=>setViewMode('simple');$('view-detailed').onclick=()=>setViewMode('detailed');
const monitors=initMonitoring($('monitor-grid'));
const resources=createResourceLoader({onRetry:n=>{$('loading').textContent=`Connection interrupted. Retrying (${n}/5)…`;}});
let meta;
try{meta=await resources.json('/api/meta');}catch(e){$('loading').textContent='Could not connect to the model server. Refresh to try again.';message(e.message);throw e;}
let current=meta.state,lastSeq=-1,runId='',samples=[],lastState=null,meshes=[],busy=false,exporting=false,robotReady=false;
function fields(id,labels,values){$(id).replaceChildren();labels.forEach((name,i)=>{const wrap=document.createElement('div');wrap.className='input-wrap';const l=document.createElement('label');l.textContent=name;const input=document.createElement('input');input.type='number';input.step='0.001';input.value=values[i];input.id=id+'-'+i;l.htmlFor=input.id;wrap.append(l,input);$(id).append(wrap);});}
function preset(){fields('target-inputs',['X','Y','Z'],meta.default.target);fields('base-inputs',['X','Y','Yaw'],meta.default.base);fields('arm-inputs',Array.from({length:7},(_,i)=>'J'+(i+1)),meta.default.arm);$('duration').value=meta.default.duration;}
preset();
const groups = new Map();
for (const [id, model] of Object.entries(meta.models)) {
  if (!groups.has(model.group)) {
    const group = document.createElement('optgroup');
    group.label = model.group;
    $('model').append(group);
    groups.set(model.group, group);
  }
  groups.get(model.group).append(new Option(model.label, id));
}
$('model').value = current.model;
function describeModel() {
  const model = meta.models[$('model').value];
  $('model-summary').textContent = model.description;
  $('model-score').textContent = model.matched ? 'Paper benchmark: ' + model.matched + ' successful' : model.group;
}
$('model').addEventListener('change', describeModel);
describeModel();
const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor(0x1b2026);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.35;
$('viewport').append(renderer.domElement);const scene=new THREE.Scene();scene.fog=new THREE.Fog(0x1b2026,9,24);
const camera=new THREE.PerspectiveCamera(42,1,.01,130);camera.up.set(0,0,1);const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.minDistance=.5;controls.maxDistance=45;controls.maxPolarAngle=Math.PI*.49;
function home(){camera.position.set(1.65,-1.8,1.45);controls.target.set(.35,0,.5);controls.update();}home();$('home').onclick=home;
scene.add(new THREE.HemisphereLight(0xe4f0ff,0x4b5359,2.6));const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(2,-4,6);scene.add(light);const fill=new THREE.DirectionalLight(0xb6dfd2,1);fill.position.set(-3,2,3);scene.add(fill);
const floor=new THREE.Mesh(new THREE.PlaneGeometry(40,40),new THREE.MeshStandardMaterial({color:0x242b31,roughness:1}));floor.position.z=-.015;scene.add(floor);const grid=new THREE.GridHelper(12,48,0x53606a,0x343f48);grid.rotation.x=Math.PI/2;grid.position.z=-.012;scene.add(grid);
const target=new THREE.Mesh(new THREE.SphereGeometry(.025,24,16),new THREE.MeshStandardMaterial({color:0xff605c,emissive:0x7a1810,emissiveIntensity:.55}));scene.add(target);
const halo=new THREE.Mesh(new THREE.TorusGeometry(.055,.002,8,60),new THREE.MeshBasicMaterial({color:0xf18a7e,transparent:true,opacity:.65}));scene.add(halo);
const eeDot=new THREE.Mesh(new THREE.SphereGeometry(.014,12,8),new THREE.MeshBasicMaterial({color:0xffab61}));scene.add(eeDot);
function trail(color){const geo=new THREE.BufferGeometry();const array=new Float32Array(10000*3);geo.setAttribute('position',new THREE.BufferAttribute(array,3).setUsage(THREE.DynamicDrawUsage));geo.setDrawRange(0,0);const line=new THREE.Line(geo,new THREE.LineBasicMaterial({color}));line.frustumCulled=false;scene.add(line);return {line,geo,array,n:0};}
const eeTrail=trail(0xffab61),baseTrail=trail(0x53b8bd);
function addTrail(t,p){if(t.n>=10000)return;t.array.set(p,t.n*3);t.n++;t.geo.attributes.position.needsUpdate=true;t.geo.setDrawRange(0,t.n);}
function clearTrails(){for(const t of [eeTrail,baseTrail]){t.n=0;t.geo.setDrawRange(0,0);}}
const loader=new ColladaLoader(),cache=new Map();
let loaded=0;const assetCount=new Set(meta.visuals.filter(v=>v.asset).map(v=>v.asset)).size;
async function asset(path){if(!cache.has(path)){const url='/assets/'+path;cache.set(path,resources.text(url).then(text=>{const d=loader.parse(text,url.slice(0,url.lastIndexOf('/')+1));d.scene.rotation.x=0;loaded++;$('loading').textContent=`Loading robot (${loaded}/${assetCount})…`;return d.scene;}).catch(e=>{cache.delete(path);throw e;}));}return (await cache.get(path)).clone(true);}
try{meshes=await Promise.all(meta.visuals.map(async v=>{const group=new THREE.Group();scene.add(group);let object;if(v.asset){object=await asset(v.asset);object.scale.multiply(new THREE.Vector3(...v.dimensions));}else{object=new THREE.Mesh(new THREE.BoxGeometry(...v.dimensions),new THREE.MeshStandardMaterial({color:new THREE.Color(...v.color.slice(0,3))}));}const local=new THREE.Group();local.position.fromArray(v.position);local.quaternion.fromArray(v.quaternion);local.add(object);group.add(local);return {link:String(v.link),group};}));robotReady=true;$('loading').hidden=true;}catch(e){$('loading').textContent='Robot assets failed to load.';const retry=document.createElement('button');retry.textContent='Reload';retry.onclick=()=>location.reload();$('loading').append(' ',retry);message(e.message);}
function resize(){if(exporting)return;const box=$('viewport').getBoundingClientRect();renderer.setSize(box.width,box.height,false);camera.aspect=box.width/box.height;camera.updateProjectionMatrix();drawChart();}new ResizeObserver(resize).observe($('viewport'));
renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera);});
$('fullscreen').onclick=()=>{if(document.fullscreenElement)document.exitFullscreen();else $('viewport').requestFullscreen();};
const phases=['Align heading','Drive','Align tool','Reach'];
function buttons(){const running=current.status==='running',paused=current.status==='paused',watching=current.can_control===false;for(const id of ['start','preview','preset'])$(id).disabled=!robotReady||busy||running||paused||watching;$('pause').disabled=busy||watching||(!running&&!paused);$('pause').textContent=paused?'Resume':'Pause';$('stop').disabled=busy||watching||(!running&&!paused);for(const e of document.querySelectorAll('aside input,aside select'))e.disabled=busy||running||paused||watching;$('export').disabled=!robotReady||busy||!['finished','stopped','error'].includes(current.status);}
function update(state){current=state;if(state.seq===lastSeq)return;lastSeq=state.seq;
 const branch=state.domain_selection?.at(-1)?.member_name;
 const branchLabels={far_specialist:'Far',high_specialist:'High',frozen_previous_domains:'Regular / near origin'};
 const model=meta.models[state.model];
 $('active-network').textContent=model.label + (branch ? ' · ' + (branchLabels[branch] || branch) : '');
 $('checkpoint').textContent=state.checkpoint_sha256 ? 'SHA-256 ' + state.checkpoint_sha256.slice(0,12) : 'Deterministic expert';
 if(state.run_id!==runId){runId=state.run_id;samples=[];clearTrails();$('json').hidden=true;$('download').hidden=true;}
 for(const v of meshes){const pose=state.poses[v.link];if(pose){v.group.position.fromArray(pose[0]);v.group.quaternion.fromArray(pose[1]);}}
 target.position.fromArray(state.target);halo.position.copy(target.position);eeDot.position.fromArray(state.ee);
 if(!samples.length||state.time>samples.at(-1).time){samples.push(state);addTrail(eeTrail,state.ee);addTrail(baseTrail,[state.base[0],state.base[1],.006]);}
 $('time').innerHTML=state.time.toFixed(2)+'<small>s</small>';$('error').innerHTML=(state.error*1000).toFixed(2)+'<small>mm</small>';$('phase').textContent=phases[state.phase]||'—';$('speed').innerHTML=(state.time>0&&state.wall_elapsed>0?(state.time/state.wall_elapsed).toFixed(2):'—')+'<small>×</small>';
 const status={ready:'Ready',running:'Running',paused:'Paused',finished:state.success?'Success':'Goal not reached',stopped:'Stopped',error:'Simulation error'};$('status').textContent=status[state.status]||state.status;
 if(['finished','stopped','error'].includes(state.status)){$('json').hidden=false;$('json').href='/runs/'+state.run_id+'.json';if(lastState?.status!==state.status)message(state.success?`Success in ${state.time.toFixed(2)} s. Final error ${(state.error*1000).toFixed(2)} mm.`:state.reason==='user_stop'?'Run stopped and trajectory saved.':'Run ended: '+state.reason);}
 lastState=state;buttons();drawChart();
}
function drawChart(){monitors.update(current,samples);const c=$('chart'),rect=c.getBoundingClientRect(),dpr=Math.min(devicePixelRatio,2);if(!rect.width)return;c.width=rect.width*dpr;c.height=rect.height*dpr;const ctx=c.getContext('2d');ctx.scale(dpr,dpr);const w=rect.width,h=rect.height,xmax=Math.max(1,Number($('duration').value),current.time),ymax=Math.max(.1,...samples.map(s=>s.error))*1.1;ctx.strokeStyle='#303943';ctx.lineWidth=1;for(let i=0;i<4;i++){const y=8+i*(h-20)/3;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}ctx.setLineDash([3,4]);ctx.strokeStyle='#718d70';const y=h-10-.01/ymax*(h-20);ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();ctx.setLineDash([]);ctx.beginPath();samples.forEach((s,i)=>{const x=s.time/xmax*w,y=h-10-s.error/ymax*(h-20);if(!i)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.strokeStyle='#ffab61';ctx.lineWidth=1.8;ctx.stroke();ctx.fillStyle='#84919c';ctx.font='9px Segoe UI';ctx.fillText(ymax.toFixed(2),4,10);$('chart-end').textContent=xmax.toFixed(0)+' s';}
function connected(state){
 update(state);
 $('connection').textContent='Simulator connected';
 $('connection-dot').style.background='#b6e0ad';
}
const events=new EventSource('/api/events');
events.onmessage=e=>connected(JSON.parse(e.data));
events.onerror=()=>{
 $('connection').textContent='Reconnecting…';
 $('connection-dot').style.background='#e9a19a';
};
function config(){const read=(id,n)=>Array.from({length:n},(_,i)=>Number($(id+'-'+i).value));for(const e of document.querySelectorAll('aside input[type=number]'))if(e.value===''||!Number.isFinite(Number(e.value)))throw Error('Enter a finite number in every field.');return {target:read('target-inputs',3),base:read('base-inputs',3),arm:read('arm-inputs',7),duration:Number($('duration').value),model:$('model').value};}
async function command(action,data={}){const r=await fetch('/api/'+action,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});const s=await r.json();if(!r.ok)throw Error(s.error);update(s);return s;}
async function invoke(action){busy=true;buttons();try{await command(action,['start','preview'].includes(action)?config():{});if(action==='start' && matchMedia('(max-width: 739px)').matches)$('viewport-card').scrollIntoView();if(action==='preview')message('Model loaded and initial state applied.');}catch(e){message(e.message);}finally{busy=false;buttons();}}
$('start').onclick=()=>invoke('start');
$('preview').onclick=()=>invoke('preview');$('pause').onclick=()=>invoke(current.status==='paused'?'resume':'pause');$('stop').onclick=()=>invoke('stop');$('preset').onchange=()=>{
 const presets={near:[0,0,1.2,30],regular:[1.5,0,.8,30],high:[1.5,0,1.4,60],far:[4,2,.8,60]};
 const selected=presets[$('preset').value];
 if (!selected) return;
 fields('target-inputs',['X','Y','Z'],selected.slice(0,3));
 $('duration').value=selected[3];
 $('preset').value='';
};
function pose(state){for(const v of meshes){const p=state.poses[v.link];if(p){v.group.position.fromArray(p[0]);v.group.quaternion.fromArray(p[1]);}}target.position.fromArray(state.target);halo.position.copy(target.position);eeDot.position.fromArray(state.ee);}
$('export').onclick=async()=>{
 busy=true;exporting=true;buttons();controls.enabled=false;
 const snapshot=current;const run=snapshot.run_id;const oldSize=renderer.getSize(new THREE.Vector2());const oldRatio=renderer.getPixelRatio();const frames=[];
 try{
  message('Reading the saved trajectory…');const record=await fetch('/runs/'+run+'.json').then(r=>r.json());const states=record.samples;
  if(states.length<2)throw Error('Too few trajectory samples.');
  const duration=record.result.time,steps=Math.ceil(duration*25)+1,total=steps+50;
  renderer.setPixelRatio(1);renderer.setSize(1280,720,false);camera.aspect=1280/720;camera.updateProjectionMatrix();clearTrails();let j=0;
  for(let k=0;k<steps;k++){
   const time=k/25;while(j+1<states.length&&states[j+1].time<=time)j++;
   const s=k===steps-1?states.at(-1):states[j];pose(s);addTrail(eeTrail,s.ee);addTrail(baseTrail,[s.base[0],s.base[1],.006]);renderer.render(scene,camera);
   const raw=atob(renderer.domElement.toDataURL('image/jpeg',.92).split(',')[1]);const bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);frames.push(new Blob([bytes],{type:'image/jpeg'}));
   if(k%25===0){message(`Rendering video: ${Math.round(k/total*100)}%`);await new Promise(resolve=>setTimeout(resolve,0));}
  }
  for(let k=0;k<50;k++)frames.push(frames.at(-1));
  const blob=new Blob(frames,{type:'application/x-mjpeg'});if(blob.size>80000000)throw Error('Recording exceeds 80 MB. Try a shorter run.');
  message('Encoding and validating MP4…');const r=await fetch('/api/video',{method:'POST',headers:{'Content-Type':'application/x-mjpeg','X-Run-Id':run,'X-Frame-Count':String(frames.length)},body:blob});const d=await r.json();if(!r.ok)throw Error(d.error);
  $('download').href=d.url;$('download').hidden=false;message(`MP4 ready: ${Number(d.validation.format.duration).toFixed(2)} s / 1280 × 720.`);
 }catch(e){message(e.message);}finally{renderer.setPixelRatio(oldRatio);renderer.setSize(oldSize.x,oldSize.y,false);camera.aspect=oldSize.x/oldSize.y;camera.updateProjectionMatrix();pose(snapshot);clearTrails();for(const s of samples){addTrail(eeTrail,s.ee);addTrail(baseTrail,[s.base[0],s.base[1],.006]);}controls.enabled=true;busy=false;exporting=false;buttons();resize();}
};
update(current);
if(['finished','stopped','error'].includes(current.status)){
 try{const saved=await fetch('/runs/'+current.run_id+'.json').then(r=>r.json());samples=saved.samples;clearTrails();for(const s of samples){addTrail(eeTrail,s.ee);addTrail(baseTrail,[s.base[0],s.base[1],.006]);}drawChart();const video=await fetch('/runs/'+current.run_id+'.video.json');if(video.ok){$('download').href='/runs/'+current.run_id+'.mp4';$('download').hidden=false;}}catch(e){message('Current state restored；Could not read saved trajectory: '+e.message);}
}
resize();



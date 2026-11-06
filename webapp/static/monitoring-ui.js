/** Compact monitors; every value comes from the latest server snapshot. */
export function initMonitoring(host){
 host.innerHTML=`<div class="monitor-card"><div class="monitor-heading"><h3>Motion speed</h3><span id="monitor-status">Waiting for samples</span></div><div class="motion-values"><div><span class="series-label ee-series">End effector</span><strong id="ee-speed">—</strong><small>m/s</small></div><div><span class="series-label base-series">Base</span><strong id="base-speed">—</strong><small>m/s</small></div></div><canvas id="speed-chart" aria-label="End-effector and base speed over the last ten seconds"></canvas><div class="monitor-note">Last 10 s. End-effector speed estimated from position samples.</div></div>
 <div class="monitor-card"><div class="monitor-heading"><h3>Joint torques</h3><span id="torque-peak">— N·m</span></div><div id="torque-bars">${Array.from({length:7},(_,i)=>`<div class="torque-row"><span>J${i+1}</span><div class="torque-track"><div class="torque-fill" id="torque-bar-${i}"></div></div><output id="torque-value-${i}">—</output></div>`).join('')}</div><div class="torque-scale"><span>−30</span><span>0</span><span>+30 N·m</span></div><div class="monitor-note">Applied torque. Centre line is zero.</div></div>
 <div class="monitor-card"><div class="monitor-heading"><h3>Position tracking</h3><span>World coordinates</span></div><table class="position-table"><thead><tr><th>Axis</th><th>Target / m</th><th>Actual / m</th><th>Error / mm</th></tr></thead><tbody>${['X','Y','Z'].map((axis,i)=>`<tr><th>${axis}</th><td id="pos-target-${i}">—</td><td id="pos-actual-${i}">—</td><td id="pos-delta-${i}">—</td></tr>`).join('')}</tbody></table><div class="hold-label"><span>Time within threshold</span><span id="hold-value">— / 500 ms</span></div><progress id="hold-progress" max="500" value="0" aria-label="Consecutive time under 1 cm"></progress><div class="monitor-note">Success requires 500 ms continuously below 1 cm.</div></div>`;
 const $=id=>host.querySelector('#'+id);
 let latest=null,history=[];
 function draw(){
  const canvas=$('speed-chart'),rect=canvas.getBoundingClientRect();if(!rect.width)return;
  const dpr=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);
  const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);const w=rect.width,h=rect.height;
  const end=latest?.time||0,start=Math.max(0,end-10),span=Math.max(1,Math.min(10,end));
  const rows=history.filter(s=>s.time>=start&&s.telemetry);const max=Math.max(.05,...rows.flatMap(s=>[s.telemetry.ee_speed_m_s,s.telemetry.base_speed_m_s]).filter(Number.isFinite))*1.15;
  ctx.lineWidth=1;ctx.strokeStyle='#303943';for(let i=0;i<3;i++){const y=8+i*(h-15)/2;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}
  for(const [key,color] of [['ee_speed_m_s','#ffab61'],['base_speed_m_s','#53b8bd']]){
   ctx.beginPath();let first=true;for(const s of rows){const value=s.telemetry[key];if(!Number.isFinite(value))continue;const x=(s.time-start)/span*w,y=h-6-value/max*(h-16);if(first){ctx.moveTo(x,y);first=false;}else ctx.lineTo(x,y);}ctx.strokeStyle=color;ctx.lineWidth=1.7;ctx.stroke();
  }
  ctx.fillStyle='#87939f';ctx.font='9px Segoe UI';ctx.fillText(max.toFixed(2),3,9);
 }
 function update(state,samples){
  latest=state;history=samples;
  const t=state.telemetry;
  $('monitor-status').textContent=t?(state.status==='paused'?'Paused · last sample':['finished','stopped','error'].includes(state.status)?'Final sample':'Live sample'):'No speed or torque samples';
  $('ee-speed').textContent=Number.isFinite(t?.ee_speed_m_s)?t.ee_speed_m_s.toFixed(3):'—';
  $('base-speed').textContent=Number.isFinite(t?.base_speed_m_s)?t.base_speed_m_s.toFixed(3):'—';
  const torques=t?.arm_torque_nm,limit=t?.torque_limit_nm||30;
  const peak=torques?Math.max(...torques.map(Math.abs)):null;
  $('torque-peak').textContent=peak===null?'— N·m':`Peak |τ| ${peak.toFixed(2)} N·m`;
  for(let i=0;i<7;i++){
   const value=torques?.[i],bar=$('torque-bar-'+i),ratio=Number.isFinite(value)?Math.min(1,Math.abs(value)/limit):0;
   bar.style.left=(value<0?50-ratio*50:50)+'%';bar.style.width=ratio*50+'%';bar.classList.toggle('near-limit',ratio>=.9);
   $('torque-value-'+i).textContent=Number.isFinite(value)?value.toFixed(2):'—';
  }
  for(let i=0;i<3;i++){
   $('pos-target-'+i).textContent=state.target[i].toFixed(4);$('pos-actual-'+i).textContent=state.ee[i].toFixed(4);
   const delta=(state.target[i]-state.ee[i])*1000;$('pos-delta-'+i).textContent=(delta>0?'+':'')+delta.toFixed(2);
  }
  const hold=Number.isFinite(t?.success_hold_s)?Math.round(t.success_hold_s*1000):null;
  $('hold-value').textContent=(hold===null?'—':hold)+' / 500 ms';$('hold-progress').value=hold||0;
  draw();
 }
 new ResizeObserver(draw).observe(host);
 return {update};
}

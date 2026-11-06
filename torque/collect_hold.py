"""Offline DAgger with continued post-arrival holding; never used by evaluation."""
from pathlib import Path
import argparse,json,hashlib,time,concurrent.futures,numpy as np

def run(task):
 s,checkpoint,out,beta,hold,noise=task;p=Path(out)/(s['id']+'.npz');sha=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
 if p.exists():
  m=json.loads(p.with_suffix('.json').read_text());assert m['sha256']==sha and m['beta']==beta and m['hold_seconds']==hold;return m
 from env import RobotEnv
 from expert import Expert
 from .optimized_policy import OptimizedPolicy,stage_features
 import pinocchio as pin
 e=RobotEnv();o=e.reset(s);initial=o.copy();student=OptimizedPolicy(checkpoint,e.dt,env=e);teacher=Expert(e)
 rng=np.random.default_rng(s['seed']+1703);perturb=np.zeros(7);first=None;held_after=0;maximum_held=0;start=time.perf_counter()
 rows={k:[] for k in ['x','y','reference','error','phase','obs','teacher_raw','teacher_previous','applied_action']}
 while True:
  predicted=student.act(o);previous=None if teacher.servo.torque is None else teacher.servo.torque.copy();label,aux=teacher.label()
  if e.k%20==0:perturb=rng.normal(0,noise,7) if np.linalg.norm(o[18:21]-o[33:36])<.1 else np.zeros(7)
  action=np.r_[predicted[:4],beta*label[4:]+(1-beta)*predicted[4:]+perturb]
  if e.k%5==0:
   measured=e.state['q'].copy();measured[4:11]=o[4:11]
   qp=e.dyn.ReoderJoints2PinVec(np.r_[o[:3],o[21:25],measured],'pos')
   g=pin.computeGeneralizedGravity(e.dyn.pin_model,e.dyn.pin_data,qp)[6:13].copy()
   raw=100*(teacher.qdes-o[4:11])+30*(teacher.velocity-o[11:18])+g
   prev=g if previous is None else previous
   # Reproduce the normal-height expert's final label before local qd augmentation.
   rate=teacher.servo.cfg.get('torque_rate',1000)*e.dt
   recreated=np.clip(prev+np.clip(np.clip(raw,-30,30)-prev,-rate,rate),-30,30)
   if e.target[2]<=1.2:assert np.max(np.abs(recreated-label[4:]))<1e-8
   values=dict(x=stage_features(o,initial,student.base.phase,student.base.reference_time),y=label[4:],reference=aux['q_des']-initial[4:11],error=np.linalg.norm(o[18:21]-o[33:36]),phase=student.base.phase,obs=o,teacher_raw=raw,teacher_previous=prev,applied_action=action)
   for k,v in values.items():rows[k].append(np.copy(v))
  o,info=e.step(action);maximum_held=max(maximum_held,e.held)
  if info['success'] and first is None:first=e.k*e.dt
  if first is not None and info['error']<.01:held_after+=1
  # Continue success episodes to expose drift/recovery rather than censor at .5s.
  if info['reason']=='safety_abort' or e.k*e.dt>=s['duration']:break
  if first is not None and e.k*e.dt>=max(12.,first+hold):break
 p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**{k:np.asarray(v,dtype='float32') for k,v in rows.items()})
 m=dict(scenario=s,sha256=sha,beta=beta,hold_seconds=hold,noise_std_nm=noise,samples=len(rows['x']),first_success_s=first,final_error_m=info['error'],success=first is not None,maximum_hold_s=maximum_held*e.dt,after_arrival_inside_seconds=held_after*e.dt,sim_seconds=e.k*e.dt,wall_s=time.perf_counter()-start,scope='assisted offline training, not student evaluation')
 p.with_suffix('.json').write_text(json.dumps(m,indent=2));e.close();return m

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scenarios',default='configs/train_new.json');p.add_argument('--checkpoint',default='runs/baseline.pt');p.add_argument('--out',required=True);p.add_argument('--beta',type=float,default=.6);p.add_argument('--hold',type=float,default=6.);p.add_argument('--noise',type=float,default=.03);p.add_argument('--offset',type=int,default=0);p.add_argument('--stride',type=int,default=1);p.add_argument('--limit',type=int);p.add_argument('--workers',type=int,default=12);a=p.parse_args()
 scenarios=json.loads(Path(a.scenarios).read_text())[a.offset::a.stride][:a.limit]
 assert all('train' in s['split'] for s in scenarios)
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
  rows=[]
  for r in pool.map(run,[(s,a.checkpoint,a.out,a.beta,a.hold,a.noise) for s in scenarios]):rows.append(r);print(json.dumps(dict(id=r['scenario']['id'],done=len(rows),success=r['success'],seconds=r['sim_seconds'])),flush=True)
 summary=dict(n=len(rows),successes=sum(r['success'] for r in rows),samples=sum(r['samples'] for r in rows),beta=a.beta,hold_seconds=a.hold,noise=a.noise,scope='offline assisted training data')
 Path(a.out).with_suffix('.summary.json').write_text(json.dumps(summary,indent=2));print('HOLD_DATA_DONE',json.dumps(summary),flush=True)

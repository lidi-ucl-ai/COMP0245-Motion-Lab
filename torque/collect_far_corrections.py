"""Training-only mixed rollouts: observe student's deviations, label with teacher."""
from pathlib import Path
import argparse,json,hashlib,time,concurrent.futures,numpy as np,torch,gc

def run(task):
 s,out,checkpoint,beta=task;dest=Path(out)/(s['id']+'.npz');sha=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
 if dest.exists():
  r=json.loads(dest.with_suffix('.json').read_text());assert r['checkpoint_sha256']==sha and r['beta']==beta;return r
 from env import RobotEnv
 from .physical_teacher import align_missing_inertias
 from .high_teacher import HighTeacher
 from .optimized_policy import OptimizedPolicy,stage_features
 from base_memory import BaseMemory
 from .learned_gravity import body_gravity
 e=RobotEnv();align_missing_inertias(e);o=e.reset(s);initial=o.copy();policy=OptimizedPolicy(checkpoint,e.dt,e);base=BaseMemory(torch.load('runs/frozen_navigation.pt',map_location='cpu',weights_only=False)['base_memory']);wheels=np.zeros(4);teacher=None;torque_state=None;first=None;start=time.perf_counter();rng=np.random.default_rng(s['seed']+941);noise=np.zeros(7)
 rows={k:[] for k in ['x','y','gravity','velocity','response','local','error','obs']}
 while True:
  raw=base.act(o);wheels+=np.clip(raw-wheels,-.008,.008)
  if teacher is None:teacher=HighTeacher(e,initial,base.reference_time)
  torque,aux=teacher.label(o)
  if torque_state is None:torque_state=aux['gravity'].copy()
  torque_state+=np.clip(torque-torque_state,-.6,.6)
  student=policy.act(o)[4:];error=np.linalg.norm(o[18:21]-o[33:36])
  if e.k%20==0:noise=rng.normal(0,.03,7) if error<.1 else np.zeros(7)
  if e.k%10==0:
   values=dict(x=np.r_[stage_features(o,initial,base.phase,base.reference_time),body_gravity(o)],y=aux['raw'],gravity=aux['gravity'],velocity=aux['velocity'],response=aux['error_response'],local=aux['local_valid'],error=error,obs=o)
   for k,v in values.items():rows[k].append(np.copy(v))
  o,info=e.step(np.r_[wheels,beta*torque_state+(1-beta)*student+noise])
  if info['success'] and first is None:first=e.k*e.dt
  if info['reason']=='safety_abort' or e.k*e.dt>=s['duration']:break
  if first is not None and e.k*e.dt>=max(12.,first+6.):break
 dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**{k:np.asarray(v,dtype='float32') for k,v in rows.items()});result=dict(scenario=s,success=first is not None,first_success_s=first,final_error_m=info['error'],samples=len(rows['x']),sim_seconds=e.k*e.dt,wall_s=time.perf_counter()-start,beta=beta,checkpoint_sha256=sha,scope='training goals only; mixed actions; pure calibrated teacher raw labels; not student accuracy')
 dest.with_suffix('.json').write_text(json.dumps(result,indent=2));e.close();gc.collect();return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scenarios',default='configs/far_train_augmented.json');p.add_argument('--checkpoint',default='runs/far_transfer/best.pt');p.add_argument('--beta',type=float,default=.5);p.add_argument('--workers',type=int,default=10);p.add_argument('--out',default='data/far_corrections');a=p.parse_args();scenarios=json.loads(Path(a.scenarios).read_text());assert all('train' in s['split'] for s in scenarios)
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
  rows=[]
  for r in pool.map(run,[(s,a.out,a.checkpoint,a.beta) for s in scenarios]):rows.append(r);print(json.dumps(dict(done=len(rows),success=r['success'],seconds=r['sim_seconds'])),flush=True)
 summary=dict(n=len(rows),successes=sum(r['success'] for r in rows),samples=sum(r['samples'] for r in rows),scope='training-only mixed trajectories, not student test performance');Path(a.out).with_suffix('.summary.json').write_text(json.dumps(summary,indent=2));print('CORRECTIONS_DONE',json.dumps(summary),flush=True)

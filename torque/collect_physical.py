from pathlib import Path
import argparse,json,time,concurrent.futures,numpy as np,torch
def run(task):
 s,out,hold,variant=task;dest=Path(out)/(s['id']+'.npz')
 if dest.exists():return json.loads(dest.with_suffix('.json').read_text())
 from env import RobotEnv
 from .markov_teacher import MarkovTeacher
 if variant=='high':
  from .high_teacher import HighTeacher
  MarkovTeacher=HighTeacher
 from .physical_teacher import align_missing_inertias
 from base_memory import BaseMemory
 from .optimized_policy import stage_features
 from .learned_gravity import body_gravity
 e=RobotEnv();align_missing_inertias(e);o=e.reset(s);initial=o.copy();base=BaseMemory(torch.load('runs/frozen_navigation.pt',map_location='cpu',weights_only=False)['base_memory']);wheels=np.zeros(4);teacher=None;torque_state=None;first=None;start=time.perf_counter();rng=np.random.default_rng(s['seed']+917);noise=np.zeros(7)
 rows={k:[] for k in ['x','y','gravity','velocity','response','local','error','obs']}
 while True:
  raw=base.act(o);wheels+=np.clip(raw-wheels,-.008,.008)
  if teacher is None:teacher=MarkovTeacher(e,initial,base.reference_time)
  torque,aux=teacher.label(o)
  if torque_state is None:torque_state=aux['gravity'].copy()
  torque_state+=np.clip(torque-torque_state,-.6,.6)
  error=np.linalg.norm(o[18:21]-o[33:36])
  if e.k%20==0:noise=rng.normal(0,.03,7) if error<.1 else np.zeros(7)
  if e.k%10==0:
   values=dict(x=np.r_[stage_features(o,initial,base.phase,base.reference_time),body_gravity(o)],y=aux['raw'],gravity=aux['gravity'],velocity=aux['velocity'],response=aux['error_response'],local=aux['local_valid'],error=error,obs=o)
   for k,v in values.items():rows[k].append(np.copy(v))
  o,info=e.step(np.r_[wheels,torque_state+noise])
  if info['success'] and first is None:first=e.k*e.dt
  if info['reason']=='safety_abort' or e.k*e.dt>=s['duration']:break
  if first is not None and e.k*e.dt>=max(12.,first+hold):break
 dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**{k:np.asarray(v,dtype='float32') for k,v in rows.items()})
 result=dict(scenario=s,first_success_s=first,success=first is not None,final_error_m=info['error'],samples=len(rows['x']),sim_seconds=e.k*e.dt,hold_seconds=hold,wall_s=time.perf_counter()-start,teacher='physical calibrated Markov torque; frozen tree base; raw labels before slew',variant=variant,scope='training only')
 dest.with_suffix('.json').write_text(json.dumps(result,indent=2));e.close();return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scenarios',default='configs/train_new.json');p.add_argument('--out',default='data/physical_rollouts');p.add_argument('--hold',type=float,default=6.);p.add_argument('--workers',type=int,default=12);p.add_argument('--variant',choices=['regular','high'],default='regular');a=p.parse_args();scenarios=json.loads(Path(a.scenarios).read_text());assert all('train' in s['split'] for s in scenarios)
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
  rows=[]
  for r in pool.map(run,[(s,a.out,a.hold,a.variant) for s in scenarios]):rows.append(r);print(json.dumps(dict(done=len(rows),success=r['success'],seconds=r['sim_seconds'])),flush=True)
 summary=dict(n=len(rows),successes=sum(r['success'] for r in rows),samples=sum(r['samples'] for r in rows),scope='training teacher data, not student accuracy')
 Path(a.out).with_suffix('.summary.json').write_text(json.dumps(summary,indent=2));print('PHYSICAL_ROLLOUTS_DONE',json.dumps(summary),flush=True)

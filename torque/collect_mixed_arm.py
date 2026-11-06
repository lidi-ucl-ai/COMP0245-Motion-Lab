from pathlib import Path
import argparse,concurrent.futures,json,time,hashlib,numpy as np
def run(task):
 s,checkpoint,out,beta=task;p=Path(out)/(s['id']+'.npz');sha=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
 if p.exists():
  r=json.loads(p.with_suffix('.json').read_text());assert r['checkpoint_sha256']==sha and r['beta']==beta;return r
 from env import RobotEnv
 from expert import Expert
 from .optimized_policy import OptimizedPolicy,stage_features
 e=RobotEnv();o=e.reset(s);initial=o.copy();student=OptimizedPolicy(checkpoint,e.dt,env=e);expert=Expert(e);assert student.hybrid
 rows={k:[] for k in ['x','y','reference','error','phase','obs','full_action']};start=time.perf_counter()
 while True:
  predicted=student.act(o);label,aux=expert.label()
  if e.k%5==0:
   x=stage_features(o,initial,student.base.phase,student.base.reference_time)
   values=dict(x=x,y=label[4:],reference=aux['q_des']-initial[4:11],error=np.linalg.norm(o[18:21]-o[33:36]),phase=student.base.phase,obs=o,full_action=label)
   for k,v in values.items():rows[k].append(np.copy(v))
  action=np.r_[predicted[:4],beta*label[4:]+(1-beta)*predicted[4:]]
  o,info=e.step(action)
  if info['done']:break
 p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**{k:np.asarray(v,dtype='float32') for k,v in rows.items()})
 r=dict(scenario=s,checkpoint=checkpoint,checkpoint_sha256=sha,beta=beta,base='frozen learned navigation, no expert base action mixing',success=info['success'],final_error_m=info['error'],steps=e.k,samples=len(rows['x']),wall_s=time.perf_counter()-start)
 p.with_suffix('.json').write_text(json.dumps(r,indent=2));e.close();return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',default='data/mixed_arm');p.add_argument('--beta',type=float,default=.6);p.add_argument('--workers',type=int,default=12);a=p.parse_args();scenarios=json.loads(Path('configs/train.json').read_text())[::4]
 assert all(s['split']=='train' for s in scenarios)
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
  rows=[]
  for r in pool.map(run,[(s,a.checkpoint,a.out,a.beta) for s in scenarios]):
   rows.append(r);print(json.dumps(dict(id=r['scenario']['id'],success=r['success'],done=len(rows))),flush=True)
 summary=dict(n=len(rows),successes=sum(r['success'] for r in rows),samples=sum(r['samples'] for r in rows),beta=a.beta,scope='training assisted data; not student test success')
 Path(a.out).with_suffix('.summary.json').write_text(json.dumps(summary,indent=2));print('MIXED_ARM_DONE',json.dumps(summary),flush=True)

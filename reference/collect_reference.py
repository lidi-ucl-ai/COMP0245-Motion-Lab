"""Replay frozen latest expert scenarios and record explicit velocity/acceleration labels."""
import argparse,concurrent.futures,json,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def run(task):
 scenario,out,student,stride=task;out=Path(out);p=out/(scenario['id']+'.npz')
 if p.exists():return json.loads(p.with_suffix('.json').read_text())
 from env import RobotEnv
 from expert import Expert
 from reference.reference_policy import features,ReferencePolicy
 e=RobotEnv();o=e.reset(scenario);initial=o.copy();ex=Expert(e)
 policy=ReferencePolicy(student,e) if student else None
 previous_q=o[4:11].copy();previous_v=np.zeros(7);previous_w=np.zeros(4)
 rows={k:[] for k in ['x_rel','x_context','y','obs','action','acceleration','phase']};start=time.perf_counter();errors=[]
 while True:
  if policy is not None:
   dq=policy.decoder.qdes;prevq=o[4:11] if dq is None else dq
   ctx=(prevq,policy.decoder.velocity,policy.decoder.wheels)
  else:ctx=(previous_q,previous_v,previous_w)
  if e.k%stride==0:
   rel=features(o,initial,*ctx,'rel');context=features(o,initial,*ctx,'context')
  a,aux=ex.label();v=ex.servo.velocity.copy();acc=(v-previous_v)/e.dt
  y=np.r_[a[1],a[0],aux['q_des']-o[4:11],v]
  if e.k%stride==0:
   for k,value in [('x_rel',rel),('x_context',context),('y',y),('obs',o),('action',a),('acceleration',acc),('phase',aux['phase'])]:rows[k].append(np.copy(value))
  previous_q=aux['q_des'].copy();previous_v=v;previous_w=a[:4].copy()
  executed=policy.act(o) if policy else a
  o,info=e.step(executed);errors.append(info['error'])
  if info['done']:break
 out.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(p,**{k:np.asarray(v,dtype='float32') for k,v in rows.items()})
 record=dict(scenario=scenario,success=info['success'],reason=info['reason'],steps=e.k,samples=len(rows['y']),final_error=info['error'],min_error=min(errors),seconds=e.k*e.dt,wall_s=time.perf_counter()-start,student=student,stride=stride)
 p.with_suffix('.json').write_text(json.dumps(record,indent=2));e.close();return record

def main():
 p=argparse.ArgumentParser();p.add_argument('--scenarios',required=True);p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=12);p.add_argument('--student');p.add_argument('--stride',type=int,default=10);p.add_argument('--limit',type=int)
 a=p.parse_args();scenarios=json.loads(Path(a.scenarios).read_text());scenarios=scenarios[:a.limit] if a.limit else scenarios
 if a.student:assert all(s['split'] in ('train','extra_train') for s in scenarios)
 results=[];start=time.perf_counter()
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
  fs=[pool.submit(run,(s,a.out,a.student,a.stride)) for s in scenarios]
  for f in concurrent.futures.as_completed(fs):
   r=f.result();results.append(r);print(json.dumps(dict(id=r['scenario']['id'],success=r['success'],samples=r['samples'],done=len(results),total=len(scenarios))),flush=True)
 summary=dict(episodes=len(results),successes=sum(r['success'] for r in results),frames=sum(r['steps'] for r in results),samples=sum(r['samples'] for r in results),wall_s=time.perf_counter()-start,groups={g:dict(n=sum(r['scenario']['group']==g for r in results),successes=sum(r['success'] and r['scenario']['group']==g for r in results)) for g in set(s['group'] for s in scenarios)})
 Path(a.out).with_suffix('.summary.json').write_text(json.dumps(summary,indent=2));print('DONE',json.dumps(summary),flush=True)
if __name__=='__main__':main()

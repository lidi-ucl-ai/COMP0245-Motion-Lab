from pathlib import Path
import argparse,concurrent.futures,json,hashlib,time,sys,importlib.abc,numpy as np,csv
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'expert','expert_planner','motion_control','markov_teacher','physical_teacher','reference_policy','student_policy','jacobian_model','robotnik_arm_control'}:raise RuntimeError('Expert/classical arm controller forbidden: '+fullname)
def run(task, policy_class=None):
 s,checkpoint,out=task;p=Path(out)/(s['id']+'.json');sha=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
 if p.exists():
  r=json.loads(p.read_text());assert r['checkpoint_sha256']==sha and r['scenario']==s;return r
 sys.meta_path.insert(0,Block())
 from env import RobotEnv
 from .optimized_policy import OptimizedPolicy
 e=RobotEnv();o=e.reset(s);policy=(policy_class or OptimizedPolicy)(checkpoint,e.dt,env=e)
 import pinocchio as pin
 def forbidden(*a,**k):raise RuntimeError('Analytic arm control API forbidden')
 for n in ['computeFrameJacobian','computeJointJacobians','getFrameJacobian','crba']:setattr(pin,n,forbidden)
 original_gravity=pin.computeGeneralizedGravity
 if not policy.reference_mode:pin.computeGeneralizedGravity=forbidden
 trace=[];mini=float('inf');exc=None;start=time.perf_counter()
 try:
  while True:
   a=policy.act(o);o,info=e.step(a);mini=min(mini,info['error'])
   if e.k%10==0:trace.append(np.r_[e.k*e.dt,info['error'],e.state['ee'],e.state['pos'],a,policy.phase,e.state['q'][4:11],e.state['qd'][4:11]])
   if info['done']:break
 except Exception as error:
  exc=str(error);info=dict(success=False,reason='exception:'+exc,error=float(np.linalg.norm(e.target-e.state['ee'])))
 r=dict(id=s['id'],scenario=s,success=info['success'],reason=info['reason'],sim_seconds=e.k*e.dt,final_error_m=info['error'],min_error_m=mini,base_goal_error_m=float(np.linalg.norm(e.base_goal-e.state['pos'][:2])),final_joint_speed_norm=float(np.linalg.norm(e.state['qd'][4:11])),checkpoint=checkpoint,checkpoint_sha256=sha,hybrid_learning_base=policy.hybrid,arm_control='learned_reference_PD_gravity' if policy.reference_mode else 'neural_torque',gravity_compensation=policy.reference_mode,gravity_calls=policy.gravity_calls,additional_damping_gain=policy.post.get('damping_gain',0.) if policy.hybrid else 0.,exception=exc,expert_modules_loaded=sorted(set(sys.modules)&{'expert','expert_planner','motion_control','reference_policy','jacobian_model'}),analytic_J_mass_APIs_blocked=True,wall_s=time.perf_counter()-start,policy_median_ms=float(np.median(policy.latencies)))
 pin.computeGeneralizedGravity=original_gravity
 assert not r['expert_modules_loaded'];p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r,indent=2));np.savez_compressed(p.with_suffix('.npz'),trace=np.asarray(trace,dtype='float32'));e.close();return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scenarios',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=12);a=p.parse_args();rows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
  fs=[pool.submit(run,(s,a.checkpoint,a.out)) for s in json.loads(Path(a.scenarios).read_text())]
  for f in concurrent.futures.as_completed(fs):
   r=f.result();rows.append(r);print(json.dumps({k:r[k] for k in ['id','success','sim_seconds','final_error_m','base_goal_error_m','exception']}),flush=True)
 summary=dict(checkpoint=a.checkpoint,n=len(rows),successes=sum(r['success'] for r in rows),mean_capped_seconds=float(np.mean([r['sim_seconds'] for r in rows])),mean_error_m=float(np.mean([r['final_error_m'] for r in rows])),mean_base_error_m=float(np.mean([r['base_goal_error_m'] for r in rows])),exceptions=sum(r['exception'] is not None for r in rows))
 (Path(a.out)/'summary.json').write_text(json.dumps(summary,indent=2));print('EVAL_DONE',json.dumps(summary),flush=True)

"""Training-only analytic labels for a separate neural force component."""
from pathlib import Path
import argparse,concurrent.futures,json,numpy as np,time

def label(task):
 idx,qs,physical=task
 from env import RobotEnv
 import pinocchio as pin
 from .learned_gravity import body_gravity
 e=RobotEnv()
 if physical:
  from .physical_teacher import align_missing_inertias
  align_missing_inertias(e)
 motor=e.default_q.copy();motor[:4]=0;motor[11:]=.1;coeff=[];original=e.dyn.pin_model.gravity.linear.copy();maxerr=0.
 for i,q in enumerate(qs):
  motor[4:11]=q;qp=e.dyn.ReoderJoints2PinVec(np.r_[0.,0.,.4,0.,0.,0.,1.,motor],'pos');c=np.empty((7,3))
  for axis in range(3):
   g=np.zeros(3);g[axis]=9.81;e.dyn.pin_model.gravity.linear=g
   c[:,axis]=pin.computeGeneralizedGravity(e.dyn.pin_model,e.dyn.pin_data,qp)[6:13]
  if i<20:
   quat=e.client.getQuaternionFromEuler([.11,-.07,.3]);o=np.zeros(65);o[21:25]=quat
   g=body_gravity(o).astype(float);e.dyn.pin_model.gravity.linear=original
   qp2=e.dyn.ReoderJoints2PinVec(np.r_[0.,0.,.4,quat,motor],'pos');actual=pin.computeGeneralizedGravity(e.dyn.pin_model,e.dyn.pin_data,qp2)[6:13]
   err=float(np.max(np.abs(c@g-actual)));maxerr=max(maxerr,err);assert err<1e-5,err
  coeff.append(c.reshape(-1))
 e.dyn.pin_model.gravity.linear=original;e.close();return idx,np.asarray(coeff,dtype='float32'),maxerr

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=200000);p.add_argument('--workers',type=int,default=4);p.add_argument('--physical',action='store_true');p.add_argument('--out',default='data/gravity');p.add_argument('--extra-folders',nargs='*',default=[]);a=p.parse_args();start=time.perf_counter()
 qs=[]
 for folder in ['original_train','mixed_arm','hold_mix','hold_lowmix','hold_pure',*a.extra_folders]:
  for path in sorted(Path('data',folder).glob('*.npz')):
   with np.load(path) as d:qs.append(d['obs'][::20,4:11])
 poses=np.concatenate(qs);rng=np.random.default_rng(20261102970);q=poses[rng.integers(0,len(poses),a.n)].copy();q+=rng.normal(0,.04,q.shape)
 low=poses.min(0)-.05;high=poses.max(0)+.05
 # Interpolate training pose bounds only; never use validation or test poses.
 count=a.n//5;q[:count]=rng.uniform(low,high,(count,7))
 chunks=np.array_split(q,a.workers);results=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
  for result in pool.map(label,[(i,q,a.physical) for i,q in enumerate(chunks)]):results.append(result);print('LABEL_PART_DONE',result[0],flush=True)
 y=np.concatenate([r[1] for r in sorted(results)]);x=np.r_[np.sin(q).T,np.cos(q).T].T.astype('float32')
 out=Path(a.out);out.mkdir(exist_ok=True);np.savez_compressed(out/'coefficients.npz',x=x,y=y,q=q)
 summary=dict(n=a.n,source_pose_samples=len(poses),max_linearity_error_nm=max(r[2] for r in results),seed=20261102970,scope='only training pose statistics and analytic offline labels',wall_s=time.perf_counter()-start)
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print('GRAVITY_DATA_DONE',json.dumps(summary),flush=True)

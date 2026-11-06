from pathlib import Path
import argparse,json,time
import numpy as np,torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesRegressor
from train_base_memory_original import pack
from base_memory import base_features,timing_features,direction_features,phase_features
from control_utils import standoff

def make_initial_labels(initial,cfg):
 rng=np.random.default_rng(20261030411);n=110000;o=initial[rng.integers(len(initial),size=n)].copy();old_yaw=o[:,3].copy();local=o[:,33:36]-o[:,:3];co=np.cos(old_yaw);si=np.sin(old_yaw);local_xy=np.c_[co*local[:,0]+si*local[:,1],-si*local[:,0]+co*local[:,1]]
 o[:,:2]=rng.uniform(-.3,.3,(n,2));o[:,3]=rng.uniform(-np.pi,np.pi,n);co=np.cos(o[:,3]);si=np.sin(o[:,3]);o[:,33]=o[:,0]+co*local_xy[:,0]-si*local_xy[:,1];o[:,34]=o[:,1]+si*local_xy[:,0]+co*local_xy[:,1]
 component=rng.integers(0,5,n);radius=np.where(component<2,rng.uniform(0,.85,n),np.where(component<4,rng.uniform(.85,1.65,n),rng.uniform(2,10,n)));theta=rng.uniform(-np.pi,np.pi,n);radius[::20]=0.;theta=np.where(radius<1e-4,0.,theta);o[:,18]=radius*np.cos(theta);o[:,19]=radius*np.sin(theta);o[:,20]=np.where(rng.random(n)<.5,rng.uniform(.2,1.05,n),rng.uniform(1.201,1.475,n))
 parking=standoff(o[:,20]);o[:,61]=o[:,18]-parking*np.cos(theta);o[:,62]=o[:,19]-parking*np.sin(theta);o[:,63]=np.sin(theta);o[:,64]=np.cos(theta);o[:,60]=0
 delta=o[:,61:63]-o[:,:2];bearing=np.arctan2(delta[:,1],delta[:,0]);yaw=o[:,3];goal=theta;wrap=lambda x:np.arctan2(np.sin(x),np.cos(x))
 cf=abs(wrap(bearing-yaw))+abs(wrap(goal-bearing));cr=abs(wrap(bearing+np.pi-yaw))+abs(wrap(goal-bearing-np.pi));labels=np.where(cr+.15<cf,4,0);T=np.maximum(1.5,np.maximum(np.linalg.norm(delta,axis=1)/cfg['base_vmax']+np.minimum(cf,cr)/cfg['yaw_max']+1.,np.linalg.norm(o[:,18:21]-o[:,33:36],axis=1)/(.3+cfg['cart_vmax'])))
 return direction_features(o),timing_features(o),labels,T

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();start=time.perf_counter();xs=[];prev=[];phases=[];ys=[];initial=[]
 for path in sorted(Path('data/train').glob('*.npz')):
  meta=json.loads(path.with_suffix('.json').read_text());assert meta['scenario']['split']=='train'
  if not meta['success']:continue # Preserve failed episodes on disk; do not imitate failed navigation loops.
  with np.load(path) as d:
   xs.append(phase_features(d['obs']));first=4 if int(d['direction'][0])<0 else 0;prev.append(np.r_[first,d['phase'][:-1]]);phases.append(d['phase']);ys.append(d['wheel_target']);initial.append(d['obs'][0].copy())
 X=np.concatenate(xs);previous=np.concatenate(prev);phase=np.concatenate(phases);Y=np.concatenate(ys);initial=np.array(initial);cfg=json.loads(Path('configs/control.json').read_text());dx,tx,dy,ty=make_initial_labels(initial,cfg)
 direction=DecisionTreeClassifier(max_depth=10,min_samples_leaf=2,class_weight='balanced',random_state=a.seed).fit(dx[:100000],dy[:100000]);duration=ExtraTreesRegressor(n_estimators=8,max_depth=14,min_samples_leaf=2,random_state=a.seed,n_jobs=2).fit(tx[:100000],ty[:100000]);packed=dict(initial_direction=pack(direction,True),duration_regression=[pack(t) for t in duration.estimators_],transition=[],regression=[],high_target_feature=True)
 for i in range(8):
  mask=previous==i
  if not mask.any():raise RuntimeError(f'Missing teacher navigation state {i}')
  t=DecisionTreeClassifier(max_depth=10,min_samples_leaf=1,class_weight='balanced',random_state=a.seed).fit(X[mask],phase[mask]);packed['transition'].append(pack(t,True));chosen=np.flatnonzero(phase==i)[::4]
  if not len(chosen):raise RuntimeError(f'Missing teacher phase output {i}')
  reg=ExtraTreesRegressor(n_estimators=8,max_depth=18,min_samples_leaf=2,random_state=a.seed,n_jobs=2).fit(X[chosen],Y[chosen]);packed['regression'].append([pack(t) for t in reg.estimators_])
 out=Path(f'checkpoints/navigation_s{a.seed}');out.mkdir(parents=True,exist_ok=True);torch.save({'base_memory':packed},out/'best.pt');summary=dict(seed=a.seed,episodes=len(initial),trajectory_frames=len(X),planning_labels_train=100000,planning_labels_validation=10000,direction_validation_accuracy=float((direction.predict(dx[100000:])==dy[100000:]).mean()),horizon_validation_mae_s=float(abs(duration.predict(tx[100000:])-ty[100000:]).mean()),seconds=time.perf_counter()-start,notes='Initial labels generated offline by training-only expert geometry, not invoked by student; wheel targets before shared slew limiting.')
 (out/'training.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

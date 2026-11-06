"""Independent learned navigation fit; no geometric planner at deployment."""
from pathlib import Path
import json,time,numpy as np,torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesRegressor
from train_base_memory_original import pack
from train_navigation import make_initial_labels
from base_memory import phase_features
start=time.perf_counter();xs=[];previous=[];phases=[];ys=[];initial=[];episode_count=0
for folder in ['data/train','data/extra_train']:
 for p in sorted(Path(folder).glob('*.npz')):
  r=json.loads(p.with_suffix('.json').read_text())
  if not r['success']:continue
  assert r['scenario']['split'] in ['train','extra_train']
  with np.load(p) as d:
   phase=d['phase'].astype(int);first=(int(phase[0])//4)*4
   xs.append(phase_features(d['obs']));previous.append(np.r_[first,phase[:-1]]);phases.append(phase);ys.append(d['action'][:,:4]);initial.append(d['obs'][0])
  episode_count+=1
X=np.concatenate(xs);previous=np.concatenate(previous);phase=np.concatenate(phases);Y=np.concatenate(ys);initial=np.array(initial);cfg=json.loads(Path('configs/control.json').read_text());dx,tx,dy,ty=make_initial_labels(initial,cfg)
direction=DecisionTreeClassifier(max_depth=10,min_samples_leaf=2,class_weight='balanced',random_state=0).fit(dx[:100000],dy[:100000]);duration=ExtraTreesRegressor(n_estimators=8,max_depth=14,min_samples_leaf=2,random_state=0,n_jobs=2).fit(tx[:100000],ty[:100000]);packed=dict(initial_direction=pack(direction,True),duration_regression=[pack(t) for t in duration.estimators_],transition=[],regression=[],high_target_feature=True)
for i in range(8):
 mask=previous==i;assert mask.any()
 t=DecisionTreeClassifier(max_depth=10,min_samples_leaf=1,class_weight='balanced',random_state=0).fit(X[mask],phase[mask]);packed['transition'].append(pack(t,True));chosen=np.flatnonzero(phase==i)[::2]
 reg=ExtraTreesRegressor(n_estimators=8,max_depth=18,min_samples_leaf=2,random_state=0,n_jobs=2).fit(X[chosen],Y[chosen]);packed['regression'].append([pack(t) for t in reg.estimators_])
out=Path('runs/navigation_reference');out.mkdir(parents=True,exist_ok=True);torch.save(dict(base_memory=packed),out/'best.pt')
summary=dict(episodes=episode_count,trajectory_samples=len(X),synthetic_initial_train=100000,synthetic_initial_validation=10000,direction_validation_accuracy=float((direction.predict(dx[100000:])==dy[100000:]).mean()),duration_validation_mae_s=float(abs(duration.predict(tx[100000:])-ty[100000:]).mean()),wall_s=time.perf_counter()-start)
(out/'metrics.json').write_text(json.dumps(summary,indent=2))
for name in ['F_rel','F_no_history_extreme','F_context_extreme']:
 p=Path('runs')/name/'model_best.pt'
 if not p.exists():continue
 d=torch.load(p,map_location='cpu',weights_only=False);d['base_memory']=packed;d['navigation']='independently_trained_reference_trees'
 output=Path('runs')/(name+'_treebase');output.mkdir(exist_ok=True);torch.save(d,output/'model_best.pt')
print(json.dumps(summary),flush=True)

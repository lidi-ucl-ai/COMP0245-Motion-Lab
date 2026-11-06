"""Fit navigation memory to training-only labels and attach it to an explicit hybrid."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesRegressor
from base_memory import base_features

def pack(model,classifier=False):
    t=model.tree_;d=dict(left=t.children_left,right=t.children_right,feature=t.feature,threshold=t.threshold,
        value=t.value[:,0,:] if classifier else t.value[:,:,0])
    if classifier:d['classes']=model.classes_
    return d

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='checkpoints/ik_hybrid_s0/best.pt');p.add_argument('--name',default='memory_ik_hybrid_s0');p.add_argument('--seed',type=int,default=0);a=p.parse_args()
    start=time.perf_counter();xs=[];prev=[];labels=[];ys=[]
    for path in sorted(Path('data/train').glob('*.npz')):
        meta=json.loads(path.with_suffix('.json').read_text());assert meta['scenario']['split']=='train'
        with np.load(path) as d:
            xs.append(base_features(d['obs']));prev.append(np.r_[0,d['phase'][:-1]]);labels.append(d['phase']);ys.append(d['action'][:,:4])
    X=np.concatenate(xs);previous=np.concatenate(prev);phase=np.concatenate(labels);Y=np.concatenate(ys)
    packed=dict(transition=[],regression=[]);nodes=0
    for i in range(4):
        mask=previous==i
        tree=DecisionTreeClassifier(max_depth=8,min_samples_leaf=1,class_weight='balanced',random_state=a.seed)
        tree.fit(X[mask],phase[mask]);packed['transition'].append(pack(tree,True));nodes+=tree.tree_.node_count
        chosen=np.flatnonzero(phase==i)[::4]
        reg=ExtraTreesRegressor(n_estimators=8,max_depth=14,min_samples_leaf=2,random_state=a.seed,n_jobs=2)
        reg.fit(X[chosen],Y[chosen]);packed['regression'].append([pack(t) for t in reg.estimators_]);nodes+=sum(t.tree_.node_count for t in reg.estimators_)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    state['base_memory']=packed
    out=Path('checkpoints')/a.name;out.mkdir(exist_ok=True)
    torch.save(state,out/'best.pt')
    summary=json.loads(Path(a.checkpoint).with_name('summary.json').read_text());summary.update(name=out.name,
        base_memory_tree_nodes=nodes,base_memory_training_seconds=time.perf_counter()-start,
        checkpoint_bytes=(out/'best.pt').stat().st_size,
        control_prior=('learned navigation transitions/wheels + learned error-to-velocity matrix + own reference + PD/gravity/limits' if state['config']['kind']=='ik_gru' else 'none; learned navigation state memory and learned direct torques'))
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()

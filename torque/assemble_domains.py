from pathlib import Path
import argparse,json,hashlib,torch,numpy as np
from sklearn.tree import DecisionTreeClassifier
def strip_navigation(d):
 d={k:v for k,v in d.items() if k not in ['navigation','navigation_path','navigation_file_sha256']}
 if d['family']=='domain_neural_policy':d['members']=[strip_navigation(m) for m in d['members']]
 return d
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--previous',required=True);p.add_argument('--specialist',required=True);p.add_argument('--stage',choices=['high','far'],required=True);p.add_argument('--out',required=True);a=p.parse_args();previous=torch.load(a.previous,map_location='cpu',weights_only=False);specialist=torch.load(a.specialist,map_location='cpu',weights_only=False)
 oldfiles=['configs/train.json','configs/near_train_augmented.json']+(['configs/high_train_augmented.json'] if a.stage=='far' else []);newfile=f'configs/{a.stage}_train_augmented.json';X=[];Y=[]
 for label,files in [(0,oldfiles),(1,[newfile])]:
  for file in files:
   for s in json.loads(Path(file).read_text()):X.append([np.hypot(*s['target'][:2]),s['target'][2]]);Y.append(label)
 if a.stage=='high':
  z=np.float32(1.05);X.extend([[.5,float(z)],[.5,float(np.nextafter(z,np.float32(np.inf)))]]);Y.extend([0,1])
 else:
  r=np.float32(1.35);z=np.float32(1.05)
  X.extend([[float(r),.6],[float(np.nextafter(r,np.float32(np.inf))),.6],[1.5,float(z)],[1.5,float(np.nextafter(z,np.float32(-np.inf)))] ]);Y.extend([0,1,0,1])
 tree=DecisionTreeClassifier(max_depth=4,min_samples_leaf=1,random_state=261102999).fit(np.asarray(X,dtype='float32'),Y);t=tree.tree_;packed=dict(left=t.children_left.copy(),right=t.children_right.copy(),feature=t.feature.copy(),threshold=t.threshold.copy(),value=t.value[:,0,:].copy(),classes=tree.classes_.copy())
 nav=previous.get('navigation')
 if nav is None:nav=torch.load(previous['navigation_path'],map_location='cpu',weights_only=False)['base_memory']
 model=dict(family='domain_neural_policy',selector=packed,member_names=['frozen_previous_domains',a.stage+'_specialist'],members=[strip_navigation(previous),strip_navigation(specialist)],navigation=nav,selection_training=dict(feature_names=['goal_radius','goal_height'],source_files=oldfiles+[newfile],samples=len(X),training_accuracy=float(tree.score(np.asarray(X,dtype='float32'),Y)),boundary_examples='synthetic domain-category boundary samples; no force/trajectory labels',labels='training target categories, not reinforcement-learned best-policy labels'),sources=dict(previous=a.previous,specialist=a.specialist,previous_sha256=hashlib.sha256(Path(a.previous).read_bytes()).hexdigest(),specialist_sha256=hashlib.sha256(Path(a.specialist).read_bytes()).hexdigest()))
 out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);torch.save(model,out);print('DOMAIN_ASSEMBLED',json.dumps(dict(out=str(out),nodes=t.node_count,training_accuracy=model['selection_training']['training_accuracy'])),flush=True)

"""Offline kinematic supervision. No test scenes or expert action queries are read."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import pinocchio as pin
from env import RobotEnv

def joints(directory):
    rows=[];local=[]
    for p in sorted(Path(directory).glob('*.npz')):
        with np.load(p) as a:
            obs=a['obs'];phase=a['phase'];active=a['reach_active'] if 'reach_active' in a else phase==3
            rows.extend([obs[active,4:11][::10],obs[~active,4:11][::100]])
            near=(obs[:,20]>1.2)&(np.linalg.norm(obs[:,39:42],axis=1)<.15)
            if near.any():local.append(obs[near,4:11][::5])
    return np.concatenate(rows),np.concatenate(local)

def main():
    start=time.time();rng=np.random.default_rng(82424)
    e=RobotEnv();e.reset(json.loads(Path('configs/train_scenarios.json').read_text())[0])
    lower=e.lower[4:11];upper=e.upper[4:11];limited=lower<upper
    output={};counts={}
    for split,n,source in [('train',360000,'data/train'),('validation',12000,'data/validation')]:
        pool,local=joints(source);half=n//2;q=np.concatenate([pool[rng.integers(len(pool),size=half)],local[rng.integers(len(local),size=n-half)]]).copy()
        scale=np.concatenate([rng.choice([.0,.03,.15,.4,1.0],size=(half,1),p=[.1,.15,.4,.25,.1]),rng.choice([0.,.003,.01,.03,.08],size=(n-half,1))])
        q+=rng.normal(size=q.shape)*scale
        q[:,limited]=np.clip(q[:,limited],lower[limited]+.05,upper[limited]-.05)
        labels=np.empty((n,3,7),dtype='float32')
        full=e.state['q'].copy()
        for i,x in enumerate(q):
            full[4:11]=x
            qp=e.dyn.ReoderJoints2PinVec(np.r_[np.zeros(3),[0,0,0,1],full],'pos')
            labels[i]=pin.computeFrameJacobian(e.dyn.pin_model,e.dyn.pin_data,qp,e.ee_id,pin.LOCAL_WORLD_ALIGNED)[:3,6:13]
        output[split+'_q']=q.astype('float32');output[split+'_j']=labels
        output[split+'_local_high']=np.arange(n)>=half;counts[split]=dict(rows=n,source_frames=len(pool),local_high_pool=len(local),local_high_rows=n-half);print(split,counts[split],flush=True)
    e.close();p=Path('data/jacobian_supervision.npz');np.savez_compressed(p,**output)
    record=dict(counts=counts,kinematic_queries=372000,expert_action_queries=0,seed=82424,wall_seconds=time.time()-start,
        file_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source='New expert train / validation episodes separate; sample actual reaching flag; base identity; m/rad',
        augmentation='Half broad q perturbations, half near-high-target states with sigma0/.003/.01/.03/.08; clipped at limits+.05; no test scenes')
    Path('data/jacobian_supervision.json').write_text(json.dumps(record,indent=2));print(record)
if __name__=='__main__':main()

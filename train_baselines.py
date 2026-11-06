"""Refit early architectures on later expert episodes; no test labels or test selection."""
from pathlib import Path
import argparse,json,hashlib,sys,time,random
import numpy as np
import torch
R = Path(__file__).resolve().parent
from models import Regressor,geometric_features

def load(folder):
 obs=[];act=[];phase=[];ref=[];velocity=[];episodes=[];offset=0
 for p in sorted(Path(folder).glob('*.npz')):
  meta=json.loads(p.with_suffix('.json').read_text())
  if not meta['success']:continue
  d=np.load(p);o=d['obs'];a=d['action'];qdes=d['q_des'];v=np.vstack([np.zeros((1,7)),np.diff(qdes,axis=0)/.001]);v=np.clip(v,-1,1).astype('float32');ph=(d['phase']%4).astype('int64')
  obs.append(o);act.append(a);phase.append(ph);ref.append((qdes-o[:,42:49]).astype('float32'));velocity.append(v)
  episodes.append(dict(id=meta['scenario']['id'],target=meta['scenario']['target'],start=offset,stop=offset+len(o),sha256=hashlib.sha256(p.read_bytes()).hexdigest()));offset+=len(o)
 return dict(obs=np.concatenate(obs),action=np.concatenate(act),phase=np.concatenate(phase),reference=np.concatenate(ref),velocity=np.concatenate(velocity),episodes=episodes)

def train(kind,data,val,plan,out):
 torch.manual_seed(0);np.random.seed(0);random.seed(0);rng=np.random.default_rng(0)
 folder=out/kind;folder.mkdir(parents=True,exist_ok=True)
 if (folder/'COMPLETE.json').exists():return
 geometry=kind=='memory_ik';temporal=kind in ['pgru','memory_ik'];netkind='ik_gru' if geometry else ('phase_gru' if temporal else 'mlp')
 x=geometric_features(data['obs']) if geometry else data['obs'];xv=geometric_features(val['obs']) if geometry else val['obs']
 indices=np.arange(21 if kind=='m1' else x.shape[1]);x=x[:,indices];xv=xv[:,indices]
 y=data['action'].copy();yv=val['action'].copy()
 if geometry:y[:,4:]=data['velocity'];yv[:,4:]=val['velocity']
 xm=x.mean(0);xs=np.maximum(x.std(0),1e-3);ym=y.mean(0);ys=np.maximum(y.std(0),np.r_[[.1]*4,[.1 if geometry else .5]*7]).astype('float32')
 rm=data['reference'].mean(0);rs=np.maximum(data['reference'].std(0),.02)
 cfg=dict(kind=netkind,input_dim=len(indices),width=128)
 if geometry:cfg.update(error_mean=xm[39:42].tolist(),error_std=xs[39:42].tolist(),arm_mean=ym[4:].tolist(),arm_std=ys[4:].tolist())
 net=Regressor(**cfg).cuda();opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-5)
 X=torch.tensor((x-xm)/xs,device='cuda');Y=torch.tensor((y-ym)/ys,device='cuda');P=torch.tensor(data['phase'],device='cuda');Q=torch.tensor((data['reference']-rm)/rs,device='cuda')
 XV=torch.tensor((xv-xm)/xs,device='cuda');YV=torch.tensor((yv-ym)/ys,device='cuda')
 budget=plan['memory_ik' if geometry else ('pgru' if temporal else 'mlp')];best=float('inf');history=[];start=time.perf_counter()
 starts=np.concatenate([np.arange(e['start'],e['stop']-128+1,16,dtype='int64') for e in data['episodes'] if e['stop']-e['start']>=128]) if temporal else None
 val_starts=np.concatenate([np.arange(e['start'],e['stop']-128+1,128,dtype='int64') for e in val['episodes'] if e['stop']-e['start']>=128]) if temporal else None
 if temporal:
  phase_pool=[starts[data['phase'][starts+96]==p] for p in range(4)]
  if not all(len(p)>0 for p in phase_pool) or not len(val_starts):
   raise ValueError('Recurrent training needs 128-step windows in every training phase and in validation')
  val_starts=rng.choice(val_starts,min(512,len(val_starts)),replace=False)
 validation_indices=rng.choice(len(xv),min(32000,len(xv)),replace=False)
 navigation=torch.load(plan['navigation_checkpoint'],map_location='cpu',weights_only=False)['base_memory'] if geometry else None
 def save(name,epoch,loss):
  d=dict(config=cfg,indices=indices.tolist(),x_mean=xm,x_std=xs,y_mean=ym,y_std=ys,state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},seed=0,epoch=epoch,validation_loss=loss,geometry=geometry,pd_prior=False,training_args=dict(model=kind,**budget),data_files=[e['id'] for e in data['episodes']],training_manifest_sha256=hashlib.sha256(json.dumps(data['episodes'],sort_keys=True).encode()).hexdigest())
  if geometry:d['base_memory']=navigation
  torch.save(d,folder/name)
 for epoch in range(1,budget['epochs']+1):
  net.train();total=0.
  for _ in range(budget['updates']):
   if temporal:
    chosen=np.array([rng.choice(phase_pool[int(k)]) for k in rng.integers(0,4,budget['batch'])]);ids=torch.tensor(chosen[:,None]+np.arange(128)[None,:],device='cuda');pred,ph,qr,_=net(X[ids]);sl=slice(32,None);pred=pred[:,sl];target=Y[ids][:,sl]
   else:
    ids=torch.randint(len(X),(budget['batch'],),device='cuda');pred,ph,qr,_=net(X[ids]);target=Y[ids]
   loss=.5*((pred[...,:4]-target[...,:4]).square().mean()+(pred[...,4:]-target[...,4:]).square().mean())
   if kind=='pgru':loss+=.1*torch.nn.functional.cross_entropy(ph[:,32:].reshape(-1,4),P[ids][:,32:].reshape(-1))+.05*(qr[:,32:]-Q[ids][:,32:]).square().mean()
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step();total+=float(loss.detach())
  net.eval();losses=[]
  with torch.no_grad():
   if temporal:
    for chunk in np.array_split(val_starts,min(16,len(val_starts))):
     ids=torch.tensor(chunk[:,None]+np.arange(128)[None,:],device='cuda');pred=net(XV[ids])[0][:,32:];target=YV[ids][:,32:];losses.append(float(.5*((pred[...,:4]-target[...,:4]).square().mean()+(pred[...,4:]-target[...,4:]).square().mean())))
   else:
    for chunk in np.array_split(validation_indices,min(8,len(validation_indices))):
     ids=torch.tensor(chunk,device='cuda');pred=net(XV[ids])[0];target=YV[ids];losses.append(float(.5*((pred[...,:4]-target[...,:4]).square().mean()+(pred[...,4:]-target[...,4:]).square().mean())))
  vl=float(np.mean(losses));history.append(dict(epoch=epoch,training=total/budget['updates'],validation=vl))
  if vl<best:best=vl;save('best.pt',epoch,vl)
  if epoch%5==0:print(json.dumps(dict(model=kind,**history[-1])),flush=True)
 save('last.pt',epoch,vl)
 metrics=dict(model=kind,architecture=cfg,budget=budget,seed=0,training_episodes=len(data['episodes']),training_frames=len(x),validation_episodes=len(val['episodes']),validation_frames=len(xv),selection='Minimum fixed validation action loss; no test access',training_time_s=time.perf_counter()-start,history=history,train_manifest=data['episodes'],validation_manifest=val['episodes'])
 (folder/'metrics.json').write_text(json.dumps(metrics,indent=2));(folder/'COMPLETE.json').write_text(json.dumps(dict(best=best,sha256=hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest())))
 print('TRAIN_COMPLETE',kind,flush=True)
 del X,Y,P,Q,XV,YV,net,opt;torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("m1", "m2", "pgru", "memory_ik"), required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=R / "runs/baseline_refits")
    parser.add_argument("--navigation-checkpoint", type=Path)
    args = parser.parse_args()
    if args.model == "memory_ik" and args.navigation_checkpoint is None:
        parser.error("Memory-IK requires --navigation-checkpoint")
    if not torch.cuda.is_available():
        parser.error("The original baseline fitter requires CUDA-enabled PyTorch")
    torch.set_num_threads(1)
    plan = json.loads((R / "configs/baseline_train_plan.json").read_text())
    plan["navigation_checkpoint"] = args.navigation_checkpoint
    data, val = load(args.train), load(args.validation)
    train_ids = {e["id"] for e in data["episodes"]}
    val_ids = {e["id"] for e in val["episodes"]}
    if train_ids & val_ids:
        parser.error("Training and validation episode IDs overlap")
    test = json.loads((R / "configs/test256.json").read_text())
    fit_targets = {tuple(e["target"]) for e in data["episodes"] + val["episodes"]}
    if fit_targets & {tuple(s["target"]) for s in test}:
        parser.error("Training or validation targets overlap with the report benchmark")
    train(args.model, data, val, plan, args.out)


if __name__ == "__main__":
    main()

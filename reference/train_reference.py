"""Original F_extreme MLP and weighted BC loss, GPU adaptation; splits by episode."""
import argparse,json,time,random,hashlib
from pathlib import Path
import numpy as np
import torch
from .original_model import MLP

def load(folder,kind,successful_only):
 xs=[];ys=[];ws=[];records=[]
 for p in sorted(Path(folder).glob('*.npz')):
  r=json.loads(p.with_suffix('.json').read_text())
  if (successful_only or r.get('student') is None) and not r['success']:continue
  with np.load(p) as d:
   x=d['x_context' if kind=='context_no_history' else 'x_'+kind]
   if kind=='context_no_history':x=np.column_stack([x[:,:63],x[:,79:81]])
   y=d['y'];o=d['obs'];err=np.linalg.norm(o[:,18:21]-o[:,33:36],axis=1)
   w=1.+10.*(err<.05)
   xs.append(x);ys.append(y);ws.append(w.astype('float32'))
  records.append(dict(id=r['scenario']['id'],n=len(x),group=r['scenario']['group']))
 return np.concatenate(xs),np.concatenate(ys),np.concatenate(ws),records

def main():
 p=argparse.ArgumentParser();p.add_argument('--features',choices=['rel','context','context_no_history'],default='rel');p.add_argument('--out',required=True);p.add_argument('--epochs',type=int,default=150);p.add_argument('--layers',type=int,default=5);p.add_argument('--hidden',type=int,default=256);p.add_argument('--seed',type=int,default=0);p.add_argument('--extra',nargs='*',default=[]);p.add_argument('--init');p.add_argument('--patience',type=int,default=25)
 a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
 device='cuda' if torch.cuda.is_available() else 'cpu';out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 x,y,w,records=load('data/train',a.features,True)
 for folder in a.extra:
  xx,yy,ww,rr=load(folder,a.features,False);x=np.concatenate([x,xx]);y=np.concatenate([y,yy]);w=np.concatenate([w,ww]);records+=rr
 xv,yv,wv,val_records=load('data/validation',a.features,False)
 assert not {r['id'] for r in records}&{r['id'] for r in val_records}
 net=MLP(x.shape[1],16,a.hidden,a.layers,'silu',0.)
 if a.init:net.load_state_dict(torch.load(a.init,map_location='cpu',weights_only=False)['state_dict'])
 else:
  ys=np.maximum(y.std(0),np.r_[.01,.01,np.full(7,.001),np.full(7,.01)])
  net.set_normalisation(x.mean(0),np.maximum(x.std(0),1e-3),y.mean(0),ys)
 net=net.to(device);X=torch.tensor(x,device=device);Y=torch.tensor(y,device=device);W=torch.tensor(w,device=device)
 XV=torch.tensor(xv,device=device);YV=torch.tensor(yv,device=device)
 opt=torch.optim.AdamW(net.parameters(),lr=.001 if not a.init else .0003,weight_decay=.0001)
 sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=8)
 control=json.loads(Path('configs/control.json').read_text());best=float('inf');bad=0;hist=[];start=time.perf_counter()
 def save(path,epoch):torch.save(dict(family='reference_bc',state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},in_dim=x.shape[1],out_dim=16,config=vars(a),control=control,epoch=epoch,mode='integrate',anchor=0.),path)
 for ep in range(1,a.epochs+1):
  net.train();perm=torch.randperm(len(X),device=device);total=0.
  for b in perm.split(2048):
   pred=net.forward_normalised(X[b]);target=(Y[b]-net.y_mean)/net.y_std
   per=(pred-target).square().mean(1);loss=(W[b]*per).sum()/W[b].sum()
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step();total+=float(loss.detach())*len(b)
  net.eval()
  with torch.no_grad():
   val=sum(float(((net(z)-yy)/net.y_std).square().mean())*len(z) for z,yy in zip(XV.split(8192),YV.split(8192)))/len(XV)
  hist.append(dict(epoch=ep,train=total/len(X),validation=val));sched.step(val)
  if val<best:best=val;bad=0;save(out/'model_best.pt',ep)
  else:bad+=1
  if ep%25==0:save(out/f'epoch_{ep:03d}.pt',ep)
  if ep%5==0:print(json.dumps(hist[-1]),flush=True)
  if bad>=a.patience:break
 result=dict(config=vars(a),device=device,input_dim=x.shape[1],output_dim=16,parameters=sum(t.numel() for t in net.parameters()),train_samples=len(x),validation_samples=len(xv),train_episodes=len(records),validation_episodes=len(val_records),best_validation_loss=best,wall_s=time.perf_counter()-start,history=hist,train_records=records,validation_records=val_records)
 (out/'metrics.json').write_text(json.dumps(result,indent=2));print('TRAIN_DONE',json.dumps({k:v for k,v in result.items() if k not in ['history','train_records','validation_records']}),flush=True)
if __name__=='__main__':main()

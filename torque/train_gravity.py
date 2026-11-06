from pathlib import Path
import argparse,json,time,numpy as np,torch
from .learned_gravity import GravityNet
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=300);p.add_argument('--out',default='runs/gravity');p.add_argument('--data',default='data/gravity');a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(902);device='cuda' if torch.cuda.is_available() else 'cpu'
 with np.load(Path(a.data)/'coefficients.npz') as d:x=d['x'];y=d['y']
 rng=np.random.default_rng(903);ids=rng.permutation(len(x));iv=ids[:20000];it=ids[20000:];net=GravityNet()
 for name,v in [('xm',x[it].mean(0)),('xs',np.maximum(x[it].std(0),.01)),('ym',y[it].mean(0)),('ys',np.maximum(y[it].std(0),.05))]:getattr(net,name).copy_(torch.tensor(v))
 net.to(device);X=torch.tensor(x,device=device);Y=torch.tensor(y,device=device);train=torch.tensor(it,device=device);val=torch.tensor(iv,device=device);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=0.);sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=15)
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);best=float('inf');bad=0;history=[];start=time.perf_counter()
 for ep in range(1,a.epochs+1):
  net.train();order=train[torch.randperm(len(train),device=device)]
  for b in order.split(2048):
   pred=net.coefficients(X[b]);loss=(((pred-Y[b])/net.ys)**2).mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
  net.eval()
  with torch.no_grad():delta=net.coefficients(X[val])-Y[val];vl=float((delta**2).mean());maximum=float(delta.abs().max())
  sched.step(vl);history.append(dict(epoch=ep,coefficient_rmse_nm=vl**.5,max_error_nm=maximum))
  if vl<best:
   best=vl;bad=0;torch.save(dict(width=256,state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},epoch=ep),out/'best.pt')
  else:bad+=1
  if ep%10==0:print(json.dumps(history[-1]),flush=True)
  if bad>=45:break
 (out/'metrics.json').write_text(json.dumps(dict(history=history,wall_s=time.perf_counter()-start,validation_scope='new kinematic samples drawn from training pose bounds; not task validation',n_train=len(it),n_val=len(iv)),indent=2));print('GRAVITY_TRAIN_DONE',json.dumps(history[-1]),flush=True)

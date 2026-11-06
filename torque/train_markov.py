"""Distill the calibrated observed-state teacher into neural raw torque outputs."""
from pathlib import Path
import argparse,json,time,hashlib,numpy as np,torch
from .optimized_policy import ArmNet
from .learned_gravity import GravityNet
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--epochs',type=int,default=140);p.add_argument('--width',type=int,default=256);p.add_argument('--separable',action='store_true');p.add_argument('--anchor',action='store_true');p.add_argument('--position-response',type=float,default=0.);p.add_argument('--seed',type=int,default=12)
 p.add_argument('--extra-data',nargs='*',default=[]);p.add_argument('--gravity-checkpoint',default='runs/gravity_physical/best.pt');p.add_argument('--invariant',action='store_true');p.add_argument('--regular-weight',type=float,default=1.);p.add_argument('--init');p.add_argument('--focus-folders',nargs='*',default=[]);p.add_argument('--focus-weight',type=float,default=1.)
 a=p.parse_args()
 torch.set_num_threads(1);torch.manual_seed(a.seed);device='cuda' if torch.cuda.is_available() else 'cpu';rows=[];files=[]
 paths=sorted(Path('data/markov_physical').glob('*.npz'))
 if Path('data/physical_rollouts.summary.json').exists():paths+=sorted(Path('data/physical_rollouts').glob('*.npz'))
 for folder in a.extra_data:
  extra=sorted(Path(folder).glob('*.npz'));assert extra,folder;paths+=extra
 for path in paths:
  with np.load(path) as d:row={k:d[k] for k in ['x','y','velocity','error','response','local']}
  weight=a.focus_weight if str(path.parent) in a.focus_folders else (1. if str(path.parent) in a.extra_data else a.regular_weight)
  row['domain_weight']=np.full(len(row['x']),weight,dtype='float32');rows.append(row);files.append(str(path))
 data={k:np.concatenate([r[k] for r in rows]) for k in rows[0]};X=torch.tensor(data['x'],device=device);A=torch.tensor(data['y'],device=device);V=torch.tensor(data['velocity'],device=device)
 gd=torch.load(a.gravity_checkpoint,map_location='cpu',weights_only=False);gnet=GravityNet(gd['width']);gnet.load_state_dict(gd['state_dict']);gnet.to(device).eval()
 with torch.no_grad():G=torch.cat([gnet(b[:,7:21],b[:,71:74]) for b in X.split(8192)])
 Y=A-G;arch=dict(width=a.width,phase_head=True,input_dim=74,separable_velocity=a.separable,terminal_anchor=a.anchor);net=ArmNet(**arch)
 for name,v in [('xm',X.mean(0)),('xs',X.std(0).clamp_min(.001)),('ym',Y.mean(0)),('ys',Y.std(0).clamp_min(.1)),('rm',V.mean(0)),('rs',V.std(0).clamp_min(.02))]:getattr(net,name).copy_(v.cpu())
 if a.init:
  previous=torch.load(a.init,map_location='cpu',weights_only=False)['actor'];assert previous['architecture']==arch
  net.load_state_dict(previous['state_dict']);a.invariant=a.invariant or bool(previous.get('ignored_world_coordinate_features',[]))
 identified=None
 if a.separable:
  rng=np.random.default_rng(20261102981);eta=rng.normal(0,.025,(30000,7));response=-30*eta;B=np.linalg.lstsq(eta,response,rcond=None)[0].T;identified=B.tolist()
  with torch.no_grad():net.skip.weight[:,21:28].copy_(torch.tensor(B,dtype=torch.float32)*net.xs[21:28][None]/net.ys[:,None])
  def freeze_velocity(grad):
   grad=grad.clone();grad[:,21:28]=0;return grad
  net.skip.weight.register_hook(freeze_velocity)
 if a.invariant:
  with torch.no_grad():net.encoder[0].weight[:,57:63]=0;net.skip.weight[:,57:63]=0
  def freeze_world_coordinates(grad):
   grad=grad.clone();grad[:,57:63]=0;return grad
  net.encoder[0].weight.register_hook(freeze_world_coordinates);net.skip.weight.register_hook(freeze_world_coordinates)
 net.to(device);rng=np.random.default_rng(20261102990);perm=rng.permutation(len(X));held=torch.tensor(perm[:16000],device=device);train=torch.tensor(perm[16000:],device=device)
 weights=(1+11*torch.tensor(data['error']<.05,dtype=torch.float32,device=device))*torch.tensor(data['domain_weight'],device=device);local=torch.tensor(data['local']>0,device=device);response=torch.tensor(data['response'],device=device)
 opt=torch.optim.AdamW(net.parameters(),lr=.00015 if a.init else .001,weight_decay=0.);sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=10);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);navsha=hashlib.sha256(Path('runs/frozen_navigation.pt').read_bytes()).hexdigest();best=float('inf');bad=0;history=[];start=time.perf_counter()
 def save(name,epoch):
  actor=dict(family='stage_gravity_residual',architecture=arch,state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},gravity=gd,input_dim=74,output_dim=7,torque_semantics='raw_before_slew',teacher='calibrated observed-state teacher',teacher_gain_distillation_matrix=identified,auxiliary_target='joint_velocity',ignored_world_coordinate_features=list(range(57,63)) if a.invariant else [],epoch=epoch,config=vars(a))
  torch.save(dict(family='tree_neural_torque',actor=actor,navigation_path='runs/frozen_navigation.pt',navigation_file_sha256=navsha,post=dict(torque_rate=600.,filter_s=0.)),out/name)
 for ep in range(1,a.epochs+1):
  net.train();order=train[torch.randperm(len(train),device=device)];total=0.
  for b in order.split(2048):
   tau,ref=net.torque(X[b]);per=(((tau-Y[b])/net.ys)**2).mean(-1);loss=(per*weights[b]).sum()/weights[b].sum()+.02*((ref-(V[b]-net.rm)/net.rs)**2).mean()
   if a.position_response and local[b].any():
    ix=b[local[b]];eta=torch.randn((len(ix),3),device=device)*.003;xp=X[ix].clone();xp[:,31:34]+=eta
    delta=net.torque(xp)[0]-tau[local[b]];target=torch.einsum('bij,bj->bi',response[ix],eta)
    loss+=a.position_response*(((delta-target)/net.ys)**2).mean()
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step();total+=float(loss.detach())*len(b)
  net.eval()
  with torch.no_grad():vl=float((((net.torque(X[held])[0]-Y[held])/net.ys)**2).mean())
  sched.step(vl);history.append(dict(epoch=ep,train=total/len(train),offline_holdout_mse=vl))
  if vl<best:best=vl;bad=0;save('best.pt',ep)
  else:bad+=1
  if ep%25==0:save(f'epoch_{ep:03d}.pt',ep)
  if ep%5==0:print(json.dumps(history[-1]),flush=True)
  if bad>=35:break
 save('last.pt',ep);metrics=dict(config=vars(a),architecture=arch,rows=len(X),local_rows=int(local.sum()),source_files=files,teacher_gain_distillation_matrix=identified,offline_validation_scope='16k temporal samples from training only',wall_s=time.perf_counter()-start,history=history)
 (out/'metrics.json').write_text(json.dumps(metrics,indent=2));print('MARKOV_TRAIN_DONE',json.dumps({k:v for k,v in metrics.items() if k not in ['history','source_files','teacher_gain_distillation_matrix']}),flush=True)

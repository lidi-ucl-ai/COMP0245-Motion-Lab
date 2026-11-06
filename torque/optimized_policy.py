"""Explicit learned-base/NN-torque combinations, with named optional constraints."""
import time,hashlib,numpy as np,torch
from pathlib import Path
from torch import nn
from .neural_policy import NeuralPolicy,features
from base_memory import BaseMemory,phase_features,direction_features,timing_features,tree_prediction
from .learned_gravity import GravityNet,body_gravity

class PhaseOnly:
 def __init__(self,packed):self.packed=packed;self.phase=0;self.started=False;self.reference_time=None
 def advance(self,o):
  if not self.started:
   t=self.packed['initial_direction'];self.phase=int(t['classes'][np.argmax(tree_prediction(t,direction_features(o)))])
   self.reference_time=float(np.clip(np.mean([tree_prediction(t,timing_features(o)) for t in self.packed['duration_regression']]),1.5,120.));self.started=True
  t=self.packed['transition'][self.phase];self.phase=int(t['classes'][np.argmax(tree_prediction(t,phase_features(o)))])
  return self.phase

def stage_features(o,initial,phase,horizon):
 x=features(o,initial,complete=True);one=np.eye(4,dtype='float32')[int(phase)%4]
 return np.r_[x,one,1. if phase<4 else -1.,horizon/15.,min(float(o[60])/max(.1,horizon),2.)].astype('float32')

class ArmNet(nn.Module):
 def __init__(self,width=128,phase_head=False,input_dim=71,precision_branch=False,separable_velocity=False,terminal_anchor=False):
  super().__init__();self.phase_head=phase_head;self.precision_branch=precision_branch;self.separable_velocity=separable_velocity;self.terminal_anchor=terminal_anchor
  for n,size in [('xm',input_dim),('xs',input_dim),('ym',7),('ys',7),('rm',7),('rs',7)]:self.register_buffer(n,torch.ones(size) if n.endswith('s') else torch.zeros(size))
  self.encoder=nn.Sequential(nn.Linear(input_dim,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
  self.head=nn.Linear(width,28 if phase_head else 7);self.skip=nn.Linear(input_dim,7,bias=False);self.reference=nn.Linear(width,7)
  nn.init.zeros_(self.skip.weight)
  if precision_branch:
   self.precision=nn.Sequential(nn.Linear(input_dim+10,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,7))
   nn.init.zeros_(self.precision[-1].weight);nn.init.zeros_(self.precision[-1].bias)
 def forward(self,x):
  z=(x-self.xm)/self.xs;encoded=z
  if self.separable_velocity:encoded=z.clone();encoded[...,21:28]=0
  h=self.encoder(encoded);y=self.head(h)
  if self.phase_head:y=(y.reshape(*y.shape[:-1],4,7)*x[...,64:68].unsqueeze(-1)).sum(-2)
  y=y+self.skip(z)
  if self.precision_branch:
   local=torch.cat([x[...,31:34]*20,x[...,21:28]*5],-1)
   gate=torch.exp(-torch.sum((x[...,31:34]/.08)**2,-1,keepdim=True))
   y=y+gate*self.precision(torch.cat([z,local],-1))
  return y,self.reference(h)
 def torque(self,x):
  p,ref=self(x);tau=p*self.ys+self.ym
  if self.terminal_anchor:
   zero=x.clone();zero[...,21:28]=0;zero[...,31:40]=0
   rest=self(zero)[0]*self.ys+self.ym;u=torch.clamp((x[...,70:71]-.9)/.2,0,1);gate=u*u*(3-2*u)
   tau=tau-gate*rest
  return tau,ref

class ReferenceNet(nn.Module):
 def __init__(self,width=256,input_dim=43):
  super().__init__()
  for n,size in [('xm',input_dim),('xs',input_dim),('ym',7),('ys',7)]:self.register_buffer(n,torch.ones(size) if n.endswith('s') else torch.zeros(size))
  self.net=nn.Sequential(nn.Linear(input_dim,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,7));self.skip=nn.Linear(input_dim,7,bias=False);nn.init.zeros_(self.skip.weight)
 def forward(self,x):
  z=(x-self.xm)/self.xs;return self.net(z)+self.skip(z)

class OptimizedPolicy:
 def __init__(self,path,dt=.001,env=None):
  torch.set_num_threads(1);d=path if isinstance(path,dict) else torch.load(path,map_location='cpu',weights_only=False);self.dt=dt;self.env=env;self.reference_mode=False;self.gravity_mode=False;self.mixture=False;self.gravity_calls=0;self.hybrid=d['family']=='tree_neural_torque';self.latencies=[];self.phase=-1
  if self.hybrid:
   navigation=d.get('navigation')
   if navigation is None:
    nav_path=Path(d['navigation_path'])
    if not nav_path.exists():nav_path=Path(__file__).resolve().parent/nav_path
    assert hashlib.sha256(nav_path.read_bytes()).hexdigest()==d['navigation_file_sha256'];navigation=torch.load(nav_path,map_location='cpu',weights_only=False)['base_memory']
   self.base=BaseMemory(navigation);self.post=d['post'];actor=d['actor'];self.stage=actor.get('family') in {'stage_neural_torque','stage_torque_ensemble','stage_gravity_residual'};self.reference_mode=actor.get('family')=='learned_pose_reference';self.gravity_mode=actor.get('family')=='stage_gravity_residual'
   if actor.get('family')=='terminal_neural_mixture':
    self.mixture=True;self.mix_config=actor['mix_config']
    self.coarse=OptimizedPolicy(dict(family='tree_neural_torque',actor=actor['coarse'],navigation=navigation,post=actor['coarse_post']),dt,env)
    self.fine=OptimizedPolicy(dict(family='tree_neural_torque',actor=actor['fine'],navigation=navigation,post=actor['fine_post']),dt,env)
    self.base=self.coarse.base;self.reset();return
   if self.reference_mode:
    assert env is not None;self.net=ReferenceNet(**actor['architecture']);self.net.load_state_dict(actor['state_dict']);self.net.eval()
   elif actor.get('family')=='stage_torque_ensemble':
    self.nets=[];self.member_gravity=[]
    for member in actor['members']:
     net=ArmNet(**member['architecture']);net.load_state_dict(member['state_dict']);net.eval();self.nets.append(net)
     gnet=None
     if member.get('gravity') is not None:
      gnet=GravityNet(member['gravity']['width']);gnet.load_state_dict(member['gravity']['state_dict']);gnet.eval()
     self.member_gravity.append(gnet)
   elif self.stage:
    self.net=ArmNet(**actor['architecture']);self.net.load_state_dict(actor['state_dict']);self.net.eval()
   else:self.actor=NeuralPolicy(actor)
   if self.gravity_mode:
    self.gnet=GravityNet(actor['gravity']['width']);self.gnet.load_state_dict(actor['gravity']['state_dict']);self.gnet.eval()
  else:self.actor=NeuralPolicy(d)
  self.reset()
 def reset(self):
  self.initial=None;self.wheels=np.zeros(4);self.torque=None;self.reference_q=None;self.latencies=[]
  if self.mixture:self.coarse.reset();self.fine.reset();self.mix_alpha=0.;return
  if self.hybrid:self.base.reset()
  if not self.hybrid or (not self.stage and not self.reference_mode):self.actor.reset()
 @torch.inference_mode()
 def act(self,o):
  start=time.perf_counter()
  if self.mixture:
   a=self.coarse.act(o);self.phase=self.coarse.phase;error=np.linalg.norm(o[18:21]-o[33:36]);cfg=self.mix_config
   enabled=(self.phase%4==3 and float(o[60])>=cfg.get('time_ratio',1.1)*self.base.reference_time)
   target=float(np.clip((cfg['radius_on']-error)/(cfg['radius_on']-cfg['radius_full']),0,1))*cfg.get('maximum_weight',1.) if enabled else 0.
   self.mix_alpha+=self.dt/(cfg.get('weight_filter_s',.04)+self.dt)*(target-self.mix_alpha)
   if self.mix_alpha<1e-5:self.fine.torque=a[4:].copy()
   b=self.fine.act(o);action=np.r_[a[:4],(1-self.mix_alpha)*a[4:]+self.mix_alpha*b[4:]]
   self.latencies.append((time.perf_counter()-start)*1000);return action
  if not self.hybrid:
   a=self.actor.act(o);self.latencies.append((time.perf_counter()-start)*1000);return a
  if self.initial is None:self.initial=o.copy()
  raw=self.base.act(o);self.phase=self.base.phase;self.wheels+=np.clip(raw-self.wheels,-8*self.dt,8*self.dt)
  if self.reference_mode:
   import pinocchio as pin
   x=stage_features(o,self.initial,self.phase,self.base.reference_time)[28:]
   if self.net.xm.numel()==40:x=np.r_[x[:3],x[6:]].astype('float32')
   r=self.net(torch.from_numpy(x)[None])[0];target=self.initial[4:11].astype(float)+(r*self.net.ys+self.net.ym).numpy()
   q=o[4:11].astype(float);qd=o[11:18].astype(float)
   if self.reference_q is None:self.reference_q=self.initial[4:11].astype(float).copy()
   previous=self.reference_q.copy();alpha=self.dt/(self.post.get('reference_filter_s',.02)+self.dt)
   self.reference_q+=np.clip(alpha*(target-self.reference_q),-self.dt,self.dt);self.reference_q=np.clip(self.reference_q,q-.15,q+.15)
   low=self.env.lower[4:11];high=self.env.upper[4:11];limited=low<high;self.reference_q=np.where(limited,np.clip(self.reference_q,low+.05,high-.05),self.reference_q)
   velocity=(self.reference_q-previous)/self.dt if self.post.get('reference_ff',False) else np.zeros(7)
   measured=self.env.state['q'].copy();measured[4:11]=q;qp=self.env.dyn.ReoderJoints2PinVec(np.r_[o[:3],o[21:25],measured],'pos');g=pin.computeGeneralizedGravity(self.env.dyn.pin_model,self.env.dyn.pin_data,qp)[6:13];self.gravity_calls+=1
   torque=100*(self.reference_q-q)+30*(velocity-qd)+g
  elif self.stage:
   x=stage_features(o,self.initial,self.phase,self.base.reference_time)
   torques=[]
   for member_index,net in enumerate(self.nets if hasattr(self,'nets') else [self.net]):
    dim=net.xm.numel()
    xi=np.r_[x,o[53:60]].astype('float32') if dim==78 else (np.r_[x,body_gravity(o),o[53:60]].astype('float32') if dim==81 else (np.r_[x,body_gravity(o)].astype('float32') if dim==74 else x))
    xt=torch.from_numpy(xi)[None];tau=net.torque(xt)[0][0]
    gnet=self.member_gravity[member_index] if hasattr(self,'member_gravity') else (self.gnet if self.gravity_mode else None)
    if gnet is not None:tau=tau+gnet(xt[:,7:21],xt[:,71:74])[0]
    torques.append(torch.clamp(tau,-30,30).numpy().astype(float))
   torque=np.mean(torques,axis=0)
  else:torque=self.actor.act(o)[4:]
  torque=torque-self.post.get('damping_gain',0.)*o[11:18]
  torque=np.clip(torque,-30,30)
  if self.torque is None:self.torque=torque.copy()
  if self.post.get('filter_s',0)>0:torque=self.torque+self.dt/(self.post['filter_s']+self.dt)*(torque-self.torque)
  rate=self.post.get('torque_rate',0)
  if rate>0:torque=self.torque+np.clip(torque-self.torque,-rate*self.dt,rate*self.dt)
  self.torque=torque.copy();self.latencies.append((time.perf_counter()-start)*1000)
  return np.r_[self.wheels,torque]

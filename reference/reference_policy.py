"""Behaviour cloning of reference actions. No Jacobian, IK or teacher at inference."""
import json,time
from pathlib import Path
import numpy as np
import torch
from .features import build_features
from .original_model import MLP

def rotate(v,yaw):
 c,s=np.cos(yaw),np.sin(yaw)
 return np.array([c*v[0]+s*v[1],-s*v[0]+c*v[1],v[2]])

def features(o,initial,prev_qdes,prev_v,prev_w,kind='rel'):
 # Preserve the original 38-D encoding, but use this project's frozen docking goal.
 x=build_features(o[:3],o[3],o[4:11],o[11:18],np.zeros(4),o[33:36],o[18:21],'rel')
 delta=rotate(np.r_[o[61:63]-o[:2],0],o[3]);angle=np.arctan2(o[63],o[64])-o[3]
 x[12:17]=[delta[0],delta[1],np.linalg.norm(delta[:2]),np.sin(angle),np.cos(angle)]
 if kind=='rel':return x
 # Causal context only: initial measurements, elapsed time and own previous references.
 context=np.r_[rotate(o[27:30],o[3]),rotate(o[30:33],o[3]),
  np.sin(initial[4:11]),np.cos(initial[4:11]),rotate(initial[33:36]-o[18:21],initial[3]),
  o[60]/30.,min(o[60]/10.,1.),prev_qdes-o[4:11],prev_v,prev_w[[1,0]],
  rotate(np.r_[initial[61:63]-initial[:2],0],initial[3])[:2]]
 full=np.r_[x,context].astype('float32')
 return np.r_[full[:63],full[79:81]].astype('float32') if kind=='context_no_history' else full

class Decoder:
 """Same joint feedback as the latest expert; learned references replace DLS entirely."""
 def __init__(self,env,cfg):self.env=env;self.cfg=cfg;self.reset()
 def reset(self):self.qdes=None;self.velocity=np.zeros(7);self.wheels=np.zeros(4);self.torque=None
 def act(self,o,y,mode='integrate',anchor=0.):
  import pinocchio as pin
  e=self.env;dt=e.dt;q=o[4:11].astype(float);qd=o[11:18].astype(float)
  low=e.lower[4:11];high=e.upper[4:11];limited=low<high
  raw=np.array([y[1],y[0],y[0],y[1]],float)
  self.wheels+=np.clip(np.clip(raw,-3.2,3.2)-self.wheels,-8*dt,8*dt)
  v=np.clip(y[9:16],-1.,1.)
  v=self.velocity+np.clip(v-self.velocity,-3*dt,3*dt)
  pos=q+np.clip(y[2:9],-.15,.15)
  if self.qdes is None or mode=='offset':self.qdes=pos
  else:self.qdes+=v*dt;self.qdes+=anchor*(pos-self.qdes)
  self.qdes=np.clip(self.qdes,q-.15,q+.15)
  self.qdes=np.where(limited,np.clip(self.qdes,low+.05,high-.05),self.qdes)
  a=(v-self.velocity)/dt;self.velocity=v.copy()
  measured=e.state['q'].copy();measured[4:11]=q
  qp=e.dyn.ReoderJoints2PinVec(np.r_[o[:3],o[21:25],measured],'pos')
  g=pin.computeGeneralizedGravity(e.dyn.pin_model,e.dyn.pin_data,qp)[6:13]
  if e.target[2]>1.2:
   cfg=self.cfg['high_reach'];M=pin.crba(e.dyn.pin_model,e.dyn.pin_data,qp)[6:13,6:13].copy();M=(M+M.T)*.5
   kp=np.array(cfg['joint_kp']);kd=2*np.sqrt(kp*np.maximum(np.diag(M),1e-5))
   tau=kp*(self.qdes-q)+kd*(v-qd)+M@a+g
  else:tau=100*(self.qdes-q)+30*(v-qd)+g
  tau=np.clip(tau,-30,30)
  if self.torque is None:self.torque=g.copy()
  self.torque+=np.clip(tau-self.torque,-600*dt,600*dt);self.torque=np.clip(self.torque,-30,30)
  return np.r_[self.wheels,self.torque]

class ReferencePolicy:
 def __init__(self,path,env,mode=None,anchor=None):
  torch.set_num_threads(1);d=torch.load(path,map_location='cpu',weights_only=False)
  self.cfg=d['config'];self.net=MLP(d['in_dim'],16,self.cfg['hidden'],self.cfg['layers'],'silu',0.)
  self.net.load_state_dict(d['state_dict']);self.net.eval();self.env=env
  self.decoder=Decoder(env,d['control']);self.kind=self.cfg['features'];self.mode=mode or d.get('mode','integrate');self.anchor=d.get('anchor',0.) if anchor is None else anchor
  self.navigation=None
  if 'base_memory' in d:
   from base_memory import BaseMemory
   self.navigation=BaseMemory(d['base_memory'])
  self.reset()
 def reset(self):
  self.initial=None;self.decoder.reset();self.latencies=[]
  if self.navigation is not None:self.navigation.reset()
 @torch.inference_mode()
 def act(self,o):
  t=time.perf_counter()
  if self.initial is None:self.initial=o.copy()
  qdes=o[4:11] if self.decoder.qdes is None else self.decoder.qdes
  x=features(o,self.initial,qdes,self.decoder.velocity,self.decoder.wheels,self.kind)
  y=self.net(torch.from_numpy(x)[None])[0].numpy()
  if self.navigation is not None:
   raw=self.navigation.act(o);y[0]=raw[1];y[1]=raw[0]
  action=self.decoder.act(o,y,self.mode,self.anchor);self.latencies.append((time.perf_counter()-t)*1000)
  return action

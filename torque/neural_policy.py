"""Pure neural11-action policy: no planner, IK, Jacobian, PD or dynamics model."""
from collections import deque
import time,numpy as np,torch
from torch import nn

def rotate(v,yaw):
 c,s=np.cos(yaw),np.sin(yaw);v=np.asarray(v)
 return np.stack([c*v[...,0]+s*v[...,1],-s*v[...,0]+c*v[...,1],v[...,2]],-1)

def features(o,initial,complete=False):
 o=np.asarray(o);initial=np.asarray(initial);q=o[...,4:11];yaw=o[...,3];dyaw=yaw-initial[...,3]
 x=np.concatenate([q,np.sin(q),np.cos(q),o[...,11:18],rotate(o[...,18:21]-o[...,:3],yaw),rotate(o[...,18:21]-o[...,33:36],yaw),rotate(o[...,27:30],yaw),rotate(o[...,30:33],yaw),np.broadcast_to(initial[...,4:11],q.shape),np.stack([o[...,60]/15.,np.minimum(o[...,60]/5.,1.)],-1),np.broadcast_to(rotate(initial[...,18:21]-initial[...,:3],initial[...,3]),(*q.shape[:-1],3)),np.broadcast_to(rotate(initial[...,18:21]-initial[...,33:36],initial[...,3]),(*q.shape[:-1],3)),np.stack([np.sin(dyaw),np.cos(dyaw)],-1)],-1)
 assert x.shape[-1]==57
 if complete:x=np.concatenate([x,o[...,:2],np.stack([np.sin(yaw),np.cos(yaw)],-1),o[...,18:21]],-1)
 return x.astype('float32')

class Residual(nn.Module):
 def __init__(self,width):super().__init__();self.net=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width))
 def forward(self,x):return torch.nn.functional.silu(x+self.net(x))

class NeuralController(nn.Module):
 def __init__(self,kind='mlp',width=128,input_dim=57):
  super().__init__();self.kind=kind;self.width=width
  for n,size in [('xm',input_dim),('xs',input_dim),('ym',11),('ys',11),('rm',7),('rs',7)]:self.register_buffer(n,torch.ones(size) if n.endswith('s') else torch.zeros(size))
  self.encoder=nn.Sequential(nn.Linear(input_dim,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU()) if kind!='res_moe' else nn.Sequential(nn.Linear(input_dim,width),nn.SiLU(),Residual(width),Residual(width),Residual(width))
  if kind=='gru':self.temporal=nn.GRU(width,width,batch_first=True)
  elif kind=='transformer':
   self.position=nn.Parameter(torch.zeros(1,16,width));nn.init.normal_(self.position,std=.02)
   self.temporal=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,4,width*2,dropout=0.,batch_first=True,norm_first=True),2,enable_nested_tensor=False)
  else:self.temporal=nn.Sequential(nn.Linear(width,width),nn.SiLU())
  self.fusion=nn.Sequential(nn.Linear(2*width,width),nn.SiLU());self.phase=nn.Linear(width,4);self.reference=nn.Linear(width,7)
  self.action=nn.Linear(width,11 if kind=='mlp' else 44);self.skip=nn.Linear(input_dim,11,bias=False)
  nn.init.zeros_(self.skip.weight)
 def context(self,e,hidden=None):
  if self.kind=='gru':return self.temporal(e,hidden)
  if self.kind=='transformer':
   n=e.shape[1];mask=torch.triu(torch.ones(n,n,device=e.device,dtype=torch.bool),diagonal=1)
   return self.temporal(e+self.position[:,:n],mask=mask),None
  return self.temporal(e),None
 def head(self,e,ctx,x):
  h=self.fusion(torch.cat([e,ctx],-1));phase=self.phase(h);raw=self.action(h)
  if self.kind!='mlp':raw=(raw.reshape(*h.shape[:-1],4,11)*phase.softmax(-1).unsqueeze(-1)).sum(-2)
  return raw+self.skip(x),phase,self.reference(h)
 def forward(self,x):
  z=(x-self.xm)/self.xs;e=self.encoder(z);ctx,_=self.context(e);return self.head(e,ctx,z)

class NeuralPolicy:
 def __init__(self,path):
  torch.set_num_threads(1);d=path if isinstance(path,dict) else torch.load(path,map_location='cpu',weights_only=False)
  if d['family']=='direct_neural_ensemble':
   self.members=[NeuralPolicy(m) for m in d['models']];self.net=None;self.reset();return
  assert d['family']=='direct_neural_actions'
  self.net=NeuralController(d['kind'],d['width'],d.get('input_dim',57));self.net.load_state_dict(d['state_dict']);self.net.eval();self.reset()
 def reset(self):
  if self.net is None:
   for m in self.members:m.reset()
   self.latencies=[];return
  self.initial=None;self.hidden=None;self.ctx=None;self.window=deque(maxlen=32 if self.net.kind=='gru' else 16);self.k=0;self.latencies=[]
 @torch.inference_mode()
 def act(self,o):
  start=time.perf_counter()
  if self.net is None:
   action=np.mean([m.act(o) for m in self.members],axis=0);self.latencies.append((time.perf_counter()-start)*1000);return action
  if self.initial is None:self.initial=o.copy()
  x=torch.from_numpy(features(o,self.initial,complete=self.net.xm.numel()==64))[None,None];z=(x-self.net.xm)/self.net.xs;e=self.net.encoder(z)
  if self.net.kind not in ['gru','transformer']:self.ctx=self.net.temporal(e)
  elif self.k%10==0:
   if not self.window:
    for _ in range(self.window.maxlen):self.window.append(e[0,0].clone())
   else:self.window.append(e[0,0].clone())
   hist=torch.stack(list(self.window))[None];self.ctx=self.net.context(hist)[0][:,-1:]
  y,_,_=self.net.head(e,self.ctx,z);action=(y[0,0]*self.net.ys+self.net.ym).numpy().astype(float)
  self.k+=1;self.latencies.append((time.perf_counter()-start)*1000)
  return np.clip(action,np.r_[[-3.2]*4,[-30.]*7],np.r_[[3.2]*4,[30.]*7])

"""Incremental frozen neural specialists selected by a learned goal-domain tree."""
import time,numpy as np,torch
from .optimized_policy import OptimizedPolicy as OriginalPolicy
from base_memory import tree_prediction

class DomainPolicy:
 last_selection=[]
 last_active_post={}
 def __init__(self,path,dt=.001,env=None,_log=None):
  d=path if isinstance(path,dict) else torch.load(path,map_location='cpu',weights_only=False);self.dt=dt;self.env=env;self.hybrid=True;self.reference_mode=False;self.gravity_mode=True;self.gravity_calls=0;self.post={};self.phase=-1
  self.log=[] if _log is None else _log
  if _log is None:DomainPolicy.last_selection=self.log
  if d['family']!='domain_neural_policy':
   self.single=OriginalPolicy(d,dt,env);self.children=[];self.reset();return
  self.single=None;self.selector=d['selector'];self.names=d['member_names'];self.children=[];nav=d['navigation']
  for model in d['members']:
   child={**model,'navigation':nav}
   if child['family']=='domain_neural_policy':self.children.append(DomainPolicy(child,dt,env,self.log))
   else:
    assert child['actor']['family'] in {'stage_neural_torque','stage_gravity_residual','stage_torque_ensemble'}
    self.children.append(OriginalPolicy(child,dt,env))
  self.reset()
 def reset(self):
  self.latencies=[];self.selected=None;self.gravity_calls=0
  if self.single is not None:self.single.reset();self.base=self.single.base;return
  for child in self.children:child.reset()
  self.base=self.children[0].base
 @torch.inference_mode()
 def act(self,o):
  start=time.perf_counter()
  if self.single is not None:child=self.single
  else:
   if self.selected is None:
    x=np.asarray([np.hypot(o[18],o[19]),o[20]],dtype='float32');p=tree_prediction(self.selector,x);self.selected=int(self.selector['classes'][np.argmax(p)])
    self.log.append(dict(member_index=self.selected,member_name=self.names[self.selected],goal_radius_m=float(x[0]),goal_height_m=float(x[1])))
   child=self.children[self.selected]
  action=child.act(o);self.base=child.base;self.phase=child.phase;self.gravity_calls=child.gravity_calls;self.active_post=getattr(child,'active_post',child.post);DomainPolicy.last_active_post=self.active_post
  self.latencies.append((time.perf_counter()-start)*1000);return action

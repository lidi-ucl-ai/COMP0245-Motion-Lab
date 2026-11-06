"""Learned navigation only: no geometric planner or expert imports."""
import numpy as np

def base_features(obs):
 o=np.asarray(obs);yaw=o[...,3];dx=o[...,61]-o[...,0];dy=o[...,62]-o[...,1];heading=np.arctan2(dy,dx)-yaw;heading=np.arctan2(np.sin(heading),np.cos(heading));goal=np.arctan2(o[...,63],o[...,64])-yaw;goal=np.arctan2(np.sin(goal),np.cos(goal))
 return np.stack([np.hypot(dx,dy),heading,abs(heading),goal,abs(goal),np.cos(heading),o[...,60]],-1).astype('float32')
def direction_features(obs):
 x=base_features(obs);difference=x[...,3]-x[...,1];difference=np.arctan2(np.sin(difference),np.cos(difference));return np.concatenate([x,np.abs(difference)[...,None]],axis=-1).astype('float32')

def phase_features(obs):
 o=np.asarray(obs);return np.concatenate([base_features(o),o[...,20:21]],axis=-1).astype('float32')
def timing_features(obs):
 o=np.asarray(obs);return np.concatenate([direction_features(o),np.linalg.norm(o[...,18:21]-o[...,33:36],axis=-1,keepdims=True)],axis=-1).astype('float32')
def tree_prediction(tree,x):
 node=0
 while tree['left'][node]!=-1:node=tree['left'][node] if x[tree['feature'][node]]<=tree['threshold'][node] else tree['right'][node]
 return tree['value'][node]
class BaseMemory:
 def __init__(self,packed):self.packed=packed;self.reset()
 def reset(self):self.phase=0;self.started=False;self.reference_time=None
 def act(self,obs):
  x=phase_features(obs) if self.packed.get('high_target_feature',False) else base_features(obs)
  if not self.started:
   if 'initial_direction' in self.packed:
    t=self.packed['initial_direction'];self.phase=int(t['classes'][np.argmax(tree_prediction(t,direction_features(obs)))])
    self.reference_time=float(np.clip(np.mean([tree_prediction(t,timing_features(obs)) for t in self.packed['duration_regression']]),1.5,120.))
   self.started=True
  t=self.packed['transition'][self.phase];self.phase=int(t['classes'][np.argmax(tree_prediction(t,x))]);return np.mean([tree_prediction(t,x) for t in self.packed['regression'][self.phase]],axis=0)

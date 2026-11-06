"""Training-only geometric navigation teacher; forbidden at student inference."""
import math
import numpy as np
def wrap(x):return float(np.arctan2(np.sin(x),np.cos(x)))

class ReversibleBase:
 def __init__(self,settings):self.cfg=settings;self.phase=0;self.direction=None;self.done=False;self.held=0
 def act(self,obs,dt):
  delta=obs[61:63].astype(float)-obs[:2];rho=float(np.linalg.norm(delta));yaw=float(obs[3]);goal_yaw=math.atan2(float(obs[63]),float(obs[64]));bearing=math.atan2(delta[1],delta[0])
  if self.direction is None:
   cost_forward=abs(wrap(bearing-yaw))+abs(wrap(goal_yaw-bearing))
   cost_reverse=abs(wrap(bearing+math.pi-yaw))+abs(wrap(goal_yaw-bearing-math.pi))
   self.direction=-1 if cost_reverse+.15<cost_forward else 1
  yaw_error=wrap(goal_yaw-yaw)
  if self.done:return np.zeros(4)
  if rho<.025:
   v=0.;w=np.clip(2.*yaw_error,-self.cfg.get('yaw_max',.8),self.cfg.get('yaw_max',.8));self.phase=2
   self.held=self.held+1 if abs(yaw_error)<.04 else 0
   if self.held*dt>=.12:self.done=True;self.phase=3;w=0.
  else:
   heading=wrap(bearing+(math.pi if self.direction<0 else 0)-yaw)
   blend=math.exp(-(rho/.16)**2)
   w=np.clip(2.*heading+.8*blend*yaw_error,-self.cfg.get('yaw_max',.8),self.cfg.get('yaw_max',.8))
   v=self.direction*min(self.cfg.get('base_vmax',.3),1.1*rho)*max(0.,math.cos(heading))
   self.phase=0 if abs(heading)>.6 else 1
  lw=(v-w*.33)/.11;rw=(v+w*.33)/.11
  wheels=np.array([rw,lw,lw,rw]);wheels*=min(1.,3.2/max(1e-12,abs(wheels).max()));return wheels

class StagedBase(ReversibleBase):
 def __init__(self,settings):
  super().__init__(settings);self.arrived=False;self.aligning=True
 def act(self,obs,dt):
  delta=obs[61:63].astype(float)-obs[:2];rho=float(np.linalg.norm(delta));yaw=float(obs[3]);goal=math.atan2(float(obs[63]),float(obs[64]));bearing=math.atan2(delta[1],delta[0]);err=wrap(goal-yaw)
  if self.direction is None:
   cf=abs(wrap(bearing-yaw))+abs(wrap(goal-bearing));cr=abs(wrap(bearing+math.pi-yaw))+abs(wrap(goal-bearing-math.pi));self.direction=-1 if cr+.15<cf else 1
  if self.done:return np.zeros(4)
  if rho<self.cfg.get('nav_arrive_m',.07):self.arrived=True
  if self.arrived:
   self.phase=2;v=0.;w=float(np.clip(2*err,-self.cfg.get('yaw_max',.65),self.cfg.get('yaw_max',.65)));self.held=self.held+1 if abs(err)<self.cfg.get('nav_yaw_tolerance_rad',.04) else 0
   if self.held*dt>=self.cfg.get('nav_settle_s',.12):self.done=True;self.phase=3;w=0.
  else:
   heading=wrap(bearing+(math.pi if self.direction<0 else 0)-yaw)
   if abs(heading)<.06:self.aligning=False
   if abs(heading)>.45:self.aligning=True
   if self.aligning:self.phase=0;v=0.;w=float(np.clip(2*heading,-self.cfg.get('yaw_max',.65),self.cfg.get('yaw_max',.65)))
   else:self.phase=1;v=self.direction*min(self.cfg.get('base_vmax',.3),1.1*rho)*max(0.,math.cos(heading));w=float(np.clip(1.5*heading,-.25,.25))
  lw=(v-w*.33)/.11;rw=(v+w*.33)/.11;a=np.array([rw,lw,lw,rw]);a*=min(1.,3.2/max(1e-12,abs(a).max()));return a


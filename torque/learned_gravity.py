"""Neural gravity torque prediction. No analytic dynamics call in this module."""
import numpy as np,torch
from torch import nn
def body_gravity(o):
 q=np.asarray(o)[...,21:25];x,y,z,w=np.moveaxis(q,-1,0)
 return np.stack([-2*(x*z-w*y),-2*(y*z+w*x),-(1-2*(x*x+y*y))],-1).astype('float32')
class GravityNet(nn.Module):
 def __init__(self,width=256):
  super().__init__()
  for n,size in [('xm',14),('xs',14),('ym',21),('ys',21)]:self.register_buffer(n,torch.ones(size) if n.endswith('s') else torch.zeros(size))
  self.net=nn.Sequential(nn.Linear(14,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,21))
 def coefficients(self,x):return self.net((x-self.xm)/self.xs)*self.ys+self.ym
 def forward(self,x,gravity):return (self.coefficients(x).reshape(*x.shape[:-1],7,3)*gravity.unsqueeze(-2)).sum(-1)

"""Supervised finite-state navigation memory and tree-regressed wheel commands.

The deployment state is the learner's own predicted phase. No expert import.
"""
import numpy as np

def base_features(obs):
    o=np.asarray(obs);yaw=o[...,3];dx=o[...,61]-o[...,0];dy=o[...,62]-o[...,1]
    heading=np.arctan2(dy,dx)-yaw;heading=np.arctan2(np.sin(heading),np.cos(heading))
    goal=np.arctan2(o[...,63],o[...,64])-yaw;goal=np.arctan2(np.sin(goal),np.cos(goal))
    return np.stack([np.hypot(dx,dy),heading,abs(heading),goal,abs(goal),np.cos(heading),o[...,60]],-1).astype('float32')

def tree_prediction(tree,x):
    node=0
    while tree['left'][node]!=-1:
        node=tree['left'][node] if x[tree['feature'][node]]<=tree['threshold'][node] else tree['right'][node]
    return tree['value'][node]

class BaseMemory:
    def __init__(self,packed):self.packed=packed;self.phase=0
    def act(self,obs):
        x=base_features(obs)
        transition=self.packed['transition'][self.phase]
        proba=tree_prediction(transition,x)
        self.phase=int(transition['classes'][int(np.argmax(proba))])
        regressors=self.packed['regression'][self.phase]
        return np.mean([tree_prediction(t,x) for t in regressors],axis=0)

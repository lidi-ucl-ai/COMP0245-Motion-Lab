"""Learn a local forward motion model; retain explicit DLS and low-level feedback."""
import time
import numpy as np
import torch
from torch import nn

class JacobianNet(nn.Module):
    def __init__(self,width=256):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(14,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,21))
    def forward(self,q):
        return self.net(torch.cat([q.sin(),q.cos()],-1)).reshape(*q.shape[:-1],3,7)

class JacobianPolicy:
    def __init__(self,path,env):
        from .base_memory import BaseMemory
        torch.set_num_threads(1)
        d=torch.load(path,map_location='cpu',weights_only=False)
        self.net=JacobianNet(**d['config']);self.net.load_state_dict(d['state_dict']);self.net.eval()
        self.base_memory=BaseMemory(d['base_memory']);self.env=env;self.settings=d['control'];self.reset()
    def reset(self):
        self.latencies=[];self.network_latencies=[];self.qdes=None;self.base_memory.phase=0
        self.velocity=np.zeros(7);self.last_j=None;self.phase=0;self.jacobian_calls=0;self.analytic_jacobian_calls=0
    @torch.inference_mode()
    def act(self,obs):
        import pinocchio as pin
        t=time.perf_counter_ns();e=self.env
        wheels=self.base_memory.act(obs);self.phase=self.base_memory.phase
        q=np.asarray(obs[4:11],dtype=float);qd=np.asarray(obs[11:18],dtype=float)
        if self.qdes is None:self.qdes=np.asarray(obs[42:49],dtype=float).copy()
        tick=time.perf_counter_ns()
        # A body-frame Jacobian depends only on arm joint angles. Translation and yaw are removed.
        J=self.net(torch.from_numpy(q.astype('float32'))).numpy().astype(float)
        self.network_latencies.append((time.perf_counter_ns()-tick)*1e-6);self.jacobian_calls+=1;self.last_j=J
        velocity=np.zeros(7)
        lower=e.lower[4:11];upper=e.upper[4:11];limited=lower<upper
        parallel=self.settings.get('parallel_mode')
        effective_target=obs[18:21].astype(float).copy()
        active=self.phase==3
        if parallel:
            measured=e.state['q'].copy();measured[4:11]=q
            qp=e.dyn.ReoderJoints2PinVec(np.r_[obs[:3],obs[21:25],measured],'pos')
            pin.forwardKinematics(e.dyn.pin_model,e.dyn.pin_data,qp);pin.updateFramePlacements(e.dyn.pin_model,e.dyn.pin_data)
            shoulder=e.dyn.pin_data.oMf[e.shoulder_id].translation.copy()
            distance=np.linalg.norm(effective_target-shoulder)
            if parallel=='always':active=True
            elif parallel=='drive':active=self.phase in (1,2,3)
            elif parallel=='near_drive':active=self.phase==3 or (self.phase in (1,2) and distance<1.15)
            else:raise ValueError(parallel)
            offset=effective_target-shoulder
            effective_target=shoulder+offset*min(1.,.85/max(1e-12,np.linalg.norm(offset)))
        if active:
            R=np.asarray(e.client.getMatrixFromQuaternion(obs[21:25])).reshape(3,3)
            err=R.T@(effective_target-obs[33:36])
            v=self.settings['cartesian_kp']*err
            v*=min(1.,self.settings['max_cartesian_speed']/max(1e-12,np.linalg.norm(v)))
            damping=np.interp(np.linalg.svd(J,compute_uv=False)[-1],[0,.15],[.1,.01])
            free=np.ones(7,dtype=bool)
            for _ in range(7):
                A=J[:,free];velocity[:]=0
                velocity[free]=A.T@np.linalg.solve(A@A.T+damping*damping*np.eye(3),v)
                blocked=limited&(((q<=lower+.05)&(velocity<0))|((q>=upper-.05)&(velocity>0)))&free
                if not blocked.any():break
                free&=~blocked
            velocity/=max(1.,abs(velocity).max())
            alpha=self.settings.get('velocity_alpha',1.)
            velocity=alpha*velocity+(1-alpha)*self.velocity
            self.qdes+=e.dt*velocity
            self.qdes=np.clip(self.qdes,q-.15,q+.15)
            self.qdes=np.where(limited,np.clip(self.qdes,lower+.05,upper-.05),self.qdes)
        else:self.qdes=np.asarray(obs[42:49],dtype=float).copy()
        self.velocity=velocity.copy()
        measured=e.state['q'].copy();measured[4:11]=q
        qp=e.dyn.ReoderJoints2PinVec(np.r_[obs[:3],obs[21:25],measured],'pos')
        gravity=pin.computeGeneralizedGravity(e.dyn.pin_model,e.dyn.pin_data,qp)[6:13]
        tau=np.clip(100*(self.qdes-q)+30*(velocity-qd)+gravity,-30,30)
        self.latencies.append((time.perf_counter_ns()-t)*1e-6)
        return np.r_[wheels,tau]

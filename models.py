"""Pure learned command regressors. Expert access is absent at inference."""
import time
import numpy as np
import torch
from torch import nn

def geometric_features(obs):
    obs=np.asarray(obs,dtype='float32')
    yaw=obs[...,3];dx=obs[...,61]-obs[...,0];dy=obs[...,62]-obs[...,1]
    c=np.cos(yaw);s=np.sin(yaw)
    heading=np.arctan2(dy,dx)-yaw
    goal_heading=np.arctan2(obs[...,63],obs[...,64])-yaw
    extra=np.stack([c*dx+s*dy,-s*dx+c*dy,np.sqrt(dx*dx+dy*dy),np.sin(heading),np.cos(heading),
                    np.sin(goal_heading),np.cos(goal_heading)],axis=-1)
    return np.concatenate([obs,extra,obs[...,4:11]-obs[...,42:49]],axis=-1).astype('float32')

def pd_prior(obs):
    obs=np.asarray(obs)
    return 100.*(obs[...,42:49]-obs[...,4:11])-30.*obs[...,11:18]

class Regressor(nn.Module):
    def __init__(self,kind='mlp',input_dim=65,width=128,error_mean=None,error_std=None,arm_mean=None,arm_std=None):
        super().__init__();self.kind=kind;self.input_dim=input_dim;self.width=width
        self.encoder=nn.Sequential(nn.Linear(input_dim,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.temporal=nn.GRU(width,width,batch_first=True) if kind in ['gru','phase_gru','ik_gru'] else None
        self.phase=nn.Linear(width,4)
        self.reference=nn.Linear(width,7)
        n=4 if kind=='phase_gru' else 1
        self.wheel=nn.Linear(width,n*4);self.arm=nn.Linear(width,n*7)
        if kind=='ik_gru':
            self.arm=nn.Linear(width,21)
            for name,value in [('error_mean',error_mean),('error_std',error_std),('arm_mean',arm_mean),('arm_std',arm_std)]:
                self.register_buffer(name,torch.tensor(value,dtype=torch.float32))
        # A learned linear path supports strong velocity feedback without an online controller.
        self.skip=nn.Linear(input_dim,11,bias=False)
    def forward(self,x,hidden=None):
        h=self.encoder(x)
        if self.temporal is not None:h,hidden=self.temporal(h,hidden)
        phase=self.phase(h);reference=self.reference(h)
        if self.kind=='phase_gru':
            gate=phase.softmax(-1)
            wheel=(self.wheel(h).reshape(*h.shape[:-1],4,4)*gate.unsqueeze(-1)).sum(-2)
            arm=(self.arm(h).reshape(*h.shape[:-1],4,7)*gate.unsqueeze(-1)).sum(-2)
        else:wheel=self.wheel(h);arm=self.arm(h)
        if self.kind=='ik_gru':
            error=x[...,39:42]*self.error_std+self.error_mean
            matrix=arm.reshape(*h.shape[:-1],7,3)
            velocity=(matrix@error.unsqueeze(-1)).squeeze(-1)*phase.softmax(-1)[...,3:4]
            arm=(velocity-self.arm_mean)/self.arm_std
            wheel=wheel+self.skip(x)[...,:4]
            action=torch.cat([wheel,arm],-1)
        else:action=torch.cat([wheel,arm],-1)+self.skip(x)
        return action,phase,reference,hidden

class Policy:
    def __new__(cls,path,env=None):
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved.get('family')=='learned_reversible_jacobian':
            from student_policy import LearnedPolicy
            return LearnedPolicy(path,env)
        if saved.get('family')=='learned_jacobian':
            from jacobian_model import JacobianPolicy
            return JacobianPolicy(path,env)
        return super().__new__(cls)
    def __init__(self,path,env=None):
        torch.set_num_threads(1)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        self.config=saved['config'];self.indices=np.array(saved['indices'])
        self.geometry=saved.get('geometry',False);self.use_pd_prior=saved.get('pd_prior',False)
        self.env=env;self.kinematic=self.config['kind']=='ik_gru';self.qdes=None
        self.sync_gate=saved.get('sync_gate',False)
        self.base_memory=None
        if 'base_memory' in saved:
            from base_memory import BaseMemory
            self.base_memory=BaseMemory(saved['base_memory'])
        self.mean=np.asarray(saved['x_mean'],dtype='float32');self.std=np.asarray(saved['x_std'],dtype='float32')
        self.ymean=np.asarray(saved['y_mean'],dtype='float32');self.ystd=np.asarray(saved['y_std'],dtype='float32')
        self.net=Regressor(**self.config);self.net.load_state_dict(saved['state_dict']);self.net.eval()
        self.hidden=None;self.latencies=[];self.network_latencies=[]
    def reset(self):
        self.hidden=None;self.latencies=[];self.network_latencies=[];self.qdes=None
        if self.base_memory is not None:self.base_memory.phase=0
    @torch.inference_mode()
    def act(self,obs):
        start=time.perf_counter_ns()
        source=geometric_features(obs) if self.geometry else np.asarray(obs)
        x=(source[self.indices]-self.mean)/self.std
        x=torch.from_numpy(x.astype('float32')).reshape(1,1,-1)
        network_start=time.perf_counter_ns()
        y,_,_,self.hidden=self.net(x,self.hidden)
        self.network_latencies.append((time.perf_counter_ns()-network_start)*1e-6)
        action=y[0,0].numpy()*self.ystd+self.ymean
        if self.base_memory is not None:action[:4]=self.base_memory.act(obs)
        if self.use_pd_prior:action[4:]+=pd_prior(obs)
        if self.kinematic:
            # Explicit hybrid extension: learned kinematic feedback, own reference
            # integration, retained PD/gravity/limit prior. No online expert/IK solve.
            import pinocchio as pin
            assert self.env is not None,'Kinematic hybrid requires the measured-state environment'
            if self.qdes is None:self.qdes=np.asarray(obs[42:49],dtype=float).copy()
            q=np.asarray(obs[4:11]);qd=np.asarray(obs[11:18]);velocity=action[4:].astype(float)
            velocity/=max(1.,float(np.max(abs(velocity))))
            if self.sync_gate and self.base_memory.phase%4!=3:
                velocity[:]=0.;self.qdes=np.asarray(obs[42:49],dtype=float).copy()
            self.qdes+=self.env.dt*velocity
            self.qdes=np.clip(self.qdes,q-.15,q+.15)
            lower=self.env.lower[4:11];upper=self.env.upper[4:11];limited=lower<upper
            self.qdes=np.where(limited,np.clip(self.qdes,lower+.05,upper-.05),self.qdes)
            qpin=self.env.state['q_pin']
            if self.env.scenario.get('noise_std',0.)>0:
                measured=self.env.state['q'].copy();measured[4:11]=q
                qpin=self.env.dyn.ReoderJoints2PinVec(np.r_[obs[:3],obs[21:25],measured],'pos')
            gravity=pin.computeGeneralizedGravity(self.env.dyn.pin_model,self.env.dyn.pin_data,qpin)[6:13]
            action[4:]=np.clip(100.*(self.qdes-q)+30.*(velocity-qd)+gravity,-30,30)
        self.latencies.append((time.perf_counter_ns()-start)*1e-6)
        return action

"""Shared DLS/PD coordination only. No base planner or direction decision here."""
import numpy as np
def ramp_reference(t,T,ramp):
 r=min(ramp,T*.25);v=1/(T-r)
 def s(u):return 10*u**3-15*u**4+6*u**5
 def integral(u):return 2.5*u**4-3*u**5+u**6
 if t<=0:return 0.,0.
 if t>=T:return 1.,0.
 if t<r:
  u=t/r;return v*r*integral(u),v*s(u)
 if t>T-r:
  u=(T-t)/r;return 1-v*r*integral(u),v*s(u)
 return v*(t-r*.5),v


class ArmServo:
 def __init__(self,env,cfg):
  self.env=env;self.cfg=dict(cfg)
  if env.target[2]>1.2:self.cfg.update(cfg.get('high_reach',{}))
  self.reset()
 def reset(self):
  self.qdes=None;self.velocity=np.zeros(7);self.torque=None;self.start_ee=None;self.T=None;self.prev_vbase=np.zeros(3);self.prev_omega=np.zeros(3);self.hold_q=None;self.navigation_phase=0
 def act(self,obs,J,horizon):
  import pinocchio as pin
  e=self.env;dt=e.dt;cfg=self.cfg
  q=obs[4:11].astype(float);qd=obs[11:18].astype(float);ee=obs[33:36].astype(float);target=obs[18:21].astype(float)
  if self.qdes is None:self.qdes=obs[42:49].astype(float).copy()
  measured=e.state['q'].copy();measured[4:11]=q;qp=e.dyn.ReoderJoints2PinVec(np.r_[obs[:3],obs[21:25],measured],'pos');pin.forwardKinematics(e.dyn.pin_model,e.dyn.pin_data,qp);pin.updateFramePlacements(e.dyn.pin_model,e.dyn.pin_data);shoulder=e.dyn.pin_data.oMf[e.shoulder_id].translation.copy()
  R=np.asarray(e.client.getMatrixFromQuaternion(obs[21:25])).reshape(3,3)
  if self.start_ee is None:self.start_ee=ee.copy();self.T=float(horizon)
  active=True
  if cfg.get('reference'):
   u,du=ramp_reference(float(obs[60]),self.T,cfg.get('ramp_s',.5));reference=self.start_ee+u*(target-self.start_ee);feed=du*(target-self.start_ee)
  else:reference=target.copy();feed=np.zeros(3)
  offset=reference-shoulder;length=np.linalg.norm(offset);reach=cfg.get('reach_limit_m',.85)
  if length>reach:reference=shoulder+offset*(reach/length);feed*=min(1.,reach/length)
  velocity=np.zeros(7);lower=e.lower[4:11];upper=e.upper[4:11];limited=lower<upper
  if active:
   world_v=cfg.get('cart_kp',1.8)*(reference-ee)+feed
   if cfg.get('compensate'):
    beta=dt/(.025+dt);self.prev_vbase+=beta*(obs[27:30]-self.prev_vbase);self.prev_omega+=beta*(obs[30:33]-self.prev_omega)
    bv=self.prev_vbase.copy();bw=self.prev_omega.copy()
    if cfg.get('limit_base_compensation',False):
     bv*=min(1.,.5/max(1e-12,np.linalg.norm(bv)));bw*=min(1.,1./max(1e-12,np.linalg.norm(bw)))
    world_v-=bv+np.cross(bw,ee-obs[:3])
   v=R.T@world_v;v*=min(1.,cfg.get('cart_vmax',.3)/max(1e-12,np.linalg.norm(v)))
   damping=np.interp(np.linalg.svd(J,compute_uv=False)[-1],[0,.15],[cfg.get('damping_max',.1),cfg.get('damping_min',.01)]);free=np.ones(7,dtype=bool)
   for _ in range(7):
    A=J[:,free];velocity[:]=0;velocity[free]=A.T@np.linalg.solve(A@A.T+damping*damping*np.eye(3),v)
    blocked=limited&(((q<=lower+.05)&(velocity<0))|((q>=upper-.05)&(velocity>0)))&free
    if not blocked.any():break
    free&=~blocked
   velocity/=max(1.,abs(velocity).max())
   alpha=dt/(cfg.get('joint_filter_s',.03)+dt);velocity=self.velocity+alpha*(velocity-self.velocity);velocity=self.velocity+np.clip(velocity-self.velocity,-cfg.get('joint_accel',4)*dt,cfg.get('joint_accel',4)*dt)
   self.qdes+=dt*velocity;self.qdes=np.clip(self.qdes,q-.15,q+.15);self.qdes=np.where(limited,np.clip(self.qdes,lower+.05,upper-.05),self.qdes)
  else:self.qdes=obs[42:49].astype(float).copy()
  if cfg.get('cartesian_terminal_hold') and target[2]>1.4:
   error=float(np.linalg.norm(target-ee))
   if self.hold_q is not None and error>.0095:self.hold_q=None;self.qdes=q.copy()
   if self.hold_q is None and self.navigation_phase==3 and error<.008 and np.linalg.norm(obs[27:30])<.02 and np.linalg.norm(obs[30:33])<.04:self.hold_q=q.copy()
   if self.hold_q is not None:self.qdes=self.hold_q.copy();velocity[:]=0.
  acceleration=(velocity-self.velocity)/dt;self.velocity=velocity.copy();gravity=pin.computeGeneralizedGravity(e.dyn.pin_model,e.dyn.pin_data,qp)[6:13]
  if cfg.get('inertia_matched_feedback'):
   mass=pin.crba(e.dyn.pin_model,e.dyn.pin_data,qp)[6:13,6:13].copy();mass=(mass+mass.T)*.5;kp=np.asarray(cfg['joint_kp']);kd=2*np.sqrt(kp*np.maximum(np.diag(mass),1e-5))
   desired_tau=np.clip(kp*(self.qdes-q)+kd*(velocity-qd)+mass@acceleration+gravity,-30,30)
  else:desired_tau=np.clip(100*(self.qdes-q)+30*(velocity-qd)+gravity,-30,30)
  if self.torque is None:self.torque=gravity.copy()
  self.torque+=np.clip(desired_tau-self.torque,-cfg.get('torque_rate',1000)*dt,cfg.get('torque_rate',1000)*dt);self.torque=np.clip(self.torque,-30,30)
  return self.torque.copy()

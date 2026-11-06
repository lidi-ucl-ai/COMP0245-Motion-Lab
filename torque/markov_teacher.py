"""Offline-only, observed-state torque teacher without an integrated q-reference."""
import numpy as np
import pinocchio as pin
from motion_control import ramp_reference

class MarkovTeacher:
 def __init__(self,env,initial,horizon,kp=1.8,kd=30.,matched=False):self.env=env;self.initial=initial.copy();self.horizon=horizon;self.kp=kp;self.kd=kd;self.matched=matched
 def label(self,o):
  e=self.env;q=o[4:11].astype(float);qd=o[11:18].astype(float);target=o[18:21].astype(float);ee=o[33:36].astype(float)
  motor=e.default_q.copy();motor[:4]=0.;motor[4:11]=q;motor[11:]=.1
  qp=e.dyn.ReoderJoints2PinVec(np.r_[o[:3],o[21:25],motor],'pos')
  pin.forwardKinematics(e.dyn.pin_model,e.dyn.pin_data,qp);pin.updateFramePlacements(e.dyn.pin_model,e.dyn.pin_data)
  J=pin.computeFrameJacobian(e.dyn.pin_model,e.dyn.pin_data,qp,e.ee_id,pin.LOCAL_WORLD_ALIGNED)[:3,6:13]
  u,du=ramp_reference(float(o[60]),self.horizon,.5);ref=self.initial[33:36]+u*(target-self.initial[33:36]);feed=du*(target-self.initial[33:36])
  shoulder=e.dyn.pin_data.oMf[e.shoulder_id].translation.copy();offset=ref-shoulder;length=np.linalg.norm(offset);reach=1.02 if target[2]>1.2 else .85
  if length>reach:ref=shoulder+offset*(reach/length);feed*=reach/length
  bv=o[27:30].astype(float).copy();bw=o[30:33].astype(float).copy();bv*=min(1.,.5/max(np.linalg.norm(bv),1e-12));bw*=min(1.,1./max(np.linalg.norm(bw),1e-12))
  v=self.kp*(ref-ee)+feed-bv-np.cross(bw,ee-o[:3]);raw_vnorm=np.linalg.norm(v);v*=min(1.,.25/max(1e-12,raw_vnorm))
  damping=np.interp(np.linalg.svd(J,compute_uv=False)[-1],[0,.15],[.1,.01]);lower=e.lower[4:11];upper=e.upper[4:11];limited=lower<upper;free=np.ones(7,dtype=bool);velocity=np.zeros(7)
  for _ in range(7):
   A=J[:,free];mapping=np.zeros((7,3));mapping[free]=A.T@np.linalg.solve(A@A.T+damping*damping*np.eye(3),np.eye(3));velocity=mapping@v
   blocked=limited&(((q<=lower+.05)&(velocity<0))|((q>=upper-.05)&(velocity>0)))&free
   if not blocked.any():break
   free&=~blocked
  max_velocity=np.max(np.abs(velocity));velocity/=max(1.,max_velocity)
  gravity=pin.computeGeneralizedGravity(e.dyn.pin_model,e.dyn.pin_data,qp)[6:13].copy();feedback=np.eye(7)*self.kd
  if self.matched:
   mass=pin.crba(e.dyn.pin_model,e.dyn.pin_data,qp)[6:13,6:13].copy();mass=(mass+mass.T)*.5;feedback=self.kd*mass
  raw=gravity+feedback@(velocity-qd)
  yaw=float(o[3]);c,s=np.cos(yaw),np.sin(yaw);rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
  wx,wy,wz=bw;skew=np.array([[0,-wz,wy],[wz,0,-wx],[-wy,wx,0]])
  response=feedback@mapping@(self.kp*np.eye(3)+skew)@rotation
  valid=(float(o[60])>=self.horizon and raw_vnorm<.18 and max_velocity<.7 and np.linalg.norm(target-ee)<.05 and not np.any(limited&((q<lower+.08)|(q>upper-.08))))
  return np.clip(raw,-30,30),dict(gravity=gravity,raw=raw,velocity=velocity,reference=ref,feedback=feedback,error_response=response,local_valid=valid)

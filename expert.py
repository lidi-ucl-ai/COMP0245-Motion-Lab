"""Training-only teacher: reversible planner and analytical Jacobian."""
from pathlib import Path
from types import SimpleNamespace
import json,math,numpy as np
import pinocchio as pin
from expert_planner import StagedBase,wrap
from motion_control import ArmServo

def teacher_direction_and_time(obs,cfg):
 delta=obs[61:63]-obs[:2];bearing=math.atan2(float(delta[1]),float(delta[0]));goal=math.atan2(float(obs[63]),float(obs[64]));yaw=float(obs[3])
 cf=abs(wrap(bearing-yaw))+abs(wrap(goal-bearing));cr=abs(wrap(bearing+math.pi-yaw))+abs(wrap(goal-bearing-math.pi))
 direction=-1 if cr+.15<cf else 1
 T=max(1.5,float(np.linalg.norm(delta))/cfg['base_vmax']+min(cf,cr)/cfg['yaw_max']+1.,float(np.linalg.norm(obs[18:21]-obs[33:36]))/(.3+cfg['cart_vmax']))
 return direction,T

class Expert:
 def __init__(self,env):
  self.env=env;self.cfg=json.loads((Path(__file__).parent/'configs/control.json').read_text());self.base=StagedBase(self.cfg);self.servo=ArmServo(env,self.cfg);self.wheels=np.zeros(4);self.base_memory=SimpleNamespace(phase=0);self.queries=0;self.overlap_steps=0;self.reach_steps=0;self.T=None;self.qdes=None;self.velocity=np.zeros(7)
  if env.target[2]>1.2:self.base.cfg={**self.cfg,**self.cfg.get('high_reach',{})}
 def label(self):
  e=self.env;o=e.latest_obs
  if self.T is None:_,self.T=teacher_direction_and_time(o,self.cfg)
  raw=self.base.act(o,e.dt);self.wheels+=np.clip(raw-self.wheels,-self.cfg['wheel_accel']*e.dt,self.cfg['wheel_accel']*e.dt)
  measured=e.state['q'].copy();measured[4:11]=o[4:11];qp=e.dyn.ReoderJoints2PinVec(np.r_[o[:3],o[21:25],measured],'pos')
  pin.forwardKinematics(e.dyn.pin_model,e.dyn.pin_data,qp);pin.updateFramePlacements(e.dyn.pin_model,e.dyn.pin_data)
  R=np.asarray(e.client.getMatrixFromQuaternion(o[21:25])).reshape(3,3)
  J=R.T@pin.computeFrameJacobian(e.dyn.pin_model,e.dyn.pin_data,qp,e.ee_id,pin.LOCAL_WORLD_ALIGNED)[:3,6:13]
  self.servo.navigation_phase=self.base.phase
  torque=self.servo.act(o,J,self.T);self.qdes=self.servo.qdes.copy();self.velocity=self.servo.velocity.copy();self.queries+=1
  state=self.base.phase+(4 if self.base.direction<0 else 0);self.base_memory.phase=state;self.overlap_steps+=int(np.linalg.norm(self.wheels)>.05);self.reach_steps+=1
  return np.r_[self.wheels,torque],dict(phase=state,q_des=self.qdes.copy(),q_init=e.init_q.copy(),queries=self.queries,reach_active=True,direction=self.base.direction,wheel_target=raw.copy(),reference_duration_s=self.T)

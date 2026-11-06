"""Student uses learned trees and learned Jacobian. Geometric planner forbidden."""
import time,numpy as np,torch
from jacobian_model import JacobianNet
from .base_memory import BaseMemory
from .motion_control import ArmServo
class LearnedPolicy:
 def __init__(self,path,env):
  d=torch.load(path,map_location='cpu',weights_only=False);torch.set_num_threads(1);self.net=JacobianNet(**d['config']);self.net.load_state_dict(d['state_dict']);self.net.eval();self.base_memory=BaseMemory(d['base_memory']);self.cfg=d['control'];self.env=env;self.servo=ArmServo(env,self.cfg);self.navigation_source='learned_direction_phase_wheel_trees';self.planner_calls=0;self.reset()
 def reset(self):
  self.base_memory.reset();self.servo.reset();self.wheels=np.zeros(4);self.qdes=None;self.velocity=np.zeros(7);self.phase=0;self.latencies=[];self.network_latencies=[];self.jacobian_calls=0;self.analytic_jacobian_calls=0
 @torch.inference_mode()
 def act(self,obs):
  begin=time.perf_counter_ns();raw=self.base_memory.act(obs);self.phase=self.base_memory.phase%4;self.wheels+=np.clip(raw-self.wheels,-self.cfg['wheel_accel']*self.env.dt,self.cfg['wheel_accel']*self.env.dt)
  tick=time.perf_counter_ns();J=self.net(torch.from_numpy(obs[4:11].astype('float32'))).numpy().astype(float);self.network_latencies.append((time.perf_counter_ns()-tick)*1e-6);self.jacobian_calls+=1
  torque=self.servo.act(obs,J,self.base_memory.reference_time);self.qdes=self.servo.qdes.copy();self.velocity=self.servo.velocity.copy();self.latencies.append((time.perf_counter_ns()-begin)*1e-6);return np.r_[self.wheels,torque]

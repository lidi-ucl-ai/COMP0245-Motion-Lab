"""Training-only expert; its memory is updated on the trajectory actually executed."""
import sys
from pathlib import Path
import numpy as np
from course.robotnik_arm_control import ArmControllerParams, BaseControllerParams, BaseNavState, arm_controller, base_controller

class Expert:
    def __init__(self,env):
        self.env=env
        self.base=BaseNavState()
        self.bp=BaseControllerParams(nav_timeout=25.)
        self.ap=ArmControllerParams(reach_limit=.85)
        self.qdes=env.init_q.copy()
        self.queries=0
        self.overlap_steps=0
        self.reach_steps=0
    def label(self):
        e=self.env; s=e.state
        if e.scenario.get('noise_std',0.)>0:
            # Reference controller receives the same noisy sensor draw as the learner.
            import pinocchio as pin
            o=e.latest_obs;s=dict(s);s['pos']=o[:3].astype(float);s['yaw']=float(o[3])
            s['q']=s['q'].copy();s['qd']=s['qd'].copy();s['q'][4:11]=o[4:11];s['qd'][4:11]=o[11:18]
            s['q_pin']=e.dyn.ReoderJoints2PinVec(np.r_[s['pos'],s['ori'],s['q']],'pos');s['ee']=o[33:36].astype(float)
            pin.forwardKinematics(e.dyn.pin_model,e.dyn.pin_data,s['q_pin']);pin.updateFramePlacements(e.dyn.pin_model,e.dyn.pin_data)
        wheels=base_controller(self.base,s['pos'],s['yaw'],e.base_goal,e.goal_yaw,e.k*e.dt,self.bp)
        mode=e.scenario.get('expert_variant','sequential')
        moving=np.linalg.norm(wheels)>.05
        if mode=='sequential':active=self.base.done
        elif mode=='drive':active=self.base.done or self.base.phase in ('drive','orient')
        elif mode=='near_drive':
            shoulder=e.dyn.pin_data.oMf[e.shoulder_id].translation
            active=self.base.done or (self.base.phase in ('drive','orient') and np.linalg.norm(e.target-shoulder)<1.15)
        elif mode=='always':active=True
        else:raise ValueError(mode)
        self.overlap_steps+=int(active and moving)
        self.reach_steps+=int(active)
        torque,self.qdes,qd=arm_controller(e.dyn,s['q_pin'],s['q'][4:11],s['qd'][4:11],self.qdes,e.init_q,
            e.target,s['ee'],e.ee_id,e.shoulder_id,e.lower[4:11],e.upper[4:11],e.dt,self.ap,reach_active=active)
        self.queries+=1
        phase=3 if self.base.done else {'align':0,'drive':1,'orient':2}[self.base.phase]
        return np.r_[wheels,torque],dict(phase=phase,q_des=self.qdes.copy(),q_init=e.init_q.copy(),queries=self.queries,reach_active=bool(active))

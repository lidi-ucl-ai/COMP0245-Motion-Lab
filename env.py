"""Measured-state environment. No expert controller is imported here.

Physics, mixed control interface and fixed fingers follow the pinned course example.
Students output 4 wheel angular velocities and 7 arm torques every 1 ms.
"""
from pathlib import Path
import contextlib
import io
import time
import numpy as np
import pybullet as pb_native
import pinocchio as pin
from control_utils import standoff as target_standoff
from simulation_and_control import pb, MotorCommands, PinWrapper

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'vendor' / 'RoboEnv'
BASE_NAMES = ['robot_front_right_wheel_joint', 'robot_front_left_wheel_joint',
              'robot_back_left_wheel_joint', 'robot_back_right_wheel_joint']
ARM_NAMES = [f'robot_j2s7s300_joint_{i}' for i in range(1, 8)]
ACTION_LIMIT = np.array([3.2]*4+[30.]*7)

def yaw_of(q):
    x,y,z,w = q
    return float(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))

class RobotEnv:
    def __init__(self, gui=False):
        with contextlib.redirect_stdout(io.StringIO()):
            self.sim = pb.SimInterface('robotnik_arm.json', conf_file_path_ext=str(ASSETS), use_gui=gui)
            self.names = self.sim.getNameActiveJoints()
            self.dyn = PinWrapper('robotnik_arm.json', 'pybullet', np.array([self.names]), ['pybullet'], False, 0, str(ASSETS))
        assert self.names[:4] == BASE_NAMES, self.names
        assert self.names[4:11] == ARM_NAMES, self.names
        assert len(self.names) == 17
        self.client = self.sim.pybullet_client
        self.rid = self.sim.bot[0].bot_pybullet
        self.ids = self.sim.bot[0].active_joint_ids
        self.dt = self.sim.GetTimeStep()
        assert abs(self.dt-.001) < 1e-12
        self.sim.SetFloorFriction(1.)
        for jid in self.ids[:4]:
            self.client.changeDynamics(self.rid,jid,contactStiffness=30000,contactDamping=1000,anisotropicFriction=[1.,.01,1.])
        self.sim.bot[0].servo_motor_model.set_motor_gains(400,30)
        # Setup-only fixed finger hold, adapted from configure_gripper_fingers.
        original = self.sim._SetMotorTorqueByIds
        def fixed_fingers(motor_ids, commands, index=0):
            original(motor_ids[:11],commands[:11],index=index)
            self.client.setJointMotorControlArray(self.rid,motor_ids[11:17],self.client.POSITION_CONTROL,targetPositions=[.1]*6,forces=[5.]*6)
        self.sim._SetMotorTorqueByIds = fixed_fingers
        self.ee_id = self.dyn.pin_model.getFrameId('robot_j2s7s300_end_effector')
        self.shoulder_id = self.dyn.pin_model.getFrameId('robot_j2s7s300_link_2')
        self.lower, self.upper = map(np.asarray,self.sim.GetBotJointsLimit())
        # Assert the expert's hard-coded Jacobian/gravity slices in the actual model.
        self.arm_v = [self.dyn.pin_model.joints[self.dyn.pin_model.getJointId(n)].idx_v for n in ARM_NAMES]
        assert self.arm_v == list(range(6,13)), self.arm_v
        self.default_q = np.asarray(self.sim.GetMotorAngles(0)).copy()
        self.default_pos, self.default_ori = self.client.getBasePositionAndOrientation(self.rid)
        self.saved = self.client.saveState()
        self.cmd = MotorCommands()
        self.modes = ['velocity']*4 + ['torque']*13
        self.target_marker = None

    def reset(self, scenario):
        if self.target_marker is not None:
            self.client.removeBody(self.target_marker)
            self.target_marker = None
        self.client.restoreState(self.saved)
        self.scenario = scenario
        self.target = np.array(scenario['target'], dtype=float)
        self.rng = np.random.default_rng(scenario['seed'])
        self.sim.step_counter=0
        self.k=0
        self.prev_action=np.zeros(11)
        q=self.default_q.copy()
        # Upstream starts joint 6 at 1.0 rad, below its URDF limit 1.13446.
        # Formal scenarios use a disclosed legal 1.25-rad nominal value.
        if not scenario.get('legacy_initial_pose',False):
            q[9]=1.25
        q[4:11] += np.asarray(scenario.get('arm_offset',[0.]*7))
        limited=self.lower < self.upper
        if not scenario.get('legacy_initial_pose',False):
            assert np.all((q[limited]>=self.lower[limited])&(q[limited]<=self.upper[limited]))
        xy_yaw=scenario.get('base',[0.,0.,0.])
        pos=np.array(self.default_pos); pos[:2]+=xy_yaw[:2]
        ori=self.client.getQuaternionFromEuler([0,0,xy_yaw[2]])
        self.client.resetBasePositionAndOrientation(self.rid,pos,ori)
        self.client.resetBaseVelocity(self.rid,[0,0,0],[0,0,0])
        for jid,angle in zip(self.ids,q):
            self.client.resetJointState(self.rid,jid,float(angle),0.)
        self.sim.ReceiveObservation()
        self.init_q=np.array(self.sim.GetMotorAngles(0))[4:11].copy()
        self.sim.SetFloorFriction(scenario.get('friction',1.))
        self.held=0
        self.min_error=float('inf')
        self.goal_yaw=float(np.arctan2(self.target[1],self.target[0])) if np.linalg.norm(self.target[:2])>1e-4 else 0.
        standoff=getattr(self, "target_standoff", target_standoff)(self.target[2])
        self.base_goal=self.target[:2]-standoff*np.array([np.cos(self.goal_yaw),np.sin(self.goal_yaw)])
        return self.observe()

    def observe(self):
        pos=np.asarray(self.sim.GetBasePosition())
        ori=np.asarray(self.sim.GetBaseOrientation())
        q=np.asarray(self.sim.GetMotorAngles(0))
        qd=np.asarray(self.sim.GetMotorVelocities(0))
        qp=self.dyn.ReoderJoints2PinVec(np.concatenate([pos,ori,q]),'pos')
        pin.forwardKinematics(self.dyn.pin_model,self.dyn.pin_data,qp)
        pin.updateFramePlacements(self.dyn.pin_model,self.dyn.pin_data)
        ee=self.dyn.pin_data.oMf[self.ee_id].translation.copy()
        yaw=yaw_of(ori)
        self.state=dict(pos=pos,ori=ori,q=q,qd=qd,q_pin=qp,ee=ee,yaw=yaw)
        # Noise is applied to deployable measurements, never to ground-truth scoring.
        noise=self.scenario.get('noise_std',0.)
        p_obs=pos+self.rng.normal(0,noise,3)
        q_obs=q[4:11]+self.rng.normal(0,noise,7)
        qd_obs=qd[4:11]+self.rng.normal(0,noise*5,7)
        ee_obs=ee+self.rng.normal(0,noise,3)
        baseline=np.r_[p_obs,yaw,q_obs,qd_obs,self.target]
        c,s=np.cos(yaw),np.sin(yaw)
        R=np.array([[c,s,0],[-s,c,0],[0,0,1.]])
        enhanced=np.r_[baseline,ori,np.sin(yaw),np.cos(yaw),self.sim.GetBaseLinVelocity(),self.sim.GetBaseAngVelocity(),
                       ee_obs,R@(self.target-p_obs),R@(self.target-ee_obs),self.init_q,self.prev_action,
                       self.k*self.dt,self.base_goal,np.sin(self.goal_yaw),np.cos(self.goal_yaw)]
        self.latest_obs=enhanced.astype(np.float32)
        return self.latest_obs

    def step(self, action):
        a=np.asarray(action,dtype=float)
        if a.shape!=(11,) or not np.isfinite(a).all():
            raise ValueError('Policy must produce a finite 11-dimensional action')
        applied=np.clip(a,-ACTION_LIMIT,ACTION_LIMIT)
        self.cmd.SetControlCmd(np.r_[applied,np.zeros(6)],self.modes)
        # Fixed, predeclared lateral perturbation; no recovery expert.
        push=self.scenario.get('push_force',0.)
        if push and 8. <= self.k*self.dt < 8.2:
            self.client.applyExternalForce(self.rid,-1,[0,push,0],self.state['pos'],self.client.WORLD_FRAME)
        self.sim.Step(self.cmd,'torque')
        self.k+=1
        self.prev_action=applied
        obs=self.observe()
        error=float(np.linalg.norm(self.target-self.state['ee']))
        self.min_error=min(self.min_error,error)
        self.held=self.held+1 if error<.01 else 0
        success=self.held>=round(.5/self.dt)
        quat=self.state['ori']
        rpy=self.client.getEulerFromQuaternion(quat)
        unsafe=(max(abs(rpy[0]),abs(rpy[1]))>1. or not np.isfinite(obs).all())
        timeout=self.k*self.dt>=self.scenario.get('duration',40.)
        reason='success' if success else ('safety_abort' if unsafe else ('timeout' if timeout else 'running'))
        return obs,dict(error=error,success=success,reason=reason,done=success or unsafe or timeout,
                        applied=applied,saturated=bool(np.any(abs(a)>ACTION_LIMIT)))

    def render(self,width=640,height=480):
        if self.target_marker is None:
            shape=self.client.createVisualShape(pb_native.GEOM_SPHERE,radius=.03,rgbaColor=[1,0,0,1])
            self.target_marker=self.client.createMultiBody(baseMass=0,baseVisualShapeIndex=shape,basePosition=self.target.tolist())
        center=(self.state['pos']+self.target)/2
        view=self.client.computeViewMatrixFromYawPitchRoll(center,2.8,50,-25,0,2)
        proj=self.client.computeProjectionMatrixFOV(55,width/height,.02,20)
        result=self.client.getCameraImage(width,height,view,proj,renderer=self.client.ER_TINY_RENDERER)
        return np.asarray(result[2]).reshape(height,width,4)[:,:,:3]

    def close(self):
        self.client.disconnect()

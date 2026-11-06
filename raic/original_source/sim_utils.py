
import os
from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin
import pybullet as p
from simulation_and_control import pb, MotorCommands, PinWrapper
from robotnik_arm_control import (
    ArmControllerParams, BaseControllerParams, BaseNavState,
    arm_controller, base_controller, configure_gripper_fingers, quaternion2bearing,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CONF_FILE = "robotnik_arm.json"

# ---- constants copied from robotnik_arm_task.py (course template) ----
FIXED_FINGER_ANGLES = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
BASE_NAV_TIMEOUT = 25.0
ARM_REACH_LIMIT = 0.85
STANDOFF_Z_POINTS = [0.1, 0.5, 1.1, 1.25]
STANDOFF_POINTS = [0.80, 0.65, 0.65, 0.35]
REACH_TOLERANCE = 0.01
MIN_TIME = 8.0                  # StopCriteria.min_time in robotnik_arm_control.py
CONTROL_MODES = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
ARM_LIMIT_MARGIN = 0.05         # ArmControllerParams.limit_margin


def base_goal(target):
    """Expert's base goal: a standoff pose facing the target (as in robotnik_arm_task.py)."""
    standoff = float(np.interp(target[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
    brg = np.arctan2(target[1], target[0]) if np.linalg.norm(target[:2]) > 1e-4 else 0.0
    return target[:2] - standoff * np.array([np.cos(brg), np.sin(brg)]), brg


@dataclass
class Env:
    sim: object
    client: object
    rid: int
    dyn: object
    ee_id: int
    shoulder_id: int
    arm_lo: np.ndarray
    arm_hi: np.ndarray
    dt: float
    cmd: object = field(default_factory=MotorCommands)

    def close(self):
        try:
            self.client.disconnect()
        except Exception:
            pass


def make_env(gui=False):
    """Simulator with the template's settings (friction, wheel contact, motor gains, fixed fingers)."""
    sim = pb.SimInterface(CONF_FILE, conf_file_path_ext=HERE, use_gui=gui)
    sim.SetFloorFriction(1.0)
    client = sim.pybullet_client
    rid = sim.bot[0].bot_pybullet
    for j in range(client.getNumJoints(rid)):
        if "wheel" in client.getJointInfo(rid, j)[1].decode():
            client.changeDynamics(rid, j, contactStiffness=30000, contactDamping=1000,
                                  anisotropicFriction=[1.0, 0.01, 1.0])
    sim.bot[0].servo_motor_model.set_motor_gains(400, 30)
    configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)

    names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    dyn = PinWrapper(CONF_FILE, "pybullet", names, ["pybullet"], False, 0, HERE)
    lower, upper = sim.GetBotJointsLimit()
    return Env(sim=sim, client=client, rid=rid, dyn=dyn,
               ee_id=dyn.pin_model.getFrameId("robot_j2s7s300_end_effector"),
               shoulder_id=dyn.pin_model.getFrameId("robot_j2s7s300_link_2"),
               arm_lo=np.asarray(lower[4:11], float), arm_hi=np.asarray(upper[4:11], float),
               dt=float(sim.GetTimeStep()))


def default_arm_q(env):
    return np.asarray(env.sim.GetMotorAngles(0), float)[4:11].copy()


def clip_to_limits(env, q_arm):
    """Keep an arm configuration inside the joint limits (with the expert's margin).
    Joints without limits (continuous joints) are left unchanged."""
    has = env.arm_lo < env.arm_hi
    return np.where(has, np.clip(q_arm, env.arm_lo + ARM_LIMIT_MARGIN, env.arm_hi - ARM_LIMIT_MARGIN), q_arm)


def set_initial_state(env, base_pos, base_ori, arm_q):
    """Place the base (position, quaternion xyzw) and the arm joints, all at rest."""
    c, rid = env.client, env.rid
    c.resetBasePositionAndOrientation(rid, list(map(float, base_pos)), list(map(float, base_ori)))
    c.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0])
    names = list(env.sim.getNameActiveJoints())
    jid = {c.getJointInfo(rid, j)[1].decode(): j for j in range(c.getNumJoints(rid))}
    for k, name in enumerate(names[4:11]):
        c.resetJointState(rid, jid[name], float(arm_q[k]), targetVelocity=0.0)
    env.sim.ReceiveObservation()


def measure(env, noise_q=0.0, noise_qd=0.0, rng=None):
    """Read the robot state and compute the end-effector position (forward kinematics).

    noise_q / noise_qd: optional Gaussian noise on the measured ARM joints (robustness tests).
    The true end-effector error is always computed from the noise-free state.
    """
    s = env.sim
    base_pos = np.asarray(s.GetBasePosition(), float)
    base_ori = np.asarray(s.GetBaseOrientation(), float)
    yaw = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
    q = np.asarray(s.GetMotorAngles(0), float)
    qd = np.asarray(s.GetMotorVelocities(0), float)

    q_pin_true = env.dyn.ReoderJoints2PinVec(np.concatenate([base_pos, base_ori, q]), "pos")
    pin.forwardKinematics(env.dyn.pin_model, env.dyn.pin_data, q_pin_true)
    pin.updateFramePlacements(env.dyn.pin_model, env.dyn.pin_data)
    ee_true = env.dyn.pin_data.oMf[env.ee_id].translation.copy()

    if noise_q > 0 or noise_qd > 0:
        q, qd = q.copy(), qd.copy()
        q[4:11] += rng.normal(0, noise_q, 7) if noise_q > 0 else 0.0
        qd[4:11] += rng.normal(0, noise_qd, 7) if noise_qd > 0 else 0.0
        q_pin = env.dyn.ReoderJoints2PinVec(np.concatenate([base_pos, base_ori, q]), "pos")
        pin.forwardKinematics(env.dyn.pin_model, env.dyn.pin_data, q_pin)
        pin.updateFramePlacements(env.dyn.pin_model, env.dyn.pin_data)
        ee = env.dyn.pin_data.oMf[env.ee_id].translation.copy()  # as seen by the controller
    else:                                                          # no noise: reuse the forward kinematics
        q_pin, ee = q_pin_true, ee_true.copy()
    tau_g = pin.computeGeneralizedGravity(env.dyn.pin_model, env.dyn.pin_data, q_pin)[6:13].copy()
    return {"base_pos": base_pos, "base_ori": base_ori, "yaw": yaw, "q": q, "qd": qd,
            "q_arm": q[4:11], "qd_arm": qd[4:11], "wheel_vel": qd[0:4], "q_pin": q_pin,
            "ee": ee, "ee_true": ee_true, "tau_g": tau_g}


def send(env, wheel, tau_arm):
    env.cmd.SetControlCmd(np.concatenate([wheel, tau_arm, np.zeros(6)]), CONTROL_MODES)
    env.sim.Step(env.cmd, "torque")


def add_target_marker(env, target):
    vis = env.client.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1.0, 0.0, 0.0, 0.85])
    env.client.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=list(map(float, target)))


class Expert:
    """The template's expert (robotnik_arm_control.py, unchanged), with its internal state."""

    def __init__(self, env, target, q_init_arm):
        self.env, self.target = env, np.asarray(target, float)
        self.goal_pos, self.goal_bearing = base_goal(self.target)
        self.state = BaseNavState()
        self.base_params = BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT)
        self.arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
        self.q_init = np.asarray(q_init_arm, float).copy()
        self.q_des = self.q_init.copy()
        self.qd_des = np.zeros(7)

    def act(self, m, t):
        """m: output of measure(). Returns wheel commands (4) and arm torques (7)."""
        wheel = base_controller(self.state, m["base_pos"], m["yaw"], self.goal_pos, self.goal_bearing,
                                t, self.base_params)
        tau, self.q_des, self.qd_des = arm_controller(
            self.env.dyn, m["q_pin"], m["q_arm"], m["qd_arm"], self.q_des, self.q_init, self.target,
            m["ee"], self.env.ee_id, self.env.shoulder_id, self.env.arm_lo, self.env.arm_hi,
            self.env.dt, self.arm_params, reach_active=self.state.done)
        return wheel, tau
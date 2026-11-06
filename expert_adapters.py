"""Put the two deterministic experts on the same scoring interface."""
import numpy as np


class OriginalExpert:
    def __init__(self, env):
        from course.robotnik_arm_control import ArmControllerParams, BaseControllerParams, BaseNavState
        self.env = env
        self.base = BaseNavState()
        self.base_params = BaseControllerParams(nav_timeout=25.)
        self.arm_params = ArmControllerParams(reach_limit=.85)
        self.qdes = env.init_q.copy()
        standoff = float(np.interp(env.target[2], [.1, .5, 1.1, 1.25], [.80, .65, .65, .35]))
        self.goal = env.target[:2] - standoff * np.array([np.cos(env.goal_yaw), np.sin(env.goal_yaw)])

    def act(self, obs):
        from course.robotnik_arm_control import arm_controller, base_controller
        e, s = self.env, self.env.state
        wheels = base_controller(self.base, s['pos'], s['yaw'], self.goal, e.goal_yaw,
                                 e.k * e.dt, self.base_params)
        torque, self.qdes, _ = arm_controller(
            e.dyn, s['q_pin'], s['q'][4:11], s['qd'][4:11], self.qdes, e.init_q,
            e.target, s['ee'], e.ee_id, e.shoulder_id, e.lower[4:11], e.upper[4:11],
            e.dt, self.arm_params, reach_active=self.base.done)
        return np.r_[wheels, torque]


class ImprovedExpert:
    def __init__(self, env):
        from expert import Expert
        self.expert = Expert(env)

    def act(self, obs):
        return self.expert.label()[0]


class ConcurrentExpert(ImprovedExpert):
    def __init__(self, env):
        from historical.concurrent_expert import Expert
        self.expert = Expert(env)

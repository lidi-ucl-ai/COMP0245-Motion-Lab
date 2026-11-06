"""Small telemetry snapshot from measured state and applied control commands."""
import numpy as np

class StateMonitor:
    def __init__(self):
        self.previous_time=None
        self.previous_ee=None
        self.ee_speed=0.

    def capture(self,t,ee,base_velocity,applied_action,held_s):
        ee=np.asarray(ee,dtype=float)
        if self.previous_time is not None and t>self.previous_time:
            self.ee_speed=float(np.linalg.norm(ee-self.previous_ee)/(t-self.previous_time))
        if self.previous_time is None or t>self.previous_time:
            self.previous_time=float(t)
            self.previous_ee=ee.copy()
        return dict(ee_speed_m_s=self.ee_speed,
            base_speed_m_s=float(np.linalg.norm(np.asarray(base_velocity,dtype=float)[:2])),
            arm_torque_nm=np.asarray(applied_action,dtype=float)[4:11].tolist(),
            torque_limit_nm=30.,success_hold_s=float(np.clip(held_s,0.,.5)))

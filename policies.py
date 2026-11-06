"""Report policy registry and adapters for the shared 11-channel simulator."""
import hashlib
import json
import sys
from functools import cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

from jacobian_model import RawJacobianNet

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor/RoboEnv/simulation_and_control'))
MODELS = json.loads((ROOT / 'checkpoints/manifest.json').read_text(encoding='utf-8'))
HISTORICAL_ENV = {'original_expert', 'parallel_expert', 'ljc_sequential', 'ljc_parallel', 'ljc_pre_high'}


def original_standoff(height):
    return np.interp(height, [.1, .5, 1.1, 1.25], [.8, .65, .65, .35])


def reset_for_policy(key, env, scenario):
    from control_utils import standoff
    env.target_standoff = original_standoff if key in HISTORICAL_ENV else standoff
    if key == 'parallel_expert':
        scenario = {**scenario, 'expert_variant': 'drive'}
    return env.reset(scenario)


@cache
def checkpoint(key):
    entry = MODELS[key]
    path = ROOT / entry['checkpoint']
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != entry['sha256']:
        raise ValueError(f'Checkpoint checksum mismatch for {key}')
    return path


class ConstantJacobian(nn.Module):
    def __init__(self, net, q):
        super().__init__()
        self.register_buffer('matrix', net(torch.as_tensor(q, dtype=torch.float32)).detach())

    def forward(self, q):
        return self.matrix


class OriginalRAIC:
    def __init__(self, path, env, integrated):
        from raic.model import BCPolicy
        self.env = env
        self.policy = BCPolicy(path, arm_integrate=integrated, dt=env.dt)

    def reset(self):
        self.policy.reset()

    def act(self, obs):
        import pinocchio as pin
        dyn = self.env.dyn
        measured = self.env.state['q'].copy()
        measured[4:11] = obs[4:11]
        qp = dyn.ReoderJoints2PinVec(np.r_[obs[:3], obs[21:25], measured], 'pos')
        gravity = pin.computeGeneralizedGravity(dyn.pin_model, dyn.pin_data, qp)[6:13].copy()
        wheels, torque = self.policy.act(obs[:3], float(obs[3]), obs[4:11], obs[11:18],
                                         np.zeros(4), obs[33:36], obs[18:21], gravity)
        return np.r_[wheels, torque]


def alter_torque(policy, variant):
    if getattr(policy, 'single', None) is not None:
        alter_torque(policy.single, variant)
    for child in getattr(policy, 'children', []):
        alter_torque(child, variant)
    nets = getattr(policy, 'nets', []) or ([policy.net] if hasattr(policy, 'net') else [])
    for net in nets:
        if variant == 'no_anchor':
            net.terminal_anchor = False
        elif variant == 'no_velocity_prior':
            with torch.no_grad():
                net.skip.weight[:, 21:28] = 0


def load_policy(key, env, initial_obs):
    entry = MODELS[key]
    kind, variant = entry['kind'], entry['variant']
    if kind == 'expert':
        from expert_adapters import OriginalExpert, ImprovedExpert, ConcurrentExpert
        expert = {'original_expert': OriginalExpert, 'improved_expert': ImprovedExpert, 'parallel_expert': ConcurrentExpert}
        return expert[key](env)

    path = checkpoint(key)
    if kind == 'raic_orig':
        policy = OriginalRAIC(path, env, variant == 'arm_integrate_true')
    elif kind == 'reference':
        from reference.reference_policy import ReferencePolicy
        policy = ReferencePolicy(path, env, mode='integrate', anchor=0.)
    elif kind == 'torque':
        from torque.domain_policy import DomainPolicy
        policy = DomainPolicy(path, env=env)
        if variant == 'far_specialist_only':
            assert policy.names[1] == 'far_specialist'
            policy.single, policy.children = policy.children[1], []
        alter_torque(policy, variant)
    elif kind == 'historical':
        if variant == 'pre_high_student':
            from historical.pre_high.student_policy import LearnedPolicy
            policy = LearnedPolicy(path, env)
        else:
            from historical.concurrent.jacobian_model import JacobianPolicy
            policy = JacobianPolicy(path, env)
    elif kind == 'jacobian':
        from student_policy import LearnedPolicy
        if variant == 'raw_angle_input':
            policy = LearnedPolicy(path, env, network=RawJacobianNet)
        else:
            policy = LearnedPolicy(path, env)
        if variant == 'no_compensation':
            policy.servo.cfg['compensate'] = False
        elif variant == 'no_high_servo':
            policy.servo.cfg = {k: v for k, v in policy.cfg.items() if k != 'high_reach'}
        elif variant == 'fixed_initial_jacobian':
            policy.net = ConstantJacobian(policy.net, initial_obs[4:11])
    else:
        from models import Policy
        policy = Policy(path, env=env)
        if variant == 'sync_gate':
            policy.sync_gate = True
    policy.reset()
    return policy

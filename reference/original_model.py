"""
model.py - The MLP regressor and a policy wrapper for closed-loop use.

The normalisation statistics are stored INSIDE the model (as buffers), so a checkpoint is
self-contained: load it, give it raw measurements, get wheel commands and arm torques.
"""
import time

import numpy as np
import torch
import torch.nn as nn

from .features import MAX_JOINT_VEL, MAX_Q_LEAD, build_features, decode_actions, resolve_feature_set

ACTIVATIONS = {"relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}


class MLP(nn.Module):
    """Fully connected network with built-in input/output standardisation.

    layers = number of hidden layers (0 gives a linear regression baseline).
    """

    def __init__(self, in_dim, out_dim, hidden=256, layers=3, act="silu", dropout=0.0):
        super().__init__()
        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_std", torch.ones(in_dim))
        self.register_buffer("y_mean", torch.zeros(out_dim))
        self.register_buffer("y_std", torch.ones(out_dim))
        mods, d = [], in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), ACTIVATIONS[act]()]
            if dropout > 0:
                mods.append(nn.Dropout(dropout))
            d = hidden
        mods.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*mods)

    def set_normalisation(self, x_mean, x_std, y_mean, y_std):
        for name, v in zip(("x_mean", "x_std", "y_mean", "y_std"), (x_mean, x_std, y_mean, y_std)):
            getattr(self, name).copy_(torch.as_tensor(v, dtype=torch.float32))

    def forward_normalised(self, x):
        """Normalised input -> normalised output (used for the training loss)."""
        return self.net((x - self.x_mean) / self.x_std)

    def forward(self, x):
        """Raw features -> outputs in physical units."""
        return self.forward_normalised(x) * self.y_std + self.y_mean


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = MLP(ckpt["in_dim"], ckpt["out_dim"], cfg["hidden"], cfg["layers"], cfg["act"], cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


class BCPolicy:
    """Closed-loop wrapper: raw measurements in, (wheel_cmds(4), torque_arm(7)) out.

    Optional mode classifier (done_checkpoint): predicts whether the base navigation is finished.
    Like the expert's state machine, the decision is LATCHED: once the classifier has said "done" for
    `done_confirm` consecutive steps, the policy stays in the done mode for the rest of the episode,
    the wheel commands are exactly 0 (as the expert's), and 'rel_mode' policies receive done = 1.
    Call reset() at the start of every episode.
    """

    def __init__(self, checkpoint_path, wheel_deadband=0.0, done_checkpoint=None,  # noqa: E501
                 done_threshold=0.5, done_confirm=20, use_qd_des=True, arm_integrate=False, dt=0.001):
        # One sample per call: multi-threading only adds overhead for such tiny matrix products
        # (typically several times faster on a laptop CPU).
        torch.set_num_threads(1)
        self.model, self.cfg = load_model(checkpoint_path)
        self.feature_set = resolve_feature_set(self.cfg["features"], self.model.x_mean.numel())
        if self.feature_set != self.cfg["features"]:
            print(f"note: {checkpoint_path} was trained with an older version of '{self.cfg['features']}' "
                  f"-> using '{self.feature_set}'")
        self.output_mode = self.cfg["output"]
        self.wheel_deadband = wheel_deadband      # see features.decode_actions
        self.use_qd_des = use_qd_des              # see features.decode_actions
        # arm_integrate (pd output only): reproduce the expert's reference INTEGRATOR instead of using the
        # predicted offset q_des - q directly:  q_des <- q_des + qd_des_pred * dt, kept within +-0.15 rad
        # of the measured joints (the expert's anti-windup). Position reference and velocity reference are
        # then consistent, and a small bias in qd_des leaves a small steady offset instead of a drift,
        # because qd_des is learned from the expert's IK, which is proportional to the end-effector error.
        self.arm_integrate, self.dt = arm_integrate, dt
        if arm_integrate and self.output_mode != "pd":
            raise ValueError("arm_integrate needs a model trained with --output pd")
        self.done_model, self.done_cfg = load_model(done_checkpoint) if done_checkpoint else (None, None)
        if self.done_model is not None:
            self.done_features = resolve_feature_set(self.done_cfg["features"], self.done_model.x_mean.numel())
        if self.feature_set.startswith("rel_mode") and self.done_model is None:
            raise ValueError("a 'rel_mode' policy needs --done-checkpoint (the mode classifier)")
        self.done_threshold, self.done_confirm = done_threshold, done_confirm
        self.infer_times = []            # seconds per call (efficiency)
        self.reset()

    def reset(self):
        self.done = False                # latched navigation-done mode
        self.done_step = None            # step at which the mode latched
        self._done_count = 0
        self._step = 0
        self.last_x_norm = None          # for out-of-distribution diagnostics
        self.last_done_prob = np.nan
        self.q_des = None                # integrated joint reference (arm_integrate)

    def act(self, base_pos, yaw, q_arm, qd_arm, wheel_vel, ee_pos, target, tau_g):
        t0 = time.perf_counter()
        meas = (base_pos, yaw, q_arm, qd_arm, wheel_vel, ee_pos, target)

        # 1) discrete mode: learned classifier + latch
        if self.done_model is not None and not self.done:
            xd = torch.from_numpy(build_features(*meas, self.done_features))[None]
            with torch.no_grad():
                self.last_done_prob = float(torch.sigmoid(self.done_model(xd))[0, 0])
            self._done_count = self._done_count + 1 if self.last_done_prob > self.done_threshold else 0
            if self._done_count >= self.done_confirm:
                self.done, self.done_step = True, self._step

        # 2) continuous control, conditioned on the mode for 'rel_mode' policies
        x = build_features(*meas, self.feature_set, nav_done=float(self.done))
        xt = torch.from_numpy(x)[None]
        with torch.no_grad():
            y = self.model(xt)[0].numpy()
            self.last_x_norm = ((xt - self.model.x_mean) / self.model.x_std)[0].numpy()
        if not np.all(np.isfinite(y)):
            raise FloatingPointError("non-finite policy output")
        if self.arm_integrate:
            q = np.asarray(q_arm, float)
            qd_des = np.clip(y[9:16], -MAX_JOINT_VEL, MAX_JOINT_VEL)
            if self.q_des is None:                             # first step: start from the predicted reference
                self.q_des = q + np.clip(y[2:9], -MAX_Q_LEAD, MAX_Q_LEAD)
            else:
                self.q_des = self.q_des + qd_des * self.dt
            self.q_des = np.clip(self.q_des, q - MAX_Q_LEAD, q + MAX_Q_LEAD)   # expert's anti-windup
            y = y.copy()
            y[2:9] = self.q_des - q
        wheel, tau = decode_actions(y, self.output_mode, q_arm, qd_arm, tau_g, self.wheel_deadband, self.use_qd_des)
        if self.done:
            wheel = np.zeros(4)          # the expert's base command in its done state
        self._step += 1
        self.infer_times.append(time.perf_counter() - t0)
        return wheel, tau

    def n_params(self):
        n = sum(p.numel() for p in self.model.parameters())
        if self.done_model is not None:
            n += sum(p.numel() for p in self.done_model.parameters())
        return n
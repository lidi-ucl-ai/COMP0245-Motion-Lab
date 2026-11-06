
import numpy as np

FEATURE_SETS = ("world", "rel", "rel_mode", "rel_wv", "nav")
# Earlier versions of 'rel' / 'rel_mode', kept so that models trained with them can still be loaded:
#   v1 (34 / 35-D): + bearing-to-goal error sin/cos, without the fine-scale end-effector error
#   v2 (32 / 33-D): without bearing-to-goal error and without the fine-scale end-effector error
LEGACY_SETS = ("rel_v1", "rel_v2", "rel_mode_v1", "rel_mode_v2")


def _layout(feature_set):
    """(with bearing-to-goal error, with fine-scale EE error, with nav_done flag, with wheel velocity)."""
    mode = feature_set.startswith("rel_mode")
    if feature_set.endswith("_v1"):
        return True, False, mode, False
    if feature_set.endswith("_v2"):
        return False, False, mode, False
    return False, True, mode, feature_set == "rel_wv"
OUTPUT_MODES = ("torque", "torque_lr", "pd", "done")

# Expert constants (robotnik_arm_control.py / robotnik_arm_task.py, unchanged)
KP, KD = 100.0, 30.0
MAX_TORQUE = 30.0
MAX_Q_LEAD = 0.15          # |q_des - q| never exceeds this in the expert
MAX_JOINT_VEL = 1.0        # |qd_des| never exceeds this in the expert
MAX_WHEEL = 3.0            # the expert's wheel commands never exceed 3 rad/s
FINE_RANGE = 0.05         # m, clip range of the fine-scale end-effector error
DIR_SOFTENING = 0.01      # m, softening of the end-effector error direction
STANDOFF_Z_POINTS = [0.1, 0.5, 1.1, 1.25]
STANDOFF_POINTS = [0.80, 0.65, 0.65, 0.35]


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _to_base(dx, dy, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def resolve_feature_set(feature_set, in_dim):
    """Feature set matching a trained model's input size (handles models trained with older versions)."""
    if len(feature_names(feature_set)) == in_dim:
        return feature_set
    family = [f for f in LEGACY_SETS if f.startswith(feature_set + "_v")]
    for f in family:
        if len(feature_names(f)) == in_dim:
            return f
    raise ValueError(f"model expects {in_dim} inputs; feature set '{feature_set}' has "
                     f"{len(feature_names(feature_set))} and no older version matches")


def feature_names(feature_set):
    if feature_set == "world":
        return (["base_x", "base_y", "base_z", "sin_yaw", "cos_yaw"]
                + [f"q{i}" for i in range(1, 8)] + [f"qd{i}" for i in range(1, 8)]
                + ["target_x", "target_y", "target_z"])
    if feature_set == "nav":
        return ["goal_rel_x", "goal_rel_y", "goal_dist", "sin_final_err", "cos_final_err"]
    head, fine, mode, wv = _layout(feature_set)
    names = (["tgt_rel_x", "tgt_rel_y", "tgt_z", "ee_err_x", "ee_err_y", "ee_err_z"]
             + (["ee_fine_x", "ee_fine_y", "ee_fine_z", "ee_dir_x", "ee_dir_y", "ee_dir_z"] if fine else [])
             + ["goal_rel_x", "goal_rel_y", "goal_dist"]
             + (["sin_head_err", "cos_head_err"] if head else [])
             + ["sin_final_err", "cos_final_err"]
             + [f"sin_q{i}" for i in range(1, 8)] + [f"cos_q{i}" for i in range(1, 8)]
             + [f"qd{i}" for i in range(1, 8)])
    if wv:
        names += [f"wheel_vel{i}" for i in range(1, 5)]
    return names + ["nav_done"] if mode else names


def output_names(output_mode):
    if output_mode == "torque":
        return ["wheel_fr", "wheel_fl", "wheel_bl", "wheel_br"] + [f"tau{i}" for i in range(1, 8)]
    if output_mode == "torque_lr":
        return ["wheel_left", "wheel_right"] + [f"tau{i}" for i in range(1, 8)]
    if output_mode == "done":
        return ["nav_done_logit"]
    return (["wheel_left", "wheel_right"] + [f"dq{i}" for i in range(1, 8)]
            + [f"qd_des{i}" for i in range(1, 8)])


def build_features(base_pos, yaw, q_arm, qd_arm, wheel_vel, ee_pos, target, feature_set, nav_done=None):
    """Works on one step (1-D inputs) or on a batch (leading dimension N).
    nav_done (bool / 0-1 array) is required for feature_set 'rel_mode'."""
    single = np.ndim(base_pos) == 1
    base_pos = np.atleast_2d(base_pos).astype(float)
    q_arm = np.atleast_2d(q_arm).astype(float)
    qd_arm = np.atleast_2d(qd_arm).astype(float)
    wheel_vel = np.atleast_2d(wheel_vel).astype(float)
    ee_pos = np.atleast_2d(ee_pos).astype(float)
    yaw = np.atleast_1d(yaw).astype(float)
    target = np.broadcast_to(np.asarray(target, dtype=float), ee_pos.shape)

    if feature_set == "world":
        F = np.column_stack([base_pos, np.sin(yaw), np.cos(yaw), q_arm, qd_arm, target])
    elif feature_set in ("rel", "rel_mode", "rel_wv", "nav") + LEGACY_SETS:
        tx, ty = _to_base(target[:, 0] - base_pos[:, 0], target[:, 1] - base_pos[:, 1], yaw)
        ex, ey = _to_base(target[:, 0] - ee_pos[:, 0], target[:, 1] - ee_pos[:, 1], yaw)
        ez = target[:, 2] - ee_pos[:, 2]
        # Expert's base goal: computed from the target only, exactly as in robotnik_arm_task.py
        standoff = np.interp(target[:, 2], STANDOFF_Z_POINTS, STANDOFF_POINTS)
        brg = np.arctan2(target[:, 1], target[:, 0])
        gdx = target[:, 0] - standoff * np.cos(brg) - base_pos[:, 0]
        gdy = target[:, 1] - standoff * np.sin(brg) - base_pos[:, 1]
        gx, gy = _to_base(gdx, gdy, yaw)
        final_err = _wrap(brg - yaw)
        nav = [gx, gy, np.hypot(gdx, gdy), np.sin(final_err), np.cos(final_err)]
        if feature_set == "nav":
            return np.column_stack(nav).astype(np.float32)[0 if single else slice(None)]
        head, fine, mode, wv = _layout(feature_set)
        e = np.column_stack([ex, ey, ez])
        cols = [tx, ty, target[:, 2], e]
        if fine:
            cols += [np.clip(e, -FINE_RANGE, FINE_RANGE), e / (np.linalg.norm(e, axis=1, keepdims=True) + DIR_SOFTENING)]
        cols += [gx, gy, np.hypot(gdx, gdy)]
        if head:
            head_err = _wrap(np.arctan2(gdy, gdx) - yaw)
            cols += [np.sin(head_err), np.cos(head_err)]
        cols += [np.sin(final_err), np.cos(final_err), np.sin(q_arm), np.cos(q_arm), qd_arm]
        F = np.column_stack(cols)
        if wv:
            F = np.column_stack([F, wheel_vel])
        if mode:
            if nav_done is None:
                raise ValueError("feature_set 'rel_mode' needs nav_done")
            F = np.column_stack([F, np.broadcast_to(np.asarray(nav_done, float), (len(F),))])
    else:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
    F = F.astype(np.float32)
    return F[0] if single else F


def features_from_episode(d, feature_set):
    """Features for every logged step of one episode file (dict-like from np.load, collect_data.py format)."""
    return build_features(d["base_pos"], d["yaw"], d["q_arm"], d["qd_arm"], d["wheel_vel"], d["ee_pos"],
                          d["target"], feature_set,
                          nav_done=d["nav_done"] if feature_set.startswith("rel_mode") else None)


def targets_from_episode(d, output_mode):
    """Regression targets for every logged step of one episode file."""
    W, tau = d["wheel_cmd"], d["torque_arm"]          # wheel order: [front_right, front_left, back_left, back_right]
    if output_mode == "torque":
        T = np.column_stack([W, tau])
    elif output_mode == "torque_lr":
        T = np.column_stack([W[:, 1], W[:, 0], tau])
    elif output_mode == "pd":
        T = np.column_stack([W[:, 1], W[:, 0], d["q_des_arm"] - d["q_arm"], d["qd_des_arm"]])
    elif output_mode == "done":
        T = np.asarray(d["nav_done"], float)[:, None]
    else:
        raise ValueError(f"output_mode must be one of {OUTPUT_MODES}")
    return T.astype(np.float32)


def decode_actions(y, output_mode, q_arm, qd_arm, tau_g, wheel_deadband=0.0, use_qd_des=True):
    """Model output (one step or batch, physical units) -> (wheel_cmds (...,4), torque_arm (...,7)).

    Safety limits are the expert's own limits: wheels +-3 rad/s, torques +-30 N*m,
    |q_des - q| <= 0.15 rad and |qd_des| <= 1 rad/s.

    wheel_deadband: if BOTH wheel sides are below this speed (rad/s), the base command is set to
    exactly 0. Justification: the expert's wheel commands are either exactly 0 (navigation done /
    holding) or larger than ~0.45 rad/s (its minimum turn rate 0.15 rad/s and minimum drive speed
    0.05 m/s), so small predicted speeds are regression error, and they make the base creep.

    use_qd_des (pd mode): if False the predicted reference velocity is ignored (qd_des = 0), i.e.
    torque = kp*(q_des-q) - kd*qd + tau_g. In the expert q_des is the INTEGRAL of qd_des, so the two are
    always consistent; the network predicts them independently, and a small bias in qd_des makes the
    kd-term drive the joints at a constant speed (slow drift away from the target). Without qd_des the
    arm is anchored by the predicted position offset only (pure damping, no velocity bias).
    """
    y = np.asarray(y, dtype=float)
    if output_mode == "torque":
        wheel, tau = y[..., 0:4], y[..., 4:11]
    else:
        lw, rw = y[..., 0], y[..., 1]
        wheel = np.stack([rw, lw, lw, rw], axis=-1)
        if output_mode == "torque_lr":
            tau = y[..., 2:9]
        else:
            dq = np.clip(y[..., 2:9], -MAX_Q_LEAD, MAX_Q_LEAD)
            qd_des = np.clip(y[..., 9:16], -MAX_JOINT_VEL, MAX_JOINT_VEL) if use_qd_des else 0.0
            tau = KP * dq + KD * (qd_des - np.asarray(qd_arm)) + np.asarray(tau_g)
    wheel = np.clip(wheel, -MAX_WHEEL, MAX_WHEEL)
    if wheel_deadband > 0:
        still = np.max(np.abs(wheel), axis=-1, keepdims=True) < wheel_deadband
        wheel = np.where(still, 0.0, wheel)
    return wheel, np.clip(tau, -MAX_TORQUE, MAX_TORQUE)
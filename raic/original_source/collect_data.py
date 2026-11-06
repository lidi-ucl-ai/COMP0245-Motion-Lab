
import argparse
import csv
import os
import time

import numpy as np

from sim_utils import (HERE, MIN_TIME, REACH_TOLERANCE, Expert, add_target_marker, clip_to_limits,
                       default_arm_q, make_env, measure, send, set_initial_state)

LOG_KEYS = ["t", "base_pos", "base_ori", "yaw", "q_arm", "qd_arm", "wheel_vel", "ee_pos",
            "wheel_cmd", "torque_arm", "tau_g", "q_des_arm", "qd_des_arm", "nav_done", "phase"]


def parse_args():
    ap = argparse.ArgumentParser(description="Collect expert demonstrations.")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "train"))
    ap.add_argument("--r-min", type=float, default=1.0)
    ap.add_argument("--r-max", type=float, default=2.0)
    ap.add_argument("--z-min", type=float, default=0.40)
    ap.add_argument("--z-max", type=float, default=0.65)
    ap.add_argument("--base-xy-noise", type=float, default=0.10, help="initial base offset, uniform +- (m)")
    ap.add_argument("--yaw-range", type=float, default=0.20, help="initial base yaw, uniform +- (rad)")
    ap.add_argument("--arm-noise", type=float, default=0.04, help="initial arm joint offsets, uniform +- (rad)")
    ap.add_argument("--hold-time", type=float, default=2.0)
    ap.add_argument("--max-time", type=float, default=40.0)
    ap.add_argument("--dart-wheel", type=float, default=0.0, help="DART: std of wheel-speed offset per side (rad/s)")
    ap.add_argument("--dart-tau", type=float, default=0.0, help="DART: std of arm-torque offset per joint (N*m)")
    ap.add_argument("--dart-period", type=float, default=0.5, help="DART: seconds between new offsets")
    ap.add_argument("--gui", action="store_true")
    return ap.parse_args()


def sample_episode(rng, args):
    r = np.sqrt(rng.uniform(args.r_min ** 2, args.r_max ** 2))       # uniform over the annulus area
    th = rng.uniform(-np.pi, np.pi)
    target = np.array([r * np.cos(th), r * np.sin(th), rng.uniform(args.z_min, args.z_max)])
    base_xy = rng.uniform(-args.base_xy_noise, args.base_xy_noise, 2)
    yaw0 = rng.uniform(-args.yaw_range, args.yaw_range)
    arm_offset = rng.uniform(-args.arm_noise, args.arm_noise, 7)
    return target, base_xy, yaw0, arm_offset


def run_expert_episode(target, base_xy, yaw0, arm_offset, args, rng):
    env = make_env(gui=args.gui)
    try:
        # initial state: template start pose + offsets
        base_pos0 = np.asarray(env.sim.GetBasePosition(), float)
        base_pos0[:2] += base_xy
        base_ori0 = np.array(env.client.getQuaternionFromEuler([0.0, 0.0, yaw0]))
        arm_q0 = clip_to_limits(env, default_arm_q(env) + arm_offset)
        set_initial_state(env, base_pos0, base_ori0, arm_q0)
        if args.gui:
            add_target_marker(env, target)

        expert = Expert(env, target, default_arm_q(env))
        log = {k: [] for k in LOG_KEYS}
        reached_t, err, t, step = None, np.nan, 0.0, 0
        period = max(1, int(round(args.dart_period / env.dt)))
        dart_w, dart_tau = np.zeros(2), np.zeros(7)
        while t < args.max_time:
            m = measure(env)
            wheel, tau = expert.act(m, t)
            err = float(np.linalg.norm(target - m["ee"]))
            for k, v in (("t", t), ("base_pos", m["base_pos"]), ("base_ori", m["base_ori"]), ("yaw", m["yaw"]),
                         ("q_arm", m["q_arm"]), ("qd_arm", m["qd_arm"]), ("wheel_vel", m["wheel_vel"]),
                         ("ee_pos", m["ee"]), ("wheel_cmd", wheel), ("torque_arm", tau), ("tau_g", m["tau_g"]),
                         ("q_des_arm", expert.q_des), ("qd_des_arm", expert.qd_des),
                         ("nav_done", expert.state.done), ("phase", expert.state.phase)):
                log[k].append(np.copy(v))
            if args.dart_wheel > 0 or args.dart_tau > 0:      # DART: perturb the EXECUTED command only
                if step % period == 0:
                    dart_w = rng.normal(0, args.dart_wheel, 2) if args.dart_wheel > 0 else np.zeros(2)
                    dart_tau = rng.normal(0, args.dart_tau, 7) if args.dart_tau > 0 else np.zeros(7)
                dl, dr = dart_w
                wheel_exec = np.clip(wheel + np.array([dr, dl, dl, dr]), -3.0, 3.0)
                if reached_t is not None:                         # no perturbation during the hold
                    wheel_exec, tau_exec = wheel, tau
                else:
                    tau_exec = np.clip(tau + dart_tau, -30.0, 30.0)
                send(env, wheel_exec, tau_exec)
            else:
                send(env, wheel, tau)
            step += 1
            t = step * env.dt
            # same success rule as the template's check_stop: base done, error < 1 cm, after 8 s
            if reached_t is None and expert.state.done and err < REACH_TOLERANCE and t > MIN_TIME:
                reached_t = t
            if reached_t is not None and t - reached_t >= args.hold_time:
                break
        out = {k: np.asarray(v) for k, v in log.items()}
        out["phase"] = out["phase"].astype("U6")
        out["nav_done"] = out["nav_done"].astype(bool)
        meta = {"target": target, "initial_base_pos": base_pos0, "initial_base_ori": base_ori0,
                "initial_arm_q": arm_q0, "success": reached_t is not None,
                "reach_time": np.nan if reached_t is None else reached_t, "hold_time": args.hold_time,
                "final_error": err, "dt": env.dt, "dart_wheel": args.dart_wheel, "dart_tau": args.dart_tau}
        return out, meta
    finally:
        env.close()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    summary = os.path.join(args.out, "summary.csv")
    new = not os.path.exists(summary)
    with open(summary, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["episode", "tx", "ty", "tz", "r", "success", "reach_time", "final_error", "steps", "wall_s"])
        for ep in range(args.start, args.start + args.n):
            path = os.path.join(args.out, f"episode_{ep:04d}.npz")
            if os.path.exists(path):
                print(f"episode {ep:4d}: exists, skipped")
                continue
            rng = np.random.default_rng([args.seed, ep])          # reproducible per episode
            target, base_xy, yaw0, arm_offset = sample_episode(rng, args)
            t0 = time.time()
            log, meta = run_expert_episode(target, base_xy, yaw0, arm_offset, args, rng)
            tmp = path[:-4] + ".partial.npz"                 # write, then rename: an interrupted run
            np.savez_compressed(tmp, **log, **meta)           # never leaves a half-written episode
            os.replace(tmp, path)
            r = float(np.hypot(*target[:2]))
            w.writerow([ep, *np.round(target, 4), round(r, 3), meta["success"], round(float(meta["reach_time"]), 3),
                        round(meta["final_error"], 5), len(log["t"]), round(time.time() - t0, 1)])
            f.flush()
            print(f"episode {ep:4d}: target={np.round(target, 2)} r={r:.2f} "
                  f"{'OK  ' if meta['success'] else 'FAIL'} reach_t={meta['reach_time']:.2f}s "
                  f"err={meta['final_error'] * 1000:.1f}mm steps={len(log['t'])} wall={time.time() - t0:.0f}s",
                  flush=True)


if __name__ == "__main__":
    main()
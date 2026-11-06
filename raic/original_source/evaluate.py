"""
evaluate.py - Closed-loop evaluation: the policy (or the expert) actually drives the robot.

For every episode, the simulator is reset to the SAME target and initial state that was recorded
for it, and the controller runs headless.
  reached : end-effector within 1 cm of the target after t >= 8 s (the template's success rule)
  held    : after reaching, the controller keeps running for --hold seconds and the end-effector
            stays within --hold-tol of the target (does it stop, or drift away again?)
  diverged: base lifted off the ground or arm spinning (rollout stopped early)

Outputs in --out:  rollouts.csv (one row per episode), summary.json

Examples
    python evaluate.py --controller expert --out eval/expert_test
    python evaluate.py --checkpoint runs/C_rel_pd/model_best.pt --out eval/C_test
    python evaluate.py --checkpoint runs/D_relmode_pd/model_best.pt --done-checkpoint runs/D_done/model_best.pt --out eval/D_test
    python evaluate.py --checkpoint ... --noise-q 0.005 --noise-qd 0.05 --out eval/D_noise      # robustness
    python evaluate.py --checkpoint ... --data data/test_ood --split all --out eval/D_ood         # generalisation

DAgger (Ross et al., AISTATS 2011): with --dagger-out DIR, the expert runs in SHADOW on every state the
policy visits and its commands are saved as labels (same format as collect_data.py). Run it on the
TRAINING split, then retrain with  train.py --extra-train DIR :
    python evaluate.py --checkpoint ... --done-checkpoint ... --split train --dagger-out data/dagger1 --out eval/dagger1
"""
import argparse
import csv
import glob
import json
import os
import time

import numpy as np

from sim_utils import MIN_TIME, REACH_TOLERANCE, Expert, default_arm_q, make_env, measure, send, set_initial_state

DAGGER_KEYS = ["t", "base_pos", "base_ori", "yaw", "q_arm", "qd_arm", "wheel_vel", "ee_pos", "wheel_cmd",
               "torque_arm", "tau_g", "q_des_arm", "qd_des_arm", "nav_done", "phase"]


def parse_args():
    ap = argparse.ArgumentParser(description="Closed-loop evaluation of a policy or the expert.")
    ap.add_argument("--controller", choices=["policy", "expert"], default="policy")
    ap.add_argument("--checkpoint", help="control model from train.py")
    ap.add_argument("--done-checkpoint", help="mode classifier from train.py --output done (latched)")
    ap.add_argument("--done-confirm", type=int, default=20, help="consecutive 'done' steps before latching")
    ap.add_argument("--done-threshold", type=float, default=0.5, help="classifier probability needed for 'done'")
    ap.add_argument("--no-qd-des", action="store_true", help="pd output: ignore predicted qd_des (see features.py)")
    ap.add_argument("--arm-integrate", action="store_true",
                    help="pd output: integrate the predicted qd_des into q_des like the expert (see model.py)")
    ap.add_argument("--wheel-deadband", type=float, default=0.0, help="rad/s, 0 = off (see features.py)")
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--split", default="test", help="test | val | train (from split.json) | all")
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--max-time", type=float, default=40.0)
    ap.add_argument("--min-time", type=float, default=MIN_TIME,
                    help="earliest time (s) at which success can count; 8 s in the course template, 0 for the "
                         "shared-benchmark rule (1 cm for 0.5 s at any time)")
    ap.add_argument("--settle", type=float, default=0.5,
                    help="reached = within 1 cm CONTINUOUSLY for this many seconds (t > 8 s). Without it, an "
                         "end-effector that only sweeps through the target while the base is still moving would count")
    ap.add_argument("--hold", type=float, default=2.0, help="seconds to keep running after reaching")
    ap.add_argument("--hold-tol", type=float, default=0.02, help="m, allowed error during the hold")
    ap.add_argument("--noise-q", type=float, default=0.0, help="rad, noise on measured arm angles")
    ap.add_argument("--noise-qd", type=float, default=0.0, help="rad/s, noise on measured arm velocities")
    ap.add_argument("--stall-time", type=float, default=6.0,
                    help="stop a rollout after t > 10 s if its minimum error has not improved by 2 mm for this "
                         "many seconds (it will not reach any more; saves most of the time of failed rollouts)")
    ap.add_argument("--workers", type=int, default=1, help="episodes simulated in parallel processes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dagger-out", help="save policy-visited states with shadow-expert labels here (DAgger)")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def episode_list(args):
    if args.split == "all":
        return sorted(os.path.basename(f) for f in glob.glob(os.path.join(args.data, "episode_*.npz")))
    with open(os.path.join(args.data, "split.json")) as f:
        names = json.load(f)[args.split]
    return names[:args.max_episodes] if args.max_episodes else names


def rollout(setup, args, policy, rng):
    env = make_env(gui=False)
    try:
        set_initial_state(env, setup["initial_base_pos"], setup["initial_base_ori"], setup["initial_arm_q"])
        target = setup["target"]
        expert = Expert(env, target, default_arm_q(env)) if policy is None else None
        shadow = Expert(env, target, default_arm_q(env)) if (policy is not None and args.dagger_out) else None
        dlog = {k: [] for k in DAGGER_KEYS} if shadow is not None else None
        if policy is not None:
            policy.reset()
        z0 = float(np.asarray(env.sim.GetBasePosition())[2])
        res = {"reached": False, "held": False, "reach_time": np.nan, "max_hold_error": np.nan,
               "reason": "time limit"}
        min_err, err, t, step, hold_errs = np.inf, np.nan, 0.0, 0, []
        best_for_stall, best_t = np.inf, 0.0
        inside_since = None
        while t < args.max_time:
            m = measure(env, args.noise_q, args.noise_qd, rng)
            err = float(np.linalg.norm(target - m["ee_true"]))
            min_err = min(min_err, err)
            if err < best_for_stall - 0.002:
                best_for_stall, best_t = err, t
            elif (args.stall_time > 0 and not res["reached"] and t > 10.0
                  and t - best_t > args.stall_time):
                res["reason"] = f"stalled (no progress for {args.stall_time:.0f} s)"
                break
            if expert is not None:
                wheel, tau = expert.act(m, t)
            else:
                try:
                    wheel, tau = policy.act(m["base_pos"], m["yaw"], m["q_arm"], m["qd_arm"], m["wheel_vel"],
                                            m["ee"], target, m["tau_g"])
                except FloatingPointError:
                    res["reason"] = "non-finite policy output"
                    break
            if shadow is not None:       # DAgger: expert label on the state the POLICY reached
                if any(not np.all(np.isfinite(v)) for v in (m["q_arm"], m["qd_arm"], m["ee"])):
                    break
                wheel_e, tau_e = shadow.act(m, t)
                for k, v in (("t", t), ("base_pos", m["base_pos"]), ("base_ori", m["base_ori"]), ("yaw", m["yaw"]),
                             ("q_arm", m["q_arm"]), ("qd_arm", m["qd_arm"]), ("wheel_vel", m["wheel_vel"]),
                             ("ee_pos", m["ee"]), ("wheel_cmd", wheel_e), ("torque_arm", tau_e), ("tau_g", m["tau_g"]),
                             ("q_des_arm", shadow.q_des), ("qd_des_arm", shadow.qd_des),
                             ("nav_done", shadow.state.done), ("phase", shadow.state.phase)):
                    dlog[k].append(np.copy(v))
            send(env, wheel, tau)
            step += 1
            t = step * env.dt

            if not res["reached"]:
                if t > args.min_time and err < REACH_TOLERANCE:
                    inside_since = t if inside_since is None else inside_since
                    if t - inside_since >= args.settle - 1e-9:
                        res.update(reached=True, reach_time=inside_since, reason="reached")
                else:
                    inside_since = None
            if res["reached"]:
                hold_errs.append(err)
                if t - res["reach_time"] >= args.hold:
                    break
            lift = float(np.asarray(env.sim.GetBasePosition())[2]) - z0
            spin = float(np.max(np.abs(m["qd_arm"])))
            if lift > 0.10 or spin > 10.0 or not np.isfinite(err):
                res["reason"] = f"diverged (base lift {lift:.2f} m, arm speed {spin:.1f} rad/s)"
                break
        if res["reached"] and hold_errs:
            res["max_hold_error"] = float(np.max(hold_errs))
            res["held"] = bool(t - res["reach_time"] >= args.hold - 1e-9 and res["max_hold_error"] < args.hold_tol)
        if dlog is not None and len(dlog["t"]) > 100:
            out = {k: np.asarray(v) for k, v in dlog.items()}
            out["phase"] = out["phase"].astype("U6")
            out["nav_done"] = out["nav_done"].astype(bool)
            res["dagger_data"] = out
        res.update(final_error=err, min_error=min_err, duration=t,
                   done_latched_at=(policy.done_step * env.dt) if (policy is not None and policy.done_step is not None)
                   else np.nan)
        return res
    finally:
        env.close()


_POLICY = None


def _make_policy(args):
    from model import BCPolicy
    return BCPolicy(args.checkpoint, wheel_deadband=args.wheel_deadband,
                    done_checkpoint=args.done_checkpoint, done_confirm=args.done_confirm,
                    done_threshold=args.done_threshold, use_qd_des=not args.no_qd_des,
                    arm_integrate=args.arm_integrate)


def run_one(job):
    """Evaluate one episode (also used inside worker processes)."""
    global _POLICY
    i, name, args = job
    if args.controller == "policy" and _POLICY is None:
        _POLICY = _make_policy(args)
    with np.load(os.path.join(args.data, name)) as d:
        setup = {k: np.asarray(d[k], float) for k in ("target", "initial_base_pos", "initial_base_ori", "initial_arm_q")}
    t0 = time.time()
    rng = np.random.default_rng([args.seed, i])                       # same noise whatever the worker count
    r = rollout(setup, args, _POLICY, rng)
    r.update(episode=name, target_x=setup["target"][0], target_y=setup["target"][1],
             target_z=setup["target"][2], target_r=float(np.hypot(*setup["target"][:2])), wall_s=time.time() - t0)
    if _POLICY is not None:
        r["infer_ms"] = float(np.mean(_POLICY.infer_times[-10000:]) * 1e3) if _POLICY.infer_times else np.nan
    return r


def main():
    args = parse_args()
    if args.controller == "policy":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for --controller policy")
        p = _make_policy(args)
        print(f"Policy: features={p.feature_set} output={p.output_mode} "
              f"mode classifier={'yes' if args.done_checkpoint else 'no'} params={p.n_params()}")
        n_params = p.n_params()
    os.makedirs(args.out, exist_ok=True)
    jobs = [(i, name, args) for i, name in enumerate(episode_list(args))]

    rows = []
    def report(r):
        rows.append(r)
        tag = "HELD" if r["held"] else ("REACH" if r["reached"] else "FAIL")
        print(f"[{len(rows):3d}/{len(jobs)}] {r['episode']} {tag:5s} reach_t={r['reach_time']:6.2f}s "
              f"min={r['min_error'] * 1000:7.1f}mm final={r['final_error'] * 1000:7.1f}mm  {r['reason']}  "
              f"({r['wall_s']:.0f}s)", flush=True)
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for r in pool.imap_unordered(run_one, jobs):
                report(r)
        rows.sort(key=lambda r: r["episode"])
    else:
        for job in jobs:
            report(run_one(job))

    keys = ["episode", "reached", "held", "reach_time", "max_hold_error", "final_error", "min_error", "duration",
            "done_latched_at", "reason", "target_x", "target_y", "target_z", "target_r", "wall_s"]
    with open(os.path.join(args.out, "rollouts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: r[k] for k in keys} for r in rows)

    reach = np.array([r["reached"] for r in rows])
    held = np.array([r["held"] for r in rows])
    rt = np.array([r["reach_time"] for r in rows if r["reached"]])
    summary = {
        "controller": args.controller, "checkpoint": args.checkpoint, "done_checkpoint": args.done_checkpoint,
        "wheel_deadband": args.wheel_deadband, "done_threshold": args.done_threshold,
        "done_confirm": args.done_confirm, "use_qd_des": not args.no_qd_des, "arm_integrate": args.arm_integrate,
        "settle_s": args.settle, "min_time_s": args.min_time, "stall_time_s": args.stall_time,
        "max_time_s": args.max_time, "hold_s": args.hold, "hold_tol_m": args.hold_tol,
        "data": args.data, "split": args.split, "max_episodes": args.max_episodes,
        "noise_q": args.noise_q, "noise_qd": args.noise_qd, "n_episodes": len(rows),
        "reach_rate": float(reach.mean()), "hold_rate": float(held.mean()),
        "median_min_error_m": float(np.median([r["min_error"] for r in rows])),
        "mean_reach_time_s": float(rt.mean()) if len(rt) else None,
        "n_diverged": int(sum(r["reason"].startswith("diverged") for r in rows)),
        "n_stalled": int(sum(r["reason"].startswith("stalled") for r in rows)),
    }
    if args.controller == "policy":
        summary.update(inference_ms_per_step=float(np.nanmean([r.get("infer_ms", np.nan) for r in rows])),
                       n_params=n_params)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

import argparse
import csv
import glob
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from features import (FEATURE_SETS, OUTPUT_MODES, decode_actions, features_from_episode,
                      output_names, targets_from_episode)
from model import MLP

NEW_FIELDS = ("base_pos", "yaw", "q_arm", "qd_arm", "wheel_vel", "ee_pos", "wheel_cmd", "torque_arm",
              "tau_g", "q_des_arm", "qd_des_arm", "nav_done")


def parse_args():
    ap = argparse.ArgumentParser(description="Train a behaviour-cloning MLP.")
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--out", default="runs/C_rel_pd")
    ap.add_argument("--features", default="rel", choices=FEATURE_SETS)
    ap.add_argument("--output", default="pd", choices=OUTPUT_MODES)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3, help="hidden layers (0 = linear regression)")
    ap.add_argument("--act", default="silu", choices=["relu", "silu", "tanh"])
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--subsample", type=int, default=10, help="use one sample every k simulation steps")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=25, help="early stopping on validation loss")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=0, help="only used when split.json is created")
    ap.add_argument("--seed", type=int, default=0, help="network initialisation / shuffling")
    ap.add_argument("--max-train-episodes", type=int, default=0,
                    help="use only the first k training episodes (data-quantity experiment; 0 = all)")
    ap.add_argument("--include-failed", action="store_true")
    ap.add_argument("--near-weight", type=float, default=0.0,
                    help="extra loss weight for samples whose end-effector is within --near-radius of the target "
                         "(the last centimetres decide success but are a small share of each episode); 0 = off")
    ap.add_argument("--near-radius", type=float, default=0.05, help="m")
    ap.add_argument("--extra-train", nargs="*", default=[],
                    help="extra folders whose episodes are added to the TRAINING set only (e.g. DAgger data)")
    return ap.parse_args()


# ==============================================================================
# Data
# ==============================================================================
def make_or_load_split(data_dir, args):
    path = os.path.join(data_dir, "split.json")
    if os.path.exists(path):
        with open(path) as f:
            split = json.load(f)
        print(f"Using existing split {path}: "
              + ", ".join(f"{k}={len(v)}" for k, v in split.items()))
        return split

    files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(data_dir, "episode_*.npz")))
    usable, skipped = [], {"old_format": 0, "failed": 0, "unreadable": 0}
    for name in files:
        if name.endswith(".partial.npz"):
            continue
        try:
            with np.load(os.path.join(data_dir, name)) as d:
                if not all(k in d.files for k in NEW_FIELDS):
                    skipped["old_format"] += 1
                    continue
                if not args.include_failed and not bool(d["success"]):
                    skipped["failed"] += 1
                    continue
                len(d["t"])                                   # forces reading: catches truncated files
        except Exception as e:
            print(f"  skipping unreadable file {name}: {type(e).__name__}")
            skipped["unreadable"] += 1
            continue
        usable.append(name)
    if not usable:
        raise SystemExit(f"No usable episodes in {data_dir} (skipped: {skipped})")

    rng = np.random.default_rng(args.split_seed)
    order = [usable[i] for i in rng.permutation(len(usable))]
    n_test = max(1, round(args.test_frac * len(order)))
    n_val = max(1, round(args.val_frac * len(order)))
    split = {"test": sorted(order[:n_test]), "val": sorted(order[n_test:n_test + n_val]),
             "train": sorted(order[n_test + n_val:])}
    with open(path, "w") as f:
        json.dump(split, f, indent=1)
    print(f"Created {path}: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])} "
          f"(skipped {skipped})")
    return split


def load_episodes(data_dir, names, args):
    """Per episode: inputs, targets and what is needed to rebuild torques (all subsampled)."""
    eps = []
    for name in names:
        with np.load(os.path.join(data_dir, name)) as d:
            idx = np.arange(0, len(d["t"]), args.subsample)
            eps.append({
                "name": name,
                "X": features_from_episode(d, args.features)[idx],
                "T": targets_from_episode(d, args.output)[idx],
                "Y": np.column_stack([d["wheel_cmd"], d["torque_arm"]])[idx].astype(np.float32),
                "q": d["q_arm"][idx], "qd": d["qd_arm"][idx], "tau_g": d["tau_g"][idx],
                "target": np.asarray(d["target"], float),
                "near": (np.linalg.norm(d["ee_pos"][idx] - d["target"], axis=1) < args.near_radius).astype(np.float32),
            })
    return eps


def stack(eps, key):
    return np.concatenate([e[key] for e in eps], axis=0)


# ==============================================================================
# Evaluation
# ==============================================================================
@torch.no_grad()
def predict(model, X, batch=16384):
    model.eval()
    out = [model(torch.from_numpy(X[i:i + batch])).numpy() for i in range(0, len(X), batch)]
    return np.concatenate(out, axis=0)


def normalised_mse(model, X, T):
    P = predict(model, X)
    y_std = model.y_std.numpy()
    return float((((P - T) / y_std) ** 2).mean()), P


def bce(logits, labels):
    """Binary cross-entropy from logits (numerically stable)."""
    return float(np.mean(np.maximum(logits, 0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))))


def split_loss(model, X, T, output_mode):
    if output_mode == "done":
        P = predict(model, X)
        return bce(P, T), P
    return normalised_mse(model, X, T)


def evaluate_classifier(model, eps):
    X, T = stack(eps, "X"), stack(eps, "T")[:, 0]
    P = predict(model, X)[:, 0] > 0
    return {"bce": bce(predict(model, X), stack(eps, "T")), "accuracy": float((P == (T > 0.5)).mean()),
            "false_done_rate": float(P[T < 0.5].mean()) if (T < 0.5).any() else None,
            "missed_done_rate": float((~P[T > 0.5]).mean()) if (T > 0.5).any() else None}


def evaluate_split(model, eps, output_mode):
    """Losses and errors in the model's own outputs AND in the torque/wheel commands sent to the
    robot, so models with different output designs can be compared on equal terms."""
    X, T, Y = stack(eps, "X"), stack(eps, "T"), stack(eps, "Y")
    loss, P = normalised_mse(model, X, T)
    wheel, tau = decode_actions(P, output_mode, stack(eps, "q"), stack(eps, "qd"), stack(eps, "tau_g"))
    return {
        "normalised_mse": loss,
        "rmse_per_output": dict(zip(output_names(output_mode),
                                    np.sqrt(((P - T) ** 2).mean(0)).astype(float).tolist())),
        "wheel_cmd_rmse": float(np.sqrt(((wheel - Y[:, :4]) ** 2).mean())),
        "arm_torque_rmse": float(np.sqrt(((tau - Y[:, 4:11]) ** 2).mean())),
        "arm_torque_rmse_per_joint": np.sqrt(((tau - Y[:, 4:11]) ** 2).mean(0)).astype(float).tolist(),
    }


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = parse_args()
    os.makedirs(os.path.join(args.out, "checkpoints"), exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    split = make_or_load_split(args.data, args)
    train_names = split["train"][:args.max_train_episodes] if args.max_train_episodes else split["train"]
    train_eps = load_episodes(args.data, train_names, args)
    for extra in args.extra_train:                       # e.g. DAgger rounds: training only, never val/test
        names = sorted(os.path.basename(f) for f in glob.glob(os.path.join(extra, "*.npz"))
                       if not f.endswith(".partial.npz"))
        train_eps += load_episodes(extra, names, args)
        print(f"Added {len(names)} episodes from {extra} to the training set")
    val_eps = load_episodes(args.data, split["val"], args)
    test_eps = load_episodes(args.data, split["test"], args)
    Xtr, Ttr = stack(train_eps, "X"), stack(train_eps, "T")
    Xva, Tva = stack(val_eps, "X"), stack(val_eps, "T")
    print(f"Features '{args.features}' ({Xtr.shape[1]}-D) -> output '{args.output}' ({Ttr.shape[1]}-D)")
    print(f"Samples: train {len(Xtr)} ({len(train_eps)} ep) | val {len(Xva)} ({len(val_eps)} ep) | "
          f"test {sum(len(e['X']) for e in test_eps)} ({len(test_eps)} ep)")

    # Normalisation from the TRAINING split only. The std floor stops near-constant inputs
    # (e.g. base height) from turning tiny disturbances into huge normalised values.
    x_std = np.maximum(Xtr.std(0), 1e-3)
    y_std = np.maximum(Ttr.std(0), 1e-6)
    model = MLP(Xtr.shape[1], Ttr.shape[1], args.hidden, args.layers, args.act, args.dropout)
    if args.output == "done":            # classifier: the output is a logit, no output scaling
        model.set_normalisation(Xtr.mean(0), x_std, np.zeros(1), np.ones(1))
    else:
        model.set_normalisation(Xtr.mean(0), x_std, Ttr.mean(0), y_std)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MLP: {args.layers} hidden layers x {args.hidden} ({args.act}), {n_params} parameters")

    def save(path, epoch):
        torch.save({"state_dict": model.state_dict(), "config": vars(args), "epoch": epoch,
                    "in_dim": Xtr.shape[1], "out_dim": Ttr.shape[1]}, path)

    Xtr_t, Ttr_t = torch.from_numpy(Xtr), torch.from_numpy(Ttr)
    Wtr_t = torch.from_numpy(1.0 + args.near_weight * stack(train_eps, "near"))       # per-sample loss weight
    if args.near_weight > 0:
        print(f"near-target samples (< {args.near_radius} m): {stack(train_eps, 'near').mean():.1%}, "
              f"weighted x{1 + args.near_weight:g}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
    hist = {"train": [], "val": []}
    best, best_epoch, bad = np.inf, 0, 0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        total = 0.0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            pred = model.forward_normalised(Xtr_t[b])
            w = Wtr_t[b]
            if args.output == "done":
                per = torch.nn.functional.binary_cross_entropy_with_logits(pred, Ttr_t[b], reduction="none").mean(1)
            else:
                per = ((pred - (Ttr_t[b] - model.y_mean) / model.y_std) ** 2).mean(1)
            loss = (w * per).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        hist["train"].append(total / len(perm))
        hist["val"].append(split_loss(model, Xva, Tva, args.output)[0])
        sched.step(hist["val"][-1])

        improved = hist["val"][-1] < best
        if improved:
            best, best_epoch, bad = hist["val"][-1], epoch, 0
            save(os.path.join(args.out, "model_best.pt"), epoch)
        else:
            bad += 1
        if epoch % args.save_every == 0:
            save(os.path.join(args.out, "checkpoints", f"epoch_{epoch:03d}.pt"), epoch)
        print(f"epoch {epoch:3d} | train {hist['train'][-1]:.5f} | val {hist['val'][-1]:.5f} | "
              f"lr {opt.param_groups[0]['lr']:.1e}{'  *best' if improved else ''}")
        if bad >= args.patience:
            print(f"Early stopping (no val improvement for {args.patience} epochs)")
            break
    train_time = time.time() - t_start
    save(os.path.join(args.out, "checkpoints", f"epoch_{epoch:03d}_last.pt"), epoch)

    # ---------------- evaluate the best model ----------------
    ckpt = torch.load(os.path.join(args.out, "model_best.pt"), weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    evaluator = evaluate_classifier if args.output == "done" else (lambda m, e: evaluate_split(m, e, args.output))
    res = {"train": evaluator(model, train_eps), "val": evaluator(model, val_eps), "test": evaluator(model, test_eps)}

    # Per-episode test errors (heavy-tailed: report median as well as mean)
    rows = []
    for e in test_eps:
        loss, _ = split_loss(model, e["X"], e["T"], args.output)
        rows.append({"episode": e["name"], "normalised_mse": loss,
                     "target_x": e["target"][0], "target_y": e["target"][1], "target_z": e["target"][2],
                     "target_r": float(np.hypot(*e["target"][:2]))})
    rows.sort(key=lambda r: r["normalised_mse"], reverse=True)
    with open(os.path.join(args.out, "test_episode_errors.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    ep_mse = np.array([r["normalised_mse"] for r in rows])

    # Inference cost of a single control step (batch 1, CPU)
    x1 = torch.from_numpy(Xva[:1])
    with torch.no_grad():
        for _ in range(200):
            model(x1)
        t0 = time.perf_counter()
        for _ in range(2000):
            model(x1)
    infer_ms = (time.perf_counter() - t0) / 2000 * 1e3

    metrics = {"config": vars(args), "best_epoch": best_epoch, "n_params": n_params,
               "inference_ms_per_step": infer_ms, "train_time_s": train_time,
               "test_episode_mse_median": float(np.median(ep_mse)),
               "test_episode_mse_max": float(ep_mse.max()), **res}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(6.5, 4))
    plt.semilogy(range(1, len(hist["train"]) + 1), hist["train"], label="train")
    plt.semilogy(range(1, len(hist["val"]) + 1), hist["val"], label="validation")
    plt.axvline(best_epoch, color="k", ls="--", lw=0.8, label=f"best (epoch {best_epoch})")
    plt.xlabel("epoch")
    plt.ylabel("binary cross-entropy (log)" if args.output == "done" else "normalised MSE (log)")
    plt.title(f"{args.features} -> {args.output}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "loss_curve.png"), dpi=150)

    print("\n================ results (best epoch {}) ================".format(best_epoch))
    for s in ("train", "val", "test"):
        r = res[s]
        if args.output == "done":
            print(f"{s:5s} | accuracy {r['accuracy']:.4f} | says done too early {r['false_done_rate']:.4f}"
                  f" | misses done {r['missed_done_rate']:.4f}")
        else:
            print(f"{s:5s} | normalised MSE {r['normalised_mse']:.4f} | wheel RMSE {r['wheel_cmd_rmse']:.3f} rad/s"
                  f" | arm torque RMSE {r['arm_torque_rmse']:.3f} N*m")
    print(f"test episodes: median loss {np.median(ep_mse):.4f}, worst {ep_mse.max():.4f} ({rows[0]['episode']})")
    print(f"{n_params} parameters | {infer_ms:.3f} ms per control step | saved to {args.out}")


if __name__ == "__main__":
    main()
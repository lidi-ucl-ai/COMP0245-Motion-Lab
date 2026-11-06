# COMP0245 Motion Lab

Local WSL2/Linux demo and benchmark for mobile-manipulator control. The policy selector includes the eight models from the matched comparison, the reported ablation configurations, the expert controllers, and the earlier student checkpoints used in the expert-improvement comparison. Each learned entry points to a frozen checkpoint in `checkpoints/manifest.json`. The loader verifies its SHA-256 before loading it.

## Start the English demo

Restore the checkpoint archive first. Follow [Release assets](RELEASE_ASSETS.md), then run either command from the folder containing this README in WSL2/Linux:

```bash
# Use Pixi to create and activate the Linux environment
pixi run demo

# Or use the existing course environment, if already installed
COMP0245_PYTHON=/path/to/RoboEnv/.pixi/envs/ml/bin/python bash run_demo.sh
```

Open [http://127.0.0.1:7870/](http://127.0.0.1:7870/). To select a different port, pass `--port 7871`. The demo listens on loopback and uses a single simulator shared by tabs connected to that server. Install `ffmpeg` (including `ffprobe`) on PATH for MP4 export. `requirements.txt` lists pip dependency equivalents. The Pixi configuration is preferred for PyBullet and Pinocchio.

1. Choose a model and enter a target in metres, or use a target preset. Start with `(0, 0, 1.2)` for the paper's near-origin illustration.
2. Use **Preview initial state** to apply the configuration. **Run** starts a new rollout. **Pause** and **Stop** control the active run.
3. The live view shows the measured end-effector and base paths, endpoint error, and achieved simulation speed. **Detailed** opens speed, torque, and position telemetry.
4. Download trajectory JSON or render an MP4 when the rollout ends. The success criterion is 3D end-effector error below 1 cm for 500 consecutive 1 ms steps.

The eight matched scores displayed in the selector come from the paper's frozen 256-target comparison. A new interactive rollout is a separate result. SNTC automatically selects one specialist within its frozen policy. Both expert controllers generate actions from rules and have no learned weights. The historical group uses the earlier controller/observation configuration for those studies.

## Models and checkpoints

| Paper name | Model key | What the policy controls |
|---|---|---|
| M1 | `m1` | Basic feedforward model for wheel speeds and joint torques |
| M2 | `m2` | Enhanced observation model for the same commands |
| P-GRU | `pgru` | Phase-aware recurrent direct-action policy |
| Memory-IK | `memory_ik` | Learned navigation and joint-motion prediction with feedback |
| RAIC | `raic_orig` | Supplied reference-action weights and base policy, arm integration disabled |
| I-RAIC | `reference` | Learned navigation trees and arm reference-action network |
| LJC | `jacobian` | Learned navigation and Jacobian, with analytical feedback for the arm |
| SNTC | `torque` | Learned navigation and region-selected neural torque policy |

The **Ablations** group contains LJC raw angles, fixed initial Jacobian, no compensation, and no high-reach servo, SNTC no anchor, no velocity prior, and far specialist only, Memory-IK SyncGate, and RAIC arm integration enabled. The raw-angle network has its own checkpoint. Other ablations reuse their parent weights.

## Evaluate frozen models

```bash
pixi run smoke
pixi run evaluate --model torque --workers 4 --out results/sntc.json
pixi run evaluate --model jac_no_compensation --suite ablation --workers 4 --out results/ljc_ablation.json
pixi run evaluate --model torque --suite robustness --workers 4 --out results/sntc_robustness.json
pixi run evaluate --model jacobian --suite extreme --workers 4 --out results/ljc_extreme.json
```

`--limit N` runs the first N scenarios. Smoke mode checks loading and 120 physics steps per selector entry. The benchmark JSON records contain per-case success, termination time, final error, and weight hash. The extreme suite scores the 48 random cases and lists the archived diagnostics separately. The `concurrent` and `high-reach` suites use distinct historical cohorts. RAIC integration was tested on the matched suite.

## Source guide

| Path | Purpose |
|---|---|
| `policies.py` | Loads the frozen models, checks weight hashes, and applies report ablations and adapters |
| `env.py`, `control_utils.py` | Measured observations, robot reset, simulation and success scoring, goal geometry |
| `models.py`, `base_memory.py` | Early action networks and shared navigation trees and memory |
| `student_policy.py`, `jacobian_model.py`, `motion_control.py` | LJC inference, Jacobian network and feedback controller |
| `expert.py`, `expert_planner.py`, `expert_adapters.py`, `course/robotnik_arm_control.py` | Improved expert, its reversible base planner, scoring adapters, and the original course controller |
| `reference/reference_policy.py`, `reference/features.py`, `reference/original_model.py` | Improved RAIC policy, observation processing, and arm network |
| `raic/model.py`, `raic/features.py` | Original RAIC policy and feature/action encoding |
| `torque/domain_policy.py`, `torque/optimized_policy.py` | SNTC routing and torque prediction with feedback and output limits |
| `torque/neural_policy.py`, `torque/learned_gravity.py` | SNTC observation encoding, network components, and learned gravity |
| `historical/` | Original runtime pieces for the earlier paired LJC-family and concurrent expert experiments |
| `collect.py`, `make_jacobian_data.py`, `train_navigation.py`, `train_base_memory_original.py`, `train_jacobian.py` | Collect Improved Expert demonstrations, generate kinematic labels, fit and serialize navigation trees, and fit the Jacobian MLP |
| `torque/train_gravity.py`, `torque/train_markov.py`, `torque/assemble_domains.py` | Fit SNTC gravity and torque components and assemble specialist routing |
| `evaluate.py`, `configs/` | Frozen evaluation runner, paper scenario files, and Improved Expert training scenarios/settings |
| `webapp/server.py`, `webapp/monitoring.py` | Local HTTP/SSE API, simulation lifecycle, telemetry and export |
| `webapp/static/` | English controls, Three.js viewer, charts, styles and vendored renderer |
| `vendor/RoboEnv/` | Simulator bindings and assets needed to render and run the Summit/Kinova task |

### Training scope

See [Baseline training](TRAINING.md) for M1, M2, P-GRU, I-RAIC, and Original RAIC entry points.

The repository contains the runtime and model training code. The frozen weights, historical recordings and checkpoints are packaged as [Release assets](RELEASE_ASSETS.md). The original LJC fitting script requires CUDA.

To collect new LJC demonstrations from the bundled scenario definitions, use an activated environment at the release root:

```bash
export PYTHONPATH="$PWD/vendor/RoboEnv/simulation_and_control${PYTHONPATH:+:$PYTHONPATH}"
python collect.py --scenarios configs/train_scenarios.json --out data/train --workers 4
python collect.py --scenarios configs/validation_scenarios.json --out data/validation --workers 4
python make_jacobian_data.py
python train_navigation.py --seed 1
python train_jacobian.py --seed 1
```

The new LJC weights are written under `checkpoints/jacobian_s1/`. The frozen demo continues to use `checkpoints/jacobian.pt`.

## Release assets

Download the model archive from the GitHub Release and extract it into this directory before starting the frozen demo. See [Release assets](RELEASE_ASSETS.md) for restoration instructions.

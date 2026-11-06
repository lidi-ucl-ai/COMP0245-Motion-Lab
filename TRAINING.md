# Training the report baselines

Use the Linux or WSL environment described in the [README](README.md). The commands below start at the project root. New runs are written to `runs/` and are separate from the frozen demo weights. Historical recordings are packaged separately as [GitHub Release assets](RELEASE_ASSETS.md).

## M1, M2, and P-GRU

`train_baselines.py` is the paper's refitting script. M1 uses the 21-element basic observation, M2 uses the enhanced observation, and P-GRU uses recurrent sequences with phase and reference supervision. The fitting budgets are in `configs/baseline_train_plan.json`. The fitter requires CUDA.

```bash
# Set DATA to the Improved Expert episode directory, with train/ and validation/.
for model in m1 m2 pgru; do
  python train_baselines.py --model "$model" --train "$DATA/train" \
    --validation "$DATA/validation" --out runs/baseline_refits
done
```

Use `windows/outputs/high_teacher_student_20261101/project/data` under the restored history. The files are paired NPZ and JSON per episode. New recordings with the same schema can also be collected using the LJC collection commands in the README. The same trainer supports the report's Memory-IK refit with `--navigation-checkpoint checkpoints/jacobian.pt`.

## I-RAIC

`reference/collect_reference.py` records the Improved Expert's reference-action labels. `reference/train_reference.py` fits the arm-reference MLP. `reference/train_reference_navigation.py` fits the navigation trees and attaches them to the trained checkpoint.

```bash
export PYTHONPATH="$PWD:$PWD/vendor/RoboEnv/simulation_and_control${PYTHONPATH:+:$PYTHONPATH}"
cd reference
# Use the archived data/ with train/, validation/ and extra_train/, or collect new episodes:
# python -m reference.collect_reference --scenarios configs/train_scenarios.json --out data/train --workers 4
# python -m reference.collect_reference --scenarios configs/validation_scenarios.json --out data/validation --workers 4
# python -m reference.collect_reference --scenarios configs/extra_train_scenarios.json --out data/extra_train --workers 4
python -m reference.train_reference --features context_no_history \
  --extra data/extra_train --out runs/F_no_history_extreme
python -m reference.train_reference_navigation
```

Use `wsl/comp0245_control_v2_20261024/experiments/reference_bc_20261102/data` under the restored history. The resulting checkpoint is `reference/runs/F_no_history_extreme_treebase/model_best.pt` with the integrated reference mode and zero anchor used by the selected run.

## Original RAIC

`raic/original_source/` contains the original collection, training and evaluation scripts. Download [RAIC-data.zip](https://github.com/lidi-ucl-ai/COMP0245-Motion-Lab/releases/download/v1.0.0/RAIC-data.zip) and extract its `data/` folder under `raic/original_source/`. The report's frozen inference checkpoint is `checkpoints/raic_orig.pt`.

`data/train/split.json` records 239 regular training, 30 validation and 30 held-out test episodes. The frozen checkpoint's configuration adds 60 training rollouts from each of `extreme_near`, `extreme_low`, `extreme_high` and `extreme_init`, for 479 training rollouts in total. The `special` folder holds target and reset configurations for diagnostics. We compared the stored 3D training and validation targets with the 256 shared benchmark goals and found zero matches within 1 micrometre.

```bash
ROOT="$PWD"
export PYTHONPATH="$ROOT/course:$ROOT/vendor/RoboEnv/simulation_and_control${PYTHONPATH:+:$PYTHONPATH}"
cd raic/original_source
ln -s ../../vendor/RoboEnv/configs configs
ln -s ../../vendor/RoboEnv/models models
python collect_data.py --help
python train.py --help
python evaluate.py --help
```

The original collector can generate fresh recordings. Its built-in defaults and the original evaluation protocol differ from the report's matched benchmark. For the report's frozen weights and common scoring, run `pixi run evaluate --model raic_orig` from the project root.

## LJC and raw-angle ablation

Follow the LJC collection and training commands in the README. To fit the seven-angle ablation on the same Jacobian labels while retaining the full model's navigation and controller settings:

```bash
python train_jacobian.py --seed 1 --raw-angles-from checkpoints/jacobian_s1/best.pt
```

The new checkpoint is written to `runs/jac_rawangle_s1/best.pt`. The frozen ablation evaluated by the report is selected with `--model jac_rawangle --suite ablation`.

## SNTC

The `torque` modules cover rollout collection, joint reference and torque relabelling, gravity fitting, arm fitting, correction rollouts, and final selector assembly. Run them from a working directory containing the SNTC `data/`, `runs/` and `configs/` trees. The checked-in scenario lists are in `torque/reproduction/configs/`.

```bash
export PYTHONPATH="$PWD:$PWD/vendor/RoboEnv/simulation_and_control${PYTHONPATH:+:$PYTHONPATH}"
cd torque/reproduction
mkdir -p data runs
# Supply runs/frozen_navigation.pt and the relevant historical training inputs.
python -m torque.collect_physical --scenarios configs/near_train_augmented.json --out data/near_rollouts
python -m torque.prepare_gravity_data --physical --extra-folders physical_rollouts near_rollouts --out data/gravity_near
python -m torque.train_gravity --data data/gravity_near --out runs/gravity_near
python -m torque.train_markov --out runs/near_invariant --separable --anchor --position-response 1 \
  --extra-data data/near_rollouts --gravity-checkpoint runs/gravity_near/best.pt \
  --invariant --regular-weight 2 --epochs 160
```

For the full historical fit, the regular-domain data include pre-existing demonstration and mixed-policy recordings. `collect_mixed_arm`, `collect_hold`, `prepare_stage_data`, and `prepare_markov_data` reproduce those collection and relabelling stages when supplied with their historical inputs. The high stage uses `collect_physical --variant high`. The far stage adds `collect_far_corrections` using the far-transfer checkpoint. `assemble_domains` first joins the regular/near and high specialists and then adds the far specialist. The [SNTC stage parameters](torque/reproduction/stages.json) record the options and selected epochs used in the report. Exact reproduction requires the original input recordings, upstream checkpoints and validation runs from the separate history archive.

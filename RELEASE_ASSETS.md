# COMP0245 Release assets

Commit the contents of `repository` to GitHub. Attach the contents of the sibling `release-assets` directory to a GitHub Release. Keep all ZIP parts and the manifest together.

| Asset | Contents |
|---|---|
| comp0245-models.zip | The twelve frozen demo checkpoints |
| comp0245-history-NN.zip | Historical training and validation data, training scripts and checkpoints, and experimental records |
| archive-manifest.json | SHA-256, size, file counts, and the mapping for shared-data links |

The history parts are separate ZIP files. Keep them together with the manifest and use Python 3.11 or later to restore them. Run the history restore in WSL/Linux on a filesystem that supports symbolic links.

```bash
# Run from the Git repository directory.
python restore_history.py ../release-assets --subset models --dest .
python restore_history.py ../release-assets --subset history --dest ~/comp0245-history
```

Use an empty directory for the history restore. An interrupted restore may leave partially extracted files. Use a fresh destination to retry. Use the option --verify-only to check the downloads before extracting. Models restore directly under checkpoints, after which the README's normal demo and evaluation commands apply.

The historical archive preserves relative paths under wsl and windows. Many original experiments assume the collection machine's original filesystem layout or an active simulator environment. Adjust those local paths before rerunning them. The archive includes the original and improved expert recordings available in the project, the LJC Jacobian supervision, SNTC specialist training and correction trajectories, and other available historical datasets. The supplied Original RAIC weights are in the model archive, and the teammate-supplied collection, fitting and evaluation scripts are in the repository. Original RAIC training trajectories were absent from both supplied packages.

| Dataset or evidence | Location after restoring the history |
|---|---|
| Improved Expert demonstrations and LJC Jacobian labels | windows/outputs/high_teacher_student_20261101/project/data |
| SNTC demonstrations, gravity samples and correction trajectories | wsl/comp0245_control_v2_20261024/experiments/torque_curriculum_20261102/data |
| SNTC fitting metrics and historical checkpoints | wsl/comp0245_control_v2_20261024/experiments/torque_curriculum_20261102/runs |
| M1, M2, and P-GRU refit recordings | windows/outputs/high_teacher_student_20261101/project/data/{train,validation} |
| I-RAIC fitting recordings | wsl/comp0245_control_v2_20261024/experiments/reference_bc_20261102/data |
| Original course baseline experiment data | wsl/COMP0245_BC_20261027_113636/final/robotnik_arm_task |
| Original project delivery data and legacy results | wsl/comp0245_control_v2_20261024/delivery/COMP0245_improved_complete |
| Study scenario definitions, robustness data, and paper sources | windows/outputs/report_model_study_20261103/latex |

Machine-specific environments, caches, private sharing state, live demo sessions, videos, logs, and redundant delivery ZIPs are excluded. See the manifest for historical links whose original targets were already broken or fell outside the archive.

[GitHub Release limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)

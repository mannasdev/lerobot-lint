# lerobot-lint

Your loss curve looks perfect but the robot pecks at the table. The bug is in your
data. Find it in 30 seconds, not 6 GPU-hours.

`lerobot-lint` is a CLI (`lelint`) and Python package (`lerobot_lint`) that checks
[LeRobot](https://github.com/huggingface/lerobot) robot-training datasets for
behavioral/kinematic data bugs — dead joints, encoder wraparound, frozen cameras,
leader/follower calibration mismatch, frame drops — before you burn GPU-hours training
on bad data.

Deterministic, rule-based checks. No ML, no training, no GPU required to run it.

## Status

Early development. See `ARCHITECTURE.md` and `CONTRIBUTING.md` (once written) for
design details.

## Install

```bash
pip install lerobot-lint
```

## Usage

```bash
lelint check <repo_id_or_path>
```

## License

Apache-2.0

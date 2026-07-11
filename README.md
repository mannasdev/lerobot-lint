# lerobot-lint

Your loss curve looks perfect but the robot pecks at the table. The bug might be in
your data, not your model.

`lerobot-lint` is a CLI (`lelint`) and Python package (`lerobot_lint`) that checks
[LeRobot](https://github.com/huggingface/lerobot) robot-training datasets for
behavioral/kinematic data bugs — dead joints, encoder wraparound, frozen telemetry,
leader/follower calibration mismatch, frame drops — before you burn GPU-hours training
on bad data.

Deterministic, rule-based checks. No ML, no training, no GPU required to run it.

## Status: works, pre-1.0, not on PyPI yet

20 checks across 5 groups are implemented and tested (130 passing tests). The CLI runs
end-to-end against real Hugging Face Hub datasets — loading, checking, console/JSON
reports, exit codes, a `guard()` context-manager API for use inside training scripts.

Not published to PyPI yet — install from source (below). See
[`PROGRESS.md`](./PROGRESS.md) for exactly what's built, what's left, and why.

**Known limitation:** camera/video checks (frozen camera, degenerate image, video
length mismatch) are implemented and unit-tested against synthetic frame data, but
real video decoding isn't wired up yet — they don't see real camera frames from an
actual dataset run today. Everything else (joint kinematics, action/state
consistency, timing, dataset hygiene) runs against real data now.

## Install (from source)

```bash
git clone https://github.com/mannasdev/lerobot-lint.git
cd lerobot-lint
pip install -e .
```

## Use it

```bash
lelint check lerobot/some-dataset --no-video
```

Auto-detects a robot profile (so101/koch joint-naming convention) from the dataset's
own metadata when `--profile` isn't passed, and always tells you which profile it
used and why — never silently. `--fail-on info|warning|error` controls what makes the
exit code nonzero (default: `error`, exit 0/1/2). `--json report.json` writes a
machine-readable report alongside the console one.

```python
import lerobot_lint

with lerobot_lint.guard("your/dataset"):
    ...  # raises before you spend GPU-hours on data with real errors
```

## A real example

Running it against [`lerobot/svla_so101_pickplace`](https://huggingface.co/datasets/lerobot/svla_so101_pickplace)
(a real SO-101 pick-place dataset) surfaces genuine recorder dropouts undisclosed in
the dataset card — 5 of 50 episodes have stretches where the robot's telemetry is
bit-identical for up to 2+ seconds while actively recording (verified by hand against
the raw data, not just this tool's own output):

```
$ lelint check lerobot/svla_so101_pickplace --no-video
Using profile: koch (auto-detected from joint names, override with --profile)
...
Errors (...)
  [FROZEN_STATE] episode 0: State frozen (identical for 67 consecutive frames) from
  frame 27 to 93 -- likely a recorder hiccup or serial dropout
```

An imitation-learning policy trained on episode 0 as-is would learn "do nothing" for
over a quarter of that episode's frames.

## What's built

- All 20 checks across 5 groups (kinematic signal, action/state consistency, timing
  integrity, camera/video, dataset hygiene) — see `lerobot_lint/checks/`.
- The dataset loader, verified against real public LeRobot datasets (including real
  SO-101 hardware, not just simulated tasks).
- The check engine: two-pass streaming, per-check crash isolation, episode- and
  dataset-scoped checks.
- The `lelint` CLI: `check` (with `--profile`, `--episodes`, `--no-video`, `--json`,
  `--verbose`, `--fail-on`), `profiles`, `version`.
- `lerobot_lint.guard()` — a context-manager API for use inside training scripts.
- Console and JSON reports.
- Robot profile auto-detection from joint-naming convention.

## What's not built yet

The A-F scorecard formula, the bug-card renderer, real video-decode wiring, PyPI
packaging/CI, and the full field study (calibrating thresholds against 30-50 real
datasets). Full list with context in [`PROGRESS.md`](./PROGRESS.md).

## License

Apache-2.0 — see [`LICENSE`](./LICENSE).

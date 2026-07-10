# lerobot-lint

Your loss curve looks perfect but the robot pecks at the table. The bug is in your
data. Find it in 30 seconds, not 6 GPU-hours.

`lerobot-lint` is a CLI (`lelint`) and Python package (`lerobot_lint`) that checks
[LeRobot](https://github.com/huggingface/lerobot) robot-training datasets for
behavioral/kinematic data bugs — dead joints, encoder wraparound, frozen cameras,
leader/follower calibration mismatch, frame drops — before you burn GPU-hours training
on bad data.

Deterministic, rule-based checks. No ML, no training, no GPU required to run it.

## Status: early, pre-release — not installable or runnable yet

All 20 planned checks are implemented and unit-tested (73 passing tests), but there
is **no CLI yet** — `lelint` is not a real command, and this package is not published
to PyPI. `pip install lerobot-lint` will not work. See
[`PROGRESS.md`](./PROGRESS.md) for exactly what's built, what's left, and in what
order.

If you're looking at this repo and want to try the tool: it's not ready. Watch/star
for when it is.

## What's built

- All 20 checks across 5 groups (kinematic signal, action/state consistency, timing
  integrity, camera/video, dataset hygiene) — see `lerobot_lint/checks/`.
- The dataset loader, verified against a real public LeRobot dataset.
- The check engine (per-check crash isolation, both episode-scoped and dataset-scoped).

## What's not built yet

The CLI itself, profiles, reports (console/JSON/scorecard), the `guard()` API, the
bug-card renderer, real video-decode wiring, and packaging/CI. Full list with context
in [`PROGRESS.md`](./PROGRESS.md).

## License

Apache-2.0 — see [`LICENSE`](./LICENSE).

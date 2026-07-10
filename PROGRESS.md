# lerobot-lint — Progress & Continuation Guide

Read this first in any new session on this project. It's the map back to
everything that's already been decided and built, so nothing gets re-derived
or re-litigated by accident.

## What this is

A CLI (`lelint`) + Python package (`lerobot_lint`) that checks LeRobot
robot-training datasets for **behavioral/kinematic** data bugs — dead joints,
encoder wraparound, frozen cameras, leader/follower calibration mismatch —
before you burn GPU-hours training on bad data. Deterministic, rule-based
checks. No ML, no training, no GPU required to run it.

One competitor exists (**trajlens**, github.com/Kunal-Somani/trajlens) that
checks dataset *format/schema* — this tool checks robot *behavior*, which
trajlens doesn't touch. Both layers matter; they're not redundant.

## Where the deeper context lives (read these if you need "why", not just "what")

- **Design doc** (full architecture, every decision + rationale, 3 rounds of
  adversarial review + 2 outside-voice passes):
  `~/.gstack/projects/Projects/mannas-unknown-design-20260710-143402.md`
- **CEO plan** (scope decisions — what got added/deferred and why, e.g. why
  the GitHub bot and live badge are v1.1 not v1):
  `~/.gstack/projects/Projects/ceo-plans/2026-07-10-lerobot-lint.md`
- **TODOS** (deferred items with context — profile versioning, paid-status-quo
  research, lerobot compat CI, CLI smoke tests, the bot, the badge, compare-mode):
  `~/.gstack/projects/Projects/lerobot-lint-TODOS.md`

These aren't duplicated here on purpose — this file is "what's built and
what's next," those are "why it's designed this way."

## Workflow rules for this project (already agreed, don't re-ask)

- **TDD, strictly.** Every function: write the test first, watch it fail
  (confirm it fails for the *right* reason), then write minimal code to pass.
  Every check has a paired positive test (inject the bug, assert it fires)
  and negative test (clean data, assert it doesn't).
- **Commit after every logical change.** Meaningful, specific commit messages.
  **No `Co-Authored-By` trailer** — commits are attributed to the user only.
  **No push** — no remote configured yet.
- Dev environment: `.venv/` (already set up, `uv`-managed). Run tests with
  `.venv/bin/python -m pytest`.

## Current state: all 20 checks complete, nothing runnable yet

```
lerobot_lint/
  types.py           EpisodeData, Finding, CameraSample, EpisodeSummary, summarize_episode()
  loader.py          real LeRobotDataset wrapper — verified against lerobot 0.6.0's
                      actual API (not memory), streams one episode at a time.
                      Tested against a live public dataset (lerobot/pusht).
  checks/
    base.py          Check + DatasetCheck ABCs, CheckRegistry + DatasetCheckRegistry,
                      both with per-check crash isolation (one broken check
                      can't take down the run — reported as a `_CRASHED` finding).
    kinematic.py      Group A (episode-scoped): DEAD_JOINT, SATURATED_JOINT,
                      JITTER, DISCONTINUITY, FROZEN_STATE, GRIPPER_INERT
    consistency.py    Group B (episode-scoped): ACTION_STATE_DIVERGENCE,
                      ACTION_RANGE_MISMATCH
    timing.py         Group C (episode-scoped): TIMESTAMP_NON_MONOTONIC,
                      FRAME_DROPS, FPS_MISMATCH
    video.py          Group D (episode-scoped, decode-agnostic): VIDEO_LENGTH_MISMATCH,
                      FROZEN_CAMERA, DEGENERATE_IMAGE — operate on CameraSample
                      (sampled frames + total count), NOT on raw video. The
                      actual av/opencv windowed-sampling decode is NOT wired up
                      yet (see "Known gaps" below).
    hygiene.py        Group E (dataset-scoped, two-pass): SHORT_EPISODE,
                      DURATION_OUTLIER, MISSING_TASK, TASK_IMBALANCE,
                      LOW_DIVERSITY (vectorized + 500-episode subsample cap
                      baked in from day one), TOO_FEW_EPISODES
  report/            empty package, nothing built yet
tests/               73 tests, all passing, mirrors lerobot_lint/ structure 1:1
```

Run `.venv/bin/python -m pytest` — should show `73 passed`.

**20 commits**, one logical unit each — `git log --oneline` to see the build
order (scaffold → types → loader → check engine → Group A → B → C → D →
two-pass accumulator → Group E).

## What's NOT built yet — in the order it should happen

1. **Profiles** (`profiles/default.yaml`, `so101.yaml`, `koch.yaml`) — per-joint
   thresholds, joint names, gripper index. Several checks already accept
   thresholds as constructor args specifically so a profile can override them
   (e.g. `GripperInertCheck(gripper_index=...)`) — check each check class's
   `__init__`/class constants before assuming a default.
2. **The CLI** (`cli.py`, `typer` app) — the thing that actually wires
   `loader.iter_episodes()` → `CheckRegistry.run_all()` per episode →
   `summarize_episode()` → `DatasetCheckRegistry.run_all()` at the end →
   a report. **Nothing currently calls any of the built code end-to-end** —
   this is the biggest remaining gap. Flags per the design doc:
   `--json out.json`, `--episodes 0:50`, `--profile so101`, `--no-video`,
   `--verbose`, `--fail-on error|warning`. Exit codes: 0=clean, 1=warnings,
   2=errors-or-load-failure (matches trajlens's convention on purpose).
3. **Reports** — `report/console.py` (rich terminal output + the A-F
   scorecard, formula still TBD per the design doc), `report/json_report.py`
   (stable schema, snapshot-tested), `report/finding_summary.py` (shared
   text-formatting helper — sanitizes control characters at the source since
   dataset-derived strings are untrusted, per the CEO review's security fix).
4. **`lerobot_lint.guard()`** — context manager, raises `LelintCheckFailedError`
   only on severity=error, raises a *distinct* `LelintCheckCrashedError` if a
   check itself crashes (not the same exception — a user needs to tell "your
   data has a bug" apart from "lelint has a bug"). Needs a conformance test
   proving it agrees with the CLI on the same fixture.
5. **`report/render.py`** — the bug-card renderer (static SVG→raster export,
   pure-Python rasterizer, no external binary — keeps the macOS/Linux platform
   promise). Plots the anomaly for spike-type findings AND static/binary ones
   (frozen camera, dead joint) — not just text, per the outside-voice fix that
   caught the original design over-indexing on the rarer bug type.
6. **Real video decode** in `loader.py` — actually calling `av`/opencv to
   produce `CameraSample` objects (windowed sampling: 3 windows of 10 truly-
   consecutive frames per camera, not evenly-spaced isolated frames — that
   was a real bug caught in the eng review). See "Known environment issue"
   below before starting this.
7. **Packaging** — CI/CD, PyPI publish workflow with a post-publish smoke test
   (install from PyPI into a clean venv, assert `import lerobot_lint;
   lerobot_lint.guard(...)` works), `CONTRIBUTING.md` + `ARCHITECTURE.md`
   (port design-doc decisions into the actual repo, not just `~/.gstack/`).
8. **Field study** — run against 30-50 real public Hub datasets, calibrate
   every threshold (all current thresholds are source-spec *starting*
   defaults, explicitly not final), produce the launch post.

## Known environment issue (hit for real, not hypothetical)

`torchcodec` (lerobot's default video-decode backend) fails to load its
FFmpeg shared libraries in this dev environment (`libavutil.56/57.dylib` not
found via Homebrew paths). This is **exactly** the environment fragility the
design doc flagged as Group D's highest risk (the "day-9 gate": `--no-video`
becomes the default if decode isn't reliable). Two implications:
- `loader.iter_episodes()` currently works fine for `states`/`actions`/
  `timestamps` (reads `hf_dataset` directly, no video touched).
- Actually wiring up video decode (step 6 above) needs either fixing the local
  ffmpeg linkage (`brew install ffmpeg` and confirm torchcodec finds it) or
  building on `av` (PyAV) directly instead of going through
  `LeRobotDataset`'s torchcodec path, which may not hit the same linkage issue
  — untested, check first.

## Naming gotcha (fixed once already, easy to reintroduce)

Package import name is **`lerobot_lint`**, not `lelint`. `lelint` is the CLI
command name only (`pip install lerobot-lint` gives you `lelint` on your PATH
and `import lerobot_lint` in Python). Got this wrong once already across an
entire CEO-review document and had to fix it — double-check any new code/docs
against this before writing `lelint.` as a Python import path.

## Quick start for a new session

```bash
cd /Users/mannas/Desktop/Projects/lerobot-lint
.venv/bin/python -m pytest        # confirm 73 passed before touching anything
git log --oneline                 # see exactly what's been built, in order
```

Then pick up at item 1 or 2 in "What's NOT built yet" above.

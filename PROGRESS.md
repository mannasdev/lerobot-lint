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
- Dev environment: `.venv/` (already set up, `uv`-managed). Run tests with
  `.venv/bin/python -m pytest`.
- **Remote is live:** `github.com/mannasdev/lerobot-lint` (public). Push when
  asked; don't push unprompted.

## Current state: the tool actually runs now

```
lerobot_lint/
  __init__.py        re-exports guard, LelintCheckFailedError, LelintCheckCrashedError
                      -- `import lerobot_lint; lerobot_lint.guard(...)` works.
  types.py           EpisodeData, Finding, CameraSample, EpisodeSummary, summarize_episode()
  config.py          Profile dataclass + load_profile() — reads profiles/*.yaml
  loader.py          real LeRobotDataset wrapper — verified against lerobot 0.6.0's
                      actual API (not memory). Streams one episode at a time.
                      Yields (index, episode_or_None, error_or_None) -- a bad
                      episode reports its own error without aborting the rest.
  engine.py          check_dataset() -- wires loader -> both check registries ->
                      two-pass accumulator -> combined findings. The ONE function
                      both cli.py and _guard.py call into -- neither duplicates
                      this wiring (that's what the conformance test depends on).
  _guard.py          the guard context manager (module named _guard, not guard --
                      see "Naming gotcha #2" below for why).
  cli.py             the `lelint` command: check/profiles/version subcommands.
                      check has --profile, --episodes, --no-video, --json,
                      --verbose. 0/1/2 exit codes (matches trajlens's convention).
  profiles/          default.yaml (generic, gripper_index=null), so101.yaml,
                      koch.yaml (both gripper_index=5, standard 6-DOF layout)
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
  report/
    console.py         terminal report, severity-grouped, ends with a summary count
    json_report.py      stable schema (schema_version=1), snapshot-tested
    finding_summary.py   shared Finding->text formatting; sanitize_text() strips
                         control characters at the source, used by both console
                         and json_report so neither has to remember to do it
tests/               110 tests, all passing, mirrors lerobot_lint/ structure 1:1
```

Run `.venv/bin/python -m pytest` — should show `110 passed`.

**Try it for real:**
```bash
.venv/bin/lelint check lerobot/pusht --episodes 0:3 --no-video
.venv/bin/lelint check lerobot/pusht --episodes 0:3 --no-video --json /tmp/report.json --verbose
.venv/bin/python -c "import lerobot_lint; lerobot_lint.guard('lerobot/pusht', episode_indices=[0], download_videos=False).__enter__()"
```

**31 commits**, one logical unit each — `git log --oneline` to see the build
order. Notable ones beyond the obvious: two loader fixes caught only by
actually wiring the engine together (real episode index vs. enumerate
position; a single bad episode no longer kills the whole generator); zero-
episode datasets now correctly treated as an error (previously silent);
and a real naming-collision bug in the package (see "Naming gotcha #2").

## IMPORTANT finding from the first real run — read before trusting any output

`lelint check lerobot/pusht` produces a **false-positive storm** on `JITTER`
(every episode, most frames). This is NOT a bug in the check or the wiring —
it's a dataset-choice mistake. `pusht`'s `observation.state`/`action` are 2D
pixel coordinates (a simulated pushing task in image space), not joint angles
in radians. `JITTER`'s 8 rad/s threshold (a real SO-101-class hobby-servo
default) is meaningless applied to pixel-position deltas — of course it fires
on nearly every frame.

**Implication:** `pusht` has been useful for validating the loader/engine
*plumbing* (real API, real streaming, real error handling) but is NOT a valid
dataset for validating the *joint-specific* checks (JITTER, DISCONTINUITY,
SATURATED_JOINT, DEAD_JOINT, GRIPPER_INERT all assume joint-radian state).
**Before trusting any calibration work or writing the launch post, find a
real SO-101/Koch/6-DOF-arm dataset** (actual joint-radian state/action) and
re-validate against that instead. This is exactly what the field study
(days 11-12 in the source spec) is for — this is that finding arriving early.

## What's NOT built yet — in the order it should happen

1. **The A-F scorecard formula** — mentioned in the source spec, not designed
   or implemented anywhere yet (`json_report.py`'s `"grade"` field is `None`
   on purpose). Needs the field study's calibration data.
2. **`report/render.py`** — the bug-card renderer (static SVG→raster export,
   pure-Python rasterizer, no external binary). Plots the anomaly for
   spike-type findings AND static/binary ones (frozen camera, dead joint).
   Can reuse `finding_summary()` for the text half.
3. **Real video decode** in `loader.py` — actually calling `av`/opencv to
   produce `CameraSample` objects (windowed sampling: 3 windows of 10 truly-
   consecutive frames per camera). See "Known environment issue" below.
4. **`--fail-on error|warning` flag** — cli.py has `--profile`, `--episodes`,
   `--no-video`, `--json`, `--verbose` now; `--fail-on` is the one remaining
   piece from the source spec's CLI surface.
5. **Auto-profile detection** (item 4 from the CEO plan) — detect robot type
   from joint naming convention, suggest a profile, always print which
   profile was used (auto-detected or not — must never be silent).
6. **Packaging** — CI/CD, PyPI publish workflow with a post-publish smoke test,
   `CONTRIBUTING.md` + `ARCHITECTURE.md`.
7. **Field study** — run against 30-50 real public Hub datasets (**with real
   arm datasets this time, not pusht** — see the finding above), calibrate
   every threshold, produce the launch post.

## Known environment issue (hit for real, not hypothetical)

`torchcodec` (lerobot's default video-decode backend) fails to load its
FFmpeg shared libraries in this dev environment (`libavutil.56/57.dylib` not
found via Homebrew paths). This is **exactly** the environment fragility the
design doc flagged as Group D's highest risk (the "day-9 gate": `--no-video`
becomes the default if decode isn't reliable). Two implications:
- `loader.iter_episodes()` currently works fine for `states`/`actions`/
  `timestamps` (reads `hf_dataset` directly, no video touched).
- Actually wiring up video decode (item 5 above) needs either fixing the local
  ffmpeg linkage (`brew install ffmpeg` and confirm torchcodec finds it) or
  building on `av` (PyAV) directly instead of going through
  `LeRobotDataset`'s torchcodec path, which may not hit the same linkage issue
  — untested, check first.

## Naming gotcha #1 (fixed once already, easy to reintroduce)

Package import name is **`lerobot_lint`**, not `lelint`. `lelint` is the CLI
command name only (`pip install lerobot-lint` gives you `lelint` on your PATH
and `import lerobot_lint` in Python). Got this wrong once already across an
entire CEO-review document and had to fix it — double-check any new code/docs
against this before writing `lelint.` as a Python import path.

## Naming gotcha #2 (a real bug, now fixed — don't reintroduce the pattern)

The `guard` context manager lives in **`lerobot_lint/_guard.py`** (underscore
prefix), not `guard.py`. Reason: `lerobot_lint/__init__.py` re-exports the
`guard` class at the package's top level (`lerobot_lint.guard(...)` is the
public API). If the module were named `guard.py`, `lerobot_lint.guard` would
ambiguously refer to either the submodule or the re-exported class — and the
class always wins once `__init__.py` has run, because `from lerobot_lint.foo
import bar` rebinds the name `bar` in the package's own namespace, overwriting
whatever the import machinery set for the submodule. This isn't just a
"don't do `from lerobot_lint import guard as x`" footgun — **even `import
lerobot_lint.guard as x` is shadowed the same way**, since Python's `import
a.b as c` resolves via attribute access on `a`, not `sys.modules['a.b']`
directly (this surprised me when I first hit it). If any future module needs
to be re-exported under the same name at the package's top level, name the
module itself something else (a leading underscore, or a distinct name) —
don't rely on any import spelling to disambiguate after the fact.

## Quick start for a new session

```bash
cd /Users/mannas/Desktop/Projects/lerobot-lint
.venv/bin/python -m pytest                              # confirm 110 passed
.venv/bin/lelint check lerobot/pusht --episodes 0:3 --no-video   # see it run
git log --oneline                                        # see what's been built
```

Then pick up at item 1 in "What's NOT built yet" above — but consider finding
a real arm dataset first (see the IMPORTANT finding section) before doing any
threshold calibration work, since pusht will mislead you.

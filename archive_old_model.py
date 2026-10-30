"""
Archive every training artifact from the CURRENT model into one timestamped
folder, leaving a genuinely clean slate for train.py to start a fresh policy
-- for use any time swing_robot.xml/swing_env.py changes structurally enough
that a previously-trained policy is no longer a valid starting point (a new
joint, a repositioned body, a changed observation/action meaning).

WHY THIS NEEDS TO BE A SEPARATE STEP, NOT JUST "RUN TRAIN.PY AGAIN":
train.py's resuming path only checks for swing_policy.zip + vec_normalize.pkl
to decide whether to resume -- but a structural model change means action[1]
(or whatever changed) doesn't mean what the old policy learned it to mean. A
policy trained against the old structure has learned nothing transferable to
the new one, so resuming from it isn't a head start, it's a bad
initialization.

Just as important and easy to miss: train.py ALSO auto-seeds its
EvalCallback's "is this a new best" baseline from any existing
best_model/best_model.zip + best_vec_normalize.pkl, evaluating that old
policy against the NEW environment. If the observation/action space SHAPES
happen to still match (e.g. still 6 obs, 2 actions), this won't crash --
it'll just quietly evaluate stale, meaningless weights against dynamics they
were never trained on, seeding the "best so far" threshold with a number
that doesn't mean anything for this model. Moving best_model/ aside (not
just swing_policy.zip) avoids that.

Also moves checkpoints/ aside: CheckpointCallback numbers files by internal
step counter, which restarts at 0 for a fresh model -- leaving old
checkpoints/ in place risks the step-number-collision confusion noted in
README.md section 5.6, just for a new lineage.

SAFETY CHECK BEFORE ARCHIVING: this used to move things immediately with no
summary and no confirmation. That silently swept away a well-supported,
far-advanced training run (20M+ steps, including a good 14-checkpoint
streak) when this script got run a second time by accident, several
structural changes after the first intentional run -- the archived
checkpoints/ folder was still sitting there intact (nothing is ever
deleted), but it took real back-and-forth to even realize what had happened.
Now prints exactly what it's about to move -- checkpoint count, step range,
and the best angle found in checkpoint_scan.csv/evaluate_log.csv if present
-- and requires typing "yes" before doing anything, so a second accidental
run gets caught at the moment it would matter, not discovered later via a
FileNotFoundError.

Run locally:
    python archive_old_model.py
"""
import csv
import glob
import os
import re
import shutil
from datetime import datetime, timezone

# Files and directories that belong to the CURRENT (about-to-be-superseded)
# model and shouldn't carry over into a fresh training run.
ARCHIVE_TARGETS = [
    "swing_policy.zip",
    "vec_normalize.pkl",
    "best_model",
    "checkpoints",
    "evaluate_log.csv",
    "checkpoint_scan.csv",
    "eval_logs",
    "tb_logs",
]

STEP_RE = re.compile(r"_(\d+)_steps\.zip$")


def summarize_checkpoints():
    """(count, min_step, max_step) for checkpoints/*.zip, or None if the
    directory doesn't exist or has nothing matching."""
    if not os.path.isdir("checkpoints"):
        return None
    paths = glob.glob(os.path.join("checkpoints", "swing_ppo_*_steps.zip"))
    steps = []
    for p in paths:
        m = STEP_RE.search(os.path.basename(p))
        if m:
            steps.append(int(m.group(1)))
    if not steps:
        return None
    return len(steps), min(steps), max(steps)


def best_angle_seen():
    """Best det_peak_angle_deg value found across checkpoint_scan.csv and
    evaluate_log.csv, or None if neither exists / has that column. Just for
    the pre-archive summary -- not used for any decision-making."""
    best = None
    for path, col in [
        ("checkpoint_scan.csv", "det_peak_angle_deg_1ep"),
        ("evaluate_log.csv", "det_peak_angle_deg"),
    ]:
        if not os.path.exists(path):
            continue
        try:
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    val = float(row[col])
                    if best is None or val > best:
                        best = val
        except (KeyError, ValueError):
            continue
    return best


def main():
    checkpoint_info = summarize_checkpoints()
    best_angle = best_angle_seen()

    print("About to archive the following (nothing is deleted, only moved):")
    for target in ARCHIVE_TARGETS:
        if os.path.exists(target):
            print(f"  - {target}")
    if not any(os.path.exists(t) for t in ARCHIVE_TARGETS):
        print("Nothing to archive -- no current-model artifacts found. "
              "Either this is already a clean slate, or this script already "
              "ran once since the last training run.")
        return

    if checkpoint_info:
        count, min_step, max_step = checkpoint_info
        print(f"\ncheckpoints/ has {count} checkpoint(s), spanning step "
              f"{min_step:,} to {max_step:,}.")
    if best_angle is not None:
        print(f"Best deterministic peak angle recorded for this model so "
              f"far: {best_angle:.1f} deg.")
    if checkpoint_info and checkpoint_info[2] > 1_000_000:
        print("\n*** This looks like substantial training progress "
              "(1M+ steps). Double check this is really what you want to "
              "archive before continuing. ***")

    confirm = input("\nType 'yes' to archive these and start clean, "
                     "anything else to cancel: ").strip().lower()
    if confirm != "yes":
        print("Cancelled -- nothing moved.")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    archive_dir = f"archive_{stamp}"

    moved = []
    for target in ARCHIVE_TARGETS:
        if os.path.exists(target):
            os.makedirs(archive_dir, exist_ok=True)
            dest = os.path.join(archive_dir, target)
            shutil.move(target, dest)
            moved.append(target)

    print(f"\nArchived into {archive_dir}/:")
    for name in moved:
        print(f"  - {name}")

    print(f"\nDone. Working directory is now clean for a fresh model.")
    print("Next: python train.py  (will take the fresh-start branch, since "
          "swing_policy.zip/vec_normalize.pkl are gone)")
    print(f"Everything just archived is still fully intact in {archive_dir}/ "
          "if you ever want to go back and re-check it -- nothing was "
          "deleted, only moved.")


if __name__ == "__main__":
    main()

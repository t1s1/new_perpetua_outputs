"""
Scan every checkpoint in checkpoints/ and report how each one performs, so
you can find where a training collapse happened and pick a good checkpoint
to recover from -- without opening evaluate.py and editing MODEL_PATH by
hand for each of (potentially) hundreds of files.

CAVEAT -- read this before trusting the numbers: checkpoints saved before
this script existed don't have their own matching VecNormalize snapshot
(see train.py's save_vecnormalize fix) -- so this evaluates every old
checkpoint's policy weights using whichever vec_normalize.pkl currently
exists (VECNORM_PATH below), which reflects normalization statistics as of
NOW, not as of when that particular checkpoint was saved. That's good
enough to see the overall SHAPE of the trend and roughly where things went
bad, but not a perfectly clean historical replay -- treat this script's
numbers as a way to narrow down candidates, not as the final word.
Checkpoints saved after the train.py fix will have their own paired
vecnormalize file and won't have this problem.

This is deliberately a FAST, ROUGH pass: one deterministic episode per
checkpoint, no averaging. That's the same single-reading noise problem
evaluate.py's 5-episode averaging exists to avoid -- fine for scanning
hundreds of files quickly, not fine as the final answer. Once you've
picked a few promising candidates from checkpoint_scan.csv, verify each
properly with evaluate.py (5 episodes, deterministic AND stochastic, with
spread) before actually promoting one.

Writes one row per checkpoint to checkpoint_scan.csv AS IT GOES, not just
at the end -- safe to Ctrl+C partway through a long scan; whatever's
already written stays there and is still readable.

SKIPS CHECKPOINTS ALREADY IN checkpoint_scan.csv: re-running this script
(e.g. after training has continued further, or after a new decline) only
scans checkpoint files it hasn't scored before, instead of re-testing
everything from scratch every time. The end-of-run summary still looks at
the FULL history (old + newly-scanned rows together), so the collapse-
point heuristic and "running best" stay meaningful across repeated runs.
Delete checkpoint_scan.csv first if you ever want a completely fresh scan.

BUG FIX -- STALE CACHE AFTER A RESCUE PROMOTION: "already scanned" used to
mean "this step number appears in checkpoint_scan.csv", full stop. That's
wrong the moment training resumes from an EARLIER promoted checkpoint
(the normal rescue workflow) -- train.py's internal step counter continues
from THAT checkpoint's own count, not from wherever the run being rescued
from had gotten to, so CheckpointCallback's next saves land on step
numbers (and therefore filenames) that already existed from the original
run, silently OVERWRITING those files on disk with completely different
weights. The step number is identical; the content isn't. This actually
happened: a rescue promotion back to a ~10.4M-step checkpoint, followed by
more training, overwrote several already-scanned files in the 10.4M-11.4M
range with new content, and the old skip-by-step-number logic never
noticed -- it just kept reporting the stale pre-overwrite numbers. Fixed
by tracking each row's file mtime at scan time and comparing against the
checkpoint's CURRENT mtime on disk: if the file changed since it was last
scanned, it gets rescanned even though its step number was already seen.

CSV SCHEMA: mirrors evaluate.py's EXPECTED_HEADER self-healing (same
reasoning: a header written once at file-creation time silently stops
describing the file the moment a column gets added, corrupting column
meaning for every row from then on unless checked). If checkpoint_scan.csv
exists with an older, different header, it gets archived to a timestamped
.bak file and a fresh one starts, rather than risk misaligned columns.

LINEAGE SPLITTING (see README.md 5.6): different train.py resume sessions
land their checkpoints on different (step % 50000) offsets, since the
internal step counter's phase relative to the 50k save interval shifts
every time the script restarts. Reading checkpoint_scan.csv in plain
chronological step order can make two unrelated resume sessions -- one
solid, one collapsed -- look like rapid, erratic alternation when they're
actually just interleaved. This script groups by that offset and reports
the longest all-good streak WITHIN each group, which is what "is there a
well-supported checkpoint to promote" actually depends on -- not the raw
chronological view.

Run locally (can take a while the first time, with hundreds of
checkpoints -- much faster on repeat runs since only new files get tested):
    python scan_checkpoints.py
"""
import csv
import glob
import os
import re
from datetime import datetime, timezone

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from swing_env import SwingEnv

CHECKPOINT_DIR = "checkpoints"
VECNORM_PATH = "vec_normalize.pkl"  # current/latest normalization stats --
                                      # see the caveat above: an
                                      # approximation for any checkpoint
                                      # that isn't the very latest one.
OUT_PATH = "checkpoint_scan.csv"

STEP_RE = re.compile(r"_(\d+)_steps\.zip$")

EXPECTED_HEADER = ["step", "checkpoint_path", "det_reward_1ep", "det_peak_angle_deg_1ep", "mtime"]

# Heuristic for flagging a likely collapse point in the printed summary:
# once the running best-so-far angle drops by more than this fraction and
# stays down for COLLAPSE_CONFIRM_COUNT checkpoints in a row, call that the
# start of the collapse. Just a pointer to look closer, not a verdict.
COLLAPSE_DROP_FRACTION = 0.5
COLLAPSE_CONFIRM_COUNT = 5

# Band for "good" when looking for a promotable streak (see
# find_best_streak below). Lower bound: well above the ~2-5 deg
# near-total-collapse range this project keeps seeing, but not so strict
# it discards a decent, not-yet-fully-converged streak -- picked from
# project history, not the reward-shaping target (80 deg) itself.
STREAK_GOOD_MIN_DEG = 55.0
# BUG FIX: this used to be a floor only (angle >= THRESHOLD), no ceiling --
# which found and recommended an 18-checkpoint "streak" that was actually
# mostly 85-124 deg, badly overshooting the then-80 deg target (overshoot
# penalty started at TARGET_ANGLE_DEG + OVERSHOOT_MARGIN_DEG, which was 90
# in swing_env.py at the time -- OVERSHOOT_MARGIN_DEG has since been
# lowered to 0, see swing_env.py 5.5a-iv, so this was already somewhat
# stale even before the line below). A checkpoint that consistently swings
# way past target isn't "even better", it's poorly calibrated and getting
# penalized hard for it -- the opposite of a converged policy, even though
# the raw angle number looks impressive. 95 (old value) allowed a little
# slack past the then-90 penalty threshold without treating wild overshoot
# as "good".
#
# UPDATED, 95.0 -> 72.5, alongside TARGET_ANGLE_DEG's redefinition in
# swing_env.py (80 -> 67.5, a deliberate priority change toward margin over
# height -- see that constant's comment). Same reasoning as above, just
# recomputed against the new target and current (0) margin: new penalty
# threshold is 67.5 + 0 = 67.5, plus the same 5 deg slack allowance =
# 72.5. STREAK_GOOD_MIN_DEG above is left unchanged -- its own comment
# already says it was picked from the collapse/near-collapse range in
# project history, not from the reward target, so it doesn't need to move
# just because the target did.
STREAK_GOOD_MAX_DEG = 72.5
STEP_INTERVAL = 50_000  # matches train.py's CheckpointCallback save_freq


def step_from_filename(path):
    m = STEP_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else -1


def run_one_episode(model):
    vec_env = DummyVecEnv([lambda: SwingEnv()])
    vec_env = VecNormalize.load(VECNORM_PATH, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    obs = vec_env.reset()
    total_reward = 0.0
    max_angle_deg = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_arr, info = vec_env.step(action)
        done = bool(done_arr[0])
        total_reward += float(reward[0])
        max_angle_deg = max(max_angle_deg, abs(info[0]["swing_angle_deg"]))
    return total_reward, max_angle_deg


def find_best_streak(results):
    """Groups results by (step % STEP_INTERVAL) -- see the module docstring's
    LINEAGE SPLITTING note -- and returns the longest run of consecutive
    (STEP_INTERVAL apart, same offset group) checkpoints that are all within
    [STREAK_GOOD_MIN_DEG, STREAK_GOOD_MAX_DEG], as a list of (step, path,
    reward, angle) tuples. Empty list if nothing qualifies. Band, not just a
    floor -- see STREAK_GOOD_MAX_DEG's comment for why a ceiling matters
    here just as much as a floor does."""
    from collections import defaultdict

    by_offset = defaultdict(list)
    for step, path, reward, angle in results:
        by_offset[step % STEP_INTERVAL].append((step, path, reward, angle))

    best_run = []
    for offset, pts in by_offset.items():
        pts = sorted(pts, key=lambda r: r[0])
        cur_run = []
        for step, path, reward, angle in pts:
            if STREAK_GOOD_MIN_DEG <= angle <= STREAK_GOOD_MAX_DEG:
                cur_run.append((step, path, reward, angle))
            else:
                if len(cur_run) > len(best_run):
                    best_run = cur_run
                cur_run = []
        if len(cur_run) > len(best_run):
            best_run = cur_run
    return best_run


def vecnorm_path_for(checkpoint_path):
    """Matching per-checkpoint VecNormalize snapshot path, per train.py's
    save_vecnormalize=True naming (swing_ppo_vecnormalize_<step>_steps.pkl,
    same directory as the checkpoint). Falls back to the current
    vec_normalize.pkl (with a caveat printed by the caller) if that
    specific file doesn't exist -- e.g. for checkpoints saved before that
    fix, or if something about the naming has changed."""
    step = step_from_filename(checkpoint_path)
    candidate = os.path.join(
        os.path.dirname(checkpoint_path), f"swing_ppo_vecnormalize_{step}_steps.pkl"
    )
    return candidate if os.path.exists(candidate) else VECNORM_PATH


def load_existing_results():
    """(step, path, reward, angle) for every row already in checkpoint_scan.csv
    -- deduplicated by step, keeping the LAST occurrence of each (a step gets
    a second row when it's rescanned after being detected as stale, per the
    mtime check in main(); later rows reflect a more recent scan) -- plus a
    {step: mtime} map of what was on disk at scan time, so main() can tell a
    genuinely-unchanged file from one that's been silently overwritten since.
    Archives and starts fresh if the header doesn't match EXPECTED_HEADER
    (schema changed since the file was created -- see module docstring),
    same self-healing pattern as evaluate.py. Empty list/dict if the file
    doesn't exist yet."""
    if not os.path.exists(OUT_PATH):
        return [], {}

    with open(OUT_PATH, newline="") as f:
        existing_header = next(csv.reader(f), None)
    if existing_header != EXPECTED_HEADER:
        backup_path = OUT_PATH + f".bak-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        os.rename(OUT_PATH, backup_path)
        print(f"!! {OUT_PATH}'s header didn't match this script's current columns "
              f"(schema changed since it was created) -- moved the old file to "
              f"{backup_path} (nothing lost, still readable by hand) and starting "
              f"a fresh scan, rather than risk misaligned columns.")
        return [], {}

    by_step = {}
    with open(OUT_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            by_step[step] = (
                step, row["checkpoint_path"],
                float(row["det_reward_1ep"]), float(row["det_peak_angle_deg_1ep"]),
                float(row["mtime"]),
            )

    results = sorted((s, p, r, a) for s, p, r, a, m in by_step.values())
    mtimes = {s: m for s, (_, _, _, _, m) in by_step.items()}
    return results, mtimes


def main():
    if not os.path.exists(VECNORM_PATH):
        print(f"{VECNORM_PATH} not found -- need it to normalize observations "
              f"for any of these checkpoints. Stopping.")
        return

    checkpoint_paths = sorted(
        glob.glob(os.path.join(CHECKPOINT_DIR, "swing_ppo_*_steps.zip")),
        key=step_from_filename,
    )
    if not checkpoint_paths:
        print(f"No checkpoints found in {CHECKPOINT_DIR}/ -- nothing to scan.")
        return

    prior_results, prior_mtimes = load_existing_results()

    # A path needs (re)scanning if its step was never scanned before, OR its
    # current on-disk mtime is newer than what was recorded at last scan --
    # the latter is the stale-overwrite case (see module docstring): same
    # step number, different content, because training resumed from an
    # earlier promoted checkpoint and CheckpointCallback reused old filenames.
    to_scan = []
    stale_count = 0
    for p in checkpoint_paths:
        step = step_from_filename(p)
        current_mtime = os.path.getmtime(p)
        cached_mtime = prior_mtimes.get(step)
        if cached_mtime is None:
            to_scan.append(p)
        elif current_mtime > cached_mtime + 1.0:  # +1s slack against float/fs precision noise
            to_scan.append(p)
            stale_count += 1

    if prior_results:
        print(f"{OUT_PATH} already has {len(prior_results)} scanned checkpoints -- "
              f"{len(to_scan) - stale_count} new one(s) and {stale_count} "
              f"previously-scanned-but-since-overwritten one(s) found in "
              f"{CHECKPOINT_DIR}/.")
        if stale_count:
            print(f"(The {stale_count} overwritten file(s) most likely happened "
                  f"because training resumed from an earlier promoted checkpoint "
                  f"and CheckpointCallback reused old step numbers/filenames -- "
                  f"see module docstring. Rescanning them, not trusting the old "
                  f"cached numbers.)")
    if not to_scan:
        print("Nothing new or changed to scan.")
        results = prior_results
    else:
        print(f"Running one deterministic episode each on {len(to_scan)} "
              f"checkpoint(s) (fast, rough pass) -- writing results to {OUT_PATH} "
              f"as they come in, so Ctrl+C at any point leaves you a usable "
              f"partial file.")

        write_header = not os.path.exists(OUT_PATH)
        new_results = []
        with open(OUT_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(EXPECTED_HEADER)

            for i, path in enumerate(to_scan, 1):
                step = step_from_filename(path)
                mtime = os.path.getmtime(path)
                model = PPO.load(path)
                reward, angle = run_one_episode(model)
                writer.writerow([step, path, f"{reward:.4f}", f"{angle:.4f}", f"{mtime:.6f}"])
                f.flush()
                new_results.append((step, path, reward, angle))
                print(f"[{i}/{len(to_scan)}] step {step}: reward {reward:.1f}, "
                      f"peak angle {angle:.1f} deg")

        # Combine old + new for the summary below, preferring the NEW result
        # for any step scanned in both (a rescanned/stale step should use its
        # fresh data, not the old cached row) -- same reasoning as
        # load_existing_results()'s own dedup-by-step-keep-last.
        merged = {s: (s, p, r, a) for s, p, r, a in prior_results}
        for s, p, r, a in new_results:
            merged[s] = (s, p, r, a)
        results = sorted(merged.values())

    print(f"\nDone. Full results ({len(results)} checkpoints total) in {OUT_PATH}.")

    # Simple collapse-point heuristic: track the running best angle seen so
    # far, and flag the first checkpoint where the angle has dropped to
    # less than COLLAPSE_DROP_FRACTION of that running best AND stays down
    # for the next COLLAPSE_CONFIRM_COUNT checkpoints in a row. Just a
    # pointer for where to look closer, not a verdict -- read the actual
    # numbers around it in checkpoint_scan.csv.
    #
    # BUG FIX: "best_so_far" used to update on ANY new max, including wild
    # overshoot spikes (e.g. 123.6 deg against an 80 deg target) -- which
    # inflated the reference point and could make an otherwise-ordinary
    # later reading look like a collapse relative to a number that was
    # never a legitimate "good" result to begin with. Capping what counts
    # toward best_so_far at STREAK_GOOD_MAX_DEG keeps the reference
    # anchored to the last genuinely well-calibrated result, same fix as
    # find_best_streak's band above, same root cause.
    best_so_far = 0.0
    best_step = None
    collapse_flagged = False
    for idx, (step, path, reward, angle) in enumerate(results):
        if angle > best_so_far and angle <= STREAK_GOOD_MAX_DEG:
            best_so_far = angle
            best_step = step
        elif not collapse_flagged and best_so_far > 0 and angle < best_so_far * (1 - COLLAPSE_DROP_FRACTION):
            window = results[idx: idx + COLLAPSE_CONFIRM_COUNT]
            if len(window) == COLLAPSE_CONFIRM_COUNT and all(
                a < best_so_far * (1 - COLLAPSE_DROP_FRACTION) for _, _, _, a in window
            ):
                print(f"\nLikely collapse starts around step {step} ({path}): peak "
                      f"angle dropped from a running best of {best_so_far:.1f} deg "
                      f"(at step {best_step}) to {angle:.1f} deg and stayed down for "
                      f"the next {COLLAPSE_CONFIRM_COUNT} checkpoints. Look at rows "
                      f"just before this step in {OUT_PATH} for good candidates.")
                collapse_flagged = True

    if not collapse_flagged:
        print(f"\nNo clear collapse pattern found by this heuristic -- running best "
              f"angle was {best_so_far:.1f} deg at step {best_step}. Sort "
              f"{OUT_PATH} by det_peak_angle_deg_1ep yourself to look for the "
              f"actual trend; this heuristic can miss softer or partial regressions.")

    # Lineage-aware streak detection -- see module docstring's LINEAGE
    # SPLITTING note. This is what actually answers "is there a
    # well-supported checkpoint to promote", as opposed to the collapse
    # heuristic above, which only looks at raw chronological step order
    # and can be confused by interleaved resume sessions.
    best_run = find_best_streak(results)
    if best_run:
        mid = best_run[len(best_run) // 2]
        mid_step, mid_path, mid_reward, mid_angle = mid
        offset = mid_step % STEP_INTERVAL
        span_steps = best_run[-1][0] - best_run[0][0]
        vecnorm = vecnorm_path_for(mid_path)
        print(f"\n=== Best supported streak (lineage-aware) ===")
        print(f"{len(best_run)} consecutive checkpoints within "
              f"[{STREAK_GOOD_MIN_DEG:.0f}, {STREAK_GOOD_MAX_DEG:.0f}] deg, "
              f"all {STEP_INTERVAL} steps apart (offset {offset}), spanning "
              f"{span_steps:,} steps: {best_run[0][0]:,} -> {best_run[-1][0]:,}.")
        print(f"Recommended candidate (middle of the streak, not an edge): "
              f"{mid_path} ({mid_angle:.1f} deg on this 1-episode scan)")
        if vecnorm == VECNORM_PATH:
            print(f"NOTE: no per-checkpoint vecnormalize file found for this one -- "
                  f"falling back to the current {VECNORM_PATH}, which is only an "
                  f"approximation for anything but the latest checkpoint (see the "
                  f"caveat at the top of this file).")
        print(f"\nTo verify and promote, edit evaluate.py's MODEL_PATH/VECNORM_PATH to:")
        print(f'    MODEL_PATH = "{mid_path[:-4]}"')  # strip .zip, evaluate.py adds it back
        print(f'    VECNORM_PATH = "{vecnorm}"')
        print(f"then run:")
        print(f"    python evaluate.py")
        print(f"and if the 5-episode numbers hold up (angle near {mid_angle:.0f} deg, "
              f"small spread, ~0 overshoot):")
        print(f"    python promote_checkpoint.py {mid_path}")
        print(f"Then revert evaluate.py's MODEL_PATH/VECNORM_PATH back to "
              f'"best_model/best_model" / "vec_normalize.pkl" before running '
              f"auto_train.py again -- it runs evaluate.py as-is.")
    else:
        print(f"\nNo streak of checkpoints within [{STREAK_GOOD_MIN_DEG:.0f}, "
              f"{STREAK_GOOD_MAX_DEG:.0f}] deg found within any single resume-session "
              f"lineage yet -- keep training, or widen the STREAK_GOOD_MIN_DEG/"
              f"STREAK_GOOD_MAX_DEG band if this feels too strict.")

    print("\nGeneral advice: pick 2-3 candidates near the highest angles (ideally a "
          "few checkpoints apart, not all clustered on one lucky episode), then "
          "verify each with evaluate.py's real 5-episode deterministic+stochastic "
          "check before trusting any single one of these numbers.")


if __name__ == "__main__":
    main()

"""
Automates the train.py -> evaluate.py -> check-the-result loop, since
that's all manual involvement has been contributing at this stage.

Each round runs train.py then evaluate.py as ordinary subprocesses --
exactly the same commands you've been typing by hand -- reads back the
deterministic result(s) evaluate.py just appended to evaluate_log.csv, and
decides one of six things: keep going, stop because the target's
genuinely been reached, stop because the last several rounds show no real
improvement (a stall), stop because the last several rounds show a real
DECLINE (a gradual multi-round slide -- see DECLINE_WINDOW below), stop
because a single round just fell off a CLIFF (a sudden one-round collapse
-- see CLIFF_MIN_PRIOR_DEG below; this is a different shape than a decline
and needs its own check, since a climb-then-cliff isn't a monotonic slide
and would slip past the decline check entirely -- confirmed this actually
happened once, undetected, before this check existed), or stop because a
single round was ERRATIC (huge spread and/or huge overshoot within that
one round's own episodes -- see ERRATIC_SPREAD_MAX_DEG below; this
catches instability that the MEAN-based checks above are blind to, since a
round can have a perfectly unremarkable mean angle while actually
alternating between near-collapse and wild overshoot underneath it --
confirmed this happened too: mean 73.04 deg looked like normal
not-yet-converged noise, but that round's own per-episode angles were
[10.4, 24.9, 100.3, 107.3, 122.3] and overshoot penalty was 1175.63, both
far outside anything seen before). All of these are treated as a signal to
come back and look at what's happening rather than keep burning rounds
blindly the way we did before finding the actuator and gamma issues, again
before finding the collapse that motivated the decline check, again before
finding the cliff that motivated that check, and again before finding the
erratic-but-mean-looks-fine round that motivated this one.

"Target reached" is deliberately NOT a single good round. A single mean
above threshold can just be a lucky evaluate.py call -- this is a
stochastic process, and evaluate.py's own per-episode numbers have shown
real swings from one run to the next. Instead this requires a round to
pass THREE checks at once (mean angle, overshoot, and spread across that
round's own episodes -- see evaluate.py's N_EVAL_EPISODES), and requires CONSECUTIVE_HITS_
REQUIRED rounds in a row to each independently pass all three before
calling it. That's "consistent across repeated checks", not "got a good
number once."

Run locally -- expect this to take a long time, since it's however long
TIMESTEPS_THIS_RUN in train.py takes, times however many rounds it runs
before stopping:
    python auto_train.py

Safe to Ctrl+C between rounds (train.py has already saved to disk by
then). Ctrl+C in the middle of a round loses that round's progress, but
not anything from previous rounds -- checkpoints, best_model/, and
evaluate_log.csv all persist independently of this script.
"""
import csv
import subprocess
import sys

LOG_PATH = "evaluate_log.csv"

MAX_ROUNDS = 3                # was 30. No longer just a safety cap -- this is
                               # now the actual intended stopping point: run
                               # a small, bounded batch of rounds, then stop
                               # and wait, rather than assume someone's free
                               # to babysit a long unattended run (and, after
                               # the collapse, to catch a growing det-vs-sto
                               # gap within a check-in or two instead of
                               # discovering it 10 rounds later). Since 3
                               # rounds is very unlikely to hit
                               # CONSECUTIVE_HITS_REQUIRED on its own from a
                               # partially-trained checkpoint, expect this to
                               # almost always end by hitting this cap, not
                               # by reaching target or stalling -- that's
                               # normal now, just rerun `python auto_train.py`
                               # again whenever there's time for another
                               # check-in; it resumes from wherever
                               # swing_policy.zip left off, same as always.
TARGET_ANGLE_DEG = 62.5       # UPDATED, 75.0 -> 62.5, alongside swing_env.py's
                               # TARGET_ANGLE_DEG redefinition (80 -> 67.5 --
                               # see that constant's comment: a deliberate
                               # priority change toward margin/robustness
                               # over height, not a bug fix). This is a
                               # SEPARATE, hand-duplicated constant, not
                               # imported from swing_env.py -- easy to miss
                               # updating alongside the real one, which is
                               # exactly what would have happened here if
                               # this weren't checked directly: this file
                               # would have kept gating "reached" on 75 deg
                               # even though the reward now actively punishes
                               # anything past ~87.5 (HEIGHT_DECAY_RANGE_DEG
                               # reaching its floor), silently blocking the
                               # new target from ever being callable as
                               # reached. Recomputed the same way as before:
                               # 5 deg below the reward cap ("close enough,
                               # doesn't need to be exact") = 67.5 - 5 = 62.5.
TARGET_OVERSHOOT_MAX = 75.0   # RAISED, 0.1 -> 75.0 (README 5.9l), after
                               # direct evidence this was gating on
                               # something other than policy quality.
                               # 0.1 assumed a converged policy should log
                               # essentially 0.00 here -- true only in a
                               # disturbance-free environment. Ablation on
                               # best_model (DISTURBANCE_PROB_PER_STEP
                               # forced to 0 vs its real 0.001/step
                               # default): disturbances OFF -> 8/8
                               # deterministic episodes converged to a
                               # tight 69.3-69.4 deg, 0.0 overshoot penalty
                               # every single time. Disturbances ON (real
                               # setting) -> peaks 71.4-73.3 deg, overshoot
                               # 0.0-3.6/episode. Confirms the reward shape
                               # itself is not the problem (proven by the
                               # disturbance-off result) -- the residual
                               # overshoot is the environment's own RNG
                               # (random mid-episode gust kicks, a
                               # deliberate robustness feature) occasionally
                               # landing while the swing is already near
                               # peak, not imprecise control. Every
                               # wall-touch count logged alongside even the
                               # worst "healthy" overshoot rounds seen so
                               # far (392, 398, 618) is 0 -- these are
                               # small, bounded, recovered-from excursions,
                               # not the runaway/wall-pinning failure mode
                               # this project has seen before. Unlike
                               # BOOTSTRAP_FAIL_MAX_ANGLE_DEG, this metric
                               # has no clean bimodal gap between "healthy"
                               # and "problem" values to calibrate against
                               # -- it's continuous disturbance noise, so
                               # this is a judgment call, not a measured
                               # boundary: recent combined (worse-of-
                               # best_model/swing_policy) readings on
                               # genuinely healthy rounds (good angle, tight
                               # spread, zero bootstrap fails) topped out at
                               # 59.75 -- 75 gives real margin above that
                               # observed max (only 6 rounds of evidence in
                               # this exact regime so far) while staying far
                               # below ERRATIC_OVERSHOOT_MAX=400 (the
                               # separate "something is badly wrong" check
                               # below, unchanged and still active) and
                               # EPISODE_OVERSHOOT_PENALTY_CAP's
                               # per-excursion ceiling of 100. Revisit if
                               # real N=15 rounds start landing consistently
                               # closer to this ceiling than to the 0-60
                               # band seen so far.
SPREAD_MAX_DEG = 15.0          # max allowed (max - min) across one round's
                               # eval episodes. A round can hit the target
                               # angle ON AVERAGE while actually alternating
                               # between a great episode and a mediocre one --
                               # this is what catches that instead of treating
                               # the average as the whole story. CAVEAT
                               # (README 5.9j): calibrated back when
                               # evaluate.py's N_EVAL_EPISODES was 5, now
                               # 15 -- max-min across more draws tends to
                               # widen even for unchanged underlying
                               # variance, so this may now be tighter (in
                               # effect) than intended. Not re-tuned yet --
                               # this gates "target reached", so watch it
                               # closely against real N=15 rounds before
                               # trusting it's still well-calibrated.
CONSECUTIVE_HITS_REQUIRED = 3  # how many rounds IN A ROW must each pass all
                               # three checks above before calling it reached.
                               # One good round can be luck; three in a row
                               # is the kind of repeated, low-variance
                               # evidence that's actually trustworthy.
STALL_ROUNDS = 5               # if the best deterministic peak angle swings
                               # by less than STALL_IMPROVEMENT_DEG across
                               # this many consecutive rounds, call it stalled
STALL_IMPROVEMENT_DEG = 1.0

DECLINE_WINDOW = 4             # a stall (above) only catches FLAT results --
                               # it explicitly would NOT catch a real, ongoing
                               # decline, since a big swing from round to
                               # round means STALL_IMPROVEMENT_DEG's "no big
                               # swing" condition is false. A real
                               # deterministic collapse was missed by every
                               # check in this file for several rounds before
                               # anyone noticed by reading the log directly --
                               # this closes that gap: if the last
                               # DECLINE_WINDOW rounds are monotonically
                               # non-increasing AND the total drop across
                               # that window is at least DECLINE_MIN_DROP_DEG,
                               # that's a real trend, not noise (pure noise
                               # doesn't reliably move in one direction for
                               # several rounds straight), so stop rather
                               # than let it keep going unattended.
DECLINE_MIN_DROP_DEG = 5.0

CLIFF_MIN_PRIOR_DEG = 20.0     # a real single-round collapse actually
                               # happened and slipped past every check above:
                               # round climbed to 55.6 deg, then the VERY
                               # NEXT round fell to 5.0 deg -- but
                               # DECLINE_WINDOW's check requires the whole
                               # window to be monotonically non-increasing,
                               # and a climb-then-cliff isn't monotonic, so
                               # it didn't fire. This check catches that
                               # specific shape directly: compares only the
                               # immediately preceding round to the current
                               # one, regardless of what came before that.
                               # CLIFF_MIN_PRIOR_DEG guards against flagging
                               # meaningless relative swings between tiny
                               # numbers early in training (e.g. 3 deg to 1
                               # deg is a big % drop but not a real event) --
                               # only counts once the prior round was already
                               # a real, non-trivial result.
CLIFF_DROP_FRACTION = 0.5      # current round below this fraction of the
                               # immediately preceding round's angle counts
                               # as a cliff. That one real event was a 91%
                               # drop in a single round -- 50% has real
                               # margin under it while still being far above
                               # anything normal noise between good rounds
                               # has shown so far.

ERRATIC_SPREAD_MAX_DEG = 90.0  # a single round's own (max - min) across its
                               # episodes above this counts as erratic,
                               # regardless of what the mean looks like.
                               # "Not yet converged" rounds have shown
                               # spreads up to the mid-50s (e.g. 54.55) and
                               # that's been treated as normal, expected
                               # noise -- not a stopping condition on its
                               # own. The event that motivated this check
                               # was a spread of 111.90 deg in one round
                               # (episodes ranging from 10.4 to 122.3 deg,
                               # back when N_EVAL_EPISODES was 5)
                               # with a perfectly unremarkable-looking mean
                               # (73.04 deg, barely different from the prior
                               # round's 75.84) -- neither the cliff nor
                               # decline check (both mean-only) had any way
                               # to catch that. 90 sits clearly above the
                               # normal noisy range seen so far and clearly
                               # below that event; tune if normal variance
                               # turns out to run higher than expected.
                               # CAVEAT (README 5.9j): evaluate.py's
                               # N_EVAL_EPISODES raised 5 -> 15 -- max-min
                               # across MORE draws tends to widen even for
                               # unchanged underlying variance (more
                               # chances to sample a tail episode), so this
                               # threshold was calibrated against the old
                               # N=5 behavior and may read too tight now.
                               # Not re-tuned yet; watch real N=15 rounds
                               # before assuming it's still well-calibrated.
ERRATIC_OVERSHOOT_MAX = 400.0  # same idea, for a single round's overshoot
                               # penalty. Values up to ~350 have shown up
                               # during an already-declining stretch (caught
                               # separately by the decline check at the
                               # time); this is for catching a spike on its
                               # own, independent of whether the mean trend
                               # also looks bad. The motivating event hit
                               # 1175.63 -- over 3x anything seen before,
                               # including that declining stretch.


BEST_MODEL_TAG = "best_model/best_model"  # MUST match evaluate.py's
                               # BEST_MODEL_PATH exactly -- same hand-
                               # duplicated-constant risk as
                               # TARGET_ANGLE_DEG below, kept separate
                               # rather than imported since this script
                               # only ever runs evaluate.py as a
                               # subprocess, never imports it.
LATEST_MODEL_TAG = "swing_policy"  # MUST match evaluate.py's
                               # LATEST_MODEL_PATH exactly, same as above.
                               # NEW (README 5.9i): this script always runs
                               # plain `evaluate.py` with no overrides, so
                               # every real round now logs TWO rows -- one
                               # under each of these tags, back to back --
                               # not one. Any OTHER model_path (an ad hoc
                               # manual check, or evaluate.py's optional
                               # EXTRA_MODEL_PATH) is still excluded from
                               # this loop's own streak/stall history,
                               # exactly as before. (read_log() no longer
                               # combines the pair -- see its own docstring,
                               # README 5.9z.)

DOMAIN_RANDOMIZATION_START_TIMESTAMP = "2026-08-16T22:58:26"  # NEW (README
                               # 5.9z). read_log() below ignores every row
                               # logged before this boundary -- the exact
                               # timestamp of the first best_model row
                               # logged under domain randomization (5.9w) +
                               # observation noise (5.9x). CAUGHT BEFORE
                               # SHIPPING: gating on best_model alone
                               # (above) means streak_from_history(), run
                               # against the FULL unfiltered history, comes
                               # back with a streak of 29 -- best_model
                               # basically never failed round_passes across
                               # dozens of rounds spanning many earlier,
                               # completely different training regimes
                               # (pre-domain-randomization, different
                               # ACTION_RATE_WEIGHT phases, etc.). Since
                               # CONSECUTIVE_HITS_REQUIRED is only 3, seeding
                               # consecutive_hits at 29 on the next
                               # invocation means a single additional
                               # passing round -- even one round of real new
                               # evidence -- would immediately trigger
                               # "Target reached AND consistent, stopping"
                               # and permanently halt further training. That
                               # would be exactly the "one good round isn't
                               # trustworthy evidence" mistake
                               # CONSECUTIVE_HITS_REQUIRED's own comment
                               # warns about, just smuggled in through stale
                               # history instead of a single lucky round.
                               # The whole point of this domain-
                               # randomization push was to gather genuine,
                               # repeated confidence THAT SPECIFICALLY
                               # regime is working -- a streak built mostly
                               # out of pre-randomization rounds doesn't
                               # answer that question at all. Every gating
                               # check (streak, cliff, decline, stall) now
                               # only ever sees rows from this boundary
                               # onward, so "target reached, stop for good"
                               # can only fire on real evidence gathered
                               # under the current regime. Not intended to
                               # be temporary -- pre-boundary history isn't
                               # going to become MORE relevant to the
                               # current regime as time passes, so this
                               # stays a permanent marker, the read-only
                               # analog of archive_old_model.py drawing a
                               # line after a structural change, just for
                               # log-based gating instead of checkpoints.


def read_log():
    """(det_peak_angle_deg, det_overshoot_penalty, det_angle_spread_deg) per
    ROUND, oldest first -- one tuple per round, from best_model's OWN row.

    CHANGED (README 5.9z): used to combine best_model + swing_policy by
    taking the WORSE of the two on each metric (5.9i), because at the time
    best_model reporting clean while the round-end state (swing_policy)
    didn't was a real failure mode invisible to a best_model-only check.
    Two things have changed since that made this gating stale rather than
    protective:

    (1) pick_resume_source() (5.9k, extended 5.9y to also catch severe
    overshoot, not just bootstrap failures) now independently guards
    EVERY round against ever training further on top of a broken
    swing_policy state -- the actual risk 5.9i's combining existed to
    prevent is now caught at the source automatically, without needing
    this script to stop the whole loop and wait for a human.

    (2) Domain randomization (5.9w) makes swing_policy's raw round-end
    snapshot inherently noisier on its own -- it's whatever the policy
    looked like immediately after training on a batch of arbitrary sampled
    physics, evaluated cold against fixed nominal physics, not the stable
    read best_model (the best-SCORING checkpoint across the whole round)
    is. Direct evidence: the first four rounds trained under domain
    randomization all showed a stable, healthy best_model (10,701-11,372
    reward, 0 bootstrap failures every round) while swing_policy's
    overshoot penalty fluctuated 5,920-12,089 with no converging trend
    (one round even showed a 69.9 deg spread and its first bootstrap
    failure) -- gating on the worse of the two meant this script stopped
    itself every single round despite the thing that actually matters
    (best_model) staying fine throughout.

    Net effect: gates ONLY on best_model's own numbers now. swing_policy's
    most recent row is still read separately (see
    latest_swing_policy_row() below) and printed in main() every round for
    visibility -- a human watching still sees it -- it just can no longer
    halt a healthy pipeline on its own.

    Backward compatible with every row logged before 5.9i (lone best_model
    rows, no swing_policy row ever followed) -- unaffected by this change,
    those were always best_model-only anyway. Falls back to spread=0.0 for
    rows logged before the spread column existed, so this doesn't crash on
    an existing log."""
    try:
        with open(LOG_PATH, newline="") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        return []
    # See DOMAIN_RANDOMIZATION_START_TIMESTAMP's own comment above -- ISO
    # timestamps compare correctly as plain strings, no parsing needed.
    rows = [r for r in rows if r.get("timestamp", "") >= DOMAIN_RANDOMIZATION_START_TIMESTAMP]

    def extract(row):
        return (
            float(row["det_peak_angle_deg"]),
            float(row["det_overshoot_penalty"]),
            float(row.get("det_angle_spread_deg") or 0.0),
        )

    gating = []
    for row in rows:
        if row.get("model_path") != BEST_MODEL_TAG:
            continue  # not a best_model row (ad hoc check, EXTRA_MODEL_PATH,
                       # or swing_policy's own row -- read separately by
                       # latest_swing_policy_row() below for reporting, not
                       # gating) -- ignore and move on.
        gating.append(extract(row))
    return gating


def latest_swing_policy_row():
    """(det_peak_angle_deg, det_overshoot_penalty, det_angle_spread_deg,
    det_bootstrap_fail_count) for the MOST RECENT swing_policy row, or None
    if there isn't one yet. READ-ONLY, for main()'s printed visibility
    (README 5.9z) -- never used for any stop/continue decision, see
    read_log()'s own comment above for why gating moved to best_model
    alone. Falls back to 0 for bootstrap_fail_count on rows logged before
    that column existed (5.9h)."""
    try:
        with open(LOG_PATH, newline="") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        return None
    rows = [r for r in rows if r.get("timestamp", "") >= DOMAIN_RANDOMIZATION_START_TIMESTAMP]
    for row in reversed(rows):
        if row.get("model_path") == LATEST_MODEL_TAG:
            return (
                float(row["det_peak_angle_deg"]),
                float(row["det_overshoot_penalty"]),
                float(row.get("det_angle_spread_deg") or 0.0),
                int(row.get("det_bootstrap_fail_count") or 0),
            )
    return None


def streak_from_history(log_rows):
    """How many TRAILING rows of persisted history already pass all three
    checks, scanning back from the most recent. Used to seed
    consecutive_hits on startup instead of always starting at 0 -- with
    MAX_ROUNDS now small, a real streak will usually span separate
    invocations of this script (separate check-ins) rather than completing
    within a single run, so it needs to survive being restarted."""
    hits = 0
    for det_angle, det_overshoot, det_spread in reversed(log_rows):
        if (det_angle >= TARGET_ANGLE_DEG and det_overshoot <= TARGET_OVERSHOOT_MAX
                and det_spread <= SPREAD_MAX_DEG):
            hits += 1
        else:
            break
    return hits


def declining_trend(log_rows):
    """None if no real decline, otherwise a message explaining what was
    found. Looks at the last DECLINE_WINDOW rows of persisted history: if
    the deterministic angle went down (or held flat) every single round
    across that whole window, AND the total drop from the start of the
    window to the end is at least DECLINE_MIN_DROP_DEG, that's a real,
    sustained trend -- ordinary noise doesn't reliably move the same
    direction for that many rounds in a row. This is what catches a
    collapse-in-progress, as distinct from STALL_ROUNDS above (which only
    catches a FLAT read, and would say nothing about a steady decline,
    since a steady decline is by definition a big swing, just the wrong
    direction)."""
    window = log_rows[-DECLINE_WINDOW:]
    if len(window) < DECLINE_WINDOW:
        return None
    angles = [a for a, _, _ in window]
    monotonic_non_increasing = all(angles[i] >= angles[i + 1] for i in range(len(angles) - 1))
    total_drop = angles[0] - angles[-1]
    if monotonic_non_increasing and total_drop >= DECLINE_MIN_DROP_DEG:
        return (f"Deterministic peak angle declined every round for the last "
                f"{DECLINE_WINDOW} rounds ({[f'{a:.1f}' for a in angles]}), a "
                f"{total_drop:.1f} deg total drop. That's a real trend, not "
                f"noise -- this is the early shape of the collapse found "
                f"earlier in this project, not a normal noisy dip.")
    return None


def single_round_cliff(log_rows):
    """None if no cliff, otherwise a message explaining what was found.
    Compares only the two most recent rows -- deliberately NOT a window
    check like declining_trend above, because a climb-then-cliff shape
    isn't monotonic and would slip past that check entirely. This is what
    actually happened once already: 55.6 deg one round, 5.0 deg the very
    next. Catches a sudden single-round collapse regardless of what came
    before it."""
    if len(log_rows) < 2:
        return None
    prior_angle = log_rows[-2][0]
    current_angle = log_rows[-1][0]
    if prior_angle >= CLIFF_MIN_PRIOR_DEG and current_angle < prior_angle * CLIFF_DROP_FRACTION:
        return (f"Deterministic peak angle fell from {prior_angle:.1f} deg to "
                f"{current_angle:.1f} deg in a single round -- a sudden "
                f"collapse, not a gradual trend. This shape wouldn't have "
                f"tripped the decline check (which needs a monotonic multi-"
                f"round slide), so it needs its own check.")
    return None


def erratic_round(log_rows):
    """None if the most recent round looks normal, otherwise a message
    explaining what was found. Looks ONLY at the single most recent row's
    own spread and overshoot -- deliberately independent of every other
    check in this file, all of which key off the MEAN angle. A round can
    have a perfectly ordinary-looking mean while its own episodes
    actually swung between near-total collapse and wild overshoot -- that
    instability is real information the mean alone throws away."""
    if not log_rows:
        return None
    det_angle, det_overshoot, det_spread = log_rows[-1]
    problems = []
    if det_spread > ERRATIC_SPREAD_MAX_DEG:
        problems.append(f"spread {det_spread:.1f} deg (max allowed {ERRATIC_SPREAD_MAX_DEG})")
    if det_overshoot > ERRATIC_OVERSHOOT_MAX:
        problems.append(f"overshoot penalty {det_overshoot:.1f} (max allowed {ERRATIC_OVERSHOOT_MAX})")
    if not problems:
        return None
    return (f"This round's mean angle ({det_angle:.1f} deg) looks unremarkable, but "
            f"{' and '.join(problems)} -- that's real instability the mean-based "
            f"checks (cliff, decline, stall) can't see, since they only look at "
            f"how the mean moves round to round, not how consistent any single "
            f"round's own episodes were.")


def run(cmd):
    print(f"\n=== running: {' '.join(cmd)} ===", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n!! {cmd[1]} exited with an error (code {result.returncode}) -- "
              f"stopping. This needs a human look, not another automated round.")
        sys.exit(1)


def main():
    log_at_start = read_log()
    rows_before = len(log_at_start)

    consecutive_hits = streak_from_history(log_at_start)
    if consecutive_hits:
        print(f"Resuming an existing streak from prior check-ins: "
              f"{consecutive_hits}/{CONSECUTIVE_HITS_REQUIRED} consecutive rounds "
              f"already passing all three checks.")

    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n########## ROUND {round_num}/{MAX_ROUNDS} ##########")
        run([sys.executable, "train.py"])
        run([sys.executable, "evaluate.py"])

        rows = read_log()
        if len(rows) < rows_before:
            # BUG FIX: evaluate_log.csv was archived and reset SINCE this
            # invocation started -- rows_before refers to row counts in a
            # file that's since been moved aside, so slicing rows[rows_before:]
            # against the new (smaller) file comes back empty even though
            # evaluate.py DID just log a real row. Confirmed this actually
            # happened: evaluate.py's own schema-migration self-heal (its
            # EXPECTED_HEADER check, added alongside 5.5a-x's new columns)
            # fired mid-round here, archiving the old log and starting a
            # fresh one -- and this script then reported "didn't grow" and
            # stopped, even though evaluate.py's own printed output said
            # "logged this run to evaluate_log.csv" moments earlier. Same
            # underlying trigger `promote_checkpoint.py`'s reset was already
            # known to cause (see the "Known limitation" note in README
            # 5.4) -- that case just never hit this SPECIFIC check before,
            # since a promotion happens between invocations, not mid-round
            # inside one. Treat every row now in the file as new relative to
            # the stale pre-reset count, and rebase rows_before to 0 so any
            # further rounds later in this same invocation keep slicing
            # correctly against the new, smaller file.
            print(f"\nNote: evaluate_log.csv has fewer rows than expected "
                  f"({len(rows)}, was tracking {rows_before} from before this "
                  f"round) -- it was archived and reset since this invocation "
                  f"started, not a logging failure. Continuing.")
            rows_before = 0
        new_rows = rows[rows_before:]
        if not new_rows:
            print("\nevaluate_log.csv didn't grow after that round -- "
                  "something's wrong with the logging, stopping.")
            sys.exit(1)

        # CHANGED (README 5.9z): these are best_model's OWN numbers now,
        # not the worse of best_model and swing_policy -- see read_log()'s
        # comment above for why. swing_policy's own most recent numbers are
        # printed right below, separately, for visibility only -- they no
        # longer affect round_passes or any of the stop checks further
        # down.
        det_angle, det_overshoot, det_spread = new_rows[-1]
        print(f"\n--- round {round_num} best_model result: peak {det_angle:.2f} "
              f"deg, overshoot penalty {det_overshoot:.2f}, spread "
              f"{det_spread:.2f} deg ---")

        swing_policy_row = latest_swing_policy_row()
        if swing_policy_row:
            sp_angle, sp_overshoot, sp_spread, sp_fails = swing_policy_row
            print(f"    (swing_policy, round-end state, informational only: "
                  f"peak {sp_angle:.2f} deg, overshoot penalty {sp_overshoot:.2f}, "
                  f"spread {sp_spread:.2f} deg, {sp_fails} bootstrap failure(s) -- "
                  f"pick_resume_source() in train.py already protects the next "
                  f"round against resuming from this if it looks broken, see "
                  f"README 5.9y/5.9z)")

        round_passes = (
            det_angle >= TARGET_ANGLE_DEG
            and det_overshoot <= TARGET_OVERSHOOT_MAX
            and det_spread <= SPREAD_MAX_DEG
        )
        if round_passes:
            consecutive_hits += 1
            print(f"Passes all three checks (angle, overshoot, spread) -- "
                  f"{consecutive_hits}/{CONSECUTIVE_HITS_REQUIRED} consecutive rounds so far.")
        else:
            if consecutive_hits > 0:
                print(f"Breaks the streak (was {consecutive_hits}/"
                      f"{CONSECUTIVE_HITS_REQUIRED}) -- resetting to 0. A single good "
                      f"round doesn't count on its own; this needs repeats.")
            consecutive_hits = 0

        if consecutive_hits >= CONSECUTIVE_HITS_REQUIRED:
            print(f"\nTarget reached AND consistent: {CONSECUTIVE_HITS_REQUIRED} rounds "
                  f"in a row each hit >= {TARGET_ANGLE_DEG} deg, overshoot <= "
                  f"{TARGET_OVERSHOOT_MAX}, and spread <= {SPREAD_MAX_DEG} deg. This is "
                  f"repeated, low-variance evidence, not one lucky reading. Stopping -- "
                  f"check it out with mjpython watch_policy.py.")
            return

        # All checks below use the full persisted history (rows), not just
        # this invocation's new_rows -- with MAX_ROUNDS now small, a genuine
        # stall or decline will often only become visible across several
        # separate check-ins, not within one run's handful of rounds.
        cliff_msg = single_round_cliff(rows)
        if cliff_msg:
            print(f"\n{cliff_msg} Stopping here rather than assuming it'll "
                  f"self-correct -- come back and look at what happened this "
                  f"round before running more.")
            return

        erratic_msg = erratic_round(rows)
        if erratic_msg:
            print(f"\n{erratic_msg} Stopping here rather than treating a "
                  f"reasonable-looking mean as the whole story -- worth checking "
                  f"scan_checkpoints.py for how this developed at finer "
                  f"resolution than one evaluate.py round per check-in gives.")
            return

        decline_msg = declining_trend(rows)
        if decline_msg:
            print(f"\n{decline_msg} Stopping here rather than letting it run "
                  f"further unattended -- this is the point to come back and "
                  f"look at what's happening, e.g. with scan_checkpoints.py "
                  f"to find a checkpoint from before the decline started.")
            return

        recent = [a for a, _, _ in rows[-STALL_ROUNDS:]]
        if len(recent) >= STALL_ROUNDS and (max(recent) - min(recent)) < STALL_IMPROVEMENT_DEG:
            print(f"\nNo real improvement (< {STALL_IMPROVEMENT_DEG} deg swing) over "
                  f"the last {STALL_ROUNDS} rounds ({[f'{a:.1f}' for a in recent]}). "
                  f"Stopping here rather than continuing to guess -- this is the "
                  f"point to come back and look at what's happening.")
            return

    print(f"\nHit the {MAX_ROUNDS}-round safety cap without reaching target, a "
          f"clear stall, a clear decline, or a cliff. Worth a manual check-in "
          f"at this point regardless.")


if __name__ == "__main__":
    main()

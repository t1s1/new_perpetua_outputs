"""
Run the trained policy for one full 120s episode -- no training, no
viewer -- and report peak swing angle and total reward. This is the
number to actually decide "keep training or not" on, rather than
eyeballing the noisy ep_rew_mean curve or a subjective impression of the
viewer.

Runs BOTH deterministic (the mean action, no exploration noise -- what
watch_policy.py and real deployment would use) and stochastic (sampled
with exploration noise -- what training rollouts actually execute, and
what ep_rew_mean is computed from) and prints both. If ep_rew_mean looks
good during training but deterministic evaluation looks bad, a large gap
between these two numbers is the smoking gun: it means the policy hasn't
yet converged to a mean behavior that pumps reliably on its own, and is
still leaning on exploration noise to occasionally stumble into reward
during rollouts. If that's what's happening, the fix is to keep training
the same policy (don't restart) -- the mean should keep catching up to
what the noisy rollouts are already finding as training continues.

The target is 67.5 degrees (UPDATED from 80 -- see swing_env.py's
TARGET_ANGLE_DEG comment for why: not a bug fix, a deliberate redefinition
after watching the policy cruise past the old 80 target as normal,
no-incident behavior. Explicit preference now is regular, robust swinging
that maxes out around 65-70 and stops there, over a taller policy that
risks any excursion past it), not "as high as possible" -- swing_env.py's
reward stops rewarding height beyond TARGET_ANGLE_DEG and penalizes angles
past TARGET_ANGLE_DEG + OVERSHOOT_MARGIN_DEG (still 67.5, since the margin
is 0).

Run locally:
    python evaluate.py

Also appends one row per run to evaluate_log.csv (timestamp, both reward/
peak-angle/overshoot numbers) -- this is what auto_train.py reads to
decide whether to keep looping automatically, and it's also just a
persistent history of every check you've done, instead of only living in
scrollback.

Also reports time-to-90% (STEADY_STATE_CYCLES comment below): how many
seconds into the episode it takes to reach 90% of that episode's own
steady-state pumping amplitude, i.e. how quickly it gets "up to speed"
rather than just where it eventually settles. Informational only for
now, like time-in-band/wall-touches before it -- not wired into
auto_train.py's pass/fail gating.
"""
import csv
import os
from datetime import datetime, timezone

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from swing_env import SwingEnv, TARGET_ANGLE_DEG

LOG_PATH = "evaluate_log.csv"

# NEW. peak_angle_deg (the single highest angle reached anywhere in the
# episode) turned out to be a misleading proxy for "is this a good
# policy" -- confirmed by actually watching watch_policy.py twice: a
# policy that looked like it had converged onto the wall (peak_angle_deg
# 163-165 deg, near-zero spread across all 10 eval episodes, exactly the
# pattern that had previously meant "stuck") was, on direct visual
# inspection, mostly a clean 80-90 deg oscillation with only 1-3 brief
# excursions per episode actually reaching the limit. One rare event is
# enough to pull the single-max statistic all the way to 165 regardless
# of how good the other ~99% of the episode was -- it can't distinguish
# "lived at the wall" from "touched it once." These two new metrics are
# meant to see what peak_angle_deg can't: TARGET_BAND_HALF_WIDTH_DEG
# defines a "does this step look like the target behavior" band around
# TARGET_ANGLE_DEG, and time_in_band_frac (below) reports what fraction
# of the WHOLE episode was spent there -- a majority-behavior statistic
# that a handful of rare steps can't dominate the way a single max can.
# WALL_TOUCH_THRESHOLD_DEG/wall_touch_count instead give the rare-event
# side of the picture explicitly, but as a COUNT of distinct excursions
# (only counting a fresh crossing from below the threshold to above it,
# not every step spent above it) rather than a single worst-case angle --
# closer to what "I counted 3 bumps" actually means than any single
# max-angle number can be. Deliberately reported ALONGSIDE the existing
# metrics, not replacing them, and NOT yet wired into auto_train.py's
# pass/fail gating -- don't have enough rounds of this new data yet to
# pick sensible thresholds for that; see it accumulate in
# evaluate_log.csv first.
TARGET_BAND_HALF_WIDTH_DEG = 10.0  # UPDATED, 15.0 -> 10.0, alongside
                              # TARGET_ANGLE_DEG's redefinition in
                              # swing_env.py (80 -> 67.5). Tightened, not
                              # just recentered: at the old width (15) the
                              # band's upper edge would sit at 77.5,
                              # counting angles close to the OLD target --
                              # exactly what's now supposed to be avoided --
                              # as "in band". 10 keeps the upper edge at
                              # 77.5 deg -- comfortably shy of where
                              # HEIGHT_DECAY_RANGE_DEG starts meaningfully
                              # cutting into reward (87.5), consistent with
                              # the new, tighter tolerance for anything past
                              # target.
WALL_TOUCH_THRESHOLD_DEG = 150.0  # Unchanged -- this is about proximity to
                              # the physical joint limit (165, swing_robot.xml
                              # 5.8), not the reward target, so it doesn't
                              # need to move just because the target did.

# NEW. Direct checkpoint sampling (README 5.9h) found deterministic
# (no-exploration-noise) evaluation sometimes never gets the swing pumping
# at all -- distinct from ordinary "hasn't converged yet" underperformance,
# and from overshoot. STOCHASTIC evaluation of the exact same checkpoint
# had zero such failures (0/8), while deterministic had 4/8 -- and a
# seed-kick-magnitude sweep on the same checkpoint found no threshold
# effect (near-identical kicks landed on opposite outcomes), ruling out
# "the random seed kick was just too weak" as the explanation. This
# threshold exists to measure how often that happens, since nothing did
# before -- it was found by accident via ad hoc sampling, not by anything
# evaluate.py reported. 20 deg sits in a clean gap actually observed
# between the two clusters: failed episodes topped out at 6.7-19.5 deg,
# every successful one reached 71-83 deg -- nothing lands in between, so
# this isn't a fine-tuned cutoff, just a line drawn in the empty space
# between two clearly separate outcomes. Like time-in-band/wall-touches/
# time-to-90pct before it: informational only for now, NOT wired into
# auto_train.py's pass/fail gating -- and even at N_EVAL_EPISODES=15 (was
# 5, README 5.9j) per side, a single round's fraction is still a coarse
# (~6.7%-resolution) estimate, not a precise rate; trust the trend across
# rounds more than any one round's number.
BOOTSTRAP_FAIL_MAX_ANGLE_DEG = 20.0

# NEW. "How quickly does it get up to speed" -- requested after a one-off
# ad hoc analysis (extrema-based per-cycle peak envelope, same technique as
# the earlier per-cycle pumping-ceiling investigations) showed the policy
# reaching ~90% of its own steady-state amplitude in about 92s of a 360s
# episode, then holding flat for the rest. That was a hand-run script, not
# a tracked metric -- this wires the same measurement into every regular
# evaluate.py round instead of a one-off.
#
# Deliberately per-CYCLE, not per fixed time interval or per raw step:
# reading angle at arbitrary time steps is exactly the fixed-time-sampling
# mistake that produced a misleading "stalled" read earlier in this
# project (see README) -- a snapshot can land anywhere in an oscillation
# regardless of how the envelope is actually trending. Extrema (turning
# points where the SIGNED angle trace changes direction) are computed on
# the signed trace, not abs(angle) -- taking abs() first would invent a
# spurious extra "turning point" at every zero-crossing (where the true
# signed trace passes smoothly through 0, but its abs value has a sharp
# corner), roughly doubling the apparent cycle count with fake extrema.
STEADY_STATE_CYCLES = 10     # trailing half-cycles averaged to define
                              # "steady state" amplitude -- matches the
                              # ad hoc investigation's own choice, not
                              # re-derived from anything more principled.
                              # Small enough to reflect recent behavior,
                              # large enough that one unlucky half-cycle
                              # doesn't swing the reference number.
MIN_CYCLES_FOR_STEADY_STATE = 10  # below this many half-cycles in the whole
                              # episode, "steady state" isn't a meaningful
                              # concept yet (nothing to average) -- report
                              # n/a rather than a number computed from too
                              # little data to mean anything.

# Two fixed checkpoints get evaluated every run now, not just one -- see
# the __main__ block at the bottom of this file (README 5.9i). Each pair
# below must use ITS OWN matching vec_normalize file, not just whichever
# one happens to be lying around: train.py always saves swing_policy.zip
# and vec_normalize.pkl together as one pair, while best_model.zip and
# best_model/best_vec_normalize.pkl are a separate pair kept in sync by
# SaveVecNormalizeOnBest -- a normalized-observation model is only
# meaningful together with the SAME run's stats, not just any recent ones.
#
# BUG FIX: the single active MODEL_PATH/VECNORM_PATH pair this file used
# to have evaluated best_model/best_model.zip's WEIGHTS against
# vec_normalize.pkl -- the swing_policy/latest-state stats, NOT
# best_model's own paired best_vec_normalize.pkl. Every best_model report
# and logged row this whole project has gone through that mismatch until
# now. Not separately re-measured here how much it actually mattered
# (VecNormalize's running stats tend to move slowly once training's past
# its early phase), but it's a real correctness bug, fixed here as part of
# adding the second (latest-state) checkpoint below, since that required
# doing this exact paired-path handling correctly anyway.
BEST_MODEL_PATH = "best_model/best_model"
BEST_VECNORM_PATH = "best_model/best_vec_normalize.pkl"

# NEW (README 5.9i). Every real problem serious enough to need a manual
# checkpoint-sampling investigation so far has shared one root cause: this
# file only ever checked best_model, never swing_policy.zip -- the raw
# state training actually resumes from, which has repeatedly turned out to
# look meaningfully worse (heavy overshoot, even bootstrap failures) while
# best_model alone reported clean. Checking both every round, automatically,
# closes that blind spot at the source instead of relying on a human
# noticing the report looks suspiciously good and sampling checkpoints by
# hand to check.
LATEST_MODEL_PATH = "swing_policy"
LATEST_VECNORM_PATH = "vec_normalize.pkl"

# Optional THIRD check, alongside the two standard ones above -- e.g. a
# rescue candidate from scan_checkpoints.py, or any other one-off
# checkpoint. None (default) skips this entirely. Replaces the old "hand-
# edit MODEL_PATH/VECNORM_PATH at the top of the file" manual-check
# workflow (README section 6) now that those two names are spoken for
# above -- set both of these instead, run `python evaluate.py`, and
# remember to set them back to None afterward so a stray extra check
# doesn't keep running (and logging) on every subsequent round.
EXTRA_MODEL_PATH = None
EXTRA_VECNORM_PATH = None

# Fallback default read by run_episode/run_multiple below when called
# without an explicit vecnorm_path -- used by this project's own ad hoc
# checkpoint-sampling scripts (`import evaluate; evaluate.MODEL_PATH =
# ...; evaluate.VECNORM_PATH = ...`, the pattern used throughout this
# project's own investigations) instead of threading a parameter through
# by hand every time. Defaults to the best_model pairing.
MODEL_PATH = BEST_MODEL_PATH
VECNORM_PATH = BEST_VECNORM_PATH
N_EVAL_EPISODES = 15  # UPDATED, 5 -> 15 (README 5.9j). Each episode gets
                       # its own randomized seed kick (see swing_env.py's
                       # reset()) -- a single episode's result is partly
                       # just "which direction/magnitude did that one
                       # random kick happen to be", which can look like
                       # wild policy instability between separate
                       # evaluate.py runs when it's really just which draw
                       # got sampled. Averaging over several episodes gives
                       # a number that reflects the policy, not the draw --
                       # but 5 turned out not to be enough of them: the
                       # round right after 5.9i shipped saw the two checks
                       # disagree sharply (best_model's 5-episode draw hit
                       # 48.9 deg mean/61.5 deg spread and tripped a
                       # decline alarm; a fresh 10-episode resample of that
                       # SAME checkpoint came back 71.4 deg mean/8.7 deg
                       # spread -- clean). Raised well past the minimum
                       # needed to fix that one incident since evaluation
                       # compute is cheap relative to a training round.
                       # NOTE: this also changes what SPREAD_MAX_DEG (15,
                       # auto_train.py) and ERRATIC_SPREAD_MAX_DEG (90,
                       # auto_train.py) mean -- max-min across MORE draws
                       # tends to widen even for a policy with unchanged
                       # underlying variance (more chances to sample a
                       # tail episode), so both were calibrated against
                       # the old N=5 behavior and may now need
                       # re-examining once real N=15 data accumulates.
                       # Deliberately not guessing new values for those
                       # without evidence, same discipline as every other
                       # threshold in this project.


def compute_time_to_90pct(angle_trace, dt):
    """(time_to_90pct_s, steady_state_deg), both None if the episode didn't
    contain enough half-cycles to define a steady state. See the
    STEADY_STATE_CYCLES comment above for why this uses per-cycle extrema
    on the SIGNED trace rather than raw per-step samples or abs(angle)."""
    angles = np.asarray(angle_trace)
    if len(angles) < 3:
        return None, None
    d = np.diff(angles)
    sign = np.sign(d)
    sign[sign == 0] = 1
    turning_idx = np.where(np.diff(sign) != 0)[0] + 1
    if len(turning_idx) < MIN_CYCLES_FOR_STEADY_STATE:
        return None, None
    cycle_times = turning_idx * dt
    cycle_peaks = np.abs(angles[turning_idx])
    steady_state = float(np.mean(cycle_peaks[-STEADY_STATE_CYCLES:]))
    threshold = 0.9 * steady_state
    hit_idx = int(np.argmax(cycle_peaks >= threshold))
    if cycle_peaks[hit_idx] < threshold:  # argmax returns 0 if nothing hit
        return None, steady_state
    return float(cycle_times[hit_idx]), steady_state


def run_episode(model, deterministic, vecnorm_path=None):
    # vecnorm_path defaults to the module-level VECNORM_PATH global rather
    # than requiring every caller to pass it -- keeps this project's own ad
    # hoc checkpoint-sampling scripts (which set evaluate.MODEL_PATH/
    # VECNORM_PATH directly, then call run_multiple) working unchanged,
    # while evaluate_and_log() below (README 5.9i) passes it explicitly so
    # two different checkpoints can be evaluated in the same run without
    # mutating shared globals in between.
    if vecnorm_path is None:
        vecnorm_path = VECNORM_PATH
    vec_env = DummyVecEnv([lambda: SwingEnv()])
    vec_env = VecNormalize.load(vecnorm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    # dt read directly off the compiled model rather than hardcoded, so this
    # can't silently drift out of sync if timestep/frame_skip ever change --
    # same self-syncing discipline as swing_env.py's arm_length.
    inner_env = vec_env.venv.envs[0]
    dt = inner_env.frame_skip * inner_env.model.opt.timestep

    obs = vec_env.reset()
    total_reward = 0.0
    max_angle_deg = 0.0
    total_overshoot_penalty = 0.0
    total_steps = 0
    steps_in_band = 0
    wall_touch_count = 0
    past_wall_threshold = False  # tracks the PREVIOUS step's state, so a
                                  # touch is counted once per crossing from
                                  # below the threshold to above it, not
                                  # once per step spent above it
    angle_trace = []  # SIGNED swing_angle_deg per step -- fed to
                       # compute_time_to_90pct after the episode ends
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done_arr, info = vec_env.step(action)
        done = bool(done_arr[0])
        total_reward += float(reward[0])
        angle_deg_signed = info[0]["swing_angle_deg"]
        angle_trace.append(angle_deg_signed)
        angle_deg = abs(angle_deg_signed)
        max_angle_deg = max(max_angle_deg, angle_deg)
        total_overshoot_penalty += info[0]["overshoot_penalty"]

        total_steps += 1
        if abs(angle_deg - TARGET_ANGLE_DEG) <= TARGET_BAND_HALF_WIDTH_DEG:
            steps_in_band += 1
        past_wall_now = angle_deg >= WALL_TOUCH_THRESHOLD_DEG
        if past_wall_now and not past_wall_threshold:
            wall_touch_count += 1
        past_wall_threshold = past_wall_now

    time_in_band_frac = steps_in_band / total_steps if total_steps else 0.0
    time_to_90pct, steady_state_deg = compute_time_to_90pct(angle_trace, dt)
    return (total_reward, max_angle_deg, total_overshoot_penalty,
            time_in_band_frac, wall_touch_count, time_to_90pct, steady_state_deg)


def run_multiple(model, deterministic, n_episodes, vecnorm_path=None):
    rewards, angles, overshoots, band_fracs, wall_touches = [], [], [], [], []
    ramp_times, steady_states = [], []
    for _ in range(n_episodes):
        r, a, o, b, w, rt, ss = run_episode(model, deterministic, vecnorm_path=vecnorm_path)
        rewards.append(r)
        angles.append(a)
        overshoots.append(o)
        band_fracs.append(b)
        wall_touches.append(w)
        ramp_times.append(rt)
        steady_states.append(ss)
    return rewards, angles, overshoots, band_fracs, wall_touches, ramp_times, steady_states


EXPECTED_HEADER = [
    "timestamp", "model_path",
    "det_reward", "det_peak_angle_deg", "det_overshoot_penalty",
    "det_angle_spread_deg",
    "sto_reward", "sto_peak_angle_deg", "sto_overshoot_penalty",
    "det_time_in_band_frac", "det_wall_touch_count",
    "sto_time_in_band_frac", "sto_wall_touch_count",
    "det_time_to_90pct_s", "det_steady_state_deg",
    "sto_time_to_90pct_s", "sto_steady_state_deg",
    "det_bootstrap_fail_count", "sto_bootstrap_fail_count",
]


def evaluate_and_log(model_path, vecnorm_path):
    """Run the full report (both eval modes, printed) for ONE checkpoint
    and append ONE row to evaluate_log.csv under model_path. Unchanged
    schema/row-writing logic from before this was split out -- README
    5.9i's change is calling this twice (once per checkpoint) from
    __main__ below, not anything about what a single call does."""
    model = PPO.load(model_path)

    try:
        std = model.policy.log_std.exp().detach().cpu().numpy()
        print(f"current action std (exploration noise level): {std}")
    except AttributeError:
        pass  # not all policy architectures expose log_std the same way; skip

    results = {}
    angles_by_side = {}
    for label, deterministic in [("DETERMINISTIC (mean action)", True), ("STOCHASTIC (sampled)", False)]:
        rewards, angles, overshoots, band_fracs, wall_touches, ramp_times, steady_states = run_multiple(
            model, deterministic, N_EVAL_EPISODES, vecnorm_path=vecnorm_path)
        mean_reward = sum(rewards) / len(rewards)
        mean_angle = sum(angles) / len(angles)
        mean_overshoot = sum(overshoots) / len(overshoots)
        mean_band_frac = sum(band_fracs) / len(band_fracs)
        mean_wall_touches = sum(wall_touches) / len(wall_touches)
        valid_ramp_times = [t for t in ramp_times if t is not None]
        mean_ramp_time = sum(valid_ramp_times) / len(valid_ramp_times) if valid_ramp_times else None
        valid_steady_states = [s for s in steady_states if s is not None]
        mean_steady_state = sum(valid_steady_states) / len(valid_steady_states) if valid_steady_states else None
        bootstrap_fail_count = sum(1 for a in angles if a < BOOTSTRAP_FAIL_MAX_ANGLE_DEG)
        results[deterministic] = (mean_reward, mean_angle, mean_overshoot, mean_band_frac,
                                   mean_wall_touches, mean_ramp_time, mean_steady_state,
                                   bootstrap_fail_count)
        angles_by_side[deterministic] = angles
        print(f"\n-- {label} ({N_EVAL_EPISODES} episodes) --")
        print(f"per-episode rewards      : {[f'{r:.1f}' for r in rewards]}")
        print(f"per-episode peak angles  : {[f'{a:.1f}' for a in angles]}")
        print(f"per-episode time-in-band : {[f'{b:.2f}' for b in band_fracs]} "
              f"(fraction of episode within {TARGET_ANGLE_DEG:.0f}+-{TARGET_BAND_HALF_WIDTH_DEG:.0f} deg)")
        print(f"per-episode wall touches : {wall_touches} "
              f"(distinct excursions >= {WALL_TOUCH_THRESHOLD_DEG:.0f} deg)")
        print(f"per-episode time-to-90%  : {[f'{t:.1f}' if t is not None else 'n/a' for t in ramp_times]} "
              f"(s to reach 90% of that episode's own steady-state amplitude, "
              f"n/a if fewer than {MIN_CYCLES_FOR_STEADY_STATE} half-cycles happened)")
        print(f"mean reward              : {mean_reward:.2f}")
        print(f"mean peak swing angle    : {mean_angle:.2f} deg")
        print(f"mean overshoot penalty   : {mean_overshoot:.2f} (want ~0)")
        print(f"mean time-in-band        : {mean_band_frac:.2%} (want high -- this is what "
              f"peak angle can't show)")
        print(f"mean wall touches        : {mean_wall_touches:.1f} (want ~0, but a low count "
              f"here with a high time-in-band above is a much better sign than peak angle "
              f"alone suggests)")
        if mean_ramp_time is not None:
            print(f"mean time-to-90%         : {mean_ramp_time:.1f} s (want low -- how quickly "
                  f"it gets up to speed, not just where it ends up; steady-state amplitude "
                  f"used as the reference was {mean_steady_state:.1f} deg)")
        else:
            print(f"mean time-to-90%         : n/a (no episode this side ran long enough / "
                  f"cycled enough to establish a steady state)")
        print(f"bootstrap failures       : {bootstrap_fail_count}/{N_EVAL_EPISODES} (peak angle "
              f"< {BOOTSTRAP_FAIL_MAX_ANGLE_DEG:.0f} deg -- never got pumping going at all, "
              f"distinct from just not reaching target; want 0. Only {N_EVAL_EPISODES} episodes, "
              f"so trust the trend across rounds more than any single round's count)")

    print("\ntarget is ~80 deg, not maximum possible height")
    print("large gap between the two means above => policy hasn't converged yet, keep training")
    print("large SPREAD within one side's own per-episode numbers => inconsistent across")
    print("different random seed kicks, a separate issue from det-vs-sto convergence")
    print("peak angle can be dominated by a single rare excursion -- time-in-band and wall")
    print("touches (above) reflect the REST of the episode, which peak angle alone can't")

    (det_reward, det_angle, det_overshoot, det_band_frac, det_wall_touches,
     det_ramp_time, det_steady_state, det_bootstrap_fails) = results[True]
    (sto_reward, sto_angle, sto_overshoot, sto_band_frac, sto_wall_touches,
     sto_ramp_time, sto_steady_state, sto_bootstrap_fails) = results[False]

    # Spread (max - min) of the deterministic per-episode peak angles -- the
    # piece a single mean can't tell you. A mean can land right on target while
    # still hiding a policy that's great on some seed kicks and poor on others;
    # that's not "arrived", it's a coin flip that happens to average out. This
    # gets logged alongside the mean so auto_train.py can require BOTH a good
    # mean AND a tight spread before calling anything reached, not just one.
    det_angles = angles_by_side[True]
    det_angle_min = min(det_angles)
    det_angle_max = max(det_angles)
    det_angle_spread = det_angle_max - det_angle_min
    print(f"\ndeterministic spread across episodes: {det_angle_spread:.2f} deg "
          f"(min {det_angle_min:.1f}, max {det_angle_max:.1f}) -- a good mean with "
          f"a wide spread here isn't a converged policy, it's a lucky average")

    # BUG FIX: a prior version of this script added det_angle_spread_deg as a
    # new column but only ever wrote the header when the file didn't exist yet
    # -- so against an already-existing log, new 9-field rows got silently
    # appended under the OLD 8-column header, shifting every column name from
    # sto_reward onward by one (sto_reward's header ended up pointing at
    # det_angle_spread_deg's values, sto_peak_angle_deg's header at sto_reward's
    # values, and so on), with the true sto_overshoot_penalty left with no
    # header at all. auto_train.py's det_peak_angle_deg/det_overshoot_penalty
    # reads were unaffected (they sit before the inserted column), but its
    # det_angle_spread_deg read silently came back as 0.0 the entire time,
    # every column-adding change to this schema. Rather than trust a header
    # written once at file-creation time to still describe the file, check it
    # on every single run: if it doesn't match what this script is about to
    # write, archive the old log (nothing is discarded) and start a fresh one
    # with a correct header, instead of silently corrupting column meaning
    # again the next time a column gets added.
    log_exists = os.path.exists(LOG_PATH)
    if log_exists:
        with open(LOG_PATH, newline="") as f:
            existing_header = next(csv.reader(f), None)
        if existing_header != EXPECTED_HEADER:
            backup_path = LOG_PATH + f".bak-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
            os.rename(LOG_PATH, backup_path)
            print(f"\n!! evaluate_log.csv's header didn't match this script's current "
                  f"columns (schema changed since it was created) -- moved the old "
                  f"file to {backup_path} (nothing lost, still readable by hand) and "
                  f"starting a fresh log with the correct header, rather than risk "
                  f"silently misaligning columns again.")
            log_exists = False

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(EXPECTED_HEADER)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(), model_path,
            f"{det_reward:.4f}", f"{det_angle:.4f}", f"{det_overshoot:.4f}",
            f"{det_angle_spread:.4f}",
            f"{sto_reward:.4f}", f"{sto_angle:.4f}", f"{sto_overshoot:.4f}",
            f"{det_band_frac:.4f}", f"{det_wall_touches:.2f}",
            f"{sto_band_frac:.4f}", f"{sto_wall_touches:.2f}",
            f"{det_ramp_time:.4f}" if det_ramp_time is not None else "",
            f"{det_steady_state:.4f}" if det_steady_state is not None else "",
            f"{sto_ramp_time:.4f}" if sto_ramp_time is not None else "",
            f"{sto_steady_state:.4f}" if sto_steady_state is not None else "",
            str(det_bootstrap_fails), str(sto_bootstrap_fails),
        ])
    print(f"\nlogged this run to {LOG_PATH} under model_path={model_path}")


# BUG FIX: this whole evaluation used to sit at plain module level, so
# `import evaluate` -- done repeatedly in this project's own ad hoc
# checkpoint-sampling scripts, to reuse run_episode/run_multiple/
# compute_time_to_90pct without copy-pasting them -- silently ran a full
# LIVE evaluation of whatever MODEL_PATH happened to be set above and
# appended a real row to evaluate_log.csv, every single time, as a side
# effect of the import statement alone. Confirmed this actually happened:
# three import-only invocations while investigating the 2026-08-11 round
# silently added three extra best_model rows to the log with no train.py
# run behind them, which would have corrupted auto_train.py's streak/
# decline/cliff history (it can't tell a real round's row from an import's
# row). Caught before auto_train.py ran again; the three spurious rows
# were removed from evaluate_log.csv by hand. Guarding the whole thing
# behind `if __name__ == "__main__":` makes `import evaluate` safe to do
# for its functions/constants -- this is now the only way to make this
# file actually re-run its evaluation.
if __name__ == "__main__":
    # NEW (README 5.9i): evaluates BOTH best_model AND swing_policy (the
    # round-end state training actually resumes from) every single run,
    # not just best_model -- every real problem serious enough to need a
    # manual checkpoint-sampling investigation so far turned out to be
    # this exact gap (best_model reporting clean while the round-end state
    # didn't). See auto_train.py's read_log() for how it combines the two
    # resulting rows per round (worse-of-both on angle/overshoot/spread)
    # into its cliff/decline/stall/erratic checks.
    print("=" * 70)
    print(f"Evaluating BEST_MODEL_PATH = {BEST_MODEL_PATH!r}")
    print("=" * 70)
    evaluate_and_log(BEST_MODEL_PATH, BEST_VECNORM_PATH)

    print("\n" + "=" * 70)
    print(f"Evaluating LATEST_MODEL_PATH = {LATEST_MODEL_PATH!r} (round-end / "
          f"current resume point)")
    print("=" * 70)
    evaluate_and_log(LATEST_MODEL_PATH, LATEST_VECNORM_PATH)

    if EXTRA_MODEL_PATH is not None:
        print("\n" + "=" * 70)
        print(f"Evaluating EXTRA_MODEL_PATH = {EXTRA_MODEL_PATH!r}")
        print("=" * 70)
        evaluate_and_log(EXTRA_MODEL_PATH, EXTRA_VECNORM_PATH)

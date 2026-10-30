"""
Train a PPO policy against SwingEnv using Stable-Baselines3.

No hand-coded pumping logic anywhere in this pipeline: the policy starts
as random noise and has to discover both how to pump the swing *and* how
to bootstrap a pump from a dead stop, purely from the reward defined in
swing_env.py (height gained each step).

Run locally -- this needs stable-baselines3 and torch, which were too
large to install in the sandbox used to build/validate the environment
itself, so this script is untested beyond a read-through. Report back
what happens; the most likely first failures are dependency versions,
not logic.

    pip install stable-baselines3
    python train.py

Training runs headless, no viewer -- rendering would only slow it down.
Once you have a saved model, use watch_policy.py to load it and see it
run in the interactive viewer.

WHAT TO WATCH: model.learn() prints a table periodically. The number that
matters is rollout/ep_rew_mean -- mean total reward per episode over the
last several completed episodes. It's the raw (unnormalized) height-gain
reward from swing_env.py, summed over an episode, so it's directly
readable as "how much cumulative height is this policy racking up." Watch
it trend across printed rows: rising means keep training; flat near a low
value for a long stretch points more toward the cold-start problem (see
swing_env.py's docstring) than a training-duration problem.

RESUMING: rerunning this script will automatically continue training an
existing swing_policy.zip rather than starting a fresh random policy, as
long as swing_policy.zip and vec_normalize.pkl are both still present in
this directory. Delete them (or move them aside) if you actually want to
start over from scratch.

PPO training is NOT guaranteed to improve monotonically -- deterministic
(mean-action) performance can regress during continued training even
while the noisy rollout reward keeps looking fine. To stop that from
silently costing progress, this script runs an EvalCallback that
periodically evaluates the policy DETERMINISTICALLY on a separate eval
environment during training, and saves the best one it's ever seen to
best_model/. evaluate.py checks both best_model AND swing_policy
automatically every round now (README 5.9i), no manual editing needed.

RESUME SOURCE (README 5.9k): resuming used to always mean "continue from
swing_policy.zip", unconditionally. Direct checkpoint investigation found
real cases where swing_policy ended a round meaningfully worse than a
checkpoint EvalCallback had already found earlier that SAME round -- a
sharp, single-update transition invisible to every PPO training
diagnostic (value_loss, std, explained_variance, approx_kl, clip_fraction
all read completely normal at the exact transition). See
pick_resume_source() below: this script now checks the most recent
evaluate_log.csv comparison between best_model and swing_policy, and
resumes from best_model instead if swing_policy showed bootstrap failures
that best_model didn't -- so training doesn't keep building on a state
already known to be worse than one already in hand.

FIXED BUG: the first version of this safety net didn't actually protect
across separate resumed runs -- EvalCallback's best-score tracking resets
to -inf every time this script starts fresh, so a mediocre evaluation
early in a new run could silently overwrite a genuinely better
best_model.zip from a previous run. This has now happened for real (a
good checkpoint was lost this way). Fixed by evaluating any existing
best_model.zip at startup and seeding the callback's baseline with its
real score, so it can only be beaten, not carelessly overwritten.
"""
import csv
import math
import os

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from swing_env import SwingEnv, DOMAIN_RANDOMIZATION_ENABLED, OBSERVATION_NOISE_ENABLED


class SaveVecNormalizeOnBest(BaseCallback):
    """Keeps best_model/best_vec_normalize.pkl in sync with best_model.zip.

    EvalCallback saves the policy weights whenever it finds a new best, but
    not the VecNormalize running statistics -- and since those stats keep
    drifting throughout training, loading best_model.zip together with
    whatever vec_normalize.pkl the run ends on (rather than what was
    current at the moment that checkpoint was actually the best) can itself
    cause a mismatch that looks like bad performance. This saves a matching
    snapshot every time EvalCallback finds a new best.
    """

    def __init__(self, vec_env, save_path):
        super().__init__()
        self.vec_env = vec_env
        self.save_path = save_path

    def _on_step(self):
        self.vec_env.save(self.save_path)
        return True


class LogStdClipCallback(BaseCallback):
    """Hard structural cap on the policy's action-distribution log_std.

    Added in README 5.5a-xxii after ent_coef tuning alone proved unable to
    reliably hold log_std in a safe range: raising it (0.01 -> 0.03)
    overshot into saturation (log_std.exp() = [2.21, 2.79]), and after
    directly resetting log_std to a sane 0.30 and lowering ent_coef to
    0.015, std STILL climbed back to [0.93, 1.08] within one further round
    -- two independent starting points climbing back toward saturation
    under the exact same coefficient value is strong evidence that no
    single ent_coef setting is a reliably stable resting point here on its
    own. Gaussian entropy is unbounded (grows without limit as std grows),
    so any positive ent_coef supplies some ongoing upward pressure, while
    the [-1, 1] action clip in swing_env.py's step() destroys the clean
    policy-gradient signal needed to self-correct once std gets large
    enough that actions start saturating it -- there's no natural brake.

    This adds a hard, structural safety net instead of relying on that
    balance: once per rollout (_on_rollout_start, which fires right after
    the previous rollout's train() call finishes its gradient updates and
    before the next batch of experience gets collected -- so it always acts
    on the freshly-updated parameters), log_std is clamped in place to
    [log(LOG_STD_MIN_STD), log(LOG_STD_MAX_STD)]. This bounds BOTH failure
    modes this project has now hit: full collapse toward near-deterministic,
    can't-explore behavior (the original problem ent_coef was added in
    5.5a-xviii to fix, back when ent_coef didn't exist at all) on the low
    end, and saturated, near-random noise (5.5a-xx/xxi) on the high end --
    regardless of what ent_coef or the reward landscape happen to be doing
    at the time. ent_coef still influences WHERE std settles within that
    range, but can no longer single-handedly cause a full collapse or
    explosion.

    _on_rollout_start also fires once before the very first rollout of any
    model.learn() call, including a freshly resumed run -- so this
    immediately clamps whatever log_std a loaded checkpoint already has.
    Simply adding this callback and resuming training is enough to correct
    a checkpoint that's already out of range; no separate one-off reset
    script needed going forward (5.5a-xxi's manual reset was a one-time
    fix for a checkpoint that already existed before this callback did).
    """

    def __init__(self, min_std, max_std, verbose=0):
        super().__init__(verbose)
        self.min_log_std = math.log(min_std)
        self.max_log_std = math.log(max_std)

    def _on_step(self):
        # Required abstract method on BaseCallback -- this callback's actual
        # work happens once per rollout in _on_rollout_start below, not on
        # every individual env step.
        return True

    def _on_rollout_start(self):
        with torch.no_grad():
            self.model.policy.log_std.data.clamp_(self.min_log_std, self.max_log_std)


TIMESTEPS_THIS_RUN = 1_000_000  # how many additional steps to train this
                                  # call, whether starting fresh or resuming.
                                  # Raised from 500k -- longer episodes and
                                  # a harder credit-assignment problem mean
                                  # this run needs more experience, not less.
N_ENVS = 8                       # parallel environments; lower this if your
                                  # machine has fewer than ~8 cores
GAMMA = 0.997  # pulled down again from 0.999. 0.999 gives a ~20s effective
                # horizon (1/(1-gamma) env-steps * ~0.02s/step); 0.997 cuts
                # that to ~6.7s -- still enough to span roughly 2 full
                # oscillation periods of this pendulum (~2.8-3.3s each), but
                # a meaningfully shorter, less variance-prone value-
                # estimation problem. NOTE: this was set once before and
                # appeared not to fix the recurring collapse -- but that
                # conclusion turned out to be unreliable, because the
                # resuming code path below wasn't passing gamma to
                # PPO.load(), so every resumed run silently kept using
                # whichever gamma was baked into the loaded checkpoint,
                # never actually picking up this value. Fixed now (see
                # PPO.load() call below) -- this is the first run where
                # 0.997 will really be in effect.
TARGET_KL = 0.03  # NEW. TensorBoard showed train/approx_kl spiking to 0.4+
                    # repeatedly (healthy PPO updates are typically ~0.01-
                    # 0.05) and train/clip_fraction chronically elevated at
                    # 15-28% for most of training, not just during collapses
                    # -- i.e. the policy is persistently trying to move
                    # further per update than the trust region comfortably
                    # allows. SB3's PPO defaults target_kl to None, meaning
                    # NO circuit breaker exists to stop a batch's gradient
                    # epochs early if KL divergence blows up mid-update --
                    # exactly the kind of runaway single-update jump that
                    # would explain a sudden single-round collapse. Setting
                    # this makes SB3 stop taking further epochs on a batch
                    # once approx_kl exceeds it, directly targeting the
                    # mechanism the TensorBoard charts actually showed,
                    # rather than another indirect guess like gamma or
                    # reward normalization.
LOG_STD_MIN_STD = 0.15  # Lower bound for LogStdClipCallback above -- keeps
                # a baseline amount of exploration alive no matter what
                # ent_coef/the reward landscape are doing, structurally
                # ruling out a full repeat of the pre-ent_coef entropy
                # collapse (5.5a-xviii). Not zero: some exploration should
                # always remain possible.
LOG_STD_MAX_STD = 0.6  # Upper bound -- stays under the point where a large
                # fraction of sampled actions start landing on step()'s
                # [-1, 1] hard clip. 5.5a-xx identified roughly 0.3-0.8 as
                # the "meaningful variation without saturating" range;
                # picked the lower half of that (0.6) given how much
                # ent_coef alone has now overshot in this project (up to
                # std=5.15) -- leaves more headroom before the cap itself
                # is doing the only thing standing between the policy and
                # saturation.
ENT_COEF = 0.005  # LOWERED AGAIN, 0.015 -> 0.005, alongside adding the
                # LogStdClipCallback hard cap above -- see README 5.5a-xxii
                # for the full evidence. In short: two INDEPENDENT starting
                # points -- the original saturated checkpoint, and a
                # freshly log_std-reset checkpoint (5.5a-xxi) -- both
                # climbed back toward saturation ([0.93, 1.08] within a
                # single further round from the reset's 0.30) under this
                # exact same 0.015 value. That's stronger evidence than a
                # single bad reading: 0.015 isn't a stable resting value
                # here, not just unlucky timing with one particular
                # starting state. With the hard cap now doing the
                # structural job of bounding BOTH collapse and saturation
                # regardless of this constant, ent_coef's role shifts from
                # "the only thing standing between collapse and explosion"
                # to "fine-tuning exploration pressure within an already-
                # safe range" -- so trying a more conservative value here
                # is much lower-risk than it would have been without the
                # cap in place. Watch model.policy.log_std.exp() same as
                # before: with the cap active it can't exceed
                # LOG_STD_MAX_STD (0.6) or fall below LOG_STD_MIN_STD
                # (0.15) regardless, so the interesting question now is
                # where WITHIN that range it settles, and whether
                # evaluate.py performance keeps recovering now that actions
                # aren't saturating.
                #
                # [PREVIOUS reasoning below, kept for history]
                # LOWERED, 0.03 -> 0.015. 0.03 (previous reasoning kept
                # below) overshot badly -- checked the policy's actual
                # action std directly (model.policy.log_std.exp(), the same
                # thing evaluate.py prints at startup) rather than just
                # inferring from train/entropy_loss, and it came back
                # [2.21, 2.79]. Action range is [-1, 1] (width 2) -- a
                # Gaussian with std over 2 has the large majority of its
                # mass outside that range, meaning sampled actions are
                # mostly landing on the hard clip in step()'s
                # np.clip(action, -1.0, 1.0) regardless of what the mean
                # says. In practice, stochastic actions had become close to
                # a coin flip between the two extremes each step -- not
                # exploration around a promising strategy, just noise
                # incapable of producing the smooth, phase-locked knee/lean
                # rhythm a real pump needs. Explains why entropy_loss kept
                # falling (-4.09 -> -4.68, still dropping, far past even the
                # 0.01 run's -1.62 low) while BOTH evaluate.py sides stayed
                # collapsed to the same tight, low cluster (2.14 deg spread)
                # instead of improving -- and why train/clip_fraction and
                # approx_kl both dropped too (2.3-6%, 0.003-0.006, lower
                # than either the 0.01 or the pre-fix runs): with actions
                # this saturated, there's little clean advantage signal left
                # for the policy gradient to learn from, so the mean barely
                # updates either. No PPO-health crisis alongside this
                # (explained_variance, value_loss both stayed fine) -- this
                # was specifically the entropy bonus overshooting into a
                # degenerate, saturated-noise regime, not a trust-region or
                # value-function problem.
                #
                # 0.01 -> 0.03 turned out to be a much more sensitive jump
                # than the OVERSHOOT_WEIGHT/WALL_TOUCH_PENALTY precedent
                # that motivated it -- going from clearly too weak to
                # clearly counterproductive without an obviously good middle
                # ground yet visible. 0.015 is a much smaller step off the
                # original 0.01 (not another 3x-style jump) given that
                # sensitivity. Watch model.policy.log_std.exp() directly
                # this time, not just train/entropy_loss -- something in
                # the rough 0.3-0.8 std range (meaningful per-step
                # variation without saturating the [-1, 1] clip) is the
                # target, not just "entropy_loss lower than before."
                #
                # PREVIOUS reasoning for the 0.01 -> 0.03 raise (context for
                # why that was tried, kept for history): RAISED, 0.01 ->
                # below) worked initially -- entropy_loss fell sharply
                # (2.60 -> -1.62) and both evaluate.py sides improved
                # substantially, stochastic reaching a clean 5/5 real-height
                # episodes at one point -- but the effect didn't hold.
                # entropy_loss reversed and climbed back steadily over the
                # following rounds: -1.62 -> -1.00 -> -0.33, roughly two
                # thirds of the way back toward the pre-fix collapsed range
                # (+2.6) and still climbing at the last check, not
                # leveling off. rollout/ep_rew_mean fell in step (53.6-55.8,
                # well under the 190-220 range training produced during the
                # good stretch), and evaluate.py's most recent round showed
                # BOTH deterministic AND stochastic collapsed to the same
                # tight, low cluster (5.3-13.4 deg / 6.7-12.7 deg) -- worse
                # than the first collapse this project saw, where stochastic
                # kept finding real height even when deterministic didn't.
                # No PPO-health crisis alongside any of this (approx_kl,
                # clip_fraction, explained_variance all stayed healthy
                # throughout) -- this is specifically an exploration
                # problem, and 0.01 wasn't enough to hold the line against
                # whatever keeps pulling the policy back toward a low-
                # effort, easily-repeatable strategy. Rather than another
                # small nudge, jumping 3x for a clean read -- same reasoning
                # this project has used before when a first, moderate value
                # proved insufficient (OVERSHOOT_WEIGHT: 0.02 -> 0.08 -> 0.3;
                # WALL_TOUCH_PENALTY: 20 -> 200) rather than incrementing by
                # small amounts and spending many more rounds finding out
                # each step was still too gentle. 0.03 is still a
                # commonly-used value in continuous-action PPO (not an
                # extreme escalation) -- reasoned, not tuned to this
                # environment specifically. Watch train/entropy_loss the
                # same way as before: if it holds well below the pre-fix
                # range this time instead of climbing back, that's the
                # signal this was enough; if it resumes the same climb, the
                # problem is more likely the mechanism (a flat ent_coef
                # bonus alone) than the magnitude.
                #
                # ORIGINAL reasoning for adding this constant at all (still
                # accurate, only the magnitude changed): was never set --
                # SB3's PPO defaults ent_coef to 0.0,
                # meaning there has been NO explicit pressure keeping
                # exploration alive for this entire project's training
                # history. Added after tb_logs/PPO_1's train/entropy_loss
                # (SB3 logs the negative of policy entropy under this name,
                # so a RISING value means entropy is FALLING) showed a real,
                # sustained, still-accelerating climb over roughly the last
                # 17M steps: ~1.0 at step 61.1M -> 1.83 (64.5M) -> 2.16
                # (67.9M) -> 2.34 (71.2M) -> 2.56 (78.0M, latest at the time
                # this was added), with no sign of leveling off across that
                # whole stretch. Ties directly to a pattern showing up in
                # evaluate.py rounds over the same window: deterministic
                # spread across episodes narrowing round over round (51 ->
                # 99 -> 34 -> 15 deg) while the MEAN stayed low (16-44 deg,
                # nowhere near TARGET_ANGLE_DEG) -- not convergence toward a
                # good answer, convergence toward a low-effort, easily-
                # repeatable ~15-20 deg oscillation, with the action
                # distribution getting too confident/narrow to keep
                # exploring toward the harder-to-discover, much more
                # rewarding real-height strategy that stochastic sampling
                # (still meaningfully wider than deterministic, e.g. 58-71
                # deg in the round that prompted this) shows is still
                # findable. With nothing opposing it, this collapse was
                # plausibly always going to happen eventually and would
                # likely have kept getting worse with more training under
                # the old settings, not self-corrected. 0.01 is a common
                # starting value for continuous-action PPO's entropy bonus
                # -- reasoned, not tuned to this specific environment yet;
                # watch train/entropy_loss over the next several rounds to
                # see whether it actually arrests the climb, and whether
                # the deterministic spread starts finding real height again
                # rather than just staying narrow at a low value.
MODEL_PATH = "swing_policy"
VECNORM_PATH = "vec_normalize.pkl"
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_DIR = "best_model"
N_EVAL_EPISODES_FOR_BEST = 5  # was 3. Raised to match evaluate.py's own
                                # N_EVAL_EPISODES: the "is this a new best"
                                # gate that decides whether to overwrite
                                # best_model.zip was being decided on the SAME
                                # small-sample noise problem evaluate.py's
                                # 5-episode averaging exists to fix -- but
                                # that fix only ever applied to the reporting
                                # tool, not to this actual gate. A collapsing/
                                # degraded policy getting lucky on 3 episodes
                                # was enough to overwrite a genuinely better
                                # best_model.zip with a worse one, which is
                                # consistent with real evaluate_log.csv data
                                # showing best_model's own deterministic score
                                # crater across several consecutive
                                # auto_train.py rounds despite the "can only
                                # be beaten, not overwritten" seeding fix
                                # already in place below.
EVAL_LOG_PATH = "evaluate_log.csv"
BEST_MODEL_PATH = os.path.join(BEST_MODEL_DIR, "best_model")
BEST_VECNORM_PATH = os.path.join(BEST_MODEL_DIR, "best_vec_normalize.pkl")


RESUME_OVERSHOOT_MAX = 400.0  # NEW (README 5.9y) -- see pick_resume_source's
                               # own docstring below for the incident that
                               # motivated this. Deliberately matches auto_
                               # train.py's ERRATIC_OVERSHOOT_MAX exactly
                               # (same bar: "this round's own overshoot is
                               # bad enough to call unambiguously broken,
                               # not just noisy"), NOT imported from there --
                               # train.py is invoked as a subprocess BY auto_
                               # train.py, so importing back the other way
                               # would be a backwards dependency; this
                               # project already keeps independent per-file
                               # copies of shared-in-spirit constants (e.g.
                               # auto_train.py's own TARGET_ANGLE_DEG isn't
                               # imported from swing_env.py either), so a
                               # second local copy here matches existing
                               # precedent rather than introducing a new
                               # pattern. If ERRATIC_OVERSHOOT_MAX ever
                               # changes there, update this to match by
                               # hand.


def pick_resume_source():
    """(model_path, vecnorm_path, reason) to resume training FROM -- NOT
    where this run will eventually SAVE to (that's always MODEL_PATH/
    VECNORM_PATH ("swing_policy"/"vec_normalize.pkl"), unchanged by this).

    NEW (README 5.9k). Used to unconditionally resume from swing_policy --
    whatever state training happened to end on, regardless of whether it
    was any good. Direct checkpoint investigation found real, repeated
    cases where swing_policy ended a round meaningfully worse than a
    checkpoint EvalCallback had already found earlier that SAME round --
    including one sharp transition (clean bootstrapping across 5
    consecutive checkpoints spanning 665,000 steps, then real bootstrap
    failures starting in the very next 50,000-step checkpoint interval and
    persisting through the round's end) that showed NOTHING unusual in any
    PPO training diagnostic checked at that exact point (value_loss, std,
    explained_variance, approx_kl, clip_fraction, policy_gradient_loss,
    loss -- all completely ordinary). This failure mode is invisible to
    every standard health check; unconditionally resuming from
    swing_policy means every subsequent round just keeps building on top
    of whatever state training happened to end on, good or silently
    broken.

    Criterion 1 (original, 5.9k): compares evaluate_log.csv's MOST RECENT
    logged row for "best_model/best_model" against the most recent for
    "swing_policy" (both written every round since 5.9i). If swing_policy
    shows ANY bootstrap failures (det or sto) while best_model shows none,
    resume from best_model instead. Deliberately reuses
    BOOTSTRAP_FAIL_MAX_ANGLE_DEG's already-established clean gap
    (evaluate.py: failed episodes topped out at 6.7-19.5 deg, every
    success reached 71+ deg, nothing in between) rather than inventing a
    new numeric threshold -- "any failures at all, when the alternative
    has none" is already a meaningful signal at that separation, not
    noise.

    Criterion 2, NEW (README 5.9y): compares the same two rows' overshoot
    penalty (max of det/sto on each side). If swing_policy's overshoot is
    past RESUME_OVERSHOOT_MAX while best_model's isn't, resume from
    best_model instead -- same structure as criterion 1, different failure
    mode. MOTIVATING INCIDENT: the very first round trained under domain
    randomization (5.9w) + observation noise (5.9x) produced a swing_policy
    with ZERO bootstrap failures (so criterion 1 alone would have called it
    "at least as reliable as best_model" and resumed from it) but det_
    overshoot_penalty of 12,089.63 and sto_overshoot_penalty of 7,305.06 --
    peak angles 79-81 deg, well past target, with TIGHT spread (3.87 deg)
    and zero bootstrap failures, i.e. a policy that pumps hard and
    consistently, just too high, not one that's unstable or failing to
    bootstrap. That's a real, recognizable failure mode this project has
    seen before (the OVERSHOOT_WEIGHT 0.3->0.6->1.2 saga, swing_env.py) --
    just invisible to a check that only ever looked at bootstrap-failure
    counts. Without this criterion, the very next round would have resumed
    training from that overshooting state instead of the healthy
    best_model (which itself scored BETTER than before this round: 11,372
    reward vs. the prior ~9,690-9,910 range, 0 bootstrap failures, faster
    ramp-up) -- compounding a problem this check exists specifically to
    catch, and losing any clean read on whether it was a one-round
    adaptation blip or something needing a real reward-shape response.

    Both criteria independently gate the SAME fallback (resume from
    best_model) -- either one firing is sufficient, matching each other's
    logic rather than requiring both simultaneously, since they're
    detecting two different, independent ways a round-end state can be
    quietly worse than the safe, known-good checkpoint.

    Falls back to the historical default (resume from swing_policy) if the
    log doesn't have both rows yet, predates the bootstrap-fail-count
    columns, or the chosen files don't actually exist on disk -- same
    fail-open-to-old-behavior discipline as auto_train.py's read_log()
    pairing logic."""
    default = (MODEL_PATH, VECNORM_PATH, "default (no comparison data available)")
    try:
        with open(EVAL_LOG_PATH, newline="") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        return default

    def last_row(tag):
        for row in reversed(rows):
            if row.get("model_path") == tag:
                return row
        return None

    best_row = last_row("best_model/best_model")
    latest_row = last_row("swing_policy")
    if best_row is None or latest_row is None:
        return default
    if "det_bootstrap_fail_count" not in best_row or "det_bootstrap_fail_count" not in latest_row:
        return default  # log predates the bootstrap-fail-count columns (5.9h)

    def total_fails(row):
        return (int(row.get("det_bootstrap_fail_count") or 0)
                + int(row.get("sto_bootstrap_fail_count") or 0))

    def worst_overshoot(row):
        return max(float(row.get("det_overshoot_penalty") or 0.0),
                    float(row.get("sto_overshoot_penalty") or 0.0))

    best_fails = total_fails(best_row)
    latest_fails = total_fails(latest_row)
    best_overshoot = worst_overshoot(best_row)
    latest_overshoot = worst_overshoot(latest_row)

    bootstrap_regression = latest_fails > 0 and best_fails == 0
    overshoot_regression = (latest_overshoot > RESUME_OVERSHOOT_MAX
                             and best_overshoot <= RESUME_OVERSHOOT_MAX)

    if ((bootstrap_regression or overshoot_regression)
            and os.path.exists(BEST_MODEL_PATH + ".zip")
            and os.path.exists(BEST_VECNORM_PATH)):
        reasons = []
        if bootstrap_regression:
            reasons.append(f"{latest_fails} bootstrap failure(s) (best_model: 0)")
        if overshoot_regression:
            reasons.append(
                f"overshoot penalty {latest_overshoot:.1f} (best_model: "
                f"{best_overshoot:.1f}, max allowed {RESUME_OVERSHOOT_MAX:.0f})"
            )
        return (
            BEST_MODEL_PATH, BEST_VECNORM_PATH,
            f"swing_policy's most recent evaluation showed " + " and ".join(reasons) +
            f" -- resuming from best_model instead to avoid building on a "
            f"state known to be broken"
        )
    return (MODEL_PATH, VECNORM_PATH,
            "swing_policy looked at least as reliable as best_model")


def main():
    # SubprocVecEnv (what make_vec_env uses by default for n_envs > 1) runs
    # each environment in its own process. On macOS this requires the
    # multiprocessing entry point to be guarded behind __main__, which is
    # why everything below lives inside main() rather than at module level
    # -- the same category of platform gotcha as the mjpython requirement
    # for the viewer earlier.
    # domain_randomization=True (README 5.9w) + observation_noise=True
    # (README 5.9x) -- ONLY on this, the actual rollout-collection env,
    # deliberately NOT on eval_env or prior_best_env below, for the same
    # reason in both cases: training needs exposure to the sampled physical-
    # parameter range AND the gyro-drift sensing gap to generalize to both;
    # "best model" selection and resume-source comparisons need to keep
    # meaning performance against the real nominal build with exact angle
    # feedback, same fixed/exact conditions evaluate.py's whole gating stack
    # (auto_train.py) has always assumed -- changing those too would
    # silently invalidate every threshold calibrated against the existing
    # evaluate_log.csv history. (Whether eval should EVENTUALLY also grow an
    # optional noisy/randomized mode to directly measure robustness is an
    # open question, same as 5.9w's -- deliberately not resolved here.)
    vec_env = make_vec_env(SwingEnv, n_envs=N_ENVS,
                            env_kwargs=dict(domain_randomization=DOMAIN_RANDOMIZATION_ENABLED,
                                             observation_noise=OBSERVATION_NOISE_ENABLED))

    resuming = os.path.exists(MODEL_PATH + ".zip") and os.path.exists(VECNORM_PATH)
    if resuming:
        print(f"Found existing {MODEL_PATH}.zip -- continuing training, not starting over.")
        # NEW (README 5.9k): resume FROM best_model instead of swing_policy
        # when the two have diverged -- see pick_resume_source()'s
        # docstring for why. resume_model_path/resume_vecnorm_path are used
        # below ONLY to decide what to load; MODEL_PATH/VECNORM_PATH remain
        # unchanged as this run's SAVE destination at the end.
        resume_model_path, resume_vecnorm_path, resume_reason = pick_resume_source()
        print(f"Resuming from {resume_model_path}.zip ({resume_reason}).")
        vec_env = VecNormalize.load(resume_vecnorm_path, vec_env)
        # BUG FIX: a saved VecNormalize carries whatever norm_reward it had
        # when it was saved -- older checkpoints have norm_reward=True baked
        # in. Force it to match the current setting below (False) rather
        # than silently inheriting the old one, so resuming an older run
        # actually gets the fix, not a stale mix of old and new behavior.
        vec_env.norm_reward = False
        # BUG FIX: PPO.load() restores hyperparameters (gamma, target_kl,
        # clip_range, etc.) from whatever was baked into the saved .zip at
        # the time it was written -- passing NO override here meant every
        # resumed run silently kept using the OLD gamma (whatever the
        # loaded checkpoint was originally trained with), not the current
        # GAMMA constant above. Since this script is almost always
        # "resuming" (checkpoint files exist after the very first run),
        # this means the gamma change made earlier likely never actually
        # took effect in any of the training that followed it -- the
        # "gamma didn't fix the recurring collapse" conclusion was
        # premature, because gamma was probably never really changed.
        # Explicit overrides here make sure this script's current constants
        # actually apply on every run, resumed or not. ent_coef added
        # alongside gamma/target_kl for the exact same reason -- this is
        # the code path that runs almost every time (checkpoint files exist
        # after the very first run), so ENT_COEF has to be passed here
        # explicitly or it silently never takes effect, same bug class as
        # gamma's above.
        model = PPO.load(resume_model_path, env=vec_env, gamma=GAMMA, target_kl=TARGET_KL, ent_coef=ENT_COEF)
    else:
        # PPO is sensitive to the relative scale of OBSERVATIONS -- our six
        # observations span very different ranges (radians, unbounded
        # angular velocity, a +-0.1m slide position) -- so norm_obs stays
        # on. This means the saved model is only meaningful together with
        # the saved normalization statistics -- see watch_policy.py for how
        # both get loaded back together.
        #
        # norm_reward=False: was True. Checkpoint scans across the full
        # training history showed the deterministic policy becoming
        # increasingly BIMODAL over time -- long stretches of good, stable
        # performance early on, giving way to a pattern of isolated good
        # checkpoints surrounded by near-total collapse, with the good
        # moments getting rarer as training continued (eventually 3.4M+
        # steps with zero good checkpoints at all). Reward normalization
        # divides by a running estimate of reward VARIANCE -- and variance
        # is exactly what spikes as a policy becomes more inconsistent. That
        # creates a plausible feedback loop: more bimodal -> higher reward
        # variance -> normalizer shrinks the training signal further ->
        # noisier gradient information right when consistency matters most
        # -> inconsistency reinforces itself instead of correcting. This
        # would explain why instability got WORSE over training time rather
        # than settling down. Safe to disable: the raw per-step height
        # reward here is already small and bounded (roughly -0.15 to ~1.23,
        # given arm_length, TARGET_ANGLE_DEG, and HEIGHT_OVERSHOOT_FLOOR in
        # swing_env.py -- was ~0 to ~1.65 before TARGET_ANGLE_DEG's 80 -> 67.5
        # redefinition and HEIGHT_OVERSHOOT_FLOOR's negative floor, same
        # conclusion either way) -- nowhere near the scale that reward
        # normalization exists to handle, unlike the observations.
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
        model = PPO(
            "MlpPolicy", vec_env, gamma=GAMMA, target_kl=TARGET_KL,
            ent_coef=ENT_COEF, verbose=1, tensorboard_log="./tb_logs",
        )

    # save_vecnormalize=True: periodic checkpoints were only ever saving
    # policy weights, not the matching VecNormalize running stats -- unlike
    # best_model.zip, which gets a paired best_vec_normalize.pkl via
    # SaveVecNormalizeOnBest below. That gap meant that after a training
    # collapse, there was no way to accurately re-evaluate an old checkpoint
    # against the normalization statistics that were actually current when
    # it was saved -- only whichever vec_normalize.pkl happens to exist now,
    # which may have drifted a lot since (especially post-collapse, if a
    # stalled policy fed it a very different, narrower distribution of
    # observations for a long stretch). Every checkpoint saved from now on
    # gets its own matching swing_ppo_vecnormalize_<step>_steps.pkl, so a
    # rollback like this doesn't require guessing again.
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix="swing_ppo",
        save_vecnormalize=True,
    )

    # Separate single-env eval environment, normalized but frozen (training=
    # False, norm_reward=False) -- EvalCallback syncs its VecNormalize obs
    # stats from the training env automatically before each evaluation, as
    # long as both are VecNormalize-wrapped.
    # domain_randomization deliberately left at its default (off, README
    # 5.9w) -- "best" should keep meaning "best against the real nominal
    # build", not "best on whatever physics this particular eval sample
    # happened to draw".
    eval_env = make_vec_env(SwingEnv, n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    save_vecnorm_on_best = SaveVecNormalizeOnBest(
        vec_env, os.path.join(BEST_MODEL_DIR, "best_vec_normalize.pkl")
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path="eval_logs",
        eval_freq=max(20_000 // N_ENVS, 1),
        n_eval_episodes=N_EVAL_EPISODES_FOR_BEST,
        deterministic=True,
        callback_on_new_best=save_vecnorm_on_best,
    )

    # BUG FIX: EvalCallback's "best score so far" tracking (best_mean_reward)
    # lives only in memory for this one process. Since this script gets
    # re-run fresh every time you resume, a brand-new EvalCallback starts
    # that tracking at -inf every single run -- meaning the FIRST evaluation
    # in a new run, even a mediocre one, looks like "a new best" relative to
    # that blank slate, and silently overwrites a genuinely better
    # best_model.zip left over from a previous run. This defeats the entire
    # point of the safety net. If a best_model already exists on disk,
    # evaluate it now and seed the callback's baseline with its real score,
    # so it can only be overwritten by something that's actually better.
    best_model_zip = os.path.join(BEST_MODEL_DIR, "best_model.zip")
    best_vecnorm_path = os.path.join(BEST_MODEL_DIR, "best_vec_normalize.pkl")
    if os.path.exists(best_model_zip) and os.path.exists(best_vecnorm_path):
        prior_best_env = make_vec_env(SwingEnv, n_envs=1)
        prior_best_env = VecNormalize.load(best_vecnorm_path, prior_best_env)
        prior_best_env.training = False
        prior_best_env.norm_reward = False
        prior_best_model = PPO.load(best_model_zip)
        prior_best_reward, _ = evaluate_policy(
            prior_best_model, prior_best_env,
            n_eval_episodes=N_EVAL_EPISODES_FOR_BEST, deterministic=True
        )
        eval_callback.best_mean_reward = prior_best_reward
        print(f"Seeded best-so-far tracking from existing best_model.zip: "
              f"{prior_best_reward:.2f} (can only be overwritten by something better)")

    log_std_clip_callback = LogStdClipCallback(LOG_STD_MIN_STD, LOG_STD_MAX_STD)
    callback = CallbackList([checkpoint_callback, eval_callback, log_std_clip_callback])

    model.learn(
        total_timesteps=TIMESTEPS_THIS_RUN,
        callback=callback,
        progress_bar=True,  # needs `pip install tqdm rich` if not already present
        reset_num_timesteps=not resuming,
    )

    model.save(MODEL_PATH)
    vec_env.save(VECNORM_PATH)
    print(f"Saved policy to {MODEL_PATH}.zip and normalization stats to {VECNORM_PATH}")
    print(f"Best deterministic checkpoint (if any improvement found this run) saved to "
          f"{BEST_MODEL_DIR}/best_model.zip + {BEST_MODEL_DIR}/best_vec_normalize.pkl")
    print("Watch it with: mjpython watch_policy.py")


if __name__ == "__main__":
    main()

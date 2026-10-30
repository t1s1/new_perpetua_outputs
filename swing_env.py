"""
Gymnasium wrapper around swing_robot.xml.

Observation (6): swing angle, swing angular velocity, lean angle, lean
angular velocity, knee angle, knee angular velocity -- i.e. the full
state (position + velocity) of every degree of freedom in the model.
This is what a real deployment would read from the IMU (swing angle/
velocity) and servo feedback (lean, knee position/velocity).

Action (2): [lean_ctrl, knee_ctrl], each in [-1, 1], written directly to
the two motors -- lean_hinge and knee_hinge (a rotating leg-mass joint,
replacing the earlier sliding counterweight -- see swing_robot.xml and
README.md for the diagram-driven redesign). No hand-coded pumping logic
anywhere in this file -- the policy has to discover it.

NOTE ON THIS BEING A FRESH MODEL: swapping cg_slide (prismatic) for
knee_hinge (rotational) changes what action[1] physically does -- a
policy trained against the old slide joint has learned nothing
transferable to this joint's dynamics. Training resumes from scratch
against this model, not from the locked-in phase-1 checkpoint.

NOTE ON ASYMMETRIC LEAN: lean_hinge's range is -5/+80 deg, not
symmetric -- refined from an initial -25/+55 visual read of a reference
diagram to the user's own reasoned estimate of real hip range of motion
(negative = forward lean, positive = backward recline), matching real
swing-pumping technique. This means action=0 no longer maps to "hinge at
0 rad"; _scale_to_ctrl() maps the normalized [-1,1] action to the
actuator's real [lo, hi] range via a proper affine transform (not a
symmetric multiply, which would have been wrong here), and the
observation space specifies low/high separately per dimension instead of
mirroring a single array.

NOTE ON THE SECOND PHYSICAL REDESIGN (rope/seat/torso/leg, from real
hardware dimensions): another fresh-model event, same "learned nothing
transferable" caveat as the knee swap above -- archive_old_model.py
moved the prior lineage (111.6M steps) aside before this promotion.
Structurally different from every prior version of this model: the rope
now attaches RIGIDLY (no joint) to a combined seat/upper-leg body at an
OFFSET point along its length -- 0.06m from the hip (lean_hinge), 0.14m
from the knee (knee_hinge) -- rather than the rope attaching directly at
the hip the way it used to. self.arm_length (used in the reward's height
math below) is now read directly off the compiled model instead of
hardcoded, specifically to avoid this exact class of sync bug recurring
a third time. See swing_robot.xml's own top-of-file comment for the full
body tree and each dimension's provenance. Actuator forcerange values on
lean_motor/knee_motor are grounded in real candidate servo torque specs
(Feetech STS3215 C044 variant, synchronized pairs/triples) discussed
with the user, not arbitrary -- see swing_robot.xml's actuator comment.

Reward: height gained by the bottom of the swing arm relative to hanging
straight down, L * (1 - cos(swing_angle)), capped at TARGET_ANGLE_DEG (see
below). Episodes start with a small randomized seed kick rather than
exact dead rest -- see reset() for why: an earlier version started at
exact rest with no kick at all, hoping the policy would learn to
bootstrap its own symmetry-breaking motion, and instead it learned to use
its own per-step exploration noise as a substitute for that (confirmed by
a huge gap between deterministic and stochastic evaluation reward in
evaluate.py), never converging to a real deterministic strategy.

episode_seconds defaults to 360, raised from 240 alongside the second
physical redesign (rope/seat/torso/leg from real hardware dimensions).
Direct evidence this needed to move again, not just left at 240 out of
inertia: the same hand-coded reference control law used throughout this
project's physical-parameter investigations, run against the new geometry
with the placeholder actuator torque values now in swing_robot.xml,
doesn't reach TARGET_ANGLE_DEG (67.5) until roughly t=284s (interpolated
between a measured 62.5deg at t=257s and 78.2deg at t=342s) -- already
PAST the previous 240s cutoff. This is the same "episode truncates before
patience pays off" bug this project already hit once (120 -> 240, below,
kept for history), recurring for the same underlying reason: a geometry
change shifted the pumping timeline and the episode length hadn't been
re-checked against it. 360s gives real headroom past the ~284s
target-reach point, so a policy that finds an equally good or better
strategy has room to actually bank reward for holding there, not just
touching it once as time runs out -- same reasoning as the original
120->240 fix, just re-applied with fresh numbers for the new geometry.

ORIGINAL reasoning for 240 (kept for history, same mechanism, different
numbers): raised from 120 after widening the lean range to +-45deg (an
earlier, now-superseded geometry) shifted manual_pump.py's own
steady-state timeline too (~56deg at the 120s mark, ~78deg by 180s, ~93deg
by 240s, settling near 103deg around 300s) -- a trained policy plateaued
right around 60deg for several runs in a row, almost exactly manual_pump's
value at the old 120s cutoff, the first time this exact bug appeared.

ACTION_RATE_PENALTY: a small cost proportional to how much the action
changed since the previous step. Nothing about height-only reward cares
whether the action sequence is smooth or jittery -- a policy that jitters
its lean back and forth is exactly as good as one that holds steady, as
far as cumulative height is concerned. This penalty makes jitter costly
while leaving deliberate, larger changes at an actual phase switch mostly
unaffected, since those happen once per switch rather than repeatedly.
Tune the weight if needed: too high and it'll start suppressing real
phase-switch transitions along with the jitter; too low and the jitter
persists.

TARGET_ANGLE_DEG / OVERSHOOT_WEIGHT: height reward stops increasing past
the target angle (no benefit to going higher), and a gentle quadratic
penalty kicks in past TARGET_ANGLE_DEG + OVERSHOOT_MARGIN_DEG (some slack
before the penalty starts, so ordinary variation around the target isn't
punished). This matters because nothing about the rigid-rod pendulum in
swing_robot.xml discourages high angles on its own -- a real chain swing
goes slack and behaves unpredictably as it approaches horizontal, which
is the actual physical reason not to swing "too high", but our model
doesn't simulate that at all, so the reward has to encode the limit
explicitly rather than relying on the physics to self-regulate.

DISTURBANCE_PROB_PER_STEP / DISTURBANCE_MAX_KICK: a mid-episode disturbance
(random angular-velocity kick to the swing itself), standing in for a
real-world gust hitting the swing while already moving -- a different event
from reset()'s seed kick, which only ever perturbs the starting condition.
NOW ON (was off/0 while a working baseline policy didn't exist yet -- same
staged-rollout reasoning as ACTION_RATE_WEIGHT: get a working, consistent
policy first (unperturbed), THEN turn this on and resume training that
policy so it also learns to reject disturbances, not just tolerate their
absence. A locked-in baseline checkpoint now exists (see README.md), so
this has been turned on and training is resuming against it.
"""
import re

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

ACTION_RATE_WEIGHT = 1.0   # SETTLED HERE, 1.33 -> 1.0 (README 5.9s). 1.33
                           # (below) was the next bisection point but was
                           # never actually trained -- the user called the
                           # search off first: "good enough is good enough."
                           # 1.0 was the last value with a fully clean
                           # track record (4/4 rounds, 5.9m/n) and had
                           # already independently satisfied the actual
                           # goal (watch_policy.py looked acceptably
                           # human-plausible, 5.9n's "no changes needed").
                           # Live swing_policy.zip/vec_normalize.pkl/
                           # best_model/ were already sitting at the 1.0
                           # backup (never overwritten, since no training
                           # ran between the 1.78->1.33 restore and this
                           # decision) -- verified byte-identical via
                           # md5sum before making this the settled value,
                           # so no further file changes were needed, only
                           # this constant.
                           #
                           # PREVIOUS reasoning below (1.78 -> 1.33, kept
                           # for history -- this bisection point was never
                           # actually run):
                           # BISECTED AGAIN, 1.78 -> 1.33 (README 5.9q).
                           # 5 rounds at 1.78: 2 clean (H, J), 1 severe
                           # collapse (I: 14/15 det, 12/15 sto bootstrap
                           # failures), 2 moderate (K, L: only 2/15 det
                           # fails each, but spread blew out to 56-62 deg).
                           # Notably better than 3.16 on two counts --
                           # best_model's own stochastic reward stayed
                           # STABLE across all 5 rounds (8487/7634/7887/
                           # 7803/7863, no decline), unlike 3.16's clean
                           # -30% slide, and no round was as severe as
                           # 3.16's repeated near-total collapses. But
                           # swing_policy still showed real trouble in 3 of
                           # 5 rounds, not just 1 -- clearly worse than
                           # 1.0's fully clean 4/4 record. Verified via
                           # pick_resume_source() that round L's moderate
                           # instability was ALSO a fresh start (from a
                           # clean best_model, after round K's mild fail
                           # triggered the resume-from-best_model path) --
                           # so this recurring pattern isn't just residual
                           # damage from round I's collapse, it's a real,
                           # if inconsistent, property of this weight.
                           #
                           # Reading: 1.78 sits in a genuine middle zone --
                           # not broadly reward-damaging like 3.16, but not
                           # reliably clean like 1.0 either. 1.33 =
                           # sqrt(1.0 * 1.78), same geometric-bisection
                           # logic as 5.9o/5.9p, against the narrowed
                           # bounds: 1.0 confirmed clean (4/4 rounds, no
                           # exceptions so far), 1.78 confirmed unreliable
                           # (3/5 rounds with real swing_policy trouble,
                           # ranging mild to severe). Restored swing_policy
                           # .zip/vec_normalize.pkl/best_model/ from
                           # action_rate_weight_1.0_confirmed_good_backup_
                           # 20260814T220735/ again (verified byte-
                           # identical via md5sum) -- same known-good
                           # anchor point used for every bisection so far.
                           #
                           # Given how narrow the search window now is
                           # (1.0-1.78), and that 1.0 was already
                           # confirmed to satisfy the user's actual goal
                           # (motion looked acceptably human-plausible,
                           # 5.9n), this bisection is now mostly about
                           # PRECISELY characterizing the limit for
                           # external justification rather than searching
                           # for meaningfully more smoothing headroom --
                           # worth keeping in mind if further rounds start
                           # feeling like diminishing returns.
                           #
                           # PREVIOUS reasoning below (3.16 -> 1.78, kept
                           # for history):
                           # BISECTED AGAIN, 3.16 -> 1.78 (README 5.9p).
                           # 3.16 initially looked plausibly safe after ONE
                           # round (round C: fully clean), then one
                           # collapse (round D: swing_policy 13/15 det,
                           # 13/15 sto bootstrap failures) that looked like
                           # it could be the pre-existing, already-
                           # documented capture-basin bug (5.9k) recurring
                           # rather than new evidence against 3.16 --
                           # best_model's own stochastic reward was still
                           # healthy (6921) that round, unlike the clearly
                           # broad damage seen at weight=10.0. That reading
                           # was WRONG, or at least incomplete -- 3 more
                           # rounds (E, F, G) settled it. Verified via
                           # pick_resume_source() directly (not assumed)
                           # that EVERY one of those 3 rounds started fresh
                           # from a CLEAN best_model, not compounding on a
                           # broken checkpoint -- and swing_policy collapsed
                           # again anyway in 2 of the 3 (round E: 14/15 det,
                           # 13/15 sto; round F: 13/15 det, 10/15 sto),
                           # with round G a partial-but-not-clean recovery
                           # (3/15 det, 0/15 sto fails, but spread blew out
                           # to 53 deg). A fresh restart from a known-good
                           # state repeatedly failing again within one
                           # round is a materially different (and worse)
                           # pattern than 5.9k's original one-off,
                           # eventually-persisting collapse -- this looks
                           # actively weight-driven, not coincidental
                           # recurrence of the old bug. Second, independent
                           # line of evidence pointing the same way:
                           # best_model's own stochastic reward declined
                           # steadily across all 5 rounds at this weight
                           # (7053 -> 6921 -> 6250 -> 5974 -> 4906, ~-30%
                           # total) even though it never failed a hard
                           # threshold -- a gradual erosion signature, not
                           # present at all in the original 5.9k incident.
                           #
                           # 1.78 = sqrt(1.0 * 3.16), same geometric-
                           # bisection reasoning as 5.9o, applied to the
                           # NEW confirmed bounds: 1.0 still confirmed good
                           # (4 clean rounds, 5.9m/n), 3.16 now confirmed
                           # bad (3.16's own README entry, 5.9o, corrected
                           # by this one). Restored swing_policy.zip/
                           # vec_normalize.pkl/best_model/ from
                           # action_rate_weight_1.0_confirmed_good_backup_
                           # 20260814T220735/ AGAIN (verified byte-
                           # identical via md5sum) -- same backup as
                           # before, since it remains the last unambiguous
                           # known-good state; not worth trusting the more
                           # recent but weight-3.16-exposed best_model
                           # given its declining reward trend.
                           #
                           # DECISION CRITERIA, revised: a single clean
                           # round is NOT sufficient evidence 1.78 is safe
                           # (3.16 looked fine after 1 round too) --
                           # requires multiple rounds, ideally with at
                           # least one verified pick_resume_source() fresh-
                           # start test, before calling it either way. Also
                           # watch best_model's own stochastic reward trend
                           # across rounds, not just bootstrap-fail counts
                           # -- that gradual decline was real signal 3.16's
                           # first-round read missed.
                           #
                           # PREVIOUS reasoning below (10.0 -> 3.16, kept
                           # for history -- includes the now-superseded
                           # "might be the old bug" read from round D
                           # alone):
                           # BISECTED, 10.0 -> 3.16 (README 5.9o), after 10.0
                           # was confirmed too strong -- not a suspicion,
                           # directly measured across 2 consecutive real
                           # training rounds: swing_policy (round-end
                           # state) bootstrap failures went 12/15 det,
                           # 14/15 sto (round 1) -> 14/15 det, 15/15 sto,
                           # EVERY stochastic episode (round 2). Matches
                           # the exact "give up and stop moving" collapse
                           # signature this constant's own staged-rollout
                           # history warned about. best_model itself held
                           # on angle/spread/overshoot both rounds (0
                           # bootstrap fails), but its stochastic reward
                           # went sharply negative both times (-1505, then
                           # -1583, worsening) -- the action-rate penalty
                           # overwhelming reward under exploration noise
                           # even there. auto_train.py's own cliff check
                           # independently confirmed this was a real,
                           # sudden collapse, not noise (round 1's
                           # combined angle: 70.0 -> 14.2 deg in one
                           # round, tripping CLIFF_DROP_FRACTION=0.5 by a
                           # wide margin).
                           #
                           # 3.16 = sqrt(1.0 * 10.0), the GEOMETRIC mean of
                           # the confirmed-good (1.0, README 5.9m/n) and
                           # confirmed-bad (10.0, above) bounds -- not the
                           # arithmetic mean (5.5). Geometric is the
                           # correct bisection here for two reasons: (1)
                           # every step change in this constant's history
                           # has been multiplicative (0.02->1.0 was 50x,
                           # 1.0->10.0 was 10x), so a search across this
                           # parameter is naturally log-scaled, and (2) the
                           # penalty itself scales with the SQUARE of the
                           # action delta, so cost at the arithmetic
                           # midpoint (5.5) would sit far closer to 10.0's
                           # failure regime than to 1.0's confirmed-safe
                           # one in actual effect, not equidistant the way
                           # a linear reading of the two numbers suggests.
                           # Restored swing_policy.zip/vec_normalize.pkl/
                           # best_model/ from
                           # action_rate_weight_1.0_confirmed_good_backup_
                           # 20260814T220735/ before this change (verified
                           # byte-identical via md5sum) -- training resumes
                           # from the last confirmed-good state, not
                           # compounding on either broken round. The two
                           # failed rounds' checkpoints (under checkpoints/,
                           # ~31.1M-31.4M steps) and evaluate_log.csv rows
                           # were deliberately left untouched (not
                           # archived/deleted) -- kept as the direct
                           # evidence backing this decision.
                           #
                           # DECISION CRITERIA for the next result (same
                           # bootstrap-fail-count basis as
                           # BOOTSTRAP_FAIL_MAX_ANGLE_DEG, 5.9h): any real
                           # bootstrap failures on swing_policy again ->
                           # 3.16 is still too strong, bisect down toward
                           # 1.0-3.16 next. Clean across a full round
                           # (matching 1.0's own clean record) -> 3.16 is
                           # safe, bisect UP toward 3.16-10.0 to keep
                           # narrowing, if still worth pinning down further.
                           #
                           # PREVIOUS reasoning below (1.0 -> 10.0, kept for
                           # history -- this is the failed attempt 3.16 is
                           # responding to):
                           # RAISED AGAIN, 1.0 -> 10.0 (README 5.9n).
                           # 1.0 (below) held up for 4 rounds -- angle,
                           # spread, and bootstrap reliability all
                           # unaffected, `auto_train.py` re-hit "target
                           # reached" -- and the user confirmed
                           # watch_policy.py looked genuinely better,
                           # "still strange, but not out of the range of
                           # human movement." User's stated approach:
                           # deliberately push past the point of real
                           # benefit to find where it stops helping, then
                           # back off -- rather than creep in small steps.
                           # Measured the CURRENT (1.0-trained) checkpoint
                           # before jumping again, not just reusing the old
                           # baseline numbers: real per-step qvel envelope
                           # barely moved (knee mean 216 deg/s, was 206 --
                           # essentially unchanged, even slightly higher),
                           # but the raw action-delta distribution this
                           # weight actually multiplies DID shrink
                           # meaningfully -- mean 0.0176 (was 0.0278, -37%),
                           # p99 0.17 (was 0.31, -45%), max 0.93 (was 1.20).
                           # Reading: 1.0 measurably smoothed the COMMANDED
                           # signal (less small-scale target jitter, likely
                           # the visible improvement) without yet
                           # constraining the actual swing envelope's gross
                           # speed -- consistent with only a partial,
                           # early-stage effect, not a plateau. At 1.0,
                           # even the current p90 event only costs ~3.7% of
                           # a good step's height reward (~1.5-1.6) -- most
                           # of the room to matter was still unused.
                           # 10.0 was chosen (not a smaller bump) using
                           # that same fresh distribution: mean-case cost
                           # rises to ~12% of a step's height reward (still
                           # net-positive to move, shouldn't trigger the
                           # "give up and stop moving" collapse the
                           # original 0.0->0.02 staged rollout guarded
                           # against), but p99 events now cost ~113% of a
                           # full step's reward and the worst observed
                           # event ~6x -- strong enough that if 10.0 is
                           # past the useful range, it should show up
                           # clearly (angle dropping, ramp time ballooning,
                           # or genuinely timid/frozen motion in
                           # watch_policy.py) rather than being another
                           # ambiguous small-step read.
                           #
                           # BACKUP: current 1.0-trained state (confirmed
                           # good -- this is now the fallback to return to,
                           # not the original pre-smoothing lineage) copied
                           # to action_rate_weight_1.0_confirmed_good_backup_
                           # <timestamp>/ before this change.
                           #
                           # PREVIOUS reasoning below (0.02 -> 1.0, still
                           # accurate history):
                           # RAISED, 0.02 -> 1.0 (README 5.9m), branch
                           # experiment: the user watched watch_policy.py
                           # and flagged the motion as "spastic" -- not
                           # human-plausible, especially at the knee.
                           # Quantified before touching this: rolled out
                           # best_model deterministic and compared real
                           # joint qvel against manual_pump.py's smooth
                           # hand-coded law in the same environment.
                           # Oscillation FREQUENCY was nearly identical
                           # (~1.8 Hz trained vs ~1.9-2.3 Hz reference) --
                           # not thrashing faster -- but mean angular
                           # velocity was 2-4x higher (knee: 206 deg/s
                           # trained vs 51 deg/s reference; max 607 vs 223
                           # deg/s), i.e. same rhythm, much sharper snaps
                           # within each cycle. Root cause: measured the
                           # ACTUAL raw action-delta distribution
                           # (sum((action-prev_action)**2), exactly what
                           # this weight multiplies) from that same
                           # rollout -- mean 0.028, p90 0.079, p99 0.31,
                           # max 1.20 (theoretical max for a full both-
                           # axes flip is 8). At the OLD weight (0.02),
                           # even the worst observed step only cost 0.024
                           # -- negligible next to per-step height reward
                           # near target (~1.5-1.6), so there was
                           # essentially zero disincentive against fast
                           # motion. 1.0 is a deliberately large first
                           # jump rather than a small increment -- same
                           # "get a clean read, don't under-correct again"
                           # reasoning OVERSHOOT_WEIGHT's own history used
                           # repeatedly (0.08->0.3->0.6->1.2) after smaller
                           # steps kept proving insufficient. At 1.0: mean-
                           # case cost stays small (~0.028, under 2% of a
                           # good step's height reward) so this shouldn't
                           # reproduce the "give up and stop moving"
                           # collapse the ORIGINAL reasoning below warns
                           # about -- but the rare large snaps (p99 and
                           # up) now cost up to ~1.2, comparable to an
                           # entire good step's height reward, real
                           # pressure specifically on the tail behavior
                           # that looked spastic rather than a uniform tax
                           # on all movement.
                           #
                           # THIS IS A BRANCH, NOT AN OVERWRITE: current
                           # swing_policy.zip/vec_normalize.pkl/best_model/
                           # (the fast-but-effective lineage this change
                           # might make worse) were copied to
                           # pre_action_rate_experiment_backup_<timestamp>/
                           # before this change, specifically so training
                           # can resume normally from the live files as
                           # always, with a full rollback available if
                           # smoothing costs too much swing height. Watch
                           # for: (a) motion looking smoother in
                           # watch_policy.py with peak angle/reliability
                           # holding up -- success; (b) angle/reliability
                           # dropping a lot -- 1.0 may be too strong, back
                           # off; (c) motion still looking spastic with no
                           # angle cost -- 1.0 wasn't enough, go bigger
                           # (same signature as OVERSHOOT_WEIGHT's repeat
                           # under-corrections).
                           #
                           # ORIGINAL reasoning for the 0.0 -> 0.02 staged
                           # rollout (still accurate history, kept below):
                           # ON. Was 0.0 -- staged intentionally that way
                           # (see below) until a working, competent policy
                           # existed. That condition is now met: watching
                           # watch_policy.py directly (not just summary
                           # stats -- see the note on why summary stats were
                           # misleading, right above HEIGHT_DECAY_RANGE_DEG)
                           # showed the swinging itself, once underway,
                           # reaching a clean 80-90 deg oscillation, exactly
                           # the target behavior. But that same viewing
                           # surfaced a different, previously-invisible
                           # problem this constant is meant to address: a
                           # weird, sustained knee oscillation lasting over
                           # a minute (a quarter of a 240s episode) BEFORE
                           # real swinging even begins, and only one or two
                           # swings that actually clip the hard limit --
                           # i.e. most of an episode's lost reward isn't
                           # from wall time at all (too brief to matter
                           # much either way), it's from this dead,
                           # thrashing startup phase. With no action-rate
                           # cost, jittery, high-frequency knee motion is
                           # free, so nothing was pushing the policy to
                           # resolve that thrashing into committing to
                           # pumping sooner. 0.02 is the value this
                           # constant's own original comment (below)
                           # earmarked for exactly this handoff -- a small
                           # nonzero start, not a tuned one.
                           #
                           # ORIGINAL reasoning for starting at 0 (still
                           # correct, just no longer the active phase): any
                           # nonzero value here penalizes a freshly-
                           # initialized (near-random) policy's actions
                           # before it's discovered pumping at all, when
                           # height reward is still ~0 and the penalty
                           # dominates -- the easiest way to minimize a
                           # dominant penalty is to stop moving, a trivial
                           # local optimum the policy can collapse into
                           # almost immediately, long before finding real
                           # pumping. Get a working policy first (comparable
                           # to manual_pump.py's ~80 degrees, or at least
                           # clearly climbing), THEN turn this on and resume
                           # training that already-competent policy to
                           # smooth it -- same staged approach that worked
                           # before, applied in the right order.

# TARGET REDEFINED, 80 -> 67.5 (and HEIGHT_TAPER_START_DEG,
# HEIGHT_DECAY_RANGE_DEG shifted/tightened to match -- see each below). Not a
# bug fix like everything else in this saga -- an explicit change in what
# "good" means, from the user, after watching the actual behavior: even the
# NORMAL, no-incident cruising height was already running 80-90 deg (past
# the old 80 target), with rare excursions to the 165 physical limit on top
# of that. Stated preference, directly: regular, robust swinging that maxes
# out around 65-70 deg and STOPS pumping there is worth more than swinging
# higher (even to the old 80 target) with any risk of excursions past it --
# explicitly the opposite priority from "maximize height, tolerate some
# overshoot" that every constant above and below this point was tuned
# against. Motivated in part by translation to a real rope/chain swing
# (5.7/5.8): the 165 deg hard limit that's been the focus of the last five
# sections (5.5a-vi through x) is itself a simulation artifact of modeling
# the chain as a single rigid link -- a real flexible chain wouldn't behave
# predictably anywhere near there, so treating 165 as a meaningful
# boundary to approach at all was arguably the wrong frame from the start.
# Better margin below a real physical failure mode beats a taller policy
# that occasionally probes it.
#
# 67.5 (midpoint of the user's stated 65-70 range) reuses the exact value
# HEIGHT_TAPER_START_DEG used to have, for the OLD target -- a coincidence
# worth noting, not a design choice: it just means the taper's old easing
# shape now needs its own start point moved down to keep a taper zone at
# all (see that constant below). Everything below this point in the file
# (OVERSHOOT_MARGIN_DEG, EPISODE_OVERSHOOT_PENALTY_CAP, OVERSHOOT_WEIGHT,
# PEAK_OVERSHOOT_WEIGHT, HEIGHT_OVERSHOOT_FLOOR) is UNCHANGED in value --
# all of them key off TARGET_ANGLE_DEG or a hard-limit distance
# (165 - TARGET_ANGLE_DEG) rather than a hardcoded "80", so they
# automatically get MORE aggressive against the same absolute excursion
# now that target sits lower (e.g. reaching 90 deg used to be a 10 deg
# overshoot, now it's 22.5) -- appropriate given the new priority, and
# nothing about their own reasoning needed to change, just the number they
# operate relative to. Their comments below still describe the OLD target
# in worked examples (e.g. "80 -> 120", "0.2 * 85**2 = 1445") -- read those
# as historical record of the reasoning at the time, not current numbers;
# recomputed the worst-case PEAK_OVERSHOOT_WEIGHT bound below since that
# one's worth having current: 0.2 * (165 - 67.5)**2 = 0.2 * 97.5**2 =
# 1,901 (was 1,445).
TARGET_ANGLE_DEG = 67.5      # no reward benefit to swinging higher than this
HEIGHT_TAPER_START_DEG = 55.0  # UPDATED, 67.5 -> 55.0, alongside
                              # TARGET_ANGLE_DEG's move above -- preserves
                              # the same 12.5 deg taper WIDTH the original
                              # choice below reasoned about, just anchored
                              # to the new lower target instead of the old
                              # one. Original reasoning (still the reason
                              # this mechanism exists, just applied to a
                              # lower number now):
                              # NEW. Height reward's own growth rate now
                              # starts leveling off here instead of staying
                              # steepest right up to TARGET_ANGLE_DEG. Same
                              # root cause as OVERSHOOT_MARGIN_DEG below:
                              # height = L*(1-cos(angle)) has derivative
                              # sin(angle), which INCREASES over the whole
                              # 0-80 deg range -- so the policy was being
                              # given maximum incentive to keep pumping
                              # harder right at the moment reward was about
                              # to cap out, then got nothing at all past it.
                              # Complementary to the margin fix, not a
                              # replacement: the margin controls when the
                              # PENALTY starts past target; this controls
                              # when the REWARD stops accelerating before
                              # target, i.e. easing off the accelerator
                              # instead of only ever touching the brakes.
                              # 67.5 (midpoint of the requested 65-70 range)
                              # is reasoned, not tuned -- see
                              # _height_reward() for the actual shape: exact
                              # physical height below this angle (unchanged,
                              # doesn't touch normal low-angle dynamics),
                              # eased to a flat landing (zero slope) exactly
                              # at TARGET_ANGLE_DEG above it. NOTE: the ease
                              # only matches VALUE, not slope, at this
                              # boundary -- there's a small kink in the
                              # gradient at 67.5 deg (much smaller than the
                              # full discontinuity-to-zero this replaces at
                              # 80 deg), a pragmatic simplification rather
                              # than a fully C1-smooth spline.
HEIGHT_DECAY_RANGE_DEG = 20.0  # UPDATED, 40.0 -> 20.0, alongside the target
                              # redefinition above (80 -> 67.5 deg). Tightened,
                              # not just shifted: the old 40 deg width meant
                              # reward stayed positive-but-shrinking all the
                              # way out to target+40 (was 120 deg) before
                              # hitting the floor -- a wide "meh, not great
                              # but not punished much either" zone that's a
                              # bad match for the new stated priority (regular
                              # swinging that stops around 65-70, no real
                              # tolerance for excursions past it, full stop).
                              # 20 reaches the floor at target+20 = 87.5 deg --
                              # notably, almost exactly where the OLD target
                              # (80) used to sit: what used to be "right at
                              # the goal" is now solidly in floor territory.
                              # Original reasoning below (why this mechanism
                              # decays instead of flat-capping at all) is
                              # unchanged, just applied to a narrower window:
                              # NEW. Root-cause fix for a exploit the
                              # overshoot-penalty terms alone couldn't close:
                              # height reward used to be FLAT past
                              # TARGET_ANGLE_DEG (min(angle_deg, 80) fed into
                              # the taper curve), meaning a policy resting at
                              # 165 deg (the hard joint limit, swing_robot.xml
                              # 5.8) got the exact same per-step height
                              # reward as one hovering at 80 -- with LOWER
                              # action-rate cost, since a rod wedged against
                              # a mechanical stop needs no ongoing correction
                              # to stay put, unlike an oscillating hover.
                              # Confirmed in eval data: 4/5 deterministic
                              # episodes settled at 165.4-165.5 deg (pinned
                              # at the limit, not just briefly overshooting
                              # through it) and scored HIGHER reward
                              # (3000-4884) than the one episode that stayed
                              # controlled near target (1337.6 at 96.5 deg
                              # peak) -- direct evidence the flat cap was
                              # making wall-pinning the better strategy, not
                              # just an insufficiently-punished one.
                              # EPISODE_OVERSHOOT_PENALTY_CAP and
                              # PEAK_OVERSHOOT_WEIGHT (below) are bounded by
                              # design (to avoid reintroducing the value-
                              # function blowup from 5.5a-iii) -- a bounded,
                              # mostly one-time penalty can never outweigh a
                              # flat MAXIMUM per-step reward held for
                              # thousands of remaining steps in a ~12,000-
                              # step episode. No penalty-side tweak fixes
                              # that; the flat cap itself had to change.
                              # This makes height reward DECAY past target
                              # instead of staying flat: linearly from its
                              # value at TARGET_ANGLE_DEG down to 0 over this
                              # many further degrees (80 -> 120), then
                              # clamped at 0 the rest of the way to the 165
                              # deg limit. Deliberately floored at 0, not
                              # allowed to go negative -- keeps this a bounded
                              # REDUCTION of an existing positive term (worst
                              # case, a step now earns up to ~1.65 less than
                              # before, arm_length*(1-cos(80))), not a new
                              # unbounded penalty, so it can't reintroduce
                              # the same blowup risk the episode cap exists
                              # to prevent. Once at 0, hovering at target
                              # keeps accumulating positive reward every step
                              # while pinning at the wall accumulates
                              # nothing further -- removes the "free win"
                              # regardless of episode length, which a bounded
                              # penalty term structurally cannot do on its
                              # own. Doesn't replace OVERSHOOT_WEIGHT/
                              # PEAK_OVERSHOOT_WEIGHT -- those still provide
                              # sharper, more immediate gradient signal
                              # right at the margin; this just makes sure
                              # the primary per-step term stops rewarding
                              # the exploit outright. 40 is reasoned, not
                              # tuned: roughly the same order of magnitude as
                              # the approach-side taper (12.5 deg, 67.5->80),
                              # chosen a bit more gradual on the way down to
                              # give a smooth corrective gradient rather than
                              # a cliff, and comfortably reaches 0 well
                              # before the 165 deg physical limit so there's
                              # no residual flat-max zone anywhere overshoot
                              # can actually reach.
HEIGHT_OVERSHOOT_FLOOR = -0.15  # NEW. HEIGHT_DECAY_RANGE_DEG (above) floored
                              # the decayed height reward at 0 past 120 deg --
                              # correct in that it stopped wall-pinning from
                              # being the flat-out BEST strategy (5.5a-vii),
                              # but the very next round showed 0 (neutral,
                              # "gain nothing more") still wasn't enough to
                              # dislodge the policy from the wall: all 10
                              # eval episodes (both deterministic AND
                              # stochastic, action noise included) converged
                              # to 165.4-165.7 deg -- a 0.08 deg spread, far
                              # tighter than anything seen from an
                              # intentional near-target policy so far, and
                              # scoring comparably to (sometimes above) the
                              # single good near-target episode logged the
                              # round before. Reading: the hard joint limit
                              # (5.8) is a genuinely stable physical resting
                              # state -- a PD-driven arm pressed against a
                              # mechanical stop just sits there, needing
                              # essentially no ongoing control effort and
                              # immune to the disturbance kicks that can
                              # knock a delicate near-target hover
                              # off-course -- versus precisely holding near
                              # 80 deg, which is a much harder, higher-
                              # variance control problem. A flat 0 imposes
                              # no ongoing cost for occupying that easy,
                              # robust state, so PPO had no pressure to keep
                              # searching for the harder, narrower, only-
                              # sometimes-better alternative. Extended the
                              # SAME linear decay past where it used to floor
                              # at 0, letting it continue into negative
                              # territory, then clamped there -- a small,
                              # deliberately FIXED constant rather than
                              # mirroring the peak height (~1.65) as the
                              # floor: a per-step penalty that large, summed
                              # over a large fraction of a ~12,000-step
                              # episode (self.max_steps), reproduces exactly
                              # the kind of large-magnitude outlier return
                              # 5.5a-iii's episode cap exists to prevent --
                              # only this time via the height term instead
                              # of the overshoot-penalty term, which the cap
                              # doesn't touch. -0.15 keeps the worst case
                              # (parked at the wall for nearly the whole
                              # episode) bounded around -0.15 * 12,000 =
                              # -1,800 -- comparable in magnitude to a good
                              # episode's positive return (roughly 1,500-
                              # 2,000), not an order of magnitude beyond it,
                              # while still being unambiguously worse than
                              # sitting at 0: every step spent pinned now
                              # actively drains the episode return instead
                              # of just failing to grow it. Changing only
                              # this (not also shortening
                              # HEIGHT_DECAY_RANGE_DEG) on purpose, to get a
                              # clean read on whether an active per-step
                              # cost -- as opposed to merely a bigger one-
                              # time cost -- is what a physically stable
                              # attractor actually needs to become
                              # unattractive.
OVERSHOOT_MARGIN_DEG = 5.0    # RESTORED (partially), 0.0 -> 5.0, after the
                              # retarget (5.5a-xi) plus WALL_TOUCH_PENALTY
                              # (5.5a-xii) together plausibly overshot the
                              # other direction: 4 straight rounds of
                              # DECLINING deterministic peak angle (103 -> 90
                              # -> 21 -> 15 deg), with the STOCHASTIC side
                              # also weakening in the last round (only 1/5
                              # episodes finding real height, down from 3/5
                              # the round before) -- a real, sustained trend,
                              # not the single-round noise this project has
                              # learned to wait out elsewhere. Wall touches
                              # were already at 0 in both of the declining
                              # rounds, so WALL_TOUCH_PENALTY itself isn't
                              # the active constraint being tripped --
                              # something upstream of it plausibly made even
                              # ATTEMPTING a real pump look too risky.
                              # Prime suspect: with margin at 0, the
                              # quadratic OVERSHOOT_WEIGHT penalty (not just
                              # the gentler height-reward decay) starts the
                              # instant a step crosses TARGET_ANGLE_DEG, and
                              # since target moved down 12.5 deg (80 -> 67.5,
                              # 5.5a-xi), that penalty is now considerably
                              # harsher at any given ABSOLUTE angle than it
                              # used to be -- e.g. at 85 deg, overshoot_deg
                              # is now 17.5 instead of 5, a ~12x larger
                              # quadratic penalty for physically the same
                              # amount of overshoot. A policy with still-
                              # imprecise control (still readapting through
                              # several stacked landscape changes: the
                              # retarget, then the wall-touch penalty) may
                              # rationally have learned that a real pump
                              # attempt risks a large, immediate penalty on
                              # any imprecise landing, making "barely move"
                              # look safer despite scoring far worse when it
                              # actually succeeds (per the stochastic data:
                              # up to 982.8 reward when a real pump lands,
                              # vs ~15-65 for barely moving at all).
                              #
                              # 5.0 is a partial, not full, restoration
                              # (original pre-5.5a-iv value was 10.0) --
                              # deliberately kept smaller than before given
                              # the new tighter-tolerance priority (5.5a-xi)
                              # is still the goal, this is just giving a
                              # still-adapting policy some slack to land
                              # near target without an immediate quadratic
                              # penalty for modest imprecision, not
                              # reopening a wide coast zone. Safer to
                              # restore now than it was to remove originally
                              # (5.5a-iv's concern) because the mechanisms
                              # added AFTER that fix -- HEIGHT_TAPER_START_DEG
                              # easing off before target, and
                              # HEIGHT_DECAY_RANGE_DEG reducing reward
                              # immediately past target regardless of this
                              # margin -- already supply continuous
                              # corrective pressure on their own; this margin
                              # only delays the ADDITIONAL quadratic penalty
                              # by a few degrees, not the height reward's own
                              # response to overshoot. Deliberately not
                              # touching HEIGHT_DECAY_RANGE_DEG or
                              # WALL_TOUCH_PENALTY this round, to get a clean
                              # read on whether this alone is enough.
EPISODE_OVERSHOOT_PENALTY_CAP = 100.0  # REDESIGNED (3rd iteration) + value
                              # lowered, 500.0 -> 100.0, after directly
                              # measuring why repeated overshoot within an
                              # episode wasn't costing anything past the
                              # first excursion. Name kept despite the
                              # semantics changing (see below) rather than
                              # renamed throughout, to avoid rewriting the
                              # many historical comments elsewhere in this
                              # file that correctly describe the OLD
                              # per-episode behavior as it stood when they
                              # were written.
                              #
                              # THE BUG THIS FIXES: the budget below used to
                              # reset only once per EPISODE (in reset()), not
                              # once per overshoot EXCURSION. Direct
                              # measurement against a real trained policy
                              # (best_model, post-5.9e) found the budget
                              # exhausts in about 4 steps (0.08s) at a
                              # typical ~10 deg overshoot excess under
                              # OVERSHOOT_WEIGHT=1.2 -- but that same episode
                              # had 118 separate overshoot excursions
                              # totaling 2,187 steps (43.7s) past the
                              # margin, and an UNCAPPED running total of
                              # 21,553. So in practice the cap was only ever
                              # taxing the FIRST excursion each episode
                              # (about 0.08s out of 43.7s of overshoot
                              # time) -- the other 117 excursions that
                              # episode were entirely free from this term.
                              # That's the direct answer to "what's the
                              # incentive to overshoot": repeating it costs
                              # nothing once the first excursion has spent
                              # the budget, so a policy that's going to
                              # overshoot at all loses nothing by doing it
                              # on every single cycle instead of just once.
                              #
                              # THE FIX: self.cumulative_overshoot_penalty
                              # (below) now resets at the START of every
                              # fresh excursion (the step where overshoot_deg
                              # first goes from 0 to positive), not just once
                              # in reset() -- mirroring exactly how
                              # WALL_TOUCH_PENALTY already detects a "fresh
                              # crossing" for its own event counting. This
                              # means every excursion gets its own full
                              # budget, so overshoot genuinely costs
                              # something on every cycle it happens, not
                              # just the first.
                              #
                              # WHY THE VALUE ALSO DROPPED, 500 -> 100 (this
                              # isn't just "same cap, applied more often"):
                              # 500 was sized to be "clearly negative...
                              # without being anywhere near the five- or
                              # six-figure range that was corrupting the
                              # value function" (original reasoning below)
                              # -- AS A PER-EPISODE TOTAL. Reapplying that
                              # same 500 per excursion, with ~100-120
                              # excursions/episode typical for a policy that
                              # overshoots every cycle, would let the worst-
                              # case episode total climb back up to
                              # 50,000-60,000 -- right back in the range
                              # that caused the original value_loss spike to
                              # 1,269,687 this constant exists to prevent
                              # (see that history below). 100 per excursion
                              # keeps the worst-case realistic total
                              # (~100-120 excursions x 100 = 10,000-12,000)
                              # well clear of that danger zone while still
                              # being far larger, in TOTAL, than the old
                              # design's effective ~500-per-episode ceiling
                              # -- and, unlike the old design, that total is
                              # now actually reachable by genuine repeated
                              # overshoot rather than mostly unused budget
                              # sitting idle after the first excursion.
                              # 100 also keeps most of the quadratic shape
                              # intact through a typical excursion (reaches
                              # cap around overshoot_deg ~9.1 deg, close to
                              # but not before the ~10-13 deg peaks observed,
                              # so the sharp early-excursion gradient signal
                              # this mechanism was built to preserve still
                              # applies through most of each excursion's
                              # depth, not just its first couple steps).
                              # Reasoned, not precisely tuned -- watch for
                              # the same signature the original 500 value's
                              # sizing was worried about (value_loss spikes,
                              # single-episode returns in the tens of
                              # thousands negative) after training resumes;
                              # if this reopens that problem, this needs to
                              # come back down, not go back to the old
                              # once-per-episode design that caused the
                              # under-taxing bug in the first place.
                              #
                              # ORIGINAL reasoning below, kept for history
                              # (describes the once-per-EPISODE design this
                              # replaces -- the cap's PURPOSE, bounding
                              # worst-case regression-target magnitude, is
                              # unchanged; only WHEN it resets changed):
                              # Caps the CUMULATIVE overshoot penalty across
                              # a whole episode, not just one step. Motivated
                              # by real TensorBoard data: train/value_loss
                              # spiked to 1,269,687 (typical range 0.6-25) at
                              # the exact step of an episode that returned
                              # -65,289 (peak angle 123.8 deg, far past
                              # target). PPO's value function trains with
                              # plain MSE loss, extremely sensitive to
                              # outlier regression targets -- an episode
                              # return that far outside the normal range can
                              # dominate a whole batch's value loss and
                              # corrupt that batch's advantage estimates
                              # (GAE is computed FROM the value function),
                              # plausibly feeding a destabilizing policy
                              # update -- a believable proximate cause for at
                              # least some of this project's "sudden single-
                              # round collapse" pattern. TRIED clip_range_vf
                              # in train.py first (bounds how far the value
                              # PREDICTION can move per update) and it made
                              # things WORSE (value_loss went from 2,943.9 to
                              # 1,269,687 with it active) -- wrong tool: it
                              # constrains movement relative to the OLD
                              # prediction, not the loss against the actual
                              # return, so a wildly outlying return still
                              # dominates the MSE regardless of clipping.
                              # Reverted that.
                              #
                              # FIRST ATTEMPT at capping the reward itself
                              # (also insufficient, superseded by this):
                              # a flat PER-STEP cap of 50.0. That bounds the
                              # instantaneous penalty but nothing bounds how
                              # many consecutive steps it can accumulate
                              # over -- with a ~12,000-step episode, SUSTAINED
                              # overshoot (not a single spike) can still rack
                              # up tens of thousands in cumulative penalty
                              # even with every single step capped at 50.
                              # Confirmed in practice: worst single-episode
                              # return went from -65,289 (uncapped) to
                              # -26,742 (per-step cap active) -- real
                              # improvement, same order-of-magnitude problem,
                              # since per-step and per-episode are genuinely
                              # different constraints once episodes run this
                              # long.
                              #
                              # This version instead tracks a per-episode
                              # overshoot penalty BUDGET (self.
                              # cumulative_overshoot_penalty, reset in
                              # reset()) that depletes as it's spent: each
                              # step's applied penalty is
                              # min(raw_penalty, cap - spent_so_far), floored
                              # at 0. Early steps of an excursion are
                              # penalized at full strength (preserving the
                              # sharp gradient signal right at the margin,
                              # same as the per-step version did), but once
                              # the episode has already accumulated enough
                              # penalty to clearly mark it as bad, further
                              # sustained overshoot stops adding more -- which
                              # is what actually bounds the worst-case
                              # EPISODE return, regardless of excursion
                              # length. 500.0 is a reasoned value, not
                              # precisely tuned: clearly negative enough to
                              # distinguish a bad episode from a good one
                              # (good episodes run a few hundred to a few
                              # thousand, positive), without being anywhere
                              # near the five- or six-figure range that was
                              # corrupting the value function.
OVERSHOOT_WEIGHT = 1.2       # DOUBLED AGAIN, 0.6 -> 1.2 (after the knee
                              # range/mass change, 5.9d). Direct evidence 0.6
                              # wasn't holding on this lineage either: sampled
                              # 6 checkpoints per round across TWO consecutive
                              # 1M-step rounds (3.1M -> 4.05M, then 4.11M ->
                              # 5.06M) and every single sampled episode (18 in
                              # the first round, 18 in the second, 36 total)
                              # hit EPISODE_OVERSHOOT_PENALTY_CAP's full 500
                              # budget with zero exceptions -- flat across
                              # ~2M steps, not a transient still-converging
                              # read. Genuine improvement did show up in that
                              # window (the earlier "sometimes fails to
                              # bootstrap into pumping at all" bimodality
                              # fully resolved, and peaks tightened from an
                              # 87-94 deg spread down to a consistent 78-86
                              # deg band) -- but overshoot severity itself
                              # never budged, a stronger non-improving signal
                              # than either previous OVERSHOOT_WEIGHT raise
                              # saw before it resolved on its own. Doubling
                              # again rather than a bigger jump, matching the
                              # 0.3->0.6 raise's own reasoning (a bigger jump
                              # was reserved for getting a first clean read
                              # after multiple prior UNDER-corrections; this
                              # is only the second correction attempt on this
                              # lineage). Same two signatures to watch for as
                              # both times before: if 1.2 still isn't enough,
                              # the same flat-near-cap pattern persists; if
                              # it's too strong, the failure mode flips to a
                              # policy that converges cleanly but plateaus
                              # noticeably BELOW target instead of
                              # approaching it.
                              #
                              # PREVIOUS reasoning below, kept for history
                              # (measured against the geometry/lineage active
                              # at each prior point, not necessarily this one
                              # -- the mechanism is the same, the specific
                              # numbers/checkpoints referenced are not):
                              #
                              # DOUBLED, 0.3 -> 0.6 (second physical redesign,
                              # 5.9). Direct evidence 0.3 wasn't holding on
                              # that geometry: sampled 6 checkpoints spanning
                              # an entire 1M-step training round (4.1M ->
                              # 5.06M steps) and every one hit at or near
                              # EPISODE_OVERSHOOT_PENALTY_CAP's full 500
                              # budget on deterministic eval episodes, flat
                              # across the whole round -- not improving, not
                              # a transient still-converging pattern. Peak
                              # angles consistently 75-82 deg, well past the
                              # 72.5 deg safe margin (TARGET_ANGLE_DEG +
                              # OVERSHOOT_MARGIN_DEG), with real time-in-band
                              # (24-29%) and strongly positive reward (mean
                              # ~9,027) alongside it -- a policy that found
                              # real height fast and stably over-climbs by a
                              # consistent amount, not one that's unstable or
                              # regressing. Same class of problem as the
                              # original 0.08 -> 0.3 raise below (a policy
                              # rationally deciding the extra height reward
                              # from climbing past the margin is worth the
                              # penalty), recurring on a geometry whose
                              # hand-coded reference law (5.9) pumps
                              # noticeably more readily/aggressively than the
                              # old one once given adequate actuator torque
                              # -- plausibly needs more correcting pressure
                              # here for the same reason. A straight doubling
                              # this time, not a bigger jump like 0.08->0.3
                              # was -- that jump was reasoned specifically to
                              # get a clean read after two prior under-corrections
                              # on the OLD geometry; this is the first
                              # attempt at correcting overshoot on the NEW
                              # one, so a more moderate step is the
                              # appropriate starting point again. That 0.6
                              # value did eventually resolve on its own on
                              # the pre-5.9d lineage (see evaluate_log.csv
                              # history around 2026-08-11), which is exactly
                              # why 1.2 above is another doubling rather than
                              # an escalation -- 0.6 as a mechanism has
                              # precedent for working, just not yet on this
                              # specific lineage within the rounds observed
                              # so far.
                              # RAISED AGAIN, 0.08 -> 0.3 -- 0.08 (4x the
                              # original 0.02) was itself a reasoned first
                              # step, not a tuned value, and it wasn't
                              # enough: training resumed from a promoted
                              # ~57 deg checkpoint (swing_ppo_10458080_steps)
                              # drifted deeper into overshoot on EVERY
                              # subsequent check-in rather than correcting --
                              # 84 deg/857 penalty, then 108.7 deg/4389
                              # penalty, then 95.6 deg/2105 penalty with a
                              # 62.6 deg spread -- the same resonant
                              # coast-and-amplify pattern as the original
                              # 0.02 -> 0.08 fix (see above), just slower to
                              # show up. Three separate drift episodes across
                              # two different weight values now -- treating
                              # this as under-corrected rather than a fluke.
                              # 0.3 is a bigger jump than the last one (not
                              # just another doubling) specifically to try to
                              # get a clean read: if this is still too gentle
                              # to hold the policy near target, that should
                              # show up unambiguously; if it's too strong,
                              # the failure mode flips to something new and
                              # recognizable -- a policy that converges
                              # cleanly (low spread, near-zero overshoot) but
                              # plateaus noticeably BELOW target instead of
                              # approaching it, distinct from anything seen
                              # in this project's history so far (which has
                              # been closer to binary: ~65-77 deg or
                              # near-total collapse, nothing that looks like
                              # a deliberately timid middle ground). Watch
                              # for that specific signature after resuming
                              # training under this value -- if it shows up,
                              # back this down; if training instead keeps
                              # producing the same high-overshoot/high-spread
                              # instability as before, 0.3 wasn't the fix and
                              # the problem is more likely something other
                              # than this weight.

PEAK_OVERSHOOT_WEIGHT = 0.2  # NEW. Closes a gap EPISODE_OVERSHOOT_PENALTY_CAP
                              # (above) opened: that budget makes the
                              # per-step overshoot penalty exhaustible, so
                              # once a bad episode has spent its 500, every
                              # further step past the margin is free --
                              # 81 deg and 165 deg (the joint's hard physical
                              # limit, swing_robot.xml) become reward-
                              # equivalent for the rest of the episode.
                              # Confirmed this isn't just theoretical: at
                              # OVERSHOOT_WEIGHT=0.3 the budget exhausts in
                              # roughly 15-20 steps once overshoot_deg
                              # reaches ~10 (500 / (0.3 * 10**2)) -- a tiny
                              # fraction of a ~12,000-step episode -- so the
                              # "nothing left to lose, may as well keep
                              # climbing" zone is reachable almost
                              # immediately in any episode that overshoots
                              # at all, not just an unlucky worst case.
                              # This term is deliberately NOT drawn from that
                              # same budget: it tracks self.peak_overshoot_deg
                              # (the worst overshoot_deg reached so far this
                              # episode, reset in reset()) and charges the
                              # MARGINAL cost only when a new peak is set --
                              # PEAK_OVERSHOOT_WEIGHT * (new_peak**2 -
                              # old_peak**2) -- so it's a one-time charge per
                              # new worst point, not a per-step cost, and
                              # can't be exhausted by sustained duration the
                              # way the budgeted penalty can. Total cost for
                              # an episode is bounded by construction at
                              # PEAK_OVERSHOOT_WEIGHT * max_possible_overshoot
                              # **2 regardless of how long the episode spends
                              # up there -- and since swing_hinge's range is
                              # now hard-capped at 165 deg (swing_robot.xml),
                              # max_possible_overshoot is itself bounded at
                              # 165 - (TARGET_ANGLE_DEG + OVERSHOOT_MARGIN_DEG)
                              # = 85 deg, so the worst-case one-time charge is
                              # a known, finite 0.2 * 85**2 = 1445 -- clearly
                              # punishing (comparable to or exceeding a
                              # typical good episode's return) without
                              # reintroducing the unbounded-regression-target
                              # problem EPISODE_OVERSHOOT_PENALTY_CAP was
                              # added to fix. A modest, quickly-corrected
                              # excursion to say 90 deg (overshoot_deg=10)
                              # only costs 0.2 * 10**2 = 20 by contrast --
                              # this term stays small unless the policy
                              # actually pushes to a new, worse extreme.
                              # 0.2 is a reasoned starting point (same
                              # "reasoned, not tuned" caveat as the other
                              # weights above), chosen to land the 165 deg
                              # worst case in the same rough magnitude as a
                              # typical episode's return rather than dwarfing
                              # or being dwarfed by it.

WALL_TOUCH_THRESHOLD_DEG = 150.0  # NEW. Matches evaluate.py's diagnostic
                              # constant of the same name/value -- kept as
                              # its own separate constant here rather than a
                              # shared import, same reasoning as
                              # TARGET_ANGLE_DEG's intentional duplication
                              # into auto_train.py (5.5a-xi): swing_env.py
                              # and evaluate.py are meant to stay
                              # independently readable. Used below to detect
                              # and penalize REPEATED close approaches to the
                              # physical joint limit (165 deg, swing_robot.xml
                              # 5.8) within a single episode -- a gap neither
                              # EPISODE_OVERSHOOT_PENALTY_CAP nor
                              # PEAK_OVERSHOOT_WEIGHT closes (see
                              # WALL_TOUCH_PENALTY directly below for why).
WALL_TOUCH_PENALTY = 200.0   # RAISED, 20.0 -> 200.0 (5.5a-xv), after the
                              # 20.0 value (below, original reasoning kept
                              # for the mechanism's own history) proved
                              # insufficient: a fresh checkpoint_scan.csv
                              # pass with all of OVERSHOOT_MARGIN_DEG=5,
                              # OVERSHOOT_WEIGHT=0.3, PEAK_OVERSHOOT_WEIGHT,
                              # and this term already active still showed 3
                              # of the last 4 checkpoints in the lineage
                              # settling at 157.9-165.4 deg -- the exact
                              # wall-pinning attractor this term was built to
                              # eliminate, recurring despite it. Root cause:
                              # OVERSHOOT_WEIGHT's effect saturates long
                              # before a policy reaches the wall --
                              # EPISODE_OVERSHOOT_PENALTY_CAP's 500 budget
                              # exhausts within ~15-20 steps of overshoot,
                              # around overshoot_deg~10 (absolute angle
                              # ~82.5 deg) -- so by 150+ deg that term is
                              # already spent and contributes nothing
                              # further. Raising OVERSHOOT_WEIGHT again would
                              # just re-run an experiment that's already
                              # failed twice (0.08 -> 0.3, drift resumed
                              # anyway both times, see that constant's
                              # comment); it sharpens the early/moderate
                              # range, not the approach to the wall. This
                              # term is the one that actually acts out there,
                              # so it's the one raised.
                              #
                              # 200 chosen to make even a SINGLE touch
                              # unambiguously costly rather than "barely
                              # different from an isolated excursion" (the
                              # explicit description of 20's effect below,
                              # which is exactly the problem): 200 is ~10-13%
                              # of a typical good episode's return (1,500-
                              # 2,000, see EPISODE_OVERSHOOT_PENALTY_CAP's
                              # comment) on its own, for the FIRST touch
                              # alone. At the 6-touch count seen in the most
                              # recent stochastic eval round, cost is 1,200 --
                              # already more than double EPISODE_OVERSHOOT_
                              # PENALTY_CAP's entire budget, appropriately
                              # dominant since repetition is specifically
                              # what's being targeted. At the highest touch
                              # count ever observed in this project (28,
                              # motivating the original 20.0 value), cost is
                              # 5,600 -- large and clearly worse than any
                              # good episode, but still nowhere near the
                              # five-to-six-figure range (1,269,687 value
                              # loss off a single -65,289 return) that
                              # EPISODE_OVERSHOOT_PENALTY_CAP exists to
                              # prevent reproducing. Deliberately NOT also
                              # raising OVERSHOOT_WEIGHT or PEAK_OVERSHOOT_
                              # WEIGHT this round, to get a clean read on
                              # whether this term alone -- the one actually
                              # scoped to the wall specifically -- is enough,
                              # same single-variable discipline as
                              # OVERSHOOT_MARGIN_DEG's restoration above.
                              #
                              # ORIGINAL reasoning for the mechanism itself
                              # (still accurate, only the magnitude changed):
                              # Motivated by real post-retarget (5.5a-xi)
                              # eval data: individual DETERMINISTIC episodes
                              # (the mode that matters for real deployment,
                              # not just noisy stochastic rollouts) logging
                              # 28 and 8 distinct wall touches each
                              # (evaluate.py's wall_touch_count metric,
                              # 5.5a-x) -- not one bad excursion, a sustained
                              # pattern of repeatedly bouncing off the limit
                              # throughout the episode. The opposite of the
                              # "regular, robust swinging" explicitly
                              # requested in 5.5a-xi, and neither existing
                              # overshoot mechanism actually discourages it:
                              # EPISODE_OVERSHOOT_PENALTY_CAP's budget
                              # exhausts within the first ~15-20 steps of
                              # overshoot (see PEAK_OVERSHOOT_WEIGHT's
                              # comment above) and stays exhausted for the
                              # rest of the episode regardless of how many
                              # more times the wall gets touched, and
                              # PEAK_OVERSHOOT_WEIGHT only charges once, on
                              # the step that sets a NEW worst-ever
                              # overshoot_deg -- a second, third, or
                              # twenty-eighth touch to a similar or lower
                              # height than the first doesn't set a new
                              # peak, so costs nothing under either
                              # mechanism. This term closes that specific
                              # gap: tracks self.past_wall_threshold (reset
                              # in reset()) and charges a FIXED cost once per
                              # distinct crossing from below
                              # WALL_TOUCH_THRESHOLD_DEG to above it --
                              # mirrors exactly how evaluate.py's diagnostic
                              # wall_touch_count is computed, just now
                              # feeding the actual reward instead of only
                              # being reported. Fixed and per-EVENT rather
                              # than scaling with severity or duration on
                              # purpose -- proportional to how often it
                              # happens, not how extreme any single instance
                              # is (that's already PEAK_OVERSHOOT_WEIGHT's
                              # job). Naturally bounded, not just capped by
                              # convention: the swing's own physical
                              # oscillation period (roughly 2.8s per full
                              # cycle near its natural frequency, see 5.5a-x)
                              # limits how many times it could possibly
                              # re-touch the limit within one ~240s episode
                              # even in the worst case, so this can't
                              # reproduce the unbounded-outlier-return
                              # problem 5.5a-iii exists to prevent.
                              # Complementary to EPISODE_OVERSHOOT_PENALTY_CAP
                              # and PEAK_OVERSHOOT_WEIGHT, not a replacement
                              # -- those cover a single excursion's duration
                              # and severity; this specifically covers
                              # repetition, which neither of them do.

WALL_TERMINATE_AFTER_TOUCHES = 2  # NEW (5.5a-xvii). 5.5a-xvi ended the
                              # episode on the FIRST crossing into
                              # WALL_TOUCH_THRESHOLD_DEG, which cleanly
                              # solved what it targeted (repeated in-episode
                              # bouncing: 11 touches -> 1, confirmed) but
                              # coincided with the checkpoint-scan catch
                              # rate (fraction of deterministic episodes
                              # reaching real height) collapsing from a
                              # steady 50-75% (steps 40M-54.2M) to 0% across
                              # three separate checkpoints spanning 57.8M-
                              # 64.9M. Investigated directly rather than
                              # assumed: action magnitude in the failing
                              # checkpoint's episodes is NOT reduced (knee
                              # action amplitude 0.93-1.0, same or higher
                              # than the working checkpoint's 0.92-0.95) --
                              # ruling out "learned to hold back / play it
                              # safe" as the mechanism, since a genuinely
                              # risk-averse policy would show LOWER effort,
                              # not equal-or-higher effort producing far
                              # less height. Looks instead like a timing/
                              # coordination regression: the working
                              # checkpoint's successful episodes showed a
                              # tight, locked rhythm (knee action period
                              # ~64 steps, angle response ~85 steps,
                              # closely matched); the failing checkpoint's
                              # episodes are full-effort but never establish
                              # that lock. Separately confirmed via
                              # tb_logs/PPO_1's rollout/ep_len_mean (actual
                              # training rollout episode length, not eval):
                              # flat at 12000 (no early terminations) for
                              # nearly this project's entire history, then a
                              # real, measurable dip (10920-11980) in
                              # exactly the 55.7M-62.3M window right after
                              # 5.5a-xvi took effect -- confirming a genuine
                              # burst of early-terminated training episodes
                              # coincided with the catch-rate collapse --
                              # but that dip had already subsided back to
                              # ~12000 by 64.4M onward, meaning this doesn't
                              # look like an ONGOING data-starvation effect
                              # happening right now, more like a turbulent
                              # transition window the policy hasn't
                              # recovered from since.
                              #
                              # Directly relevant: at the working 54.2M
                              # checkpoint, of episodes that reached real
                              # height (peak >=50 deg), HALF ended by
                              # riding straight into the wall -- action
                              # amplitude was statistically indistinguishable
                              # between the ones that stopped short (~100-
                              # 115 deg) and the ones that didn't (150+ deg).
                              # At this stage of skill, "commit to a real
                              # pump" and "risk touching the wall" were
                              # substantially the SAME strategy, not
                              # separable ones -- meaning 5.5a-xvi's
                              # first-touch termination was landing on a
                              # large fraction of genuinely successful
                              # climbs, not just the failures it was meant
                              # to stop.
                              #
                              # This change: don't terminate on the first
                              # crossing, only the second. The first crossing
                              # still charges WALL_TOUCH_PENALTY (200) as
                              # before -- a real, immediate cost -- but the
                              # episode is allowed to continue, preserving
                              # the chance for a trajectory that briefly
                              # overshot into wall territory as part of an
                              # otherwise-good climb to settle back down and
                              # keep running (and keep contributing full-
                              # length training experience) rather than
                              # being cut off instantly. A SECOND distinct
                              # crossing -- the signature of actual repeated
                              # bouncing, which is what originally motivated
                              # 5.5a-xii/xvi, not a single overshoot -- still
                              # ends the episode immediately. 2 is a
                              # reasoned starting point, not tuned: it's the
                              # smallest change that still makes sustained
                              # repetition structurally impossible (the
                              # original 11/28-touch pattern would still be
                              # cut short almost immediately) while giving
                              # up the strict "not even once" guarantee.
                              # Whether this actually restores the catch
                              # rate is an open question this change alone
                              # doesn't answer -- needs real training time
                              # under it to find out, not just a smoke test.
DISTURBANCE_PROB_PER_STEP = 0.001  # ON. Was 0.0 -- see README.md section 7
                                    # for the staging rationale (a working,
                                    # locked-in checkpoint now exists:
                                    # checkpoints/swing_ppo_15489696_steps,
                                    # promoted into swing_policy.zip /
                                    # best_model/, so it's safe to layer
                                    # robustness training on top rather than
                                    # fighting the cold-start problem at the
                                    # same time). 0.001 per env-step, with
                                    # ~12000 steps per 240s episode, works out
                                    # to roughly 12 disturbances per episode
                                    # on average -- about one every 20s,
                                    # frequent enough to actually be learned
                                    # from without being so constant that the
                                    # policy never gets a clean stretch to
                                    # pump in between. Revisit if training
                                    # struggles to hold any height at all
                                    # (too frequent) or barely reacts to
                                    # disturbances at eval time (too rare).
DISTURBANCE_MAX_KICK = 0.15         # rad/s added (or subtracted, random sign)
                                    # to swing angular velocity when a
                                    # disturbance fires -- roughly a stronger,
                                    # untimed version of reset()'s seed kick,
                                    # standing in for a gust hitting the swing
                                    # mid-motion rather than at a dead stop.

# DOMAIN RANDOMIZATION (roadmap item 3, README 5.9r/5.9w). Every physical
# parameter this project has actually measured sensitivity for gets
# resampled fresh EVERY EPISODE (in reset(), not once at startup) so the
# policy has to work across the real range of plausible hardware, not just
# the one nominal build swing_robot.xml happens to specify. Same staging
# logic already used for DISTURBANCE_PROB_PER_STEP above: this is layered
# on top of an existing working, target-reaching policy (best_model /
# swing_policy.zip), not something a cold-start policy has to fight
# alongside learning to pump at all -- turned ON now for that reason, not
# staged off first.
#
# Ranges below come directly from three dedicated sensitivity sweeps
# (investigate_mass_sensitivity.py, investigate_damping_sensitivity.py,
# investigate_air_drag_sensitivity.py -- README 5.9r and 5.9v), not
# guesses:
#   - rope_mass, seat_mass: LOW sensitivity across the full tested range --
#     used as-is, no need to be conservative.
#   - leg_mass: MODERATE, smooth response -- full tested range, EXCEPT
#     staying above 0.9 kg, where the hand-coded reference law itself went
#     unstable (a control-law artifact, not real physics, but no reason to
#     train through a region the sensitivity data itself can't speak to).
#   - torso_mass, head_mass: HIGH sensitivity, each with a SHARP CLIFF
#     (torso: clean through 3.0, collapsed by 3.5; head: clean through 0.6,
#     collapsed by 0.7) plausibly caused by both bodies sharing one fixed
#     8.1 Nm lean-servo torque budget. User's explicit call: stay clear of
#     both cliffs entirely rather than pin them down more precisely first
#     -- conservative, asymmetric-from-baseline ranges (only below or at
#     the current nominal value, not above), revisit if a stronger lean
#     servo is chosen later.
#   - swing/lean/knee_damping: LOW sensitivity across every tested range --
#     used as-is.
#   - air_density: continuous, steady decline with no cliff, but a real
#     one -- even bare air (1.2 kg/m^3) alone drops the fixed hand-coded
#     law's ceiling well under target (52.9 deg vs 67.5 target), a floor
#     on what's achievable, not a proven RL limit, since best_model has
#     never trained against any drag before now. Real air is the hard
#     LOWER bound (can't build something draggier than bare air minus
#     drag), user's own framing: real hardware can only be made draggier,
#     never less draggy. User's explicit call on the upper bound: 2.4
#     kg/m^3 (2x real air) -- leaves real headroom for a retrained policy
#     to still reach target, versus 4.8+ where the fixed-law ceiling (25.2
#     deg) suggested target-reaching may not be reachable even retrained.
#
# Implementation: rebuilds swing_robot.xml's TEXT with new sampled values
# substituted in (via the same unique-anchor + bounded-regex approach the
# investigate_*.py sweep scripts already use and verify), then recompiles
# a fresh MjModel from that string -- NOT direct post-compile mutation of
# model.body_mass/dof_damping/opt.density, which for mass specifically
# would leave body_inertia inconsistent with the new mass (MuJoCo doesn't
# recompute inertia tensors from a bare mass edit) -- a correctness risk
# flagged before any of this was implemented, avoided by always going
# through a real recompile instead. Every recompiled model is verified
# against the sampled values before being trusted (see _randomize_physics
# below), matching this project's own established discipline for this
# exact kind of substitution.
# IMPORTANT SCOPE NOTE, added after catching this the first time through:
# this does NOT turn on globally for every SwingEnv() in the project just by
# being True here -- see __init__'s domain_randomization parameter below.
# evaluate.py, watch_policy.py, scan_checkpoints.py, and every investigate_
# /scratch script all construct SwingEnv() with no arguments and MUST keep
# getting fixed, nominal (baseline swing_robot.xml) physics, unchanged --
# evaluate.py's whole gating stack in auto_train.py (spread thresholds,
# streak detection, bootstrap-failure rate, TARGET_OVERSHOOT_MAX) was
# calibrated against episode-to-episode variance meaning ONLY policy
# stochasticity, never physics variance too; randomizing eval episodes would
# silently break every one of those thresholds' meaning and make them
# incomparable to the whole evaluate_log.csv history. Only train.py's actual
# ROLLOUT-COLLECTION env should opt in (env_kwargs=dict(domain_randomization=
# True) at its make_vec_env(SwingEnv, ...) call) -- eval_env and
# prior_best_env inside train.py, and every use of SwingEnv() elsewhere,
# deliberately stay off so "best" and "target reached" keep meaning
# performance on the real nominal build, not a random sample from the
# training distribution. This is this constant's RECOMMENDED value for that
# one opt-in call site, not a global switch.
DOMAIN_RANDOMIZATION_ENABLED = True
DOMAIN_RANDOMIZATION_RANGES = {
    # mass, kg -- baseline (current swing_robot.xml): rope 0.5, seat 1.5,
    # torso 3.0, head 0.4, leg 1.5
    "rope_mass": (0.25, 1.0),
    "seat_mass": (0.75, 2.5),
    "leg_mass": (1.0, 2.5),
    "torso_mass": (2.5, 3.0),
    "head_mass": (0.3, 0.6),
    # joint damping -- baseline: swing 0.01, lean 0.05, knee 0.2
    "swing_damping": (0.005, 0.04),
    "lean_damping": (0.02, 0.2),
    "knee_damping": (0.05, 0.8),
    # air density, kg/m^3 -- baseline: 0.0 (no drag modeled before 5.9v)
    "air_density": (1.2, 2.4),
}
_MASS_BASELINE = dict(rope_mass=0.5, seat_mass=1.5, torso_mass=3.0, head_mass=0.4, leg_mass=1.5)
_MASS_ANCHORS = {
    "rope_mass": 'fromto="0 0 0  0 0 -2.25"',
    "seat_mass": 'fromto="0.06 0 0  -0.14 0 0"',
    "torso_mass": 'fromto="0 0 0  0 0 0.38"',
    "head_mass": 'name="head_geom"',
    "leg_mass": 'name="leg_mass"',
}
_DAMPING_JOINT_NAMES = {"swing_damping": "swing_hinge", "lean_damping": "lean_hinge", "knee_damping": "knee_hinge"}
_DAMPING_ANCHORS = {
    "swing_damping": 'name="swing_hinge" type="hinge" axis="0 1 0" pos="0 0 0"',
    "lean_damping": 'name="lean_hinge" type="hinge" axis="0 1 0" pos="0 0 0"',
    "knee_damping": 'name="knee_hinge" type="hinge" axis="0 1 0"',
}
_OPTION_ANCHOR = '<option gravity="0 0 -9.81" timestep="0.005"/>'
_AIR_VISCOSITY = 1.85e-5  # held fixed, see README 5.9v -- negligible next to
                          # the density-driven quadratic drag term at these
                          # speeds/sizes, so only density is randomized.

# OBSERVATION NOISE (anticipating gyro-derived swing angle on real hardware
# -- README 5.9t/5.9x). No direct swing-angle sensor is practical for the
# real build (a pivot rotary encoder was ruled out) -- swing_angle will
# actually come from integrating a seat-mounted gyro (5.9t), which drifts
# over an episode instead of being exact the way this sim's qpos reading is.
# ONLY swing_angle is noised, nothing else in the observation: swing_
# VELOCITY comes directly off the gyro's own rate output, not an integrated/
# derived quantity, so there's no equivalent drift mechanism for it (and no
# investigated number to calibrate against); lean/knee angle+velocity come
# from each servo's own position feedback, a completely different sensing
# path 5.9t never flagged a gap for. Not adding unverified noise to those
# just to be thorough -- only touching the one specific, actually-
# investigated gap.
#
# Modeled as a random WALK in swing_angle, re-zeroed at the start of every
# episode (matching the deployment plan's "re-zero the gyro coming out of
# the rest phase", 5.9u) -- deliberately NOT i.i.d. per-step noise, since
# gyro-integration error is fundamentally a DRIFT process (errors compound
# over time as the integral accumulates), not independent per-sample
# measurement noise. The per-step standard deviation below is a genuine
# physical drift-RATE constant (radians of drift per env-step, independent
# of episode length), sized directly from 5.9t's own measurement -- pure
# gyro-integration tracked true angle to 0.32 deg RMSE over a 240s episode
# (12,000 env-steps at this project's 0.02s/step). A Wiener-process random
# walk's std grows as sigma*sqrt(n_steps), so sigma = 0.32deg / sqrt(12000)
# reproduces that exact measured result at 240s, and naturally gives more
# total drift over this project's actual 360s (18,000-step) episodes
# (expected final std ~0.39 deg) -- not a separately re-tuned number, the
# same rate applied for longer just because the episode is longer.
# IMPORTANT CAVEAT, same as 5.9t's own: 5.9t's 0.32 deg figure came from
# MuJoCo's noise-free synthetic gyro, so this is a floor on real drift, not
# a ceiling -- an actual MEMS gyro adds its own sensor noise/bias
# instability on top of pure integration error, not modeled here for lack
# of any real hardware to characterize it against yet. Reasoned placeholder,
# flagged for revisit once real sensor data exists, same discipline this
# project has used for every other not-yet-measurable physical placeholder.
OBSERVATION_NOISE_ENABLED = True
_ANGLE_NOISE_REFERENCE_RMSE_DEG = 0.32   # 5.9t's directly measured result
_ANGLE_NOISE_REFERENCE_N_STEPS = 12000   # 5.9t's test: 240s / (4 * 0.005s)
ANGLE_NOISE_RANDOM_WALK_STD_RAD_PER_STEP = np.deg2rad(
    _ANGLE_NOISE_REFERENCE_RMSE_DEG / np.sqrt(_ANGLE_NOISE_REFERENCE_N_STEPS)
)


def _set_attr_after_anchor(xml, anchor, attr_name, new_value, window_chars):
    # Shared substitution primitive for both mass and damping -- unique-
    # anchor + bounded-regex, exactly matching the already-verified approach
    # in investigate_mass_sensitivity.py / investigate_damping_sensitivity.py.
    count = xml.count(anchor)
    assert count == 1, f"anchor {anchor!r} appears {count} times, expected exactly 1"
    anchor_pos = xml.index(anchor)
    window = xml[anchor_pos:anchor_pos + window_chars]
    m = re.search(rf'{attr_name}="([\d.]+)"', window)
    assert m is not None, f'no {attr_name}="..." found within {window_chars} chars after anchor {anchor!r}'
    old_start = anchor_pos + m.start(1)
    old_end = anchor_pos + m.end(1)
    return xml[:old_start] + f"{new_value}" + xml[old_end:]


def _build_randomized_xml(base_xml, params):
    xml = base_xml
    for key, anchor in _MASS_ANCHORS.items():
        xml = _set_attr_after_anchor(xml, anchor, "mass", params[key], 300)
    for key, anchor in _DAMPING_ANCHORS.items():
        xml = _set_attr_after_anchor(xml, anchor, "damping", params[key], 200)
    assert xml.count(_OPTION_ANCHOR) == 1, (
        "<option> anchor not found -- swing_robot.xml's <option> line changed "
        "since this was written, update _OPTION_ANCHOR to match")
    new_option = (f'<option gravity="0 0 -9.81" timestep="0.005" '
                  f'density="{params["air_density"]}" viscosity="{_AIR_VISCOSITY}"/>')
    xml = xml.replace(_OPTION_ANCHOR, new_option)
    return xml


class SwingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, xml_path="swing_robot.xml", episode_seconds=360.0, frame_skip=4,
                 domain_randomization=False, observation_noise=False):
        super().__init__()
        # OFF by default, deliberately -- an instance-level opt-in, not tied
        # to the DOMAIN_RANDOMIZATION_ENABLED module constant above. Every
        # existing SwingEnv() call site across this project (evaluate.py,
        # watch_policy.py, scan_checkpoints.py, every investigate_*.py sweep)
        # expects fixed, nominal physics and must keep getting it unchanged --
        # see DOMAIN_RANDOMIZATION_ENABLED's own comment above for why. Only
        # train.py's rollout-collection env passes domain_randomization=True
        # explicitly.
        self._domain_randomization = domain_randomization
        # Same OFF-by-default, instance-level opt-in pattern, same reason --
        # see OBSERVATION_NOISE_ENABLED's comment above. Only train.py's
        # rollout-collection env passes observation_noise=True explicitly;
        # evaluate.py and everything else keeps seeing exact qpos-based
        # swing_angle, unchanged, same as before this existed.
        self._observation_noise = observation_noise
        # Keep the raw XML TEXT around, not just the compiled model -- domain
        # randomization (below) needs to recompile a fresh model from
        # substituted text every episode, and reading it via xml_path (not a
        # hardcoded "swing_robot.xml") keeps this consistent with whatever
        # file this instance was actually constructed against. Read
        # unconditionally (cheap, a few KB) even when domain_randomization is
        # off, so flipping it on later for an existing instance would work
        # too -- not currently done anywhere, but no reason to make it harder.
        with open(xml_path) as f:
            self._base_xml_text = f.read()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        # Baseline structural counts + total mass, captured from this first,
        # un-randomized compile -- used by _randomize_physics() below as a
        # cheap invariant check (structure unchanged across randomized
        # variants, since only numeric attribute VALUES are substituted, see
        # _build_randomized_xml) and to verify the intended mass delta
        # actually landed, same discipline investigate_mass_sensitivity.py
        # already established.
        self._baseline_njnt = self.model.njnt
        self._baseline_nu = self.model.nu
        self._baseline_total_mass = float(self.model.body_mass.sum())

        self.frame_skip = frame_skip  # physics steps per env step (action held constant)
        self.max_steps = int(episode_seconds / (self.model.opt.timestep * frame_skip))
        self.step_count = 0

        self.swing_dof = self.model.joint("swing_hinge").dofadr[0]
        self.swing_qpos = self.model.joint("swing_hinge").qposadr[0]
        self.lean_dof = self.model.joint("lean_hinge").dofadr[0]
        self.lean_qpos = self.model.joint("lean_hinge").qposadr[0]
        self.knee_dof = self.model.joint("knee_hinge").dofadr[0]
        self.knee_qpos = self.model.joint("knee_hinge").qposadr[0]

        self.lean_act = self.model.actuator("lean_motor").id
        self.knee_act = self.model.actuator("knee_motor").id

        # Actuators are position (PD servo) controlled: ctrl means "target
        # position", in the joint's own units (radians) -- not a
        # normalized torque. Read the real physical range straight from
        # the compiled model instead of duplicating it here, so this can't
        # drift out of sync with the XML. The RL action space stays
        # normalized to [-1, 1] (better for training); actions get
        # rescaled to these physical ranges in step() via _scale_to_ctrl().
        #
        # BUG FIX: this used to read only ctrlrange[1] (the upper bound)
        # and multiply -- correct ONLY when a range is symmetric about
        # zero (lo == -hi). lean_hinge's range is now asymmetric (-25/+55
        # deg, see swing_robot.xml), so that approach would silently
        # produce wrong targets: e.g. action=-1 would map to -0.9599 rad
        # (mirroring the upper bound) instead of the real lower bound,
        # -0.4363 rad -- overshooting the joint's actual mechanical limit
        # by more than double. Storing full [lo, hi] pairs and doing a
        # proper affine map instead.
        self.lean_ctrl_range = self.model.actuator_ctrlrange[self.lean_act]
        self.knee_ctrl_range = self.model.actuator_ctrlrange[self.knee_act]

        # length of the swing arm/rope -- used only to convert angle into a
        # physically meaningful height for the reward. UPDATED from the
        # hardcoded 2.0 to reading this directly off the compiled model
        # (the "seat" body's attachment offset from "swing_arm", i.e. the
        # rope length) -- this file's own prior comment already flagged
        # the hardcoded-number approach as a sync risk ("if you change the
        # arm length in the XML, update this too"), and the second physical
        # redesign (rope now 2.25m, was 2.0m) is exactly the kind of change
        # that risk warned about -- reading it directly closes that gap
        # permanently rather than trusting every future geometry edit to
        # remember this line.
        self.arm_length = abs(self.model.body("seat").pos[2])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # NOTE: previously built as a single obs_high array with bounds
        # implicitly mirrored (low = -high) -- correct only while every
        # dimension was symmetric about zero. lean_angle no longer is
        # (-25/+55 deg), so low/high are now specified separately per
        # dimension instead of one array negated.
        #
        # These bounds are hand-copied from swing_robot.xml's joint ranges,
        # not read off the compiled model -- a known sync risk (same class
        # as arm_length used to be before it was made to read the model
        # directly). knee_angle's high bound UPDATED, deg2rad(60) ->
        # deg2rad(90), to match knee_hinge's range -60/+90 in the XML (the
        # user asked for full extension up to +90). Low bound (-60)
        # unchanged.
        # 5.9ae: low bound now -45 (real knee mount binds there).
        obs_low = np.array(
            [-np.pi, -np.inf, np.deg2rad(-5), -np.inf, np.deg2rad(-45), -np.inf], dtype=np.float32
        )
        obs_high = np.array(
            [np.pi, np.inf, np.deg2rad(80), np.inf, np.deg2rad(90), np.inf], dtype=np.float32
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    def _get_obs(self):
        return np.array(
            [
                self.data.qpos[self.swing_qpos],
                self.data.qvel[self.swing_dof],
                self.data.qpos[self.lean_qpos],
                self.data.qvel[self.lean_dof],
                self.data.qpos[self.knee_qpos],
                self.data.qvel[self.knee_dof],
            ],
            dtype=np.float32,
        )

    def _apply_observation_noise(self, true_obs):
        # Returns a COPY with swing_angle (index 0 only) perturbed by the
        # current drift bias -- see OBSERVATION_NOISE_ENABLED's comment
        # above for the model and why only this one component is touched.
        # CRITICAL: this must ONLY ever be applied to the array actually
        # returned to the policy/caller -- reward and termination logic in
        # step() reads swing_angle from its OWN separate, never-noised
        # `obs` local, on purpose. Reward shaping is a training-time proxy
        # for the real physical task and has to stay grounded in ground
        # truth; only what the POLICY perceives should carry the sensing
        # gap it'll actually have to work with on real hardware. Noising
        # the reward's own input by accident here would silently corrupt
        # training against a fictitious task, not just make it harder.
        if not self._observation_noise:
            return true_obs
        noisy = true_obs.copy()
        noisy[0] += self._angle_noise_bias
        return noisy

    @staticmethod
    def _scale_to_ctrl(action_component, ctrl_range):
        # Maps normalized action in [-1, 1] to the actuator's real
        # [lo, hi] physical range via a simple affine transform.
        # action=-1 -> lo, action=+1 -> hi, action=0 -> the range's
        # midpoint (not necessarily 0 -- for an asymmetric range like
        # lean_hinge's, the midpoint itself is offset from zero, which is
        # correct: it's just where "no preference either way" lands).
        # Reduces exactly to the old symmetric-only behavior when
        # lo == -hi, so this is a strict generalization, not a behavior
        # change for any joint that stays symmetric (e.g. knee_hinge).
        lo, hi = ctrl_range
        return lo + (action_component + 1.0) * 0.5 * (hi - lo)

    @staticmethod
    def _height_reward(angle_deg, arm_length):
        # See HEIGHT_TAPER_START_DEG's comment above for why this exists.
        # True physical height (arm_length * (1 - cos(angle))) below
        # HEIGHT_TAPER_START_DEG, unchanged -- doesn't touch ordinary
        # low-angle dynamics at all. Above it, eased toward a FLAT landing
        # (zero slope) exactly at TARGET_ANGLE_DEG instead of arriving
        # there at the curve's steepest point and then dropping straight
        # to zero. Continuous in VALUE at the HEIGHT_TAPER_START_DEG
        # boundary; NOT continuous in slope there (a small kink, not a
        # fully smooth spline) -- a pragmatic simplification, much smaller
        # than the discontinuity it replaces at TARGET_ANGLE_DEG.
        capped_deg = min(angle_deg, TARGET_ANGLE_DEG)
        if capped_deg <= HEIGHT_TAPER_START_DEG:
            base = arm_length * (1.0 - np.cos(np.deg2rad(capped_deg)))
        else:
            theta0 = np.deg2rad(HEIGHT_TAPER_START_DEG)
            theta1 = np.deg2rad(TARGET_ANGLE_DEG)
            h0 = arm_length * (1.0 - np.cos(theta0))
            h1 = arm_length * (1.0 - np.cos(theta1))
            frac = (capped_deg - HEIGHT_TAPER_START_DEG) / (TARGET_ANGLE_DEG - HEIGHT_TAPER_START_DEG)
            eased = 1.0 - (1.0 - frac) ** 2  # ease-out: slope -> 0 as frac -> 1
            base = h0 + (h1 - h0) * eased

        if angle_deg <= TARGET_ANGLE_DEG:
            return base
        # See HEIGHT_DECAY_RANGE_DEG's comment above -- decay `base` back
        # toward 0 past target instead of holding it flat, so resting at
        # the joint limit can't out-earn hovering at target over a long
        # episode. Past where this decay would cross zero, floor at
        # HEIGHT_OVERSHOOT_FLOOR instead of at exactly 0 -- see that
        # constant's comment above for why flat-zero wasn't enough on its
        # own (didn't dislodge the policy from the physically stable
        # resting state at the joint limit).
        overshoot = angle_deg - TARGET_ANGLE_DEG
        decay_frac = 1.0 - overshoot / HEIGHT_DECAY_RANGE_DEG
        if decay_frac >= 0.0:
            return base * decay_frac
        return HEIGHT_OVERSHOOT_FLOOR

    def _randomize_physics(self):
        # Resamples every DOMAIN_RANDOMIZATION_RANGES parameter uniformly
        # and recompiles self.model/self.data from scratch against the
        # result. Called from reset(), AFTER super().reset(seed=seed) (so
        # self.np_random is the one actually seeded/advanced for this
        # episode), BEFORE mj_resetData (which needs self.model/self.data to
        # already be the ones for this episode).
        #
        # Cached indices/values from __init__ (swing_dof, lean_act,
        # lean_ctrl_range, arm_length, etc.) are NOT recomputed here on
        # purpose -- only numeric attribute VALUES change between variants
        # (mass, damping, option density), never body/joint/actuator
        # structure or order, so those cached integers and arrays stay valid
        # against every recompiled model. njnt/nu are checked below as a
        # cheap, cheap-to-violate invariant rather than trusted blindly.
        params = {key: self.np_random.uniform(lo, hi) for key, (lo, hi) in DOMAIN_RANDOMIZATION_RANGES.items()}
        xml = _build_randomized_xml(self._base_xml_text, params)
        model = mujoco.MjModel.from_xml_string(xml)
        assert model.njnt == self._baseline_njnt and model.nu == self._baseline_nu, (
            "randomized model's joint/actuator count changed from baseline -- "
            "structure shouldn't be able to change from a value-only substitution")
        expected_mass_delta = sum(params[k] - _MASS_BASELINE[k] for k in _MASS_ANCHORS)
        actual_mass_delta = float(model.body_mass.sum()) - self._baseline_total_mass
        assert abs(actual_mass_delta - expected_mass_delta) < 1e-6, (
            f"expected total mass delta {expected_mass_delta:.4f}, got {actual_mass_delta:.4f}")
        for key, joint_name in _DAMPING_JOINT_NAMES.items():
            dof = model.joint(joint_name).dofadr[0]
            assert abs(model.dof_damping[dof] - params[key]) < 1e-9, (
                f"{key}: expected dof_damping {params[key]}, got {model.dof_damping[dof]}")
        assert abs(model.opt.density - params["air_density"]) < 1e-9, (
            f"expected opt.density {params['air_density']}, got {model.opt.density}")
        self.model = model
        self.data = mujoco.MjData(model)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._domain_randomization:
            self._randomize_physics()
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        # Tracks how much of EPISODE_OVERSHOOT_PENALTY_CAP's budget has been
        # spent so far in the CURRENT overshoot excursion -- see that
        # constant's comment above for the per-excursion redesign. Reset
        # here (episode start) AND at the start of every fresh excursion in
        # step() below (via past_overshoot_threshold's crossing detection) --
        # this is deliberately NOT part of the observation, since it's a
        # reward-shaping bookkeeping detail, not real physical state.
        self.cumulative_overshoot_penalty = 0.0
        # Tracks whether the PREVIOUS step was already past the overshoot
        # margin (TARGET_ANGLE_DEG + OVERSHOOT_MARGIN_DEG), so step() can
        # detect a FRESH excursion (crossing from below the margin to above
        # it) the same way past_wall_threshold below detects a fresh wall
        # crossing -- that's what triggers cumulative_overshoot_penalty's
        # per-excursion reset. Also reward-shaping bookkeeping, not real
        # physical state, so also excluded from the observation.
        self.past_overshoot_threshold = False
        # Tracks the worst (largest) overshoot_deg reached so far this
        # episode, for PEAK_OVERSHOOT_WEIGHT's marginal-cost term -- see
        # that constant's comment above. Also reward-shaping bookkeeping,
        # not real physical state, so also excluded from the observation.
        self.peak_overshoot_deg = 0.0
        # Tracks whether the PREVIOUS step was already past
        # WALL_TOUCH_THRESHOLD_DEG, for WALL_TOUCH_PENALTY's crossing
        # detection -- see that constant's comment above. Also reward-
        # shaping bookkeeping, not real physical state, so also excluded
        # from the observation.
        self.past_wall_threshold = False
        # Counts distinct wall crossings this episode, for
        # WALL_TERMINATE_AFTER_TOUCHES (5.5a-xvii) -- see that constant's
        # comment below. Also reward-shaping/termination bookkeeping, not
        # real physical state, so also excluded from the observation.
        self.wall_touch_count_this_episode = 0
        # Gyro-drift observation-noise bias (swing_angle only) -- zeroed
        # every episode, matching the deployment plan's "re-zero the gyro
        # coming out of the rest phase" (5.9u). See
        # OBSERVATION_NOISE_ENABLED's comment above. No-op when
        # self._observation_noise is False.
        self._angle_noise_bias = 0.0
        # Small randomized seed kick, not exact dead rest. Even the
        # hand-coded control law in manual_pump.py needed one of these to
        # ever get going (its qvel=0.05 seed) -- asking the *policy* to
        # invent its own symmetry-breaking perturbation from an exactly
        # symmetric state turned out to be a much harder problem than
        # reacting to and amplifying small existing motion. evaluate.py
        # caught the policy solving that the wrong way: relying on its own
        # per-step exploration noise as a substitute seed kick (huge gap
        # between deterministic and stochastic evaluation), rather than
        # learning a real, repeatable strategy. Randomized sign/magnitude
        # each episode so it has to actually react to what it's given,
        # not memorize one fixed response.
        seed_kick = self.np_random.uniform(-0.08, 0.08)
        self.data.qvel[self.swing_dof] = seed_kick
        mujoco.mj_forward(self.model, self.data)
        return self._apply_observation_noise(self._get_obs()), {}

    def step(self, action):
        # action stays normalized [-1, 1] for the policy and for the
        # action-rate penalty below; only the physical ctrl write gets
        # rescaled to the actuators' real position units.
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.data.ctrl[self.lean_act] = self._scale_to_ctrl(action[0], self.lean_ctrl_range)
        self.data.ctrl[self.knee_act] = self._scale_to_ctrl(action[1], self.knee_ctrl_range)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # Mid-episode disturbance -- a stand-in for a real-world gust hitting
        # the swing while it's already moving, as opposed to reset()'s seed
        # kick, which only ever perturbs a dead-stop starting condition. Now
        # on -- see DISTURBANCE_PROB_PER_STEP above for the current value
        # and why it was safe to enable now.
        disturbed = False
        if DISTURBANCE_PROB_PER_STEP > 0 and self.np_random.random() < DISTURBANCE_PROB_PER_STEP:
            kick = self.np_random.uniform(-DISTURBANCE_MAX_KICK, DISTURBANCE_MAX_KICK)
            self.data.qvel[self.swing_dof] += kick
            mujoco.mj_forward(self.model, self.data)
            disturbed = True

        obs = self._get_obs()
        angle = obs[0]
        angle_deg = float(np.degrees(abs(angle)))

        # Advance the gyro-drift bias once per env-step (not per physics
        # substep -- the policy only observes at this cadence, so this is
        # the right rate to accumulate drift at). See
        # OBSERVATION_NOISE_ENABLED's comment above. No-op when
        # self._observation_noise is False. Deliberately does NOT touch
        # `obs`/`angle`/`angle_deg` above -- everything below this point
        # that computes reward or termination keeps reading the ground-
        # truth values already captured, not whatever gets returned to the
        # caller at the end of this function.
        if self._observation_noise:
            self._angle_noise_bias += self.np_random.normal(0.0, ANGLE_NOISE_RANDOM_WALK_STD_RAD_PER_STEP)

        # No reward benefit past the target, and the growth rate tapers off
        # starting at HEIGHT_TAPER_START_DEG rather than staying steepest
        # right up to it -- see that constant's comment above and
        # _height_reward() for why.
        height = self._height_reward(angle_deg, self.arm_length)

        overshoot_deg = max(0.0, angle_deg - (TARGET_ANGLE_DEG + OVERSHOOT_MARGIN_DEG))
        # Fresh-excursion detection, mirroring WALL_TOUCH_PENALTY's crossing
        # check below exactly: a NEW excursion starts the step overshoot_deg
        # goes from 0 to positive (crossing up through the margin), not
        # every step spent above it. This is what makes
        # EPISODE_OVERSHOOT_PENALTY_CAP's budget reset per-excursion instead
        # of once per episode -- see that constant's comment above for why
        # the once-per-episode version let every excursion after the first
        # go untaxed.
        overshoot_now = overshoot_deg > 0.0
        if overshoot_now and not self.past_overshoot_threshold:
            self.cumulative_overshoot_penalty = 0.0
        self.past_overshoot_threshold = overshoot_now

        # Drawn from EPISODE_OVERSHOOT_PENALTY_CAP's per-EXCURSION budget --
        # see that constant's comment above for why (bounds the regression
        # target PPO's value function trains against at the EXCURSION
        # level, not just per step, so a single excursion can't rack up an
        # unbounded cumulative penalty even though each individual step's
        # raw penalty is uncapped -- while still letting every separate
        # excursion in the episode contribute, unlike the old once-per-
        # episode design). Whatever's left of the budget this excursion;
        # never goes negative.
        raw_overshoot_penalty = OVERSHOOT_WEIGHT * (overshoot_deg**2)
        remaining_budget = max(0.0, EPISODE_OVERSHOOT_PENALTY_CAP - self.cumulative_overshoot_penalty)
        overshoot_penalty = min(raw_overshoot_penalty, remaining_budget)
        self.cumulative_overshoot_penalty += overshoot_penalty

        # PEAK_OVERSHOOT_WEIGHT's marginal cost -- see that constant's
        # comment above. Charged once, only when this step sets a new
        # worst-ever overshoot_deg for the episode; zero otherwise. Not
        # drawn from EPISODE_OVERSHOOT_PENALTY_CAP's budget on purpose, so
        # it can't be exhausted the way overshoot_penalty above can.
        peak_penalty = 0.0
        if overshoot_deg > self.peak_overshoot_deg:
            peak_penalty = PEAK_OVERSHOOT_WEIGHT * (
                overshoot_deg**2 - self.peak_overshoot_deg**2
            )
            self.peak_overshoot_deg = overshoot_deg

        # WALL_TOUCH_PENALTY's fixed per-event cost -- see that constant's
        # comment above. Charged once per distinct crossing from below
        # WALL_TOUCH_THRESHOLD_DEG to above it (mirrors evaluate.py's
        # wall_touch_count computation exactly), not per step spent above
        # it and not scaled by how far past -- repetition is the thing
        # being penalized here, severity is PEAK_OVERSHOOT_WEIGHT's job.
        past_wall_now = angle_deg >= WALL_TOUCH_THRESHOLD_DEG
        new_wall_crossing = past_wall_now and not self.past_wall_threshold
        wall_touch_penalty = WALL_TOUCH_PENALTY if new_wall_crossing else 0.0
        if new_wall_crossing:
            self.wall_touch_count_this_episode += 1
        self.past_wall_threshold = past_wall_now

        action_rate_penalty = ACTION_RATE_WEIGHT * float(
            np.sum((action - self.prev_action) ** 2)
        )
        reward = (
            float(height) - action_rate_penalty - overshoot_penalty
            - peak_penalty - wall_touch_penalty
        )
        self.prev_action = action

        self.step_count += 1
        # NEW (5.5a-xvi). Was unconditionally False -- every mechanism
        # above (OVERSHOOT_WEIGHT, PEAK_OVERSHOOT_WEIGHT, WALL_TOUCH_PENALTY)
        # only discourages reaching the wall, none of them PREVENT an
        # episode from continuing to bounce in and out of it once it's
        # there: a single episode logged 11 distinct WALL_TOUCH_PENALTY
        # charges (5.5a-xv's 200 each, -2,200 total, matching a real
        # -2,239.9 episode return) because nothing ended the episode after
        # the first crossing. No penalty MAGNITUDE closes that gap -- it
        # only makes each additional touch more expensive, which still
        # requires the policy to have already adapted before it stops
        # accumulating them within a single episode. Ending the episode on
        # the first crossing instead makes repeated in-episode wall contact
        # structurally impossible rather than merely costly, and removes
        # the wall's "stable resting state" property outright (see
        # HEIGHT_OVERSHOOT_FLOOR's comment) -- there's no more sitting
        # there accumulating whatever the remaining episode length would
        # have offered, good or bad, since there's no remaining episode.
        # Uses WALL_TOUCH_THRESHOLD_DEG (150 deg), the same threshold
        # WALL_TOUCH_PENALTY and evaluate.py's wall_touch_count already use
        # -- one more reuse of "proximity to the physical joint limit,
        # not the reward target" rather than a new number. Deliberately
        # does NOT extend to the graduated overshoot terms (OVERSHOOT_
        # WEIGHT, PEAK_OVERSHOOT_WEIGHT) below 150 deg -- excursions that
        # overshoot without reaching the wall are still a "keep training,
        # get better at not doing this" problem, not a "this must never
        # happen" one; only the wall itself gets the hard line.
        # 5.5a-xvii: was `terminated = past_wall_now` (ended on the FIRST
        # crossing). Now waits for the SECOND -- see
        # WALL_TERMINATE_AFTER_TOUCHES's comment above for the full
        # reasoning (first-touch termination was landing on genuinely
        # successful climbs, not just repeated-bouncing failures).
        terminated = self.wall_touch_count_this_episode >= WALL_TERMINATE_AFTER_TOUCHES
        truncated = self.step_count >= self.max_steps
        info = {
            "height": height,
            "swing_angle_deg": np.degrees(angle),
            "action_rate_penalty": action_rate_penalty,
            "overshoot_penalty": overshoot_penalty,
            "wall_touch_penalty": wall_touch_penalty,
            "disturbed": disturbed,
        }
        # NOISE APPLIED ONLY HERE, on the way out -- obs/angle/angle_deg
        # above (height, overshoot, wall-touch, termination) all already
        # finished reading the ground-truth values before this line. See
        # _apply_observation_noise's own comment for why that ordering
        # matters.
        return self._apply_observation_noise(obs), reward, terminated, truncated, info


if __name__ == "__main__":
    # Quick manual smoke test with random actions -- no training, just
    # confirms the environment doesn't crash and shapes are sane.
    env = SwingEnv()
    obs, info = env.reset()
    print("obs shape:", obs.shape, "obs:", obs)
    total_reward = 0.0
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print("ran 200 random steps ok, total reward:", total_reward)

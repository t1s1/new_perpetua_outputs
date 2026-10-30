"""
Direct empirical check on the user's question: is the reward actually
pulling toward height, or could something in the current reward shaping
(ACTION_RATE_WEIGHT, overshoot/wall penalties, disturbances) be making a
real pump not worth attempting even though the policy could in principle
find one?

Rather than reason about the pieces in isolation, runs manual_pump.py's
own known-good hand-coded control law (the same sign-based
push-with-velocity / knee-tuck law used as this project's original,
non-learned proof that pumping is physically achievable at all) through
the CURRENT swing_env.py's actual step() function -- same reward,
same penalties, same disturbances a trained policy would face -- and
reports the real cumulative episode reward and peak angle it earns,
headless (no viewer, so it can run in the sandbox).

If this scores dramatically higher than the trained policy's current
~8 deg / 30-110 reward episodes, that's direct evidence the reward
function itself still rewards real height a lot more than staying low --
the bottleneck is the policy not finding/holding the strategy, not the
reward discouraging it. If it scores comparably or worse, that's evidence
something in the current penalty stack is actually working against
reaching height even when the underlying physical strategy is sound.
"""
import numpy as np

from swing_env import SwingEnv

env = SwingEnv()
obs, info = env.reset(seed=0)

lean_lo, lean_hi = env.lean_ctrl_range
knee_lo, knee_hi = env.knee_ctrl_range

total_reward = 0.0
max_angle_deg = 0.0
component_totals = {"height": 0.0, "action_rate_penalty": 0.0, "overshoot_penalty": 0.0, "wall_touch_penalty": 0.0}
n_wall_touches = 0
step = 0
terminated = truncated = False

while not (terminated or truncated):
    swing_angle = env.data.qpos[env.swing_qpos]
    swing_vel = env.data.qvel[env.swing_dof]

    # Same control law as manual_pump.py, translated from raw ctrl writes
    # into normalized [-1,1] actions (what SwingEnv.step expects), since
    # manual_pump.py writes ctrl directly and predates the normalized
    # action space.
    lean_ctrl_target = lean_hi * np.clip(swing_vel * 3.0, -1, 1)  # matches manual_pump.py's use of lean_hi as its scale
    # invert _scale_to_ctrl to get the normalized action that produces this ctrl target
    lean_action = 2.0 * (lean_ctrl_target - lean_lo) / (lean_hi - lean_lo) - 1.0

    knee_ctrl_target = knee_hi * (-np.cos(2 * swing_angle) if abs(swing_angle) < 1.4 else 0)
    knee_action = 2.0 * (knee_ctrl_target - knee_lo) / (knee_hi - knee_lo) - 1.0

    action = np.clip([lean_action, knee_action], -1.0, 1.0).astype(np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    angle_deg = abs(float(info["swing_angle_deg"]))
    max_angle_deg = max(max_angle_deg, angle_deg)
    component_totals["height"] += float(info["height"])
    component_totals["action_rate_penalty"] += float(info["action_rate_penalty"])
    component_totals["overshoot_penalty"] += float(info["overshoot_penalty"])
    component_totals["wall_touch_penalty"] += float(info["wall_touch_penalty"])
    if info["wall_touch_penalty"] > 0:
        n_wall_touches += 1
    step += 1

print(f"Manual-pump-law episode (headless, current reward function): {step} steps")
print(f"  ended via: {'terminated (wall)' if terminated else 'truncated (time limit)'}")
print(f"  total reward: {total_reward:.1f}")
print(f"  peak angle: {max_angle_deg:.1f} deg")
print(f"  wall touches: {n_wall_touches}")
print(f"  reward components (summed): {component_totals}")
print()
print("For comparison, the current trained policy's recent evaluate.py numbers:")
print("  deterministic mean reward ~39, mean peak angle ~8 deg")
print("  stochastic    mean reward ~109, mean peak angle ~12 deg")

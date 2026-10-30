"""
Tests two specific things the "does wall-avoidance suppress height-gain
generally" question needs data on, comparing an OLD checkpoint (still
catching real pumps ~75% of the time) against the CURRENT one (0% catch
rate in the last several checkpoints tested):

1. Among SUCCESSFUL (strong) episodes, how often does reaching real height
   actually involve crossing the wall threshold (terminating early) versus
   reaching real height and staying clear of it? If real pumping and wall
   risk are entangled even in the working regime, that's evidence a
   mechanism punishing wall contact could plausibly suppress attempts at
   real height too, not just the excess overshoot past it.

2. Is the CURRENT checkpoint's action magnitude generally more subdued than
   the OLD checkpoint's, even among episodes that fail to reach real
   height either way? If the new checkpoint is quieter/more passive even
   in its failures, that points at general suppression rather than the
   same old phase-lock miss rate recurring at a worse ratio.

Usage: python3 investigate_wall_entanglement.py <checkpoint_step> <n_episodes>
"""
import json
import sys
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from swing_env import SwingEnv, WALL_TOUCH_THRESHOLD_DEG

RESULTS_PATH = "wall_entanglement_by_checkpoint.jsonl"

step = sys.argv[1]
n_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 4

model_path = f"checkpoints/swing_ppo_{step}_steps"
vecnorm_path = f"checkpoints/swing_ppo_vecnormalize_{step}_steps.pkl"

t_start = time.time()
model = PPO.load(model_path)
vec_env = DummyVecEnv([lambda: SwingEnv()])
vec_env = VecNormalize.load(vecnorm_path, vec_env)
vec_env.training = False
vec_env.norm_reward = False

episodes = []
for ep in range(n_episodes):
    obs = vec_env.reset()
    max_angle = 0.0
    knee_actions = []
    lean_actions = []
    steps = 0
    done = False
    ended_via_wall_termination = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        lean_actions.append(float(action[0][0]))
        knee_actions.append(float(action[0][1]))
        obs, reward, done_arr, info = vec_env.step(action)
        done = bool(done_arr[0])
        angle_deg = abs(float(info[0]["swing_angle_deg"]))
        max_angle = max(max_angle, angle_deg)
        steps += 1
    # If the episode ended before max_steps (12000) and its peak reached the
    # wall threshold, it ended via terminated=True (the wall crossing),
    # not truncated (time limit) -- steps < 12000 alongside peak >= threshold
    # is the tell, since evaluate.py doesn't expose terminated/truncated
    # directly through the VecEnv `done` flag.
    ended_via_wall_termination = steps < 12000 and max_angle >= WALL_TOUCH_THRESHOLD_DEG

    outcome = "strong" if max_angle >= 50.0 else ("weak" if max_angle < 20.0 else "mid")
    episodes.append({
        "peak_angle_deg": max_angle,
        "outcome": outcome,
        "steps": steps,
        "ended_via_wall_termination": ended_via_wall_termination,
        "mean_abs_knee_action": float(np.mean(np.abs(knee_actions))),
        "mean_abs_lean_action": float(np.mean(np.abs(lean_actions))),
    })

n_strong = sum(1 for e in episodes if e["outcome"] == "strong")
n_weak = sum(1 for e in episodes if e["outcome"] == "weak")
strong_wall_term = sum(1 for e in episodes if e["outcome"] == "strong" and e["ended_via_wall_termination"])

result = {
    "step": int(step),
    "n_episodes": n_episodes,
    "n_strong": n_strong,
    "n_weak": n_weak,
    "strong_that_hit_wall": strong_wall_term,
    "episodes": episodes,
    "elapsed_sec": time.time() - t_start,
}
with open(RESULTS_PATH, "a") as f:
    f.write(json.dumps(result) + "\n")

print(f"step {step}: strong={n_strong} (of which {strong_wall_term} ended via wall) weak={n_weak}")
for e in episodes:
    print(f"  peak={e['peak_angle_deg']:6.1f} outcome={e['outcome']:5s} "
          f"steps={e['steps']:5d} wall_term={e['ended_via_wall_termination']} "
          f"knee_amp={e['mean_abs_knee_action']:.3f} lean_amp={e['mean_abs_lean_action']:.3f}")

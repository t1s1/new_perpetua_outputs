"""
Follow-up to investigate_coldstart.py: that script showed WHICH episodes
diverge into strong vs weak pumping and roughly WHEN (first 2-3 sim
seconds), but only looked at the resulting angle, not what the policy was
actually DOING differently. This captures the actual knee/lean actions
(not just angle) at every step through the critical early window, so the
two groups' control behavior can be compared directly, not just their
outcome.

Usage: python3 investigate_coldstart_actions.py <n_episodes_this_call>
"""
import json
import sys
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from swing_env import SwingEnv

MODEL_PATH = "best_model/best_model"
VECNORM_PATH = "vec_normalize.pkl"
RESULTS_PATH = "coldstart_actions_episodes.jsonl"
WINDOW_STEPS = 300  # ~6 sim seconds at this env's timestep*frame_skip --
                     # covers the divergence window investigate_coldstart.py
                     # found (visible by ~2-3s, clearly split by ~6s), plus
                     # margin. Full per-step resolution here, not
                     # downsampled -- action-level detail is the point.

n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 5

model = PPO.load(MODEL_PATH)
vec_env = DummyVecEnv([lambda: SwingEnv()])
vec_env = VecNormalize.load(VECNORM_PATH, vec_env)
vec_env.training = False
vec_env.norm_reward = False
raw_env = vec_env.venv.envs[0]

for ep in range(n_episodes):
    t0 = time.time()
    obs = vec_env.reset()
    seed_kick = float(raw_env.data.qvel[raw_env.swing_dof])

    angle_window = []
    lean_action_window = []
    knee_action_window = []
    total_reward = 0.0
    max_angle = 0.0
    step = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        if step < WINDOW_STEPS:
            lean_action_window.append(float(action[0][0]))
            knee_action_window.append(float(action[0][1]))
        obs, reward, done_arr, info = vec_env.step(action)
        done = bool(done_arr[0])
        total_reward += float(reward[0])
        angle_deg = float(info[0]["swing_angle_deg"])
        max_angle = max(max_angle, abs(angle_deg))
        if step < WINDOW_STEPS:
            angle_window.append(round(angle_deg, 3))
        step += 1

    outcome = "strong" if max_angle >= 50.0 else ("weak" if max_angle < 20.0 else "mid")
    result = {
        "episode_index_this_call": ep,
        "seed_kick": seed_kick,
        "total_reward": total_reward,
        "peak_angle_deg": max_angle,
        "outcome": outcome,
        "angle_window": angle_window,
        "lean_action_window": [round(x, 4) for x in lean_action_window],
        "knee_action_window": [round(x, 4) for x in knee_action_window],
        "elapsed_sec": time.time() - t0,
    }
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"ep {ep}: seed_kick={seed_kick:+.4f} outcome={outcome:5s} "
          f"peak={max_angle:6.1f} reward={total_reward:8.1f} "
          f"elapsed={result['elapsed_sec']:.1f}s")

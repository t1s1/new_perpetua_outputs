"""
One-off investigation: does whether an episode ends up "weak" (barely
pumping) or "strong" (real height) trace back to the reset() seed kick, or
does it diverge later regardless of how the episode started? Runs N
deterministic episodes against best_model/best_model, recording the exact
seed kick (qvel right after reset(), before any policy action) and a
downsampled angle-vs-time trajectory for each, appended to a jsonl file so
it survives being run in several chunked invocations (this sandbox kills
any process that outstays a single ~45s tool call).

Usage: python3 investigate_coldstart.py <n_episodes_this_call>
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
RESULTS_PATH = "coldstart_episodes.jsonl"
TRAJ_SAMPLE_EVERY = 10  # record angle every 10 env steps, not every one --
                          # keeps the jsonl readable while still resolving
                          # the early transient (frame_skip means each env
                          # step is already several sim substeps)

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
    # Read the actual seed kick MuJoCo state directly, right after reset(),
    # before any step -- this is the exact same value reset() drew from
    # np_random.uniform(-0.08, 0.08), not a reconstruction.
    seed_kick = float(raw_env.data.qvel[raw_env.swing_dof])

    trajectory = []  # (step, angle_deg, disturbed_this_step)
    total_reward = 0.0
    max_angle = 0.0
    disturbance_steps = []
    step = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_arr, info = vec_env.step(action)
        done = bool(done_arr[0])
        total_reward += float(reward[0])
        angle_deg = float(info[0]["swing_angle_deg"])
        max_angle = max(max_angle, abs(angle_deg))
        if info[0]["disturbed"]:
            disturbance_steps.append(step)
        if step % TRAJ_SAMPLE_EVERY == 0:
            trajectory.append(round(angle_deg, 3))
        step += 1

    outcome = "strong" if max_angle >= 50.0 else ("weak" if max_angle < 20.0 else "mid")
    result = {
        "episode_index_this_call": ep,
        "seed_kick": seed_kick,
        "total_reward": total_reward,
        "peak_angle_deg": max_angle,
        "outcome": outcome,
        "n_disturbances": len(disturbance_steps),
        "first_disturbance_step": disturbance_steps[0] if disturbance_steps else None,
        "traj_sample_every": TRAJ_SAMPLE_EVERY,
        "trajectory": trajectory,
        "elapsed_sec": time.time() - t0,
    }
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"ep {ep}: seed_kick={seed_kick:+.4f} outcome={outcome:5s} "
          f"peak={max_angle:6.1f} reward={total_reward:8.1f} "
          f"disturbances={len(disturbance_steps)} elapsed={result['elapsed_sec']:.1f}s")

"""
Second follow-up: does the phase-lock catch rate (fraction of episodes that
establish a real pump vs stay weak) actually improve across the recent
training lineage, or has it been flat? Runs a handful of DETERMINISTIC
episodes against a given checkpoint and records the strong/weak/mid split,
appended per-checkpoint to a jsonl so several checkpoints spread across the
recent lineage can be compared. Each invocation does ONE checkpoint's worth
of episodes (kept small -- 4 -- to fit the sandbox's ~45s-per-call limit
alongside model-load overhead for a fresh checkpoint each time).

Usage: python3 investigate_coldstart_trend.py <checkpoint_step> <n_episodes>
"""
import glob
import json
import sys
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from swing_env import SwingEnv

RESULTS_PATH = "coldstart_trend_by_checkpoint.jsonl"

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

outcomes = []
for ep in range(n_episodes):
    obs = vec_env.reset()
    max_angle = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_arr, info = vec_env.step(action)
        done = bool(done_arr[0])
        max_angle = max(max_angle, abs(float(info[0]["swing_angle_deg"])))
    outcome = "strong" if max_angle >= 50.0 else ("weak" if max_angle < 20.0 else "mid")
    outcomes.append({"peak_angle_deg": max_angle, "outcome": outcome})

n_strong = sum(1 for o in outcomes if o["outcome"] == "strong")
n_weak = sum(1 for o in outcomes if o["outcome"] == "weak")
n_mid = sum(1 for o in outcomes if o["outcome"] == "mid")

result = {
    "step": int(step),
    "n_episodes": n_episodes,
    "n_strong": n_strong,
    "n_weak": n_weak,
    "n_mid": n_mid,
    "outcomes": outcomes,
    "elapsed_sec": time.time() - t_start,
}
with open(RESULTS_PATH, "a") as f:
    f.write(json.dumps(result) + "\n")
print(f"step {step}: strong={n_strong} weak={n_weak} mid={n_mid} "
      f"(of {n_episodes}), elapsed={result['elapsed_sec']:.1f}s")

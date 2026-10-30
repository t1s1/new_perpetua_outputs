"""
Can the trained policy live with a smaller knee tuck range WITHOUT retraining?

Real hardware binds at a ~45 deg tuck (sim knee angle -45), but the policy was
trained with knee_hinge range -60..+90 (action -1 -> -60 deg target). Here the
trained best_model is run unchanged while:
  (1) the knee TARGET is clipped to >= L degrees (what the Pi code would do), and
  (2) the sim joint's own lower limit is moved to L (the mechanical stop),
for several L. Also reports how often the unclipped policy asks for < -45.

Read-only: does not modify any project file or evaluate_log.csv.
Usage: python3 investigate_knee_limit.py <group>   (group 0..1)
"""
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from swing_env import SwingEnv
from evaluate import compute_time_to_90pct

N_EPISODES = 3
LO_DEG, HI_DEG = -60.0, 90.0
LIMITS = [None, -55.0, -50.0, -45.0, -40.0, -35.0]
GROUPS = [LIMITS[0:3], LIMITS[3:6]]


def run(model, limit):
    vec = DummyVecEnv([lambda: SwingEnv()])
    vec = VecNormalize.load("best_model/best_vec_normalize.pkl", vec)
    vec.training = False
    vec.norm_reward = False
    inner = vec.venv.envs[0]
    dt = inner.frame_skip * inner.model.opt.timestep
    if limit is not None:
        jid = inner.model.joint("knee_hinge").id
        inner.model.jnt_range[jid, 0] = np.deg2rad(limit)
    obs = vec.reset()
    total, peak, ovs, trace, done = 0.0, 0.0, 0.0, [], False
    n, below45 = 0, 0
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        a = np.clip(a, -1.0, 1.0).copy()
        tgt = LO_DEG + (a[0, 1] + 1.0) * 0.5 * (HI_DEG - LO_DEG)
        n += 1
        below45 += tgt < -45.0
        if limit is not None:
            tgt = max(tgt, limit)
            a[0, 1] = 2.0 * (tgt - LO_DEG) / (HI_DEG - LO_DEG) - 1.0
        obs, r, d, info = vec.step(a)
        done = bool(d[0])
        total += float(r[0])
        ang = info[0]["swing_angle_deg"]
        trace.append(ang)
        peak = max(peak, abs(ang))
        ovs += info[0]["overshoot_penalty"]
    _, steady = compute_time_to_90pct(trace, dt)
    return total, peak, ovs, steady, 100.0 * below45 / n


if __name__ == "__main__":
    g = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    model = PPO.load("best_model/best_model")
    print(f"{'knee floor':>10s} {'reward':>8s} {'peak':>7s} {'steady':>7s} {'overshoot':>10s} {'%steps<-45 asked':>17s}")
    for L in GROUPS[g]:
        rows = np.array([run(model, L) for _ in range(N_EPISODES)])
        m = rows.mean(axis=0)
        name = "none" if L is None else f"{L:.0f}"
        print(f"{name:>10s} {m[0]:8.0f} {m[1]:7.1f} {m[3]:7.1f} {m[2]:10.0f} {m[4]:16.1f}%")

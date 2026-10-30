"""
How much per-leg variation can the trained policy tolerate? (deployment-time
"organic motion" idea -- the two real knee servos get slightly different
versions of the single knee command the policy outputs, with NO retraining.)

LIMITATION, stated up front: swing_robot.xml has ONE knee hinge standing in
for both legs, so left/right cannot be simulated directly. This script uses
the standard first-order stand-in instead: with two independent, equal-mass
legs commanded c_L and c_R, the combined leg mass moves like a single leg
commanded (c_L + c_R) / 2. So every scenario below perturbs the AVERAGE knee
command. That also makes the answer for MIRRORED variation easy to read off:
perfectly mirrored position offsets (+d left, -d right) average to zero, i.e.
no change at all to first order (only a small second-order effect from the
joint nonlinearity, not modeled here). The scenarios therefore test the
harder, non-mirrored cases:

  offset   -- the average command drifts by a slow random offset, RMS A deg
              (worst case: both legs drift together, unmirrored).
  indep    -- each leg drifts independently, RMS A deg each; the average
              drifts with RMS A/sqrt(2). The realistic "random-seeming" case.
  delay    -- one leg lags the policy by D control steps, the other doesn't:
              average command = 0.5 * (a[t] + a[t-D]).

Drift is slow (AR(1), ~5 s correlation time), not step-by-step noise.
Uses best_model with the exact eval setup (nominal physics, exact angles).
Read-only: does not modify any project file or evaluate_log.csv.

Usage: python3 investigate_leg_variation.py <group>   (group in 0..2; each
group fits the sandbox's per-call time cap)
"""
import sys
from collections import deque

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from swing_env import SwingEnv
from evaluate import compute_time_to_90pct

KNEE_DEG_PER_ACTION_UNIT = 75.0  # ctrlrange -60..+90 deg -> half-span 75 deg
N_EPISODES = 3
CORR_TIME_S = 5.0

SCENARIOS = [
    ("baseline", dict()),
    ("offset 2deg", dict(kind="offset", A=2.0)),
    ("offset 5deg", dict(kind="offset", A=5.0)),
    ("offset 10deg", dict(kind="offset", A=10.0)),
    ("offset 20deg", dict(kind="offset", A=20.0)),
    ("indep 5deg/leg", dict(kind="indep", A=5.0)),
    ("indep 10deg/leg", dict(kind="indep", A=10.0)),
    ("indep 20deg/leg", dict(kind="indep", A=20.0)),
    ("delay 2 steps", dict(kind="delay", D=2)),
    ("delay 5 steps", dict(kind="delay", D=5)),
    ("delay 10 steps", dict(kind="delay", D=10)),
    ("delay 25 steps", dict(kind="delay", D=25)),
]
GROUPS = [SCENARIOS[0:4], SCENARIOS[4:8], SCENARIOS[8:12]]


def run(model, spec, rng):
    vec = DummyVecEnv([lambda: SwingEnv()])
    vec = VecNormalize.load("best_model/best_vec_normalize.pkl", vec)
    vec.training = False
    vec.norm_reward = False
    inner = vec.venv.envs[0]
    dt = inner.frame_skip * inner.model.opt.timestep
    phi = np.exp(-dt / CORR_TIME_S)
    kind = spec.get("kind")
    A = spec.get("A", 0.0) / KNEE_DEG_PER_ACTION_UNIT
    xL = xR = 0.0
    hist = deque(maxlen=spec.get("D", 0) + 1)

    obs = vec.reset()
    total, peak, ovs, trace, done = 0.0, 0.0, 0.0, [], False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        a = np.clip(a, -1.0, 1.0).copy()
        k = a[0, 1]
        if kind == "offset":
            xL = phi * xL + np.sqrt(1 - phi ** 2) * A * rng.standard_normal()
            k = k + xL
        elif kind == "indep":
            xL = phi * xL + np.sqrt(1 - phi ** 2) * A * rng.standard_normal()
            xR = phi * xR + np.sqrt(1 - phi ** 2) * A * rng.standard_normal()
            k = k + 0.5 * (xL + xR)
        elif kind == "delay":
            hist.append(k)
            k = 0.5 * (k + hist[0])
        a[0, 1] = np.clip(k, -1.0, 1.0)
        obs, r, d, info = vec.step(a)
        done = bool(d[0])
        total += float(r[0])
        ang = info[0]["swing_angle_deg"]
        trace.append(ang)
        peak = max(peak, abs(ang))
        ovs += info[0]["overshoot_penalty"]
    _, steady = compute_time_to_90pct(trace, dt)
    return total, peak, ovs, steady


if __name__ == "__main__":
    g = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    model = PPO.load("best_model/best_model")
    rng = np.random.default_rng(0)
    print(f"{'scenario':18s} {'reward':>8s} {'peak':>7s} {'steady':>7s} {'overshoot':>10s}")
    for name, spec in GROUPS[g]:
        rows = np.array([run(model, spec, rng) for _ in range(N_EPISODES)])
        m = rows.mean(axis=0)
        print(f"{name:18s} {m[0]:8.0f} {m[1]:7.1f} {m[3]:7.1f} {m[2]:10.0f}"
              f"   (steady range {rows[:, 3].min():.1f}-{rows[:, 3].max():.1f})")

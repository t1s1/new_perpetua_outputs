"""
Do the per-element mass limits combine? The earlier mass sweeps (5.9r) varied ONE
element at a time. This runs the deployed pipeline (pi/: self-excited start, handover,
pi/policy.npz, drift on, dead-rest start, disturbances off) on whole-robot combinations:
the corners of the 5-element mass box, its midpoint, and random points inside it.

Element ranges (kg) = the domain-randomization ranges the policy was trained over
(swing_env.DOMAIN_RANDOMIZATION_RANGES): rope 0.25-1.0, seat 0.75-2.5, leg 1.0-2.5,
torso 2.5-3.0, head 0.3-0.6. Damping at its baseline, air density 1.2 (real air).
Rope length 2.25 m. One seed per combination, 300 s.

    python3 investigate_mass_combinations.py corners 0 8 16 24     # corner indices 0-31
    python3 investigate_mass_combinations.py mid
    python3 investigate_mass_combinations.py random 8 <seed>

Per row: total mass, the five masses, time to reach 55 deg, largest angle, amplitude
(mean 5 s-peak) over the last 100 s. Target 62.5, overshoot cap 75. Read-only.
"""
import itertools
import sys
import tempfile
import os

import mujoco
import numpy as np

sys.path.insert(0, "pi")
import config as C
import swing_core as S
import swing_env
swing_env.DISTURBANCE_PROB_PER_STEP = 0.0
from swing_env import SwingEnv, DOMAIN_RANDOMIZATION_RANGES as R, _build_randomized_xml

SECS = 300.0
C.HOLD_SECONDS = 1e9
C.ACTIVE_MAX_SECONDS = 1e9
C.BOOT_TIMEOUT_S = 1e9
BASE = open("swing_robot.xml").read()
ELEMS = ["rope_mass", "seat_mass", "leg_mass", "torso_mass", "head_mass"]


def run(m, seed=1):
    params = dict(m, swing_damping=0.01, lean_damping=0.05, knee_damping=0.2, air_density=1.2)
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, dir="/tmp") as f:
        f.write(_build_randomized_xml(BASE, params))
        path = f.name
    env = SwingEnv(xml_path=path, episode_seconds=SECS + 10)
    os.unlink(path)
    env.reset(seed=seed)
    env.data.qvel[env.swing_dof] = 0.0
    mujoco.mj_forward(env.model, env.data)
    obs = env._get_obs()
    dt = env.frame_skip * env.model.opt.timestep
    ctl = S.SwingController(S.Policy("pi/policy.npz"), S.SwingEstimator(), dt, seed=3)
    t55, pk, tr = None, 0.0, []
    for k in range(int(SECS / dt)):
        out = ctl.step(obs[1], np.rad2deg(obs[2]), np.rad2deg(obs[4]), np.rad2deg(obs[4]))
        a = np.array([2 * (out["lean"] - C.LEAN_RANGE_DEG[0]) / (C.LEAN_RANGE_DEG[1] - C.LEAN_RANGE_DEG[0]) - 1,
                      2 * (0.5 * (out["knee_a"] + out["knee_b"]) - C.KNEE_RANGE_DEG[0]) / (C.KNEE_RANGE_DEG[1] - C.KNEE_RANGE_DEG[0]) - 1],
                     dtype=np.float32)
        obs, r, te, tru, info = env.step(a)
        ang = abs(info["swing_angle_deg"])
        tr.append(ang)
        pk = max(pk, ang)
        if t55 is None and out["hold_s"] > 0:
            t55 = k * dt
        if te or tru:
            break
    w = int(5.0 / dt)
    last = [max(tr[i:i + w]) for i in range(max(0, len(tr) - int(100 / dt)), len(tr) - w, w)]
    return t55, pk, float(np.mean(last))


def show(tag, m):
    t55, pk, amp = run(m)
    tot = sum(m[e] for e in ELEMS)
    print(f"{tag:>9s} total {tot:5.2f} | rope {m['rope_mass']:.2f} seat {m['seat_mass']:.2f} leg {m['leg_mass']:.2f} "
          f"torso {m['torso_mass']:.2f} head {m['head_mass']:.2f} | reach55 {'never' if t55 is None else f'{t55:.0f}s':>6s} "
          f"max {pk:5.1f} amp {amp:5.1f}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    corners = list(itertools.product(*[(0, 1)] * 5))
    if mode == "corners":
        for i in map(int, sys.argv[2:]):
            m = {e: R[e][b] for e, b in zip(ELEMS, corners[i])}
            show(f"c{i:02d}", m)
    elif mode == "inner":     # torso/head/leg at their 25% / 75% points (middle half of each range), rope & seat at mid
        for i in map(int, sys.argv[2:]):
            b = list(itertools.product(*[(0, 1)] * 3))[i]
            m = {e: 0.5 * (R[e][0] + R[e][1]) for e in ELEMS}
            for e, bit in zip(("leg_mass", "torso_mass", "head_mass"), b):
                m[e] = R[e][0] + (0.75 if bit else 0.25) * (R[e][1] - R[e][0])
            show(f"in{i}", m)
    elif mode == "mid":
        show("mid", {e: 0.5 * (R[e][0] + R[e][1]) for e in ELEMS})
    elif mode == "random":
        n, seed = int(sys.argv[2]), int(sys.argv[3])
        rng = np.random.default_rng(seed)
        for j in range(n):
            show(f"r{j}", {e: float(rng.uniform(*R[e])) for e in ELEMS})

"""
How much can the rope (pivot -> seat attachment) length differ from the trained 2.25 m
before the deployed controller stops working? (Real range under consideration: 2.0-2.5 m.)

Method: for each length L, a temporary copy of swing_robot.xml is written with the rope
capsule, the seat attachment and the pivot height all moved together (pivot raised by
L - 2.25 so the seat stays at its usual height above the ground). The policy is NOT
retrained: it is the exact deployed pipeline from pi/ (self-excited start, handover at
10 deg, then pi/policy.npz), starting from dead rest with the sim's random disturbance
kicks OFF (as on hardware), drift on, perfect gyro bias. Everything else, including
the 0.5 kg rope mass, is unchanged -- the mass is NOT scaled with length.

Reports per length: time to hand over from the start routine, time until the recent peak
first reaches 55 deg, the largest angle seen, and the swing amplitude (windowed peak)
averaged over the last 100 s. Target is 62.5 deg; the training overshoot cap is 75.

Read-only: writes only a temp XML in /tmp. Run from the project root:
    python3 investigate_rope_length.py 1.75 2.0 2.25
"""
import os
import sys
import tempfile

import mujoco
import numpy as np

sys.path.insert(0, "pi")
import config as C
import swing_core as S
import swing_env
swing_env.DISTURBANCE_PROB_PER_STEP = 0.0
from swing_env import SwingEnv

SECS = 360.0
C.HOLD_SECONDS = 1e9          # never "finish": we want to watch the whole run
C.ACTIVE_MAX_SECONDS = 1e9
C.BOOT_TIMEOUT_S = 1e9
BASE = open("swing_robot.xml").read()


def xml_for(L):
    x = BASE
    for old, new in [('fromto="0 0 0  0 0 -2.25" size="0.015"', f'fromto="0 0 0  0 0 -{L}" size="0.015"'),
                     ('<body name="seat" pos="0 0 -2.25">', f'<body name="seat" pos="0 0 -{L}">'),
                     ('<body name="pivot" pos="0 0 3">', f'<body name="pivot" pos="0 0 {L + 0.75}">')]:
        assert x.count(old) == 1, old
        x = x.replace(old, new)
    return x


def run(L, seed=1):
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, dir="/tmp") as f:
        f.write(xml_for(L))
        path = f.name
    env = SwingEnv(xml_path=path, episode_seconds=SECS + 10)
    os.unlink(path)
    assert abs(env.arm_length - L) < 1e-9, env.arm_length
    env.reset(seed=seed)
    env.data.qvel[env.swing_dof] = 0.0
    mujoco.mj_forward(env.model, env.data)
    obs = env._get_obs()
    dt = env.frame_skip * env.model.opt.timestep
    ctl = S.SwingController(S.Policy("pi/policy.npz"), S.SwingEstimator(), dt, seed=3)
    t_hand = t55 = None
    pk, tr = 0.0, []
    for k in range(int(SECS / dt)):
        out = ctl.step(obs[1], np.rad2deg(obs[2]), np.rad2deg(obs[4]), np.rad2deg(obs[4]))
        a = np.array([2 * (out["lean"] - C.LEAN_RANGE_DEG[0]) / (C.LEAN_RANGE_DEG[1] - C.LEAN_RANGE_DEG[0]) - 1,
                      2 * (0.5 * (out["knee_a"] + out["knee_b"]) - C.KNEE_RANGE_DEG[0]) / (C.KNEE_RANGE_DEG[1] - C.KNEE_RANGE_DEG[0]) - 1],
                     dtype=np.float32)
        obs, r, te, tru, info = env.step(a)
        ang = abs(info["swing_angle_deg"])
        tr.append(ang)
        pk = max(pk, ang)
        if t_hand is None and out["mode"] == "POLICY":
            t_hand = k * dt
        if t55 is None and out["hold_s"] > 0:
            t55 = k * dt
        if te or tru:
            break
    w = int(5.0 / dt)
    last = [max(tr[i:i + w]) for i in range(max(0, len(tr) - int(100 / dt)), len(tr) - w, w)]
    return t_hand, t55, pk, float(np.mean(last)) if last else float("nan")


if __name__ == "__main__":
    print(f"{'rope L':>7s} {'handover':>9s} {'reach 55':>9s} {'max':>6s} {'last-100s amplitude':>20s}")
    for L in map(float, sys.argv[1:]):
        th, t55, pk, amp = run(L)
        f = lambda v: "never" if v is None else f"{v:.0f} s"
        print(f"{L:7.2f} {f(th):>9s} {f(t55):>9s} {pk:6.1f} {amp:20.1f}", flush=True)

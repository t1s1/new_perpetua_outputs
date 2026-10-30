"""
Run the Pi control logic (swing_core.py + config.py + policy.npz) against the
MuJoCo sim standing in for the robot, to catch conversion / sign / estimator
mistakes before any hardware is involved. Run from the PROJECT ROOT:

    python3 pi/sim_test.py [seconds] [drift 0|1] [gyro_bias_deg_per_s]

The "sensors" it feeds the controller are deliberately hardware-like: gyro =
true swing rate + a constant bias, swing angle only from integrating that
gyro, joint angles quantized to servo counts (and commands quantized the same
way), joint velocity by finite difference. Compare the printed peak/steady
angle with evaluate.py's (about 74 deg peak, reward ~9-10k).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import config as C
import swing_core as S
import swing_env
swing_env.DISTURBANCE_PROB_PER_STEP = 0.0   # real hardware has no random kicks; test from dead rest
from swing_env import SwingEnv

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 200.0
C.HOLD_SECONDS = float(os.environ.get('HOLD_S', '60'))
C.DRIFT_ENABLE = bool(int(sys.argv[2])) if len(sys.argv) > 2 else True
bias_dps = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
if len(sys.argv) > 5:
    C.JOINT_VEL_EMA = float(sys.argv[5])

env = SwingEnv(episode_seconds=secs + 10)
obs, _ = env.reset(seed=1)
import mujoco
env.data.qvel[env.swing_dof] = 0.0       # dead rest, no seed kick
mujoco.mj_forward(env.model, env.data)
obs = env._get_obs()
dt = env.frame_skip * env.model.opt.timestep
assert abs(dt - 1.0 / C.CONTROL_HZ) < 1e-9, (dt, C.CONTROL_HZ)

pol = S.Policy(os.path.join(os.path.dirname(__file__), "policy.npz"))
est = S.SwingEstimator()
ctl = S.SwingController(pol, est, dt, seed=3)
true_bias = np.deg2rad(bias_dps)
est.bias = true_bias - np.deg2rad(float(os.environ.get('RESID_DPS', '0')))           # as if a perfect still-hang calibration; residual drift is not modeled here

EXACT = len(sys.argv) > 4 and sys.argv[4] == 'exact'
lean_rng, knee_rng = C.LEAN_RANGE_DEG, C.KNEE_RANGE_DEG
peak = 0.0
tr = []
err = []
for k in range(int(secs / dt)):
    gyro = obs[1] + true_bias
    true_now = obs[0]
    lean_raw = S.lean_deg_to_raw(np.rad2deg(obs[2]))       # what a servo would report (quantized)
    knee_raw = S.knee_deg_to_raw(np.rad2deg(obs[4]))
    if EXACT:
        est.angle = obs[0]; est._prev_rate = obs[1] + true_bias - est.bias
    out = ctl.step(gyro, S.raw_to_lean_deg(lean_raw), S.raw_to_knee_deg(knee_raw), S.raw_to_knee_deg(knee_raw))
    lean_cmd = S.raw_to_lean_deg(S.lean_deg_to_raw(out["lean"]))
    knee_cmd = 0.5 * (S.raw_to_knee_deg(S.knee_deg_to_raw(out["knee_a"])) + S.raw_to_knee_deg(S.knee_deg_to_raw(out["knee_b"])))
    a = np.array([2 * (lean_cmd - lean_rng[0]) / (lean_rng[1] - lean_rng[0]) - 1,
                  2 * (knee_cmd - knee_rng[0]) / (knee_rng[1] - knee_rng[0]) - 1], dtype=np.float32)
    err.append(np.rad2deg(est.angle) - np.rad2deg(true_now))
    obs, r, term, trunc, info = env.step(a)
    ang = info["swing_angle_deg"]
    tr.append(ang)
    if k % 3000 == 0 or (out['mode'] == 'POLICY' and 'tb' not in globals()):
        if out['mode'] == 'POLICY' and 'tb' not in globals():
            tb = k * dt; print(f'  handover to policy at t={tb:.0f} s')
    peak = max(peak, abs(ang))
    if ctl.finished:
        print(f'  hold complete -> would go PASSIVE at t={k*dt:.0f} s (held {out["hold_s"]:.0f} s)'); break
    if term or trunc:
        print("episode ended early at t=%.0fs" % (k * dt)); break
tr = np.array(tr)
print(f"drift={'on' if C.DRIFT_ENABLE else 'off'}  gyro bias={bias_dps} deg/s  ->  peak {peak:.1f} deg, "
      f"mean |angle| over last 60 s {np.abs(tr[-int(60/dt):]).mean():.1f}, "
      f"estimator error vs true angle: max |err| {np.abs(err).max():.2f} deg, end {err[-1]:+.2f} deg")

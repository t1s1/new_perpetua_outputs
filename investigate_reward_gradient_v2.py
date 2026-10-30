"""
Follow-up to investigate_reward_gradient.py: that test translated
manual_pump.py's control law through the normalized action space, which
introduced a round-trip through _scale_to_ctrl's inverse -- ruling out
whether THAT translation, not the underlying physical model, explains the
surprisingly low 23 deg peak (vs the ~78-93 deg the project's own history
describes for this law). This version drives env.data.ctrl directly,
exactly like manual_pump.py itself does, bypassing SwingEnv.step()'s
action normalization entirely -- still using SwingEnv's underlying
model/data so the physics (current swing_robot.xml: knee-driven leg mass,
current lean range, current damping) are identical to what a trained
policy actually experiences, but with zero risk of a translation bug
in this diagnostic script itself.

Also logs the angle trajectory over time (not just the final peak) to see
whether amplitude is still climbing when the episode ends (a slow-growth
story) or has already leveled off early (a resonance/energy-balance
ceiling under current physical parameters).

Also tries one stronger-gain variant, to check whether 23 deg is a
this-specific-control-law tuning artifact or looks more like a hard
physical ceiling regardless of control law aggressiveness.
"""
import mujoco
import numpy as np

from swing_env import SwingEnv, TARGET_ANGLE_DEG

def run_trial(vel_gain, label):
    env = SwingEnv()
    obs, info = env.reset(seed=0)

    swing_dof = env.swing_dof
    swing_qpos = env.swing_qpos
    lean_act = env.lean_act
    knee_act = env.knee_act
    lean_lo, lean_hi = env.lean_ctrl_range
    knee_lo, knee_hi = env.knee_ctrl_range

    angle_trace = []  # (sim_time, peak-so-far)
    peak = 0.0
    for step_i in range(env.max_steps):
        swing_vel = env.data.qvel[swing_dof]
        swing_angle = env.data.qpos[swing_qpos]

        # Exactly manual_pump.py's law, driving ctrl directly (no
        # normalized-action round trip at all).
        env.data.ctrl[lean_act] = lean_hi * np.clip(swing_vel * vel_gain, -1, 1)
        env.data.ctrl[knee_act] = knee_hi * (-np.cos(2 * swing_angle) if abs(swing_angle) < 1.4 else 0)

        for _ in range(env.frame_skip):
            mujoco.mj_step(env.model, env.data)

        angle_deg = abs(float(np.degrees(env.data.qpos[swing_qpos])))
        peak = max(peak, angle_deg)
        if step_i % 500 == 0:
            angle_trace.append((round(env.data.time, 1), round(angle_deg, 1), round(peak, 1)))

    print(f"[{label}] vel_gain={vel_gain}: final peak={peak:.1f} deg (target={TARGET_ANGLE_DEG})")
    print(f"  trajectory (sim_time, current_angle, peak_so_far), every 500 steps (~10s):")
    for t, cur, pk in angle_trace:
        print(f"    t={t:6.1f}s  angle={cur:6.1f}  peak_so_far={pk:6.1f}")
    print()

run_trial(3.0, "original manual_pump.py gain")
run_trial(6.0, "2x gain")
run_trial(10.0, "~3.3x gain, aggressive")

"""
Follow-up to investigate_reward_gradient_v2.py, prompted by a direct
methodological challenge: that script found the hand-coded pumping law's
LEAN command saturates against the -5 deg limit 95% of the time it's
wanted (vs only 25% on the +80 deg side) and treated that as evidence the
mechanism is crippled. But real swing-pumping technique is legitimately
asymmetric (recline hard at one extreme, tuck at the bottom, not a
mirror-image motion at the other extreme) -- so measuring "how often does
each HALF-swing individually saturate" isn't a fair test on its own. The
real question is whether a COMPLETE cycle (both halves together, however
asymmetric) nets real height gain, not whether the two halves look alike.

This script detects actual swing extrema (local maxima/minima of the
swing angle, found via velocity sign changes -- the moment it's about to
reverse direction) rather than sampling on a fixed timer (which was
aliasing against the ~2.8-3.3s oscillation period in the previous
script's every-10s snapshots). Each pair of consecutive extrema (one
positive-side, one negative-side swing) is treated as one complete cycle,
and reports the actual physical height reached at each extreme, so
cycle-over-cycle net gain is visible directly -- whatever the asymmetry
between the two halves looks like.
"""
import mujoco
import numpy as np

from swing_env import SwingEnv


def height_of(angle_deg, arm_length=2.0):
    return arm_length * (1.0 - np.cos(np.deg2rad(abs(angle_deg))))


def run_and_extract_extrema(vel_gain, label):
    env = SwingEnv()
    obs, info = env.reset(seed=0)
    swing_dof, swing_qpos = env.swing_dof, env.swing_qpos
    lean_act, knee_act = env.lean_act, env.knee_act
    lean_hi = env.lean_ctrl_range[1]
    knee_hi = env.knee_ctrl_range[1]

    prev_vel = env.data.qvel[swing_dof]
    extrema = []  # (sim_time, signed angle_deg)
    for step_i in range(env.max_steps):
        swing_vel = env.data.qvel[swing_dof]
        swing_angle = env.data.qpos[swing_qpos]
        env.data.ctrl[lean_act] = lean_hi * np.clip(swing_vel * vel_gain, -1, 1)
        env.data.ctrl[knee_act] = knee_hi * (-np.cos(2 * swing_angle) if abs(swing_angle) < 1.4 else 0)
        for _ in range(env.frame_skip):
            mujoco.mj_step(env.model, env.data)
        vel_now = env.data.qvel[swing_dof]
        if prev_vel != 0 and (prev_vel > 0) != (vel_now > 0):
            extrema.append((env.data.time, float(np.degrees(env.data.qpos[swing_qpos]))))
        prev_vel = vel_now

    print(f"=== {label} (vel_gain={vel_gain}) ===")
    print(f"{len(extrema)} half-swing extrema detected over {env.data.time:.0f}s "
          f"({len(extrema)//2} complete cycles)")
    print(f"{'cycle':>5} {'t(s)':>7} {'side+':>8} {'side-':>8} {'cycle_max_deg':>14} "
          f"{'cycle_max_h(m)':>15} {'net_gain_h_vs_prev_cycle':>25}")

    prev_cycle_max_h = None
    for i in range(0, len(extrema) - 1, 2):
        t_a, a_deg = extrema[i]
        t_b, b_deg = extrema[i + 1]
        pos_extreme = a_deg if a_deg > 0 else (b_deg if b_deg > 0 else None)
        neg_extreme = a_deg if a_deg < 0 else (b_deg if b_deg < 0 else None)
        cycle_max_deg = max(abs(a_deg), abs(b_deg))
        cycle_max_h = height_of(cycle_max_deg)
        net_gain = "" if prev_cycle_max_h is None else f"{cycle_max_h - prev_cycle_max_h:+.4f}"
        cycle_idx = i // 2
        # Print every cycle for the first 10, then every 5th, to keep output
        # manageable across a ~40-cycle episode while still showing the
        # early transient (most informative part) in full.
        if cycle_idx < 10 or cycle_idx % 5 == 0:
            print(f"{cycle_idx:5d} {t_b:7.1f} "
                  f"{pos_extreme if pos_extreme is not None else float('nan'):8.2f} "
                  f"{neg_extreme if neg_extreme is not None else float('nan'):8.2f} "
                  f"{cycle_max_deg:14.2f} {cycle_max_h:15.4f} {net_gain:>25}")
        prev_cycle_max_h = cycle_max_h
    print()


run_and_extract_extrema(3.0, "original manual_pump.py gain")
run_and_extract_extrema(10.0, "aggressive gain")

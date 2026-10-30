"""
Step 2: drive it with a hand-coded rule, no learning yet.

Same model, but now a simple sign-based control law pumps the swing:
push the lean joint in the direction the swing is currently moving
(the "self-excited oscillator" idea from our design discussion), and
rotate the knee-driven leg mass in/out twice per cycle at the bottom
and top (same functional shape as the old sliding-counterweight
heuristic, just driving a rotational joint now instead of a slide).

This has no learning in it at all -- it's here so you can see the
mechanism work (or not) before RL enters the picture, and so you have
a hand-written baseline to compare a trained policy against later.

Run locally:
    pip install mujoco
    python manual_pump.py

On macOS, GUI windows must run on the main thread (a Cocoa restriction),
so MuJoCo's passive viewer needs its special launcher instead of plain
python, or the render loop can desync from this script's pacing entirely
(symptom: changing SLOWMO_FACTOR has no visible effect, motion looks like
strobing rather than smooth):
    mjpython manual_pump.py
"""
import time

import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("swing_robot.xml")
data = mujoco.MjData(model)

swing_hinge_id = model.joint("swing_hinge").id
lean_act_id = model.actuator("lean_motor").id
knee_act_id = model.actuator("knee_motor").id

# Actuators are now position (PD servo) controlled, not raw torque motors
# -- ctrl means "target position" in the joint's own units (radians,
# meters), not a normalized +-1 value. Scale this control law's output
# (still shaped the same way, still roughly +-1) by each actuator's real
# range, read from the compiled model rather than hardcoded.
lean_ctrl_scale = model.actuator_ctrlrange[lean_act_id][1]
knee_ctrl_scale = model.actuator_ctrlrange[knee_act_id][1]

# tiny seed kick so there's something to amplify from rest
data.qvel[model.joint("swing_hinge").dofadr[0]] = 0.05

# ---- playback speed, both knobs manual -----------------------------------
# FRAME_DT: how often the viewer redraws (60Hz is a normal display rate).
# SUBSTEPS_PER_FRAME: how many 5ms physics steps run between redraws.
#   Left at "real time" (frame_dt / timestep) this advances sim time at the
#   same rate as the physics model intends. Lowering it makes each redraw
#   cover less sim time, which *also* slows things down, but couples the
#   slowdown to render granularity -- fine, but SLOWMO_FACTOR below is the
#   cleaner knob if you just want "watch this more slowly" without changing
#   anything about the physics.
# SLOWMO_FACTOR: independent of the above -- stretches how long the script
#   *waits* between frames, without changing how much sim time each frame
#   covers. 1.0 = real time. 6.0 = six times slower than real time. This is
#   the one to turn up if the pumped-up swing is moving too fast to watch.
FRAME_DT = 1.0 / 60.0
SUBSTEPS_PER_FRAME = max(1, round(FRAME_DT / model.opt.timestep))
SLOWMO_FACTOR = 1.0
# ----------------------------------------------------------------------------

'''
print(f"[diag] model.opt.timestep = {model.opt.timestep}")
print(f"[diag] SUBSTEPS_PER_FRAME = {SUBSTEPS_PER_FRAME}")
print(f"[diag] target wall time / frame = {FRAME_DT * SLOWMO_FACTOR:.4f}s "
      f"(target fps = {1.0 / (FRAME_DT * SLOWMO_FACTOR):.2f})")
      '''

frame_count = 0
last_report = time.time()
sim_time_at_last_report = data.time

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        frame_start = time.time()

        for _ in range(SUBSTEPS_PER_FRAME):
            swing_vel = data.qvel[model.joint("swing_hinge").dofadr[0]]
            swing_angle = data.qpos[model.joint("swing_hinge").qposadr[0]]

            # lean: push in the direction of current swing velocity (anti-damping)
            data.ctrl[lean_act_id] = lean_ctrl_scale * np.clip(swing_vel * 3.0, -1, 1)

            # knee: pull leg in near the bottom (angle ~ 0), push out near the extremes
            data.ctrl[knee_act_id] = knee_ctrl_scale * (
                -np.cos(2 * swing_angle) if abs(swing_angle) < 1.4 else 0
            )

            mujoco.mj_step(model, data)

        viewer.sync()
        frame_count += 1

        now = time.time()
        if now - last_report >= 1.0:
            real_elapsed = now - last_report
            sim_elapsed = data.time - sim_time_at_last_report
            '''print(f"[diag] {frame_count:4d} frames/{real_elapsed:.2f}s real "
                  f"({frame_count / real_elapsed:6.1f} fps)  |  "
                  f"sim advanced {sim_elapsed:6.3f}s  |  "
                  f"sim/real ratio = {sim_elapsed / real_elapsed:6.3f}x  "
                  f"(target = {1.0 / SLOWMO_FACTOR:.3f}x)")
                  '''
            frame_count = 0
            last_report = now
            sim_time_at_last_report = data.time

        target_wall_time = FRAME_DT * SLOWMO_FACTOR
        elapsed = time.time() - frame_start
        if elapsed < target_wall_time:
            time.sleep(target_wall_time - elapsed)

"""
Step 1: just look at the thing.

Loads swing_robot.xml and opens MuJoCo's interactive viewer with no control
applied at all -- the swing just hangs under gravity. Click-drag any body
with the mouse (ctrl+click to apply a force) to nudge it and watch the
pendulum physics. This is purely to build intuition for the model before
any code drives it.

Run locally (not in a headless sandbox -- this needs a display):
    pip install mujoco
    python view_model.py

On macOS, use the mjpython launcher instead of plain python -- Cocoa
requires GUI windows on the main thread, and without it the render loop
can run desynced from this script's pacing code entirely:
    mjpython view_model.py
"""
import time

import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("swing_robot.xml")
data = mujoco.MjData(model)

# Pace rendering to a target frame rate, NOT to the physics timestep (see
# manual_pump.py for why: pacing off a 5ms physics step leaves no slack for
# viewer.sync()'s own overhead and causes fast/slow cycling).
FRAME_DT = 1.0 / 60.0
SUBSTEPS_PER_FRAME = max(1, round(FRAME_DT / model.opt.timestep))
# >1 = slower than real time, e.g. 6.0 = six times slower. See manual_pump.py.
SLOWMO_FACTOR = 6.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        frame_start = time.time()

        for _ in range(SUBSTEPS_PER_FRAME):
            mujoco.mj_step(model, data)

        viewer.sync()

        target_wall_time = FRAME_DT * SLOWMO_FACTOR
        elapsed = time.time() - frame_start
        if elapsed < target_wall_time:
            time.sleep(target_wall_time - elapsed)

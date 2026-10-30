"""
Damping-sensitivity sweep against the current swing_robot.xml, same
methodology and same motivation as investigate_mass_sensitivity.py (5.9r):
domain randomization (roadmap item 3) needs real sensitivity data to pick
ranges from, not guesses. Mass is covered; this covers the three explicit
joint damping values (swing_hinge=0.01, lean_hinge=0.05, knee_hinge=0.2),
representing real-world losses this project hasn't measured yet (pivot
bearing friction, hip/knee joint friction, servo internal friction) --
NOT MuJoCo contact friction, which has no effect in this model at all
(nothing ever contacts anything else during normal operation -- the swing
hangs well clear of the ground and the body chain doesn't self-intersect,
confirmed by grepping swing_robot.xml for any explicit geom friction/
contype/conaffinity override: there is none, only defaults, and no
<contact> activity is expected given the geometry).

Same discipline as the mass sweep: unique-anchor + bounded-regex
substitution (not hardcoded multi-line literals), every variant verified
against the compiled model (here: exact damping value readback via
model.dof_damping, not just that the XML text changed) before trusting
any ceiling result from it.

Same LIMITATION as before: this is the physical ceiling under a fixed
hand-coded control law, not a measurement of RL sensitivity -- a cheap
physics-only proxy for finding sensitivity/cliffs, not a substitute for
actually training under randomization.
"""
import re

import mujoco
import numpy as np

with open("swing_robot.xml") as f:
    BASE_XML = f.read()

BASELINE = dict(swing_damping=0.01, lean_damping=0.05, knee_damping=0.2)

ANCHORS = {
    "swing_damping": 'name="swing_hinge" type="hinge" axis="0 1 0" pos="0 0 0"',
    "lean_damping": 'name="lean_hinge" type="hinge" axis="0 1 0" pos="0 0 0"',
    "knee_damping": 'name="knee_hinge" type="hinge" axis="0 1 0"',
}


def set_damping(xml, anchor, new_value):
    count = xml.count(anchor)
    assert count == 1, f"anchor {anchor!r} appears {count} times, expected exactly 1"
    anchor_pos = xml.index(anchor)
    window = xml[anchor_pos:anchor_pos + 200]
    m = re.search(r'damping="([\d.]+)"', window)
    assert m is not None, f"no damping=\"...\" found within 200 chars after anchor {anchor!r}"
    old_start = anchor_pos + m.start(1)
    old_end = anchor_pos + m.end(1)
    return xml[:old_start] + f"{new_value}" + xml[old_end:]


def build_variant(**overrides):
    params = dict(BASELINE)
    params.update(overrides)
    xml = BASE_XML
    for key, anchor in ANCHORS.items():
        xml = set_damping(xml, anchor, params[key])
    return xml, params


JOINT_NAMES = {"swing_damping": "swing_hinge", "lean_damping": "lean_hinge", "knee_damping": "knee_hinge"}


def verify_variant(xml, params):
    model = mujoco.MjModel.from_xml_string(xml)
    for key, joint_name in JOINT_NAMES.items():
        dof = model.joint(joint_name).dofadr[0]
        actual = model.dof_damping[dof]
        expected = params[key]
        assert abs(actual - expected) < 1e-9, (
            f"{key}: expected dof_damping {expected}, compiled model has {actual}")
    return model


def find_ceiling(xml, vel_gain=10.0, seconds=900.0, label=""):
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    frame_skip = 4
    max_steps = int(seconds / (model.opt.timestep * frame_skip))

    swing_dof = model.joint("swing_hinge").dofadr[0]
    swing_qpos = model.joint("swing_hinge").qposadr[0]
    lean_act = model.actuator("lean_motor").id
    knee_act = model.actuator("knee_motor").id
    lean_hi = model.actuator_ctrlrange[lean_act][1]
    knee_hi = model.actuator_ctrlrange[knee_act][1]

    data.qvel[swing_dof] = 0.05
    mujoco.mj_forward(model, data)

    prev_vel = data.qvel[swing_dof]
    extrema = []
    for step_i in range(max_steps):
        swing_vel = data.qvel[swing_dof]
        swing_angle = data.qpos[swing_qpos]
        data.ctrl[lean_act] = lean_hi * np.clip(swing_vel * vel_gain, -1, 1)
        data.ctrl[knee_act] = knee_hi * (-np.cos(2 * swing_angle) if abs(swing_angle) < 1.4 else 0)
        for _ in range(frame_skip):
            mujoco.mj_step(model, data)
        vel_now = data.qvel[swing_dof]
        if prev_vel != 0 and (prev_vel > 0) != (vel_now > 0):
            extrema.append(float(np.degrees(data.qpos[swing_qpos])))
        prev_vel = vel_now

    cycle_maxes = [max(abs(extrema[i]), abs(extrema[i + 1])) for i in range(0, len(extrema) - 1, 2)]
    tail = cycle_maxes[int(len(cycle_maxes) * 0.8):]
    ceiling = sum(tail) / len(tail) if tail else float("nan")
    still_climbing = (tail[-1] - tail[0]) > 1.0 if len(tail) > 1 else True
    print(f"{label:38s} ceiling~{ceiling:6.2f} deg  (n_cycles={len(cycle_maxes)}, "
          f"last-cycle={cycle_maxes[-1]:6.2f}, still climbing: {still_climbing})")
    return ceiling


if __name__ == "__main__":
    print("Baseline damping:", BASELINE)
    print("All runs: same hand-coded control law, gain=10.0, 900s simulated. Only the named param changes.\n")

    base_xml, base_params = build_variant()
    verify_variant(base_xml, base_params)
    find_ceiling(base_xml, label="BASELINE (current swing_robot.xml)")

    sweeps = [
        ("swing_damping", [0.005, 0.02, 0.04]),
        ("lean_damping", [0.02, 0.1, 0.2]),
        ("knee_damping", [0.05, 0.4, 0.8]),
    ]
    for param, values in sweeps:
        for v in values:
            xml, params = build_variant(**{param: v})
            verify_variant(xml, params)
            label = f"{param} {BASELINE[param]} -> {v}"
            find_ceiling(xml, label=label)

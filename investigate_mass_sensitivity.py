"""
Mass-sensitivity sweep against the CURRENT swing_robot.xml (second physical
redesign -- seat/torso/head/lower_leg as separate bodies, 2.25m rope, -60/+90
knee range). The project's earlier mass-sensitivity investigation
(investigate_physical_params.py) targets the ARCHIVED pre-redesign geometry
(single torso body, 0.12m lower leg, 2.0m rope, -60/+60 knee) -- none of its
anchor strings exist in the current file, so its results don't transfer.
This is a fresh version of the same methodology against the live model.

Five mass locations tested independently, holding the other four at their
current (baseline) value:
  - rope_mass   (swing_arm body's capsule, currently 0.5 kg)
  - seat_mass   (seat/upper-leg unit, "the lap", currently 1.5 kg)
  - torso_mass  (torso capsule, currently 3.0 kg)
  - head_mass   (head sphere, currently 0.4 kg)
  - leg_mass    (knee-driven point mass, currently 1.5 kg)

METHODOLOGY, reused from investigate_physical_params.py on purpose (same
long-horizon hand-coded-control-law ceiling extraction, same discipline
learned from that script's own history: verify every substitution actually
took effect, don't trust silently -- that script's FIRST version used regex
against line-wrapped tags and silently no-opped). Uses UNIQUE-anchor-based
regex substitution here instead of hardcoded multi-line literal strings
(the old script's approach), specifically to avoid re-hitting that exact
whitespace-transcription failure mode -- and additionally verifies each
variant's compiled model.body_mass total against the expected delta, not
just that the XML text changed.

LIMITATION (same as the old script, worth restating): this measures the
PHYSICAL pumping ceiling under a fixed, simple hand-coded control law, not
how an RL-trained policy's performance would respond to the same change --
retraining for every variant would be far too expensive to sweep. Useful as
a cheap physics-only proxy, not a direct measurement of RL sensitivity.

Does NOT touch swing_robot.xml on disk -- read-only exploration.
"""
import re

import mujoco
import numpy as np

with open("swing_robot.xml") as f:
    BASE_XML = f.read()

BASELINE = dict(rope_mass=0.5, seat_mass=1.5, torso_mass=3.0, head_mass=0.4, leg_mass=1.5)

# Each anchor must appear EXACTLY once in the file -- asserted below --
# and the mass="..." attribute nearest after it (before the next geom/body
# tag) is the one being targeted.
ANCHORS = {
    "rope_mass": 'fromto="0 0 0  0 0 -2.25"',
    "seat_mass": 'fromto="0.06 0 0  -0.14 0 0"',
    "torso_mass": 'fromto="0 0 0  0 0 0.38"',
    "head_mass": 'name="head_geom"',
    "leg_mass": 'name="leg_mass"',
}


def set_mass(xml, anchor, new_value):
    count = xml.count(anchor)
    assert count == 1, f"anchor {anchor!r} appears {count} times, expected exactly 1"
    anchor_pos = xml.index(anchor)
    # Search a bounded window after the anchor for the next mass="..." --
    # bounded so this can't accidentally cross into a DIFFERENT geom's mass
    # attribute if this one doesn't have an explicit override nearby.
    window = xml[anchor_pos:anchor_pos + 300]
    m = re.search(r'mass="([\d.]+)"', window)
    assert m is not None, f"no mass=\"...\" found within 300 chars after anchor {anchor!r}"
    old_value_str = m.group(1)
    old_start = anchor_pos + m.start(1)
    old_end = anchor_pos + m.end(1)
    new_xml = xml[:old_start] + f"{new_value}" + xml[old_end:]
    assert new_xml != xml or float(old_value_str) == new_value, "substitution had no effect"
    return new_xml


def build_variant(**overrides):
    params = dict(BASELINE)
    params.update(overrides)
    xml = BASE_XML
    for key, anchor in ANCHORS.items():
        xml = set_mass(xml, anchor, params[key])
    return xml, params


def verify_variant(xml, params, label=""):
    """Compiles the variant and checks total body_mass differs from the
    baseline's total by exactly the sum of the intended per-parameter
    deltas -- direct confirmation the substitution changed real simulated
    mass, not just XML text, mirroring this project's established
    'verified against the compiled model' discipline."""
    model = mujoco.MjModel.from_xml_string(xml)
    total_mass = float(model.body_mass.sum())
    return total_mass


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
    print("Baseline masses:", BASELINE)
    print("All runs: same hand-coded control law, gain=10.0, 900s simulated. Only the named param changes.\n")

    base_xml, base_params = build_variant()
    base_total_mass = verify_variant(base_xml, base_params)
    print(f"baseline total body_mass: {base_total_mass:.4f} kg\n")

    find_ceiling(base_xml, label="BASELINE (current swing_robot.xml)")

    sweeps = [
        ("rope_mass", [0.25, 1.0, 2.0]),
        ("seat_mass", [0.75, 3.0]),
        ("torso_mass", [1.5, 6.0]),
        ("head_mass", [0.0, 0.8]),
        ("leg_mass", [0.75, 3.0]),
    ]
    for param, values in sweeps:
        for v in values:
            xml, params = build_variant(**{param: v})
            total_mass = verify_variant(xml, params)
            expected_delta = v - BASELINE[param]
            actual_delta = total_mass - base_total_mass
            assert abs(actual_delta - expected_delta) < 1e-6, (
                f"{param}={v}: expected total mass delta {expected_delta:.4f}, "
                f"got {actual_delta:.4f}")
            label = f"{param} {BASELINE[param]} -> {v}"
            find_ceiling(xml, label=label)

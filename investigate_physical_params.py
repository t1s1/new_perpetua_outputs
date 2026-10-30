"""
Follow-up to investigate_percycle_gain.py's finding that the current
physical model has a real energy-balance pumping ceiling around 55-56 deg,
short of the 67.5 deg reward target, even under an aggressive hand-coded
control gain and 900s of simulated time.

The user has real-world flexibility on mass distribution and dimensions
for an eventual physical build, and asked specifically about leg mass,
lever-arm length, and rope/chain length (arm_length). This sweeps each
physical parameter independently through the SAME long-horizon (900s)
per-cycle methodology and the SAME hand-coded control law/gain throughout,
so the only thing that varies between runs is the one physical parameter
under test.

FIRST VERSION of this script used regex substitution against
swing_robot.xml's text and got it wrong: the file's leg_mass geom tag is
line-wrapped (pos="..." on one line, rgba=... mass=... on the next), and
the regex assumed single-line spacing, so leg_mass/leg_offset changes
silently never took effect (caught by noticing 1x/2x/4x leg_mass all
producing IDENTICAL ceilings -- a strong tell that something in the
pipeline wasn't actually varying). This version uses exact literal
string.replace() against known substrings instead, with an assertion that
each substitution actually changed the text before trusting any result --
same "verify directly, don't trust silently" discipline as the rest of
this project.

Does NOT touch the actual swing_robot.xml file on disk -- read-only
exploration of the design space, not an applied change.
"""
import mujoco
import numpy as np

with open("swing_robot.xml") as f:
    BASE_XML = f.read()

# Exact literal substrings from the real file, used as replace() anchors.
LEG_BLOCK_OLD = (
    '<geom type="capsule" fromto="0 0 0  0 0 -0.12" size="0.015"\n'
    '                  rgba="0.3 0.3 0.3 1"/>\n'
    '            <geom name="leg_mass" type="sphere" size="0.06" pos="0 0 -0.12"\n'
    '                  rgba="0.2 0.2 0.8 1" mass="0.5"/>'
)
KNEE_DAMPING_OLD = (
    '<joint name="knee_hinge" type="hinge" axis="0 1 0"\n'
    '                   range="-60 60" damping="0.2"/>'
)
ARM_CAPSULE_OLD = '<geom type="capsule" fromto="0 0 0  0 0 -2" size="0.015" rgba="0.3 0.3 0.3 1"/>'
TORSO_POS_OLD = '<body name="torso" pos="0 0 -2">'
TORSO_MASS_OLD = (
    '<geom name="torso_geom" type="box" size="0.06 0.1 0.15" pos="0 0 0.15"\n'
    '                rgba="0.8 0.2 0.2 1" mass="1.0"/>'
)

assert LEG_BLOCK_OLD in BASE_XML, "leg block anchor not found -- file format changed"
assert KNEE_DAMPING_OLD in BASE_XML, "knee damping anchor not found"
assert ARM_CAPSULE_OLD in BASE_XML, "arm capsule anchor not found"
assert TORSO_POS_OLD in BASE_XML, "torso pos anchor not found"
assert TORSO_MASS_OLD in BASE_XML, "torso mass anchor not found"


def build_variant(leg_mass=0.5, leg_offset=0.12, knee_damping=0.2, arm_length=2.0, torso_mass=1.0):
    xml = BASE_XML
    leg_block_new = (
        f'<geom type="capsule" fromto="0 0 0  0 0 -{leg_offset}" size="0.015"\n'
        f'                  rgba="0.3 0.3 0.3 1"/>\n'
        f'            <geom name="leg_mass" type="sphere" size="0.06" pos="0 0 -{leg_offset}"\n'
        f'                  rgba="0.2 0.2 0.8 1" mass="{leg_mass}"/>'
    )
    xml2 = xml.replace(LEG_BLOCK_OLD, leg_block_new)
    if leg_mass != 0.5 or leg_offset != 0.12:
        assert xml2 != xml, "leg_mass/leg_offset substitution had no effect"
    xml = xml2

    knee_damping_new = (
        f'<joint name="knee_hinge" type="hinge" axis="0 1 0"\n'
        f'                   range="-60 60" damping="{knee_damping}"/>'
    )
    xml2 = xml.replace(KNEE_DAMPING_OLD, knee_damping_new)
    if knee_damping != 0.2:
        assert xml2 != xml, "knee_damping substitution had no effect"
    xml = xml2

    arm_new = f'<geom type="capsule" fromto="0 0 0  0 0 -{arm_length}" size="0.015" rgba="0.3 0.3 0.3 1"/>'
    xml2 = xml.replace(ARM_CAPSULE_OLD, arm_new)
    if arm_length != 2.0:
        assert xml2 != xml, "arm_length capsule substitution had no effect"
    xml = xml2

    torso_pos_new = f'<body name="torso" pos="0 0 -{arm_length}">'
    xml2 = xml.replace(TORSO_POS_OLD, torso_pos_new)
    if arm_length != 2.0:
        assert xml2 != xml, "arm_length torso-pos substitution had no effect"
    xml = xml2

    torso_mass_new = (
        f'<geom name="torso_geom" type="box" size="0.06 0.1 0.15" pos="0 0 0.15"\n'
        f'                rgba="0.8 0.2 0.2 1" mass="{torso_mass}"/>'
    )
    xml2 = xml.replace(TORSO_MASS_OLD, torso_mass_new)
    if torso_mass != 1.0:
        assert xml2 != xml, "torso_mass substitution had no effect"
    xml = xml2

    return xml


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
    print(f"{label:42s} ceiling~{ceiling:6.2f} deg  (n_cycles={len(cycle_maxes)}, "
          f"last-cycle={cycle_maxes[-1]:.2f}, still climbing: {still_climbing})")


if __name__ == "__main__":
    print("All runs: same hand-coded control law, gain=10.0, 900s simulated. Only the named param changes.")
    find_ceiling(build_variant(), label="BASELINE (current swing_robot.xml)")
    find_ceiling(build_variant(leg_mass=1.0), label="leg_mass 0.5 -> 1.0 (2x)")
    find_ceiling(build_variant(leg_mass=2.0), label="leg_mass 0.5 -> 2.0 (4x)")
    find_ceiling(build_variant(leg_offset=0.24), label="leg_offset 0.12 -> 0.24 (2x lever arm)")
    find_ceiling(build_variant(leg_mass=1.0, leg_offset=0.24), label="leg_mass 2x AND leg_offset 2x combined")
    find_ceiling(build_variant(knee_damping=0.05), label="knee_hinge damping 0.2 -> 0.05 (4x less)")
    find_ceiling(build_variant(arm_length=1.5), label="arm_length (rope/chain) 2.0 -> 1.5 (shorter)")
    find_ceiling(build_variant(arm_length=3.0), label="arm_length (rope/chain) 2.0 -> 3.0 (longer)")
    find_ceiling(build_variant(torso_mass=0.5), label="torso_mass 1.0 -> 0.5 (lighter rider)")

"""
Torso/head mass reduction vs. minimum lean/torso servo torque needed --
the INVERSE question from investigate_mass_sensitivity.py (5.9r). That
script held torque FIXED (at the current 8.1 Nm placeholder) and asked
"how does the ceiling change with mass" -- this one holds a TARGET
ceiling fixed (TARGET_ANGLE_DEG, 67.5 deg -- the actual reward target,
not a looser "healthy-looking" number) and asks "what's the minimum
torque needed to still reach it, at a given (reduced) torso+head mass".

Motivation, directly from the user: torso+head mass is a variable they
control (unlike, say, real-world air resistance), and the lean/torso
servo is both the most expensive/hardest-to-source actuator in this
build AND the one whose torque budget torso+head mass most directly
taxes (5.9r's cliff). If cutting torso/head mass buys a meaningful torque
reduction, that's a genuine, actionable cost lever -- worth knowing
BEFORE finalizing a servo purchase, not after.

METHOD: same hand-coded-control-law ceiling-extraction methodology as
every other sensitivity sweep in this project (LIMITATION, same as always:
this is a physics-only proxy under a fixed simple law, not a measurement
of what a retrained RL policy could do at each torque/mass combination --
retraining per grid cell would be far too expensive to sweep). Same
unique-anchor + bounded-regex substitution + compiled-model verification
discipline as investigate_mass_sensitivity.py, extended here to ALSO
substitute lean_motor's forcerange (a new anchor, verified against
model.actuator_forcerange, not just that the XML text changed).

THREE scenario sets, all reduction fractions applied to the current
swing_robot.xml baseline (torso=3.0 kg, head=0.4 kg):
  A. COMBINED  -- torso and head reduced together, same fraction. The
     user's actual question ("the torso/head combination").
  B. TORSO ONLY -- torso reduced, head held at baseline. Attribution: how
     much of set A's effect comes from torso specifically.
  C. HEAD ONLY -- head reduced, torso held at baseline. Attribution: the
     rest.

For each scenario, sweeps lean_motor forcerange over a 0.5 Nm grid from
3.0 to 8.1 Nm (8.1 = current baseline, known-good) and reports the ceiling
at each torque -- the minimum torque clearing TARGET_ANGLE_DEG (67.5 deg)
is this scenario's answer. Grid, not bisection, matching every other
sensitivity sweep in this project's own precedent (mass, damping, air
drag) -- bisection was only used for ACTION_RATE_WEIGHT, where each step
required real RL retraining; every point here is a cheap ~1.2s physics-only
sim (timed directly before committing to this grid size), so a fine grid
is cheap enough to just run outright.

Does NOT touch swing_robot.xml on disk -- read-only, exactly like every
prior investigate_*.py script.
"""
import re

import mujoco
import numpy as np

from investigate_mass_sensitivity import ANCHORS as MASS_ANCHORS, set_mass

with open("swing_robot.xml") as f:
    BASE_XML = f.read()

BASELINE = dict(rope_mass=0.5, seat_mass=1.5, torso_mass=3.0, head_mass=0.4, leg_mass=1.5)
TARGET_ANGLE_DEG = 67.5  # matches swing_env.py's actual reward target -- the
                         # real bar, not a looser "looks healthy" number.

LEAN_MOTOR_ANCHOR = 'name="lean_motor"'
assert BASE_XML.count(LEAN_MOTOR_ANCHOR) == 1


def set_lean_torque(xml, torque):
    anchor_pos = xml.index(LEAN_MOTOR_ANCHOR)
    window = xml[anchor_pos:anchor_pos + 200]
    m = re.search(r'forcerange="-?[\d.]+ -?[\d.]+"', window)
    assert m is not None, "no forcerange=\"...\" found near lean_motor anchor"
    old_start = anchor_pos + m.start()
    old_end = anchor_pos + m.end()
    return xml[:old_start] + f'forcerange="-{torque} {torque}"' + xml[old_end:]


def build_variant(torque, **mass_overrides):
    params = dict(BASELINE)
    params.update(mass_overrides)
    xml = BASE_XML
    for key, anchor in MASS_ANCHORS.items():
        xml = set_mass(xml, anchor, params[key])
    xml = set_lean_torque(xml, torque)
    return xml, params


def verify_variant(xml, params, torque):
    model = mujoco.MjModel.from_xml_string(xml)
    lean_act = model.actuator("lean_motor").id
    actual_torque = model.actuator_forcerange[lean_act][1]
    assert abs(actual_torque - torque) < 1e-9, (
        f"expected lean forcerange {torque}, compiled model has {actual_torque}")
    return model


def find_ceiling(xml, vel_gain=10.0, seconds=900.0):
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
    for _ in range(max_steps):
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
    return sum(tail) / len(tail) if tail else float("nan")


TORQUE_GRID = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.1]


def min_torque_for_target(torso_mass, head_mass, label=""):
    """Sweeps TORQUE_GRID (ascending) for this (torso_mass, head_mass) pair,
    returns (min_torque_clearing_target, ceiling_at_that_torque,
    ceiling_at_baseline_8.1). None for min_torque if nothing in the grid
    clears TARGET_ANGLE_DEG (i.e. even 8.1 Nm baseline fails at this mass)."""
    ceiling_at_baseline = None
    result = None
    for torque in TORQUE_GRID:
        xml, params = build_variant(torque, torso_mass=torso_mass, head_mass=head_mass)
        verify_variant(xml, params, torque)
        ceiling = find_ceiling(xml)
        if abs(torque - 8.1) < 1e-9:
            ceiling_at_baseline = ceiling
        if result is None and ceiling >= TARGET_ANGLE_DEG:
            result = (torque, ceiling)
        print(f"  {label:28s} torque {torque:4.1f} Nm -> ceiling {ceiling:6.2f} deg"
              f"{'  <-- min clearing target' if result and result[0] == torque else ''}")
    if result is None:
        return None, None, ceiling_at_baseline
    return result[0], result[1], ceiling_at_baseline


def summarize(name, rows):
    print(f"\n--- {name} summary ---")
    for pct, torso, head, min_t, ceiling, base_ceiling in rows:
        min_t_str = f"{min_t:.1f} Nm" if min_t is not None else "NONE IN GRID (even 8.1 Nm fails)"
        print(f"  -{pct:2d}%  torso={torso:.2f} head={head:.2f}  "
              f"min torque for {TARGET_ANGLE_DEG} deg: {min_t_str}  "
              f"(baseline 8.1 Nm ceiling here: {base_ceiling:.2f} deg)")


if __name__ == "__main__":
    # Runs ONE set per invocation (sys.argv[1] in {"A","B","C"}) -- splitting
    # this way, not as a design choice but a pragmatic one: each set is ~66
    # find_ceiling() calls at ~1.2s each (timed directly beforehand), and all
    # three together ran long enough to exceed this sandbox's per-call
    # timeout. Each set stands alone and prints its own summary, so nothing
    # is lost by running them as three separate invocations instead of one.
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "A"
    reductions = [0, 10, 20, 30, 40, 50]  # percent

    print(f"Baseline: torso={BASELINE['torso_mass']} kg, head={BASELINE['head_mass']} kg, "
          f"lean torque=8.1 Nm. Target ceiling: {TARGET_ANGLE_DEG} deg (swing_env.py's actual reward target).\n")

    if which == "A":
        print("=== SET A: COMBINED (torso + head reduced together) ===")
        rows = []
        for pct in reductions:
            frac = 1 - pct / 100
            torso = BASELINE["torso_mass"] * frac
            head = BASELINE["head_mass"] * frac
            label = f"A -{pct}% (torso {torso:.2f}, head {head:.2f})"
            min_t, ceiling, base_ceiling = min_torque_for_target(torso, head, label)
            rows.append((pct, torso, head, min_t, ceiling, base_ceiling))
        summarize("SET A (combined)", rows)

    elif which == "B":
        print("=== SET B: TORSO ONLY (head held at baseline 0.4 kg) ===")
        rows = []
        for pct in reductions:
            frac = 1 - pct / 100
            torso = BASELINE["torso_mass"] * frac
            head = BASELINE["head_mass"]
            label = f"B -{pct}% (torso {torso:.2f}, head {head:.2f})"
            min_t, ceiling, base_ceiling = min_torque_for_target(torso, head, label)
            rows.append((pct, torso, head, min_t, ceiling, base_ceiling))
        summarize("SET B (torso only)", rows)

    elif which == "C":
        print("=== SET C: HEAD ONLY (torso held at baseline 3.0 kg) ===")
        rows = []
        for pct in reductions:
            frac = 1 - pct / 100
            torso = BASELINE["torso_mass"]
            head = BASELINE["head_mass"] * frac
            label = f"C -{pct}% (torso {torso:.2f}, head {head:.2f})"
            min_t, ceiling, base_ceiling = min_torque_for_target(torso, head, label)
            rows.append((pct, torso, head, min_t, ceiling, base_ceiling))
        summarize("SET C (head only)", rows)

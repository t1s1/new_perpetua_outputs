"""
Air-drag sensitivity sweep. Adds MuJoCo's built-in fluid (aerodynamic
drag) model to swing_robot.xml via <option density="..." viscosity="..."/>
-- both default to 0 in the current file (confirmed: no override exists),
meaning ZERO air resistance is currently simulated at all. This is the
direct cause of 5.9u's finding that passive decay never settles (still
20+ deg amplitude after 10 simulated minutes with no drag active).

MuJoCo computes drag automatically per-body from each body's own inertia-
implied ellipsoid once density/viscosity are nonzero (the "implicit"
fluid model) -- no per-geom fluidshape annotation needed for a first,
reasoned-not-precisely-tuned pass, consistent with how every other
physical placeholder in this project has been handled.

Sweeps DENSITY only (viscosity held fixed at real air's ~1.85e-5 Pa*s --
negligible linear/viscous contribution at these speeds/sizes compared to
the density-driven blunt/quadratic drag term, which dominates). Density
is the practically controllable "how much air resistance" knob per the
user -- real hardware can always be made draggier (larger cross-section,
less streamlined materials) but not less draggy than bare air, so the
sweep is deliberately asymmetric around the real-air baseline (1.2
kg/m^3), going up several multiples rather than down.

Measures TWO things per density value, since this parameter affects both
phases this project cares about oppositely: (1) the active-pumping
ceiling (hand-coded control law, same methodology as every other
sensitivity sweep here) -- more drag costs more energy per cycle, makes
pumping harder; (2) passive-decay behavior (amplitude envelope over a
fixed window, starting from a real pumped-up state, actuators cut to
neutral) -- more drag settles faster, which is the whole reason this is
being added. Does NOT touch swing_robot.xml on disk -- read-only.
"""
import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from swing_env import SwingEnv
import evaluate

with open("swing_robot.xml") as f:
    BASE_XML = f.read()

OPTION_OLD = '<option gravity="0 0 -9.81" timestep="0.005"/>'
assert BASE_XML.count(OPTION_OLD) == 1

AIR_DENSITY = 1.2       # kg/m^3, real air at room temperature/sea level
AIR_VISCOSITY = 1.85e-5  # Pa*s, real air -- held fixed throughout, see docstring


def build_variant(density):
    new_option = (f'<option gravity="0 0 -9.81" timestep="0.005" '
                  f'density="{density}" viscosity="{AIR_VISCOSITY}"/>')
    xml = BASE_XML.replace(OPTION_OLD, new_option)
    assert xml != BASE_XML or density == 0.0
    return xml


def verify_variant(xml, density):
    model = mujoco.MjModel.from_xml_string(xml)
    assert abs(model.opt.density - density) < 1e-9, (
        f"expected opt.density {density}, compiled model has {model.opt.density}")
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
    print(f"{label:34s} ceiling~{ceiling:6.2f} deg  (n_cycles={len(cycle_maxes)})")
    return ceiling


def passive_decay_amplitude(density, pump_seconds=200.0, decay_seconds=300.0):
    """Returns amplitude (deg) at the START and END of the decay window,
    using the ACTUAL trained policy to pump up first (not the hand-coded
    law) -- this half of the test is about the passive phase specifically,
    which only matters once the policy has actually gotten the swing
    going for real."""
    xml = build_variant(density)
    model = verify_variant(xml, density)
    vec_env = DummyVecEnv([lambda: SwingEnv(episode_seconds=pump_seconds + decay_seconds + 10)])
    vec_env = VecNormalize.load(evaluate.BEST_VECNORM_PATH, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    policy = PPO.load(evaluate.BEST_MODEL_PATH)
    raw_env = vec_env.venv.envs[0]
    raw_env.model = model
    raw_env.data = mujoco.MjData(model)

    obs = vec_env.reset()
    env_step_seconds = raw_env.frame_skip * model.opt.timestep
    pump_steps = int(pump_seconds / env_step_seconds)
    for _ in range(pump_steps):
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

    data = raw_env.data
    data.ctrl[raw_env.lean_act] = 0.0
    data.ctrl[raw_env.knee_act] = 0.0
    decay_steps = int(decay_seconds / env_step_seconds)
    prev_vel = data.qvel[raw_env.swing_dof]
    extrema = []
    for i in range(decay_steps):
        for _ in range(raw_env.frame_skip):
            mujoco.mj_step(model, data)
        vel_now = data.qvel[raw_env.swing_dof]
        if prev_vel != 0 and (prev_vel > 0) != (vel_now > 0):
            extrema.append(abs(np.degrees(data.qpos[raw_env.swing_qpos])))
        prev_vel = vel_now
    return extrema[0] if extrema else float("nan"), extrema[-1] if extrema else float("nan")


if __name__ == "__main__":
    print("=== active-pumping ceiling vs air density (hand-coded law, 900s) ===")
    for density in [0.0, 0.6, 1.2, 2.4, 4.8, 9.6]:
        xml = build_variant(density)
        verify_variant(xml, density)
        find_ceiling(xml, label=f"density {density:.1f} kg/m^3 (real air = 1.2)")

    print()
    print("=== passive-decay amplitude, start vs end of a 300s window ===")
    print("(pumped up 200s with the TRAINED policy first -- not retrained for drag, expect")
    print(" some ceiling cost baked in here too, same policy used for every density value)")
    for density in [0.0, 0.6, 1.2, 2.4, 4.8]:
        start_amp, end_amp = passive_decay_amplitude(density)
        print(f"density {density:4.1f} kg/m^3: amplitude {start_amp:6.2f} deg -> {end_amp:6.2f} deg over 300s")

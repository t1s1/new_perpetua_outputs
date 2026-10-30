"""
Can swing_angle be recovered from an onboard IMU (accelerometer + gyro)
instead of a direct angle sensor? User confirmed a rotary encoder at the
pivot (the "obvious" direct sensor, README section 7's sensing note)
isn't practical for the real build. This tests the alternative directly
against simulation ground truth rather than reasoning about it in the
abstract.

WHY THE SEAT, NOT THE TORSO: the seat body has NO joint of its own --
it's rigidly bolted to the rope's end (swing_robot.xml). That means the
seat's own rotation is EXACTLY the swing_hinge angle, with zero
contribution from lean/knee motion, and it stays at a fixed, known radius
(2.25m, ROPE_LENGTH) from the pivot at all times. An IMU on the seat
sees a clean single-axis, fixed-radius pendulum -- an IMU on the torso
would additionally pick up lean_hinge/knee_hinge-driven accelerations,
a strictly harder reconstruction problem. Tests the easier, more
realistic-to-mount case first.

METHOD:
1. Add a <site> + MuJoCo <accelerometer>/<gyro> sensor pair to the seat
   body in a COPY of swing_robot.xml (read-only, doesn't touch the file
   on disk) -- uses MuJoCo's own physics for the synthetic sensor
   readings (includes gravity reaction, centripetal, and tangential
   acceleration correctly) rather than hand-deriving the formula, which
   is exactly the kind of thing this project's own history (investigate_
   physical_params.py's regex incident) warns against trusting without
   verification.
2. Roll out best_model deterministic, recording ground-truth swing_angle
   (qpos) alongside the synthetic accelerometer/gyro readings at every
   step.
3. Reconstruct swing_angle from ONLY the synthetic sensor stream (as a
   real onboard IMU would see it) via a model-based correction: at each
   step, subtract the KNOWN centripetal (omega^2 * r) and estimated
   tangential (r * dOmega/dt, omega and its derivative both from the
   gyro trace only, not qacc) terms from the raw accelerometer reading to
   isolate gravity, then take the gravity vector's angle in the swing
   plane. Also reports a gyro-integration-only estimate (drifts) and a
   simple complementary-filtered blend of the two, for comparison.
4. Compares all three against ground truth, in degrees, over a full
   episode.
"""
import re

import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from swing_env import SwingEnv
import evaluate

with open("swing_robot.xml") as f:
    BASE_XML = f.read()

# Insert a site + accelerometer/gyro pair on the seat body, at its local
# origin (the rope attachment point -- a defensible, realistic IMU mount
# location: rigid seat structure, not out on a limb). Anchored on the
# seat's own opening tag, unique in the file.
SEAT_OPEN = '<body name="seat" pos="0 0 -2.25">'
assert BASE_XML.count(SEAT_OPEN) == 1, "seat body anchor not found or not unique"
SEAT_WITH_SITE = SEAT_OPEN + '\n          <site name="imu_site" pos="0 0 0"/>'
XML = BASE_XML.replace(SEAT_OPEN, SEAT_WITH_SITE)
assert XML != BASE_XML, "site insertion had no effect"

SENSOR_BLOCK = (
    "  <sensor>\n"
    '    <accelerometer name="imu_accel" site="imu_site"/>\n'
    '    <gyro name="imu_gyro" site="imu_site"/>\n'
)
assert "  <sensor>\n" in XML, "sensor block anchor not found"
XML = XML.replace("  <sensor>\n", SENSOR_BLOCK, 1)

ROPE_LENGTH = 2.25  # matches swing_robot.xml -- the seat's fixed, KNOWN
                     # radius from the pivot; a real build would know this
                     # from its own construction, not something the IMU
                     # needs to estimate.

model = mujoco.MjModel.from_xml_string(XML)
accel_sensor_id = model.sensor("imu_accel").id
gyro_sensor_id = model.sensor("imu_gyro").id
accel_adr = model.sensor_adr[accel_sensor_id]
gyro_adr = model.sensor_adr[gyro_sensor_id]

vec_env = DummyVecEnv([lambda: SwingEnv(episode_seconds=240.0)])
vec_env = VecNormalize.load(evaluate.BEST_VECNORM_PATH, vec_env)
vec_env.training = False
vec_env.norm_reward = False
policy = PPO.load(evaluate.BEST_MODEL_PATH)
raw_env = vec_env.venv.envs[0]

# Swap the raw env's compiled model/data for our sensor-augmented version,
# keeping everything else (frame_skip, reset logic, reward, etc.)
# untouched -- only adding sensors, not changing any dynamics.
raw_env.model = model
raw_env.data = mujoco.MjData(model)

obs = vec_env.reset()
true_angle_deg, accel_readings, gyro_readings, times = [], [], [], []
env_step_seconds = raw_env.frame_skip * model.opt.timestep
done = False
step_count = 0
while not done and step_count < 12000:
    action, _ = policy.predict(obs, deterministic=True)
    obs, reward, done_arr, info = vec_env.step(action)
    done = bool(done_arr[0])
    if done:
        break
    step_count += 1
    true_angle_deg.append(float(np.degrees(raw_env.data.qpos[raw_env.swing_qpos])))
    accel_readings.append(raw_env.data.sensordata[accel_adr:accel_adr + 3].copy())
    gyro_readings.append(raw_env.data.sensordata[gyro_adr:gyro_adr + 3].copy())
    times.append(step_count * env_step_seconds)

true_angle_deg = np.array(true_angle_deg)
accel = np.array(accel_readings)   # (N, 3), sensor-local frame
gyro = np.array(gyro_readings)     # (N, 3), sensor-local frame
times = np.array(times)
dt = env_step_seconds
N = len(times)
print(f"captured {N} steps ({times[-1]:.1f}s)")

# Swing rotates about world/local Y -- gyro's Y component IS the swing's
# own angular velocity directly (confirmed below against ground truth,
# not assumed).
omega_y = gyro[:, 1]
true_omega_deg_s = np.degrees(np.gradient(true_angle_deg, dt))
gyro_rmse = np.sqrt(np.mean((np.degrees(omega_y) - true_omega_deg_s) ** 2))
print(f"gyro Y vs true angular velocity (finite-diff of qpos) RMSE: {gyro_rmse:.3f} deg/s "
      f"-- should be ~0 (exact rigid-body relationship, only finite-diff noise on the true side)")

# --- Estimate 1: pure gyro integration (drifts) ---
gyro_integrated_deg = np.degrees(np.cumsum(omega_y) * dt) + true_angle_deg[0]

# --- Estimate 2: accelerometer-only, with KNOWN centripetal + estimated
# tangential (from differencing the GYRO trace, not qacc -- a real IMU
# only has accel+gyro, this must use only what's actually measurable) ---
alpha_y = np.gradient(omega_y, dt)  # angular acceleration estimate, from gyro only
# In the seat's local frame, X = along the rope-perpendicular seat axis,
# Z = along the rope. Centripetal accel points from seat TOWARD the pivot
# (local +Z, since seat sits at rope's far end); tangential is
# perpendicular to the radius, in the swing plane (local X).
centripetal_local = np.stack([np.zeros(N), np.zeros(N), omega_y ** 2 * ROPE_LENGTH], axis=1)
tangential_local = np.stack([-alpha_y * ROPE_LENGTH, np.zeros(N), np.zeros(N)], axis=1)
gravity_est_local = accel - centripetal_local - tangential_local
# swing angle = angle of the local gravity vector's (X, Z) components
# relative to straight down -- matches swing_hinge's own convention
# (0 = hanging straight down).
accel_angle_deg = np.degrees(np.arctan2(gravity_est_local[:, 0], gravity_est_local[:, 2]))

# --- Estimate 3: simple complementary filter (98% gyro-integrated step,
# 2% pulled toward the accelerometer-only estimate each step -- standard
# form, not tuned) ---
comp_est = np.zeros(N)
comp_est[0] = true_angle_deg[0]
BETA = 0.02
for i in range(1, N):
    gyro_step = comp_est[i - 1] + np.degrees(omega_y[i]) * dt
    comp_est[i] = (1 - BETA) * gyro_step + BETA * accel_angle_deg[i]

def report(label, est):
    err = est - true_angle_deg
    rmse = np.sqrt(np.mean(err ** 2))
    max_err = np.max(np.abs(err))
    print(f"{label:32s} RMSE {rmse:6.2f} deg, max abs err {max_err:6.2f} deg")

print()
report("gyro-integration only (drifts)", gyro_integrated_deg)
report("accelerometer-only (corrected)", accel_angle_deg)
report("complementary filter (both)", comp_est)

# Break down accelerometer-only error by swing speed -- worst case should
# be near the bottom of the arc (highest omega, largest centripetal term
# to correct for), best case near the turning points.
speed_pct = np.percentile(np.abs(true_omega_deg_s), [25, 75])
slow_mask = np.abs(true_omega_deg_s) <= speed_pct[0]
fast_mask = np.abs(true_omega_deg_s) >= speed_pct[1]
print()
print(f"accelerometer-only error, slow moments (near turning points): "
      f"{np.sqrt(np.mean((accel_angle_deg[slow_mask] - true_angle_deg[slow_mask])**2)):.2f} deg RMSE")
print(f"accelerometer-only error, fast moments (near bottom of arc): "
      f"{np.sqrt(np.mean((accel_angle_deg[fast_mask] - true_angle_deg[fast_mask])**2)):.2f} deg RMSE")

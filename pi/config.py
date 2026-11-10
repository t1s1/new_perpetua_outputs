"""
All hardware-specific numbers for the Pi in ONE place. Edit this file, not the code.
Items marked  # SET ME  must be checked against your actual hardware.
Sim-side conventions (README 5.9ae): lean + = reclining, knee + = extending,
swing + = swinging in the direction the rider faces; angles in the sim's degrees.
"""

# ---- servo bus (FE-URT-1 adapter) ----
SERIAL_PORT = "/dev/ttyACM0"      # SET ME: `ls /dev/ttyUSB*` on the Pi
BAUDRATE = 1_000_000
ID_LEAN = 1                       # SET ME: the IDs you assigned in MotorLab
ID_KNEE_A = 2                     # SET ME: the two knees are interchangeable
ID_KNEE_B = 3                     # SET ME

# ---- joint mapping: raw = ZERO_RAW + SIGN * COUNTS_PER_DEG * sim_degrees ----
COUNTS_PER_DEG = 4096 / 360.0     # 11.378
ZERO_RAW = 2048                   # raw count at the sim zero pose (set with MotorLab "Set here")
SIGN_LEAN = -1                    # as-built: a positive raw jog leans FORWARD, sim + is reclining
SIGN_KNEE = -1                    # as-built: a positive raw jog curls UNDER, sim + is extending
LEAN_RANGE_DEG = (-5.0, 80.0)     # must equal the trained ctrlrange in swing_robot.xml
KNEE_RANGE_DEG = (-45.0, 90.0)    # trained ctrlrange after the 5.9ae change

# ---- servo settings written at every start (RAM, so they must be re-sent after each power-up) ----
TORQUE_LIMIT_LEAN = 870           # raw 0-1000 = 0-100% of stall; ~8.1 N*m cap at 7.4 V
TORQUE_LIMIT_KNEE = 600           # each knee (~5.6 N*m). Was 290 (the sim's 5.4 N*m split over two servos), but the loaded knee could not lift the leg quickly at 290; 600 from the loaded step tests (README 5.9ai); sim check: peak 76.6 -> ~80 deg, no collapse
SERVO_SPEED = 3000                # raw steps/s; above the rated no-load speed, so effectively "max"
SERVO_ACC = 150                   # 0-254, units of 100 steps/s^2; lower = gentler/less robotic. TUNE.

# ---- IMU (MPU-6050 / GY-521 on I2C bus 1) ----
IMU_ADDR = 0x68
GYRO_AXIS = "y"                   # sensor axis along the hinge (rider's right), per README mounting note
GYRO_SIGN = +1                    # CONFIRMED by the bench rock test: front-edge-up (forward swing) reads + on the Y gyro
GYRO_BIAS_CAL_S = 3.0             # still-hang time at start used to measure gyro bias

# ---- control loop ----
CONTROL_HZ = 50                   # = training: 4 physics steps x 0.005 s
JOINT_VEL_EMA = 1.0               # 1.0 = plain finite difference. Tested in the sim: 0.5 smoothing raised the peak 73 -> 77 deg (lag)

# ---- organic knee drift (README 5.9ad): independent per knee, slow ----
DRIFT_ENABLE = True
DRIFT_MAX_DEG = 5.0               # hard clip on each knee's offset
DRIFT_TAU_S = 5.0                 # correlation time
DRIFT_STD_DEG = 2.5               # stationary std before clipping

# ---- safety (run ends and servos RELAX immediately) ----
MAX_SWING_DEG = 90.0              # training stops/penalizes beyond ~75; 90 is a generous hard stop
MAX_COMM_FAILS = 5                # consecutive failed bus/IMU reads

# ---- cycle (README 5.9u): run -> passive settle -> re-zero -> repeat ----
HOLD_SECONDS = 240                # time spent NEAR THE TARGET swing (not pushing higher, not letting it fall) before going passive. PLACEHOLDER: adjust.
HOLD_MIN_DEG = 55.0               # the swing counts as 'at target' while its recent peak is at least this (sim target 62.5, policy settles ~72-75)
HOLD_WINDOW_S = 5.0               # 'recent peak' = largest |angle| over this window (about one swing period)
ACTIVE_MAX_SECONDS = 900          # build-up + hold may not exceed this; if the target is never reached it is treated as a fault
REST_RATE_DPS = 20.0              # "stopped" = |gyro rate| below this ...
REST_HOLD_S = 12.0                # ... continuously for this long (README 5.9u: 10-15 s)
REST_CAL_STD_DPS = 0.5            # then wait for the gyro to be truly quiet (std below this over GYRO_BIAS_CAL_S) before re-zeroing
MAX_FAST_FAULTS = 3               # this many runs in a row ending in a fault within FAST_FAULT_S -> give up and relax
FAST_FAULT_S = 30.0

# ---- bootstrap from dead rest (found in the sim: the policy alone stalls at ~4 deg without an outside kick) ----
BOOT_ENABLE = True                # self-excited start: lean follows the swing velocity until the swing is big enough
BOOT_HYST_DPS = 0.5               # swing rate must pass +/- this to flip the lean (noise std is ~0.03 dps; 1 dps worked in the sim, 2.9 dps did not)
BOOT_AMP = 0.6                    # lean action amplitude during the bootstrap (1.0 = full travel; 0.35 also works, slower)
BOOT_STALL_S = 3.0                # no swing-rate crossing for this long -> flip the lean anyway (some mass combinations never cross on their own)
BOOT_HANDOVER_DEG = 10.0          # hand control to the policy once the swing has reached this
BOOT_TIMEOUT_S = 150.0            # not there by now -> fault. Most likely a wrong GYRO_SIGN or a stuck joint.

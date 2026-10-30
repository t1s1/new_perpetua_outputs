# Pi setup and bring-up order

Copy the `pi/` folder to the Pi (it includes `policy.npz`). Needs only numpy, pyserial, smbus2, feetech-servo-sdk:

    sudo raspi-config        # Interface Options -> I2C -> enable, reboot
    pip3 install numpy pyserial smbus2 feetech-servo-sdk

Edit `config.py` first: serial port, the three servo IDs, and (after the bench check) `GYRO_SIGN`.

Bring-up order (each step only if the previous one passed):

1. `python3 ../mpu6050_check.py static|live|drift` -- IMU alive, which axis is which, **GYRO_SIGN** (push the seat so it swings in the direction the rider faces; that gyro rate should be positive, otherwise set `GYRO_SIGN = -1`). Drift must be under ~0.02 deg/s (marginal 0.06, bad above).
2. `python3 servo_check.py` -- all three answer; stored travel limits sane; torque limits; move joints by hand and confirm the sim-degree readout signs and zero.
3. `python3 servo_check.py --apply-torque-limits`, power-cycle the servos, run `servo_check.py` again (what persists).
4. `python3 servo_step_test.py knee 0 60` (and `lean 0 40`), unloaded then loaded.
5. `python3 run_swing.py --dry-run` -- torque stays off; move the joints/seat by hand and watch the numbers.
6. `python3 run_swing.py --cycles 1` -- first powered run. Keep clear of the seat; Ctrl+C relaxes all servos (torso and legs drop).

Cycle: ACTIVE (self-excited start, handover to the policy at 10 deg, build up, then HOLD_SECONDS spent near target; `HOLD_SECONDS` in config.py is a placeholder to adjust) -> PASSIVE (zero pose held, waits until the swing has stopped) -> re-zero gyro -> repeat. Any error in ACTIVE also goes to PASSIVE and the cycle restarts; three quick faults in a row, or a bus failure while PASSIVE, relaxes the servos and exits.

Desk test with no hardware: `python3 pi/sim_test.py 240 1 0.05` from the project root (needs mujoco and the project files).
Re-export the policy after any retraining: `python3 pi/export_policy.py` from the project root, then copy `pi/policy.npz`.

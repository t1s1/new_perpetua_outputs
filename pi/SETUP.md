# Pi setup and bring-up order

Copy the `pi/` folder to the Pi (it includes `policy.npz`). Needs only numpy, pyserial, smbus2, feetech-servo-sdk:

    sudo raspi-config        # Interface Options -> I2C -> enable, reboot
    # (pip is blocked system-wide on current Pi OS: use the virtual environment steps in "Fresh Raspberry Pi OS" below)

Edit `config.py` first: serial port, the three servo IDs, and (after the bench check) `GYRO_SIGN`.

Bring-up order (each step only if the previous one passed):

1. `python3 mpu6050_check.py static|live|drift|rock` -- IMU alive, which axis is which, **GYRO_SIGN** (`rock`: tilt the board front-edge-UP as the seat does in a forward swing, then back to level; Y should integrate positive, otherwise set `GYRO_SIGN = -1`. Done on the user's board: Y, +1). Drift must be under ~0.02 deg/s (marginal 0.06, bad above).
2. `python3 servo_scan.py` (finds the serial port and servo IDs; `servo_scan.py <port> watch` shows live positions so you can tell which ID is the lean and which are the knees -- put them in `config.py`), then `python3 servo_check.py` -- all three answer; stored travel limits sane; torque limits; move joints by hand and confirm the sim-degree readout signs and zero.
3. `python3 servo_check.py --apply-torque-limits`, power-cycle the servos, run `servo_check.py` again (what persists).
4. `python3 servo_step_test.py knee 0 60` (and `lean 0 40`), unloaded then loaded.
5. `python3 run_swing.py --dry-run` -- torque stays off; move the joints/seat by hand and watch the numbers.
6. `python3 run_swing.py --cycles 1` -- first powered run. Keep clear of the seat; Ctrl+C relaxes all servos (torso and legs drop).

**Interim instruction:** the gyro's zero point moves with its temperature, and a cold board drifted about 0.04 deg/s in the first dry run (a warm one drifted ~0.0003). Until the software handles warm-up itself (planned for the self-start work: wait for the chip temperature to settle, then calibrate), power the Pi on and wait about 10 minutes before the first run. `python3 mpu6050_check.py warm` records how long it really takes.

Cycle: ACTIVE (self-excited start, handover to the policy at 10 deg, build up, then HOLD_SECONDS spent near target; `HOLD_SECONDS` in config.py is a placeholder to adjust) -> PASSIVE (zero pose held, waits until the swing has stopped) -> re-zero gyro -> repeat. Any error in ACTIVE also goes to PASSIVE and the cycle restarts; three quick faults in a row, or a bus failure while PASSIVE, relaxes the servos and exits.

Desk test with no hardware: `python3 pi/sim_test.py 240 1 0.05` from the project root (needs mujoco and the project files).
Re-export the policy after any retraining: `python3 pi/export_policy.py` from the project root, then copy `pi/policy.npz`.

## Fresh Raspberry Pi OS (Bookworm or newer) install

    sudo apt update && sudo apt install -y python3-venv python3-numpy i2c-tools tmux
    sudo raspi-config nonint do_i2c 0                  # enable I2C (or raspi-config -> Interface Options)
    sudo usermod -aG dialout,i2c $USER                 # serial adapter + I2C access without sudo; log out and in again
    python3 -m venv --system-site-packages ~/swing-env # pip is blocked system-wide on current Pi OS, so use a venv
    source ~/swing-env/bin/activate
    pip install pyserial smbus2 feetech-servo-sdk
    python3 -c "import numpy, serial, smbus2, scservo_sdk; print('imports ok')"

Copy the folder from your computer: `scp -r pi <user>@<pi-hostname>.local:~/` (or use a USB stick). Each new terminal needs `source ~/swing-env/bin/activate`. Run `run_swing.py` inside `tmux` so a dropped SSH session cannot interrupt it (a dropped session now relaxes the servos, like Ctrl+C). Not needed on the Pi: torch, stable-baselines3, mujoco.

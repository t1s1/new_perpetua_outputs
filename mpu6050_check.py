"""
Bench check for an MPU-6050 (GY-521 board) on a Raspberry Pi, over I2C.

Purpose: confirm the board is wired and alive, see which axis is which, learn
the sign of each gyro axis, and measure how much a still gyro drifts when
integrated -- the same quantity README 5.9t/5.9x modeled (~0.32 deg RMSE over
240 s). NOT the final robot code; just a test tool. Untested on real hardware
as written (no board available in the sandbox where this was drafted) -- if
anything misbehaves, send me the error text.

Setup (once, on the Pi):
    sudo raspi-config        # Interface Options -> I2C -> enable, then reboot
    sudo apt install -y i2c-tools python3-pip
    pip3 install smbus2
    i2cdetect -y 1           # should show 68 in the grid (address 0x68)

Wiring (GY-521 -> Pi 40-pin header): VCC -> pin 1 (3.3V), GND -> pin 6,
SDA -> pin 3, SCL -> pin 5. Leave AD0 unconnected (address 0x68).

Usage:
    python3 mpu6050_check.py static   # hold still: gravity should be ~1 g on ONE axis
    python3 mpu6050_check.py live     # live readout; rotate the board by hand
    python3 mpu6050_check.py drift    # leave perfectly still for 60 s; reports integrated drift

Ranges set below: gyro +/-500 deg/s (swing peaks roughly 150 deg/s), accel
+/-4 g (a 70 deg swing sees ~2.3 g at the bottom, beyond the +/-2 g default).
"""
import sys
import time

from smbus2 import SMBus

ADDR = 0x68
PWR_MGMT_1, CONFIG, GYRO_CONFIG, ACCEL_CONFIG, DATA = 0x6B, 0x1A, 0x1B, 0x1C, 0x3B
GYRO_LSB_PER_DPS = 65.5      # for +/-500 deg/s
ACCEL_LSB_PER_G = 8192.0     # for +/-4 g


def to_signed(hi, lo):
    v = (hi << 8) | lo
    return v - 65536 if v & 0x8000 else v


def setup(bus):
    bus.write_byte_data(ADDR, PWR_MGMT_1, 0x00)   # wake up
    time.sleep(0.1)
    bus.write_byte_data(ADDR, CONFIG, 0x03)       # low-pass filter ~44 Hz
    bus.write_byte_data(ADDR, GYRO_CONFIG, 0x08)  # +/-500 deg/s
    bus.write_byte_data(ADDR, ACCEL_CONFIG, 0x08) # +/-4 g


def read(bus):
    b = bus.read_i2c_block_data(ADDR, DATA, 14)
    ax, ay, az = (to_signed(b[i], b[i + 1]) / ACCEL_LSB_PER_G for i in (0, 2, 4))
    gx, gy, gz = (to_signed(b[i], b[i + 1]) / GYRO_LSB_PER_DPS for i in (8, 10, 12))
    return (ax, ay, az), (gx, gy, gz)


def average(bus, seconds):
    n, a, g = 0, [0.0] * 3, [0.0] * 3
    end = time.time() + seconds
    while time.time() < end:
        (ax, ay, az), (gx, gy, gz) = read(bus)
        for i, v in enumerate((ax, ay, az)):
            a[i] += v
        for i, v in enumerate((gx, gy, gz)):
            g[i] += v
        n += 1
        time.sleep(0.005)
    return [x / n for x in a], [x / n for x in g]


def main(mode):
    with SMBus(1) as bus:
        who = bus.read_byte_data(ADDR, 0x75)
        print(f"WHO_AM_I = 0x{who:02X} (expect 0x68; some clones differ but still work)")
        setup(bus)
        if mode == "static":
            print("Hold the board completely still for 3 s...")
            a, g = average(bus, 3.0)
            print("accel (g):      X %+.3f  Y %+.3f  Z %+.3f   <- ONE axis should be ~ +/-1.0, others ~0" % tuple(a))
            print("gyro (deg/s):   X %+.3f  Y %+.3f  Z %+.3f   <- all near 0 (small constant = bias)" % tuple(g))
        elif mode == "live":
            print("Rotate slowly about one board axis at a time; Ctrl+C to stop.")
            while True:
                a, g = read(bus)
                print("accel g  X %+.2f Y %+.2f Z %+.2f | gyro d/s  X %+7.1f Y %+7.1f Z %+7.1f" % (*a, *g), end="\r")
                time.sleep(0.05)
        elif mode == "drift":
            print("Calibrating bias (3 s, keep still)...")
            _, bias = average(bus, 3.0)
            print("Measuring for 60 s, keep the board still...")
            ang, t_prev, t0 = [0.0] * 3, time.time(), time.time()
            while time.time() - t0 < 60.0:
                _, g = read(bus)
                now = time.time()
                for i in range(3):
                    ang[i] += (g[i] - bias[i]) * (now - t_prev)
                t_prev = now
                time.sleep(0.005)
            print("integrated drift after 60 s (deg):  X %+.3f  Y %+.3f  Z %+.3f" % tuple(ang))
            rate = max(abs(x) for x in ang) / 60.0
            verdict = "good" if rate < 0.02 else "marginal" if rate < 0.06 else "BAD"
            print(f"worst axis: {rate:.3f} deg/s -> {verdict}.  Sim test (pi/sim_test.py): a residual gyro bias of 0.05 deg/s "
                  f"still swings (peak 81, ~9 deg angle error after 200 s); 0.2 deg/s collapses the policy. "
                  f"Let the board warm up ~5 min first (bias moves with temperature). README 5.9t/5.9x assumed ~0.32 deg RMSE over 240 s.")
        else:
            print(__doc__)


if __name__ == "__main__":
    try:
        main(sys.argv[1] if len(sys.argv) > 1 else "")
    except KeyboardInterrupt:
        print()

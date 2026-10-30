"""
Run the trained swing policy on the real robot, as a repeating cycle (README 5.9u):

    ACTIVE (policy builds the swing, then holds it at target for HOLD_SECONDS) -> PASSIVE (servos hold the zero pose,
    no pumping) until the swing has settled -> RE-ZERO (gyro bias + angle) -> ACTIVE ...

Any error during ACTIVE (swing angle too large, repeated read failures) also goes to
PASSIVE and the cycle then restarts. If MAX_FAST_FAULTS runs in a row fault within
FAST_FAULT_S, or the servos can't be commanded at all, it gives up and RELAXES the
servos (torque off: torso and legs will drop). Ctrl+C always relaxes immediately.

    python3 run_swing.py --dry-run      # reads sensors, prints what it WOULD command, touches no servo
    python3 run_swing.py                # real run
    python3 run_swing.py --cycles 1     # stop after one full cycle

Logs to logs/run_<timestamp>.csv. Auto-start at power-up is not set up yet.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import swing_core as S
from hardware import BusError, Imu, ServoBus

IDS = [C.ID_LEAN, C.ID_KNEE_A, C.ID_KNEE_B]
ZERO_SPEED, ZERO_ACC = 400, 20          # gentle moves to the zero pose
DT = 1.0 / C.CONTROL_HZ


def zero_goals():
    return {C.ID_LEAN: S.lean_deg_to_raw(0.0), C.ID_KNEE_A: S.knee_deg_to_raw(0.0), C.ID_KNEE_B: S.knee_deg_to_raw(0.0)}


def go_to_zero_pose(bus, dry):
    if dry:
        return
    bus.write_goals(zero_goals(), ZERO_SPEED, ZERO_ACC)


def wait_at_zero(bus, dry, timeout=8.0):
    if dry:
        return
    g, t0 = zero_goals(), time.time()
    while time.time() - t0 < timeout:
        p = bus.read_positions()
        if all(abs(p[i] - g[i]) < 12 for i in IDS):
            return
        time.sleep(0.1)
    raise BusError(f"did not reach zero pose in {timeout:.0f} s: at {p}, wanted {g}")


def setup_servos(bus, dry):
    for i in IDS:
        if bus.ping(i) is None:
            raise BusError(f"servo id {i} does not answer")
    print("all servos answer; positions:", bus.read_positions())
    if dry:
        return
    for i in IDS:
        bus.torque(i, False)
        bus.set_torque_limit(i, C.TORQUE_LIMIT_LEAN if i == C.ID_LEAN else C.TORQUE_LIMIT_KNEE)
        print(f"  id {i}: torque limit now {bus.read2(i, 48)}")
    bus.write_goals(bus.read_positions(), ZERO_SPEED, ZERO_ACC)      # goal = where it is, so torque-on doesn't jump
    for i in IDS:
        bus.torque(i, True)
    print("moving to zero pose ...")
    go_to_zero_pose(bus, dry)
    wait_at_zero(bus, dry)


def settle_and_rezero(bus, imu, est, dry):
    """PASSIVE: hold zero pose, wait for stillness, then measure bias and zero the angle."""
    print("PASSIVE: holding zero pose, waiting for the swing to settle ...")
    go_to_zero_pose(bus, dry)
    rest, t0, last_print = S.RestDetector(), time.time(), 0.0
    while True:
        r = imu.read_gyro_rad_s() - est.bias
        if rest.update(r, DT):
            break
        if time.time() - t0 - last_print > 30:
            last_print = time.time() - t0
            print(f"  still swinging after {last_print:.0f} s (rate {np.rad2deg(r):+.1f} deg/s)")
        time.sleep(DT)
    print(f"  rate < {C.REST_RATE_DPS} deg/s for {C.REST_HOLD_S:.0f} s; waiting for the gyro to be fully quiet ...")
    while True:
        rates = []
        t1 = time.time()
        while time.time() - t1 < C.GYRO_BIAS_CAL_S:
            rates.append(imu.read_gyro_rad_s())
            time.sleep(DT)
        if np.rad2deg(np.std(rates)) < C.REST_CAL_STD_DPS:
            break
    est.bias = float(np.mean(rates))
    est.angle = 0.0
    est._prev_rate = None
    print(f"  re-zeroed: bias {np.rad2deg(est.bias):+.3f} deg/s, noise std {np.rad2deg(np.std(rates)):.3f} deg/s "
          f"(settling took {time.time() - t0:.0f} s)")


def active_run(bus, imu, ctl, log, dry, k0, args):
    """Returns the reason the run ended: 'done' (HOLD_SECONDS spent at target) or an error string."""
    fails, overruns = 0, 0
    tstart = time.perf_counter()
    k = 0
    while not ctl.finished:
        tick = tstart + k * DT
        now = time.perf_counter()
        if now < tick:
            time.sleep(tick - now)
        t_loop = time.perf_counter()
        try:
            p = bus.read_positions()
            g = imu.read_gyro_rad_s()
            fails = 0
        except Exception as e:
            fails += 1
            print("read failed:", e)
            if fails >= C.MAX_COMM_FAILS:
                return "too many consecutive read failures"
            k += 1
            continue
        lean = S.raw_to_lean_deg(p[C.ID_LEAN])
        ka, kb = S.raw_to_knee_deg(p[C.ID_KNEE_A]), S.raw_to_knee_deg(p[C.ID_KNEE_B])
        try:
            out = ctl.step(g, lean, ka, kb)
        except S.SwingController.SafetyStop as e:
            return f"safety stop: {e}"
        if not dry:
            try:
                bus.write_goals({C.ID_LEAN: S.lean_deg_to_raw(out["lean"]),
                                 C.ID_KNEE_A: S.knee_deg_to_raw(out["knee_a"]),
                                 C.ID_KNEE_B: S.knee_deg_to_raw(out["knee_b"])}, C.SERVO_SPEED, C.SERVO_ACC)
            except Exception as e:
                return f"write failed: {e}"
        ms = (time.perf_counter() - t_loop) * 1000
        overruns += ms > 1000 * DT
        log.writerow([f"{k0 + k*DT:.2f}", f"{out['swing_deg']:.2f}", f"{lean:.1f}", f"{ka:.1f}", f"{kb:.1f}",
                      f"{out['lean']:.1f}", f"{out['knee_a']:.1f}", f"{out['knee_b']:.1f}", f"{ms:.1f}"])
        if k % (5 * C.CONTROL_HZ) == 0:
            print(f"t={k*DT:6.1f}s {out['mode']:6s} held {out['hold_s']:5.0f}/{C.HOLD_SECONDS:.0f}s swing {out['swing_deg']:+6.1f}  lean {lean:+5.1f}->{out['lean']:+5.1f}  "
                  f"knees {ka:+5.1f}/{kb:+5.1f}->{out['knee_a']:+5.1f}/{out['knee_b']:+5.1f}  loop {ms:.1f} ms  overruns {overruns}")
        k += 1
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cycles", type=int, default=0, help="stop after this many cycles (0 = repeat forever)")
    ap.add_argument("--policy", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.npz"))
    args = ap.parse_args()
    dry = args.dry_run

    bus, imu = ServoBus(IDS), Imu()
    os.makedirs("logs", exist_ok=True)
    logf = open(os.path.join("logs", time.strftime("run_%Y%m%d_%H%M%S.csv")), "w", newline="")
    log = csv.writer(logf)
    log.writerow(["t", "swing_deg", "lean_deg", "knee_a_deg", "knee_b_deg", "lean_cmd", "knee_a_cmd", "knee_b_cmd", "loop_ms"])
    try:
        setup_servos(bus, dry)
        est = S.SwingEstimator()
        print(f"measuring gyro bias for {C.GYRO_BIAS_CAL_S:.0f} s -- keep the swing perfectly still ...")
        rates = []
        t0 = time.time()
        while time.time() - t0 < C.GYRO_BIAS_CAL_S:
            rates.append(imu.read_gyro_rad_s())
            time.sleep(DT)
        sd, mean = np.rad2deg(np.std(rates)), np.rad2deg(np.mean(rates))
        print(f"  bias {mean:+.3f} deg/s, noise std {sd:.3f} deg/s")
        if sd > C.REST_CAL_STD_DPS or abs(mean) > 10:
            raise BusError("gyro looks disturbed or faulty during calibration -- not starting")
        est.bias, est.angle = float(np.mean(rates)), 0.0

        pol = S.Policy(args.policy)
        cycle, fast_faults, t_total = 0, 0, 0.0
        while args.cycles == 0 or cycle < args.cycles:
            cycle += 1
            print(f"\n===== cycle {cycle}: ACTIVE until {C.HOLD_SECONDS:.0f} s spent at target (limit {C.ACTIVE_MAX_SECONDS:.0f} s) (Ctrl+C = relax servos immediately) =====")
            ctl = S.SwingController(pol, est, DT)
            t_run = time.time()
            reason = active_run(bus, imu, ctl, log, dry, t_total, args)
            t_total += time.time() - t_run
            if reason != "done":
                print("ACTIVE ended by error:", reason)
                fast_faults = fast_faults + 1 if time.time() - t_run < C.FAST_FAULT_S else 0
                if fast_faults >= C.MAX_FAST_FAULTS:
                    raise BusError(f"{fast_faults} runs in a row faulted within {C.FAST_FAULT_S:.0f} s -- giving up")
            else:
                fast_faults = 0
                print("ACTIVE time complete")
            settle_and_rezero(bus, imu, est, dry)
    except KeyboardInterrupt:
        print("\nCtrl+C")
    except (BusError, S.SwingController.SafetyStop) as e:
        print("STOP:", e)
    finally:
        print("relaxing all servos")
        if not dry:
            bus.relax_all()
        bus.close()
        logf.close()


if __name__ == "__main__":
    main()

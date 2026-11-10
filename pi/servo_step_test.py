"""
Bench test: how fast do the real servos actually follow a step? Addresses the open
question from README 5.9ae (sim knee peaks ~600 deg/s; the STS3095 is ~240 deg/s no-load
at 8.4 V). SERVOS WILL MOVE -- keep the joint clear of obstacles, and run it both with the
joint unloaded and with the real torso / leg mass attached.

    python3 servo_step_test.py kneeB 0 60      # same for the second knee (the plain "knee" tests knee A)
    python3 servo_step_test.py knee 0 60       # knee: step sim 0 -> +60 deg and back, 3 times
    python3 servo_step_test.py lean 0 40
    python3 servo_step_test.py knee 0 60 600     # optional 4th argument: torque limit to use instead of config.py's
                                                  # (to see whether the torque cap itself is throttling the speed)

Prints, for each step, the time to cover 90% of the move and the peak speed. Torque
limits from config.py are applied first, so the result includes the cap.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import swing_core as S
from hardware import ServoBus

joint, a, b = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
torque_override = int(sys.argv[4]) if len(sys.argv) > 4 else None
sid = C.ID_LEAN if joint == "lean" else (C.ID_KNEE_B if joint == "kneeB" else C.ID_KNEE_A)
to_raw = S.lean_deg_to_raw if joint == "lean" else S.knee_deg_to_raw
to_deg = S.raw_to_lean_deg if joint == "lean" else S.raw_to_knee_deg
bus = ServoBus([sid])
try:
    bus.set_torque_limit(sid, torque_override or (C.TORQUE_LIMIT_LEAN if joint == "lean" else C.TORQUE_LIMIT_KNEE))
    print("torque limit in use:", bus.read2(sid, 48))
    bus.write_goals(bus.read_positions(), 400, 20)
    bus.torque(sid, True)
    for (src, dst) in [(a, b), (b, a)] * 3:
        bus.write_goals({sid: to_raw(src)}, 400, 20)
        time.sleep(1.5)
        t0 = time.perf_counter()
        bus.write_goals({sid: to_raw(dst)}, C.SERVO_SPEED, C.SERVO_ACC)
        ts, ds = [], []
        while time.perf_counter() - t0 < 1.2:
            ds.append(to_deg(bus.read_positions()[sid]))
            ts.append(time.perf_counter() - t0)
        move = dst - src
        t90 = next((t for t, d in zip(ts, ds) if abs(d - src) >= 0.9 * abs(move)), None)
        peak = max(abs((ds[k + 2] - ds[k]) / (ts[k + 2] - ts[k])) for k in range(len(ds) - 2))
        print(f"{src:+5.0f} -> {dst:+5.0f} deg: 90% in " + (f"{t90*1000:.0f} ms" if t90 else "never within 1.2 s") +
              f", peak speed {peak:.0f} deg/s, samples {len(ts)}")
finally:
    try:       # put the config torque limit back, so an override never lingers in the servo's RAM
        bus.set_torque_limit(sid, C.TORQUE_LIMIT_LEAN if joint == "lean" else C.TORQUE_LIMIT_KNEE)
    except Exception:
        pass
    bus.relax_all()
    bus.close()

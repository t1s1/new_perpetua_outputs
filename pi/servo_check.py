"""
Read-back check of every servo's settings -- closes the loose threads on torque
limits, travel limits, mode, zero and direction. Safe: it only READS, except with
--apply-torque-limits, which writes the RAM torque limits from config.py (and reads
them back). Torque is never enabled here.

    python3 servo_check.py
    python3 servo_check.py --apply-torque-limits

How to use it for the open items:
  * Power-cycle test: run it, power the servos off and on, run it again. EEPROM values
    (min/max angle, max torque @16) should be unchanged. The RAM torque limit @48 comes
    from @16 at power-up, so if you stored a lower value in @16 via MotorLab it will
    show here; if not, it goes back to 1000 -- run_swing.py rewrites it every start anyway.
  * Zero / direction: move each joint BY HAND to its sim zero pose (torso upright /
    shin straight down) and then to each end; the printed sim degrees should read
    ~0 at zero, lean + toward recline, knee + toward extension, and stay inside the
    travel ranges. (Servos are limp when torque is off, so you can move them by hand.)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import swing_core as S
from hardware import (REG_MAX_ANGLE, REG_MAX_TORQUE, REG_MIN_ANGLE, REG_MODE,
                      REG_TEMP, REG_TORQUE_LIMIT, REG_VOLTAGE, ServoBus)

ap = argparse.ArgumentParser()
ap.add_argument("--apply-torque-limits", action="store_true")
args = ap.parse_args()

names = {C.ID_LEAN: "lean", C.ID_KNEE_A: "knee A", C.ID_KNEE_B: "knee B"}
want = {C.ID_LEAN: C.TORQUE_LIMIT_LEAN, C.ID_KNEE_A: C.TORQUE_LIMIT_KNEE, C.ID_KNEE_B: C.TORQUE_LIMIT_KNEE}
exp_raw = {  # raw travel the config expects (sim range converted), as (low, high)
    C.ID_LEAN: sorted((S.lean_deg_to_raw(C.LEAN_RANGE_DEG[0]), S.lean_deg_to_raw(C.LEAN_RANGE_DEG[1]))),
    C.ID_KNEE_A: sorted((S.knee_deg_to_raw(C.KNEE_RANGE_DEG[0]), S.knee_deg_to_raw(C.KNEE_RANGE_DEG[1]))),
    C.ID_KNEE_B: sorted((S.knee_deg_to_raw(C.KNEE_RANGE_DEG[0]), S.knee_deg_to_raw(C.KNEE_RANGE_DEG[1]))),
}
bus = ServoBus(list(names))
try:
    for i, n in names.items():
        model = bus.ping(i)
        print(f"\n[{n}] id {i}: " + ("NO ANSWER" if model is None else f"model number {model}"))
        if model is None:
            continue
        lo, hi = bus.read2(i, REG_MIN_ANGLE), bus.read2(i, REG_MAX_ANGLE)
        print(f"  stored travel limits  min {lo}  max {hi}   (config expects travel {exp_raw[i][0]}..{exp_raw[i][1]}; "
              f"limits should sit a little OUTSIDE that, but inside the mechanical stops)")
        if lo == 0 and hi == 0:
            print("  (0/0 usually means 'no limit' -- fine for multi-turn off, but then nothing but your brackets stops it)")
        elif lo > exp_raw[i][0] or hi < exp_raw[i][1]:
            print("  !! stored limits are TIGHTER than the range the policy can command -- the servo will clip commands")
        print(f"  mode {bus.read1(i, REG_MODE)} (0 = position servo)   voltage {bus.read1(i, REG_VOLTAGE)/10:.1f} V   temp {bus.read1(i, REG_TEMP)} C")
        print(f"  max torque (EEPROM @16) {bus.read2(i, REG_MAX_TORQUE)}   torque limit (RAM @48) {bus.read2(i, REG_TORQUE_LIMIT)}   config wants {want[i]}")
        if args.apply_torque_limits:
            bus.set_torque_limit(i, want[i])
            print(f"  wrote torque limit {want[i]}; read back {bus.read2(i, REG_TORQUE_LIMIT)}")
    print()
    p = bus.read_positions()
    for i, n in names.items():
        deg = S.raw_to_lean_deg(p[i]) if i == C.ID_LEAN else S.raw_to_knee_deg(p[i])
        print(f"[{n}] raw {p[i]:4d} -> sim {deg:+6.1f} deg")
finally:
    bus.close()

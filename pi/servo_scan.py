"""
Find the serial port and servo IDs. Safe: it only pings and reads; torque is never enabled,
so the servos stay limp and you can move them by hand.

    python3 servo_scan.py                    # lists serial ports, then pings IDs 1-253 on the first one found
    python3 servo_scan.py /dev/ttyUSB0       # use a specific port
    python3 servo_scan.py /dev/ttyUSB0 watch # after the scan, print live positions of the servos found:
                                             # move one joint by hand and see which ID changes (that tells you
                                             # which ID is the lean and which are the knees). Ctrl+C to stop.

Then put the port and the three IDs in config.py (SERIAL_PORT, ID_LEAN, ID_KNEE_A, ID_KNEE_B).
"""
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from hardware import ServoBus

ports = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
print("serial ports found:", ports or "none")
for p in sorted(glob.glob("/dev/serial/by-id/*")):
    print(f"   {p} -> {os.path.realpath(p)}")
if not ports and len(sys.argv) < 2:
    print("None found. Check the USB cable/adapter, and `dmesg | tail` right after plugging it in.")
    sys.exit(1)
port = sys.argv[1] if len(sys.argv) > 1 else ports[0]
watch = len(sys.argv) > 2 and sys.argv[2] == "watch"

print(f"\npinging IDs 1-253 on {port} at {C.BAUDRATE} baud (takes a minute or so) ...")
bus = ServoBus([], port=port)
found = []
try:
    odd = []
    for i in range(1, 254):
        m, res, txt = bus.ping_result(i)
        if m is not None:
            found.append(i)
            print(f"  ID {i}: answers, model number {m}")
        elif res != -6:                       # something replied, but not cleanly
            odd.append((i, res, txt))
    for i, res, txt in odd:
        print(f"  ID {i}: something replied but not cleanly: code {res} ({txt}) -- often two servos sharing this ID")
    if not found:
        print("No servo answered. Check: servo power on (separate supply), shared ground, the adapter's "
              "mode switch (UART/bus position), and the baud rate (1,000,000 is the factory default).")
        sys.exit(1)
    print(f"\nfound IDs: {found}")
    if watch:
        bus.ids = found
        from hardware import GroupSyncRead, REG_PRESENT_POS
        bus.reader = GroupSyncRead(bus.port, bus.ph, REG_PRESENT_POS, 4)
        for i in found:
            bus.reader.addParam(i)
        print("move one joint at a time by hand; Ctrl+C to stop")
        first = bus.read_positions()
        while True:
            p = bus.read_positions()
            print("  " + "   ".join(f"ID {i}: {p[i]:4d} ({p[i]-first[i]:+5d})" for i in found), end="\r")
            time.sleep(0.1)
except KeyboardInterrupt:
    print()
finally:
    bus.close()

"""
Servo bus + IMU access for the Pi. Uses the PyPI package `feetech-servo-sdk`
(import name `scservo_sdk`) only for its generic packet layer; the STS register
numbers below are from the Feetech STS memory map (community-documented, same
caveat as the torque-limit numbers in README 5.9ab) -- servo_check.py reads
them back so you can confirm they behave as expected on YOUR servos.

NOT tested on real hardware (none available where this was written).
"""
import time

import config as C

try:
    from scservo_sdk import (COMM_SUCCESS, GroupSyncRead, GroupSyncWrite,
                             PacketHandler, PortHandler)
except ImportError as _e:                 # lets swing_core / sim_test run on a dev machine
    _IMPORT_ERROR = _e
    PortHandler = PacketHandler = GroupSyncRead = GroupSyncWrite = None
    COMM_SUCCESS = 0

# ---- STS register map ----
REG_MIN_ANGLE = 9          # 2 bytes, EEPROM
REG_MAX_ANGLE = 11         # 2 bytes, EEPROM
REG_MAX_TORQUE = 16        # 2 bytes, EEPROM (0-1000)
REG_MODE = 33              # 1 byte, 0 = position servo
REG_TORQUE_ENABLE = 40     # 1 byte
REG_ACC = 41               # 1 byte; goal block is acc(1) pos(2) time(2) speed(2)
REG_GOAL_POS = 42
REG_TORQUE_LIMIT = 48      # 2 bytes, RAM (copied from 16 at power-up)
REG_PRESENT_POS = 56       # 2 bytes; speed follows at 58 (2 bytes, bit 15 = sign)
REG_VOLTAGE = 62           # 1 byte, units of 0.1 V
REG_TEMP = 63              # 1 byte, deg C


def _signed15(v):
    """Feetech sign-magnitude: bit 15 is the sign."""
    return -(v & 0x7FFF) if v & 0x8000 else v


class BusError(Exception):
    pass


class ServoBus:
    def __init__(self, ids, port=None, baud=None):
        if PortHandler is None:
            raise RuntimeError(f"servo library not importable ({_IMPORT_ERROR!r}). Is the virtual environment active "
                               f"(source ~/swing-env/bin/activate)? Otherwise: pip install feetech-servo-sdk pyserial")
        self.ids = list(ids)
        self.port = PortHandler(port or C.SERIAL_PORT)
        if not self.port.openPort():
            raise BusError(f"could not open {port or C.SERIAL_PORT}")
        if not self.port.setBaudRate(baud or C.BAUDRATE):
            raise BusError("could not set baud rate")
        self.ph = PacketHandler(0)        # 0 = little-endian (STS series)
        self.reader = GroupSyncRead(self.port, self.ph, REG_PRESENT_POS, 4)
        for i in self.ids:
            self.reader.addParam(i)

    # ---- single-servo helpers ----
    def ping(self, i):
        model, res, err = self.ph.ping(self.port, i)
        return (model if res == COMM_SUCCESS else None)

    def ping_result(self, i):
        """(model number or None, comm result code, text). Codes: 0 ok, -6 timeout (nobody answered),
        -7 corrupt reply (often two servos answering at once = duplicate ID), -3 rx fail."""
        model, res, err = self.ph.ping(self.port, i)
        return (model if res == COMM_SUCCESS else None), res, self.ph.getTxRxResult(res)

    def write1(self, i, addr, v):
        res, err = self.ph.write1ByteTxRx(self.port, i, addr, int(v))
        if res != COMM_SUCCESS:
            raise BusError(f"write1 id={i} addr={addr}: {self.ph.getTxRxResult(res)}")

    def write2(self, i, addr, v):
        res, err = self.ph.write2ByteTxRx(self.port, i, addr, int(v))
        if res != COMM_SUCCESS:
            raise BusError(f"write2 id={i} addr={addr}: {self.ph.getTxRxResult(res)}")

    def read1(self, i, addr):
        v, res, err = self.ph.read1ByteTxRx(self.port, i, addr)
        if res != COMM_SUCCESS:
            raise BusError(f"read1 id={i} addr={addr}: {self.ph.getTxRxResult(res)}")
        return v

    def read2(self, i, addr):
        v, res, err = self.ph.read2ByteTxRx(self.port, i, addr)
        if res != COMM_SUCCESS:
            raise BusError(f"read2 id={i} addr={addr}: {self.ph.getTxRxResult(res)}")
        return v

    # ---- setup ----
    def set_torque_limit(self, i, v):
        self.write2(i, REG_TORQUE_LIMIT, v)

    def torque(self, i, on):
        self.write1(i, REG_TORQUE_ENABLE, 1 if on else 0)

    def relax_all(self):
        """Best effort, never raises: torque off on every servo."""
        for i in self.ids:
            for _ in range(3):
                try:
                    self.torque(i, False)
                    break
                except Exception:
                    time.sleep(0.005)

    # ---- fast path, one bus transaction each ----
    def read_positions(self):
        """{id: raw position} for all servos in one sync read."""
        res = self.reader.txRxPacket()
        if res != COMM_SUCCESS:
            raise BusError(f"sync read: {self.ph.getTxRxResult(res)}")
        out = {}
        for i in self.ids:
            if not self.reader.isAvailable(i, REG_PRESENT_POS, 2):
                raise BusError(f"no data from id {i}")
            out[i] = self.reader.getData(i, REG_PRESENT_POS, 2) & 0x0FFF
        return out

    def write_goals(self, goals, speed, acc):
        """goals: {id: raw position}. One sync-write packet."""
        w = GroupSyncWrite(self.port, self.ph, REG_ACC, 7)
        for i, pos in goals.items():
            pos = int(max(0, min(4095, pos)))
            w.addParam(i, [int(acc) & 0xFF, pos & 0xFF, (pos >> 8) & 0xFF, 0, 0, int(speed) & 0xFF, (int(speed) >> 8) & 0xFF])
        res = w.txPacket()
        w.clearParam()
        if res != COMM_SUCCESS:
            raise BusError(f"sync write: {self.ph.getTxRxResult(res)}")

    def close(self):
        self.port.closePort()


class Imu:
    """MPU-6050 gyro (one axis used) + raw accel for the bench checks."""
    _AX = {"x": 0, "y": 1, "z": 2}

    def __init__(self, bus_num=1):
        from smbus2 import SMBus
        self.bus = SMBus(bus_num)
        a = C.IMU_ADDR
        self.bus.write_byte_data(a, 0x6B, 0x00)    # wake
        time.sleep(0.1)
        self.bus.write_byte_data(a, 0x1A, 0x03)    # DLPF ~44 Hz
        self.bus.write_byte_data(a, 0x1B, 0x08)    # gyro +/-500 dps
        self.bus.write_byte_data(a, 0x1C, 0x08)    # accel +/-4 g
        self.k = self._AX[C.GYRO_AXIS]

    def read_gyro_rad_s(self):
        """Rate about the configured axis, rad/s, NOT yet multiplied by GYRO_SIGN (swing_core does that)."""
        import math
        b = self.bus.read_i2c_block_data(C.IMU_ADDR, 0x43, 6)
        hi, lo = b[2 * self.k], b[2 * self.k + 1]
        v = (hi << 8) | lo
        v = v - 65536 if v & 0x8000 else v
        return math.radians(v / 65.5)

"""
Hardware-independent control logic: conversions, gyro angle estimator, policy,
knee drift. Pure numpy so it runs on the Pi AND can be tested against the sim
(see sim_test.py). No servo or I2C code in here.
"""
import numpy as np

import config as C


# ---------- joint <-> raw conversions ----------
def lean_deg_to_raw(deg):
    return int(round(C.ZERO_RAW + C.SIGN_LEAN * C.COUNTS_PER_DEG * deg))


def knee_deg_to_raw(deg):
    return int(round(C.ZERO_RAW + C.SIGN_KNEE * C.COUNTS_PER_DEG * deg))


def raw_to_lean_deg(raw):
    return C.SIGN_LEAN * (raw - C.ZERO_RAW) / C.COUNTS_PER_DEG


def raw_to_knee_deg(raw):
    return C.SIGN_KNEE * (raw - C.ZERO_RAW) / C.COUNTS_PER_DEG


# ---------- policy ----------
class Policy:
    def __init__(self, path="policy.npz"):
        d = np.load(path)
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = (d[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3"))
        self.mean, self.var, self.clip, self.eps = d["mean"], d["var"], float(d["clip"]), float(d["eps"])

    def act(self, obs):
        x = np.clip((np.asarray(obs, dtype=np.float64) - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip)
        h = np.tanh(x @ self.W1 + self.b1)
        h = np.tanh(h @ self.W2 + self.b2)
        return np.clip(h @ self.W3 + self.b3, -1.0, 1.0)


def action_to_target_deg(a, rng_deg):
    lo, hi = rng_deg
    return lo + (a + 1.0) * 0.5 * (hi - lo)


# ---------- swing-angle estimator (gyro integration, as trained: README 5.9t/5.9x) ----------
class SwingEstimator:
    """angle = integral of (gyro - bias). Bias measured while hanging still; angle zeroed then."""

    def __init__(self):
        self.bias = 0.0          # rad/s
        self.angle = 0.0         # rad
        self._acc, self._n = 0.0, 0
        self._prev_rate = None

    def calibrate_add(self, rate_rad_s):
        self._acc += rate_rad_s
        self._n += 1

    def calibrate_finish(self):
        self.bias = self._acc / max(self._n, 1)
        self.angle = 0.0
        return self.bias

    def update(self, rate_rad_s, dt):
        rate = rate_rad_s - self.bias
        # trapezoid rule: the sample is an instantaneous reading, so averaging with
        # the previous one removes the half-step phase lag plain summation has
        prev = rate if self._prev_rate is None else self._prev_rate
        self.angle += 0.5 * (rate + prev) * dt
        self._prev_rate = rate
        return self.angle, rate


# ---------- per-knee slow drift ----------
class KneeDrift:
    def __init__(self, rng, dt):
        self.rng = rng
        self.phi = np.exp(-dt / C.DRIFT_TAU_S)
        self.sig = C.DRIFT_STD_DEG * np.sqrt(1 - self.phi ** 2)
        self.x = np.zeros(2)

    def step(self):
        if not C.DRIFT_ENABLE:
            return np.zeros(2)
        self.x = self.phi * self.x + self.sig * self.rng.standard_normal(2)
        self.x = np.clip(self.x, -C.DRIFT_MAX_DEG, C.DRIFT_MAX_DEG)
        return self.x


# ---------- the controller ----------
class SwingController:
    """
    One call per control tick:
        targets = ctl.step(gyro_rate_rad_s, lean_deg, knee_a_deg, knee_b_deg)
    returns dict(lean=deg, knee_a=deg, knee_b=deg, swing_deg=..., action=(a_lean, a_knee)).
    Raises SafetyStop if the swing angle exceeds MAX_SWING_DEG.
    """

    class SafetyStop(Exception):
        pass

    def __init__(self, policy, est, dt, seed=None):
        self.policy, self.est, self.dt = policy, est, dt
        self.drift = KneeDrift(np.random.default_rng(seed), dt)
        self.prev = None          # previous (lean_deg, knee_mean_deg)
        self.vel = np.zeros(2)    # smoothed deg/s
        self.mode = "BOOT" if C.BOOT_ENABLE else "POLICY"
        self.t, self.peak_deg, self.boot_state = 0.0, 0.0, -1.0
        self.hold_s = 0.0
        self._since_flip = 0.0
        self._win = []            # recent |swing| samples for the windowed peak
        self.finished = False     # True once HOLD_SECONDS have been spent at target

    def step(self, gyro_rate_rad_s, lean_deg, knee_a_deg, knee_b_deg):
        swing, swing_vel = self.est.update(C.GYRO_SIGN * gyro_rate_rad_s, self.dt)
        if abs(np.rad2deg(swing)) > C.MAX_SWING_DEG:
            raise self.SafetyStop(f"swing angle {np.rad2deg(swing):.1f} deg exceeds {C.MAX_SWING_DEG}")
        knee = 0.5 * (knee_a_deg + knee_b_deg)
        cur = np.array([lean_deg, knee])
        if self.prev is None:
            raw_vel = np.zeros(2)
        else:
            raw_vel = (cur - self.prev) / self.dt
        self.prev = cur
        self.vel = C.JOINT_VEL_EMA * raw_vel + (1 - C.JOINT_VEL_EMA) * self.vel
        self.t += self.dt
        self.peak_deg = max(self.peak_deg, abs(np.rad2deg(swing)))
        if self.mode == "BOOT" and self.peak_deg >= C.BOOT_HANDOVER_DEG:
            self.mode = "POLICY"
        if self.mode == "BOOT" and self.t > C.BOOT_TIMEOUT_S:
            raise self.SafetyStop(f"bootstrap did not reach {C.BOOT_HANDOVER_DEG} deg in {C.BOOT_TIMEOUT_S:.0f} s "
                                  f"(peak {self.peak_deg:.1f}); check GYRO_SIGN and that the lean joint moves freely")
        self._win.append(abs(np.rad2deg(swing)))
        if len(self._win) > int(C.HOLD_WINDOW_S / self.dt):
            self._win.pop(0)
        if max(self._win) >= C.HOLD_MIN_DEG:
            self.hold_s += self.dt
        if self.hold_s >= C.HOLD_SECONDS:
            self.finished = True
        if self.t > C.ACTIVE_MAX_SECONDS and not self.finished:
            raise self.SafetyStop(f"target ({C.HOLD_MIN_DEG} deg) not held for {C.HOLD_SECONDS:.0f} s within {C.ACTIVE_MAX_SECONDS:.0f} s "
                                  f"(held {self.hold_s:.0f} s, peak {self.peak_deg:.1f})")
        obs = [swing, swing_vel, np.deg2rad(lean_deg), np.deg2rad(self.vel[0]), np.deg2rad(knee), np.deg2rad(self.vel[1])]
        if self.mode == "BOOT":
            h = np.deg2rad(C.BOOT_HYST_DPS)       # Schmitt trigger: only flip when the rate is clearly one way
            self._since_flip += self.dt
            if swing_vel > h and self.boot_state < 0 or swing_vel < -h and self.boot_state > 0:
                self.boot_state = -self.boot_state
                self._since_flip = 0.0
            elif self._since_flip > C.BOOT_STALL_S:      # stalled: nothing to follow, so wiggle
                self.boot_state = -self.boot_state
                self._since_flip = 0.0
            a = np.array([C.BOOT_AMP * self.boot_state, 0.0])
        else:
            a = self.policy.act(obs)
        lean_t = action_to_target_deg(a[0], C.LEAN_RANGE_DEG)
        knee_t = action_to_target_deg(a[1], C.KNEE_RANGE_DEG)
        d = self.drift.step()
        lo, hi = C.KNEE_RANGE_DEG
        return dict(lean=lean_t,
                    knee_a=float(np.clip(knee_t + d[0], lo, hi)),
                    knee_b=float(np.clip(knee_t + d[1], lo, hi)),
                    swing_deg=float(np.rad2deg(swing)), action=a, mode=self.mode, hold_s=self.hold_s)


# ---------- rest detection (README 5.9u) ----------
class RestDetector:
    """True once |rate| has stayed below REST_RATE_DPS for REST_HOLD_S continuously."""

    def __init__(self):
        self.t_quiet = 0.0

    def update(self, rate_rad_s, dt):
        if abs(np.rad2deg(rate_rad_s)) < C.REST_RATE_DPS:
            self.t_quiet += dt
        else:
            self.t_quiet = 0.0
        return self.t_quiet >= C.REST_HOLD_S

"""
Scan the checkpoints saved during the knee -45 deg fine-tune (README 5.9ae) and
rank them, so the best one can be promoted. Only looks at files modified after
CUTOFF (the start of the fine-tune), so it does NOT re-scan the ~1000 older
checkpoints the way scan_checkpoints.py would from scratch.

Runs EPISODES deterministic episodes per checkpoint (nominal physics, exact
angles, same as evaluate.py's deterministic check), paired with that
checkpoint's own vecnormalize file. Appends to knee45_scan.csv as it goes, so
Ctrl+C is safe and re-running skips what's done. Read-only otherwise.

Raw periodic checkpoints are volatile (README 5.9ac), so expect some bad ones;
the ranking is just to find candidates -- the top few then get a proper
evaluate.py-style check before promotion.

Run:  python3 scan_knee45.py
"""
import csv
import glob
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from swing_env import SwingEnv

CUTOFF = time.mktime(time.strptime("2026-10-29 16:38", "%Y-%m-%d %H:%M"))  # local time
EPISODES = 2
OUT = "knee45_scan.csv"

done = set()
if os.path.exists(OUT):
    done = {r["checkpoint"] for r in csv.DictReader(open(OUT))}
else:
    open(OUT, "w").write("checkpoint,reward,peak_deg,overshoot,mtime\n")

cks = sorted((p for p in glob.glob("checkpoints/swing_ppo_[0-9]*_steps.zip") if os.path.getmtime(p) > CUTOFF),
             key=os.path.getmtime)
print(f"{len(cks)} checkpoints since cutoff, {len(done)} already scanned")
for p in cks:
    if p in done:
        continue
    vn = p.replace("swing_ppo_", "swing_ppo_vecnormalize_").replace(".zip", ".pkl")
    if not os.path.exists(vn):
        continue
    model = PPO.load(p)
    rows = []
    for _ in range(EPISODES):
        v = DummyVecEnv([lambda: SwingEnv()])
        v = VecNormalize.load(vn, v)
        v.training = False
        v.norm_reward = False
        obs = v.reset()
        tot = peak = ov = 0.0
        finished = False
        while not finished:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, d, info = v.step(a)
            finished = bool(d[0])
            tot += float(r[0])
            peak = max(peak, abs(info[0]["swing_angle_deg"]))
            ov += info[0]["overshoot_penalty"]
        rows.append((tot, peak, ov))
    m = np.mean(rows, axis=0)
    with open(OUT, "a") as f:
        f.write(f"{p},{m[0]:.1f},{m[1]:.2f},{m[2]:.1f},{os.path.getmtime(p):.0f}\n")
    print(f"{os.path.basename(p):32s} reward {m[0]:8.0f}  peak {m[1]:6.1f}  overshoot {m[2]:8.0f}", flush=True)

print("\nTop 10 by reward:")
rows = sorted(csv.DictReader(open(OUT)), key=lambda r: -float(r["reward"]))[:10]
for r in rows:
    print(f"{os.path.basename(r['checkpoint']):32s} reward {float(r['reward']):8.0f}  peak {float(r['peak_deg']):6.1f}  overshoot {float(r['overshoot']):8.0f}")

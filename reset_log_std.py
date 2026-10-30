"""
Surgical fix for the entropy-coefficient overshoot documented in README
5.5a-xx/xxi: two rounds of ent_coef reduction (0.03 -> 0.015) FAILED to
reverse action-std saturation -- model.policy.log_std.exp() kept climbing
([2.21,2.79] -> [2.94,4.15] -> [3.65,5.15]) even after the coefficient was
halved. Direct inspection of the Adam optimizer state (policy.optimizer,
saved/reloaded on every PPO.load() same as the weights) showed this isn't
leftover momentum from the 0.03 era -- with default betas (0.9, 0.999) the
effective memory window is ~1000 steps against 7.4M total, so exp_avg/
exp_avg_sq already reflect only the current regime. The problem is the
baked-in log_std WEIGHT itself: once std is this far past the [-1,1] action
range, sampled actions land on step()'s hard clip almost regardless of the
mean, which destroys the clean advantage signal needed for the policy
gradient to pull it back down on any reasonable timescale, while Gaussian
entropy is unbounded so any positive ent_coef keeps some upward pressure on
top. Ordinary training was not going to fix this on its own.

This script directly overwrites log_std with a sane, small value instead of
waiting on gradient descent, and clears the Adam optimizer state for that
one parameter (its old exp_avg/exp_avg_sq were calibrated for gradients
around a completely different weight regime -- leaving them in place risks
a distorted first update at the new operating point).

Does NOT touch any other policy/value weights, VecNormalize stats, or
training hyperparameters. Run once, then resume normal train.py rounds.
"""
import shutil

import torch
from stable_baselines3 import PPO

MODEL_PATH = "swing_policy"
BACKUP_PATH = "swing_policy_pre_logstd_reset_backup.zip"
TARGET_STD = 0.3  # within the 0.3-0.5 "healthy" range discussed; chose the
                   # more conservative end given how badly this overshot before

shutil.copy(MODEL_PATH + ".zip", BACKUP_PATH)
print(f"Backed up {MODEL_PATH}.zip -> {BACKUP_PATH}")

model = PPO.load(MODEL_PATH)

before = model.policy.log_std.data.clone()
print("log_std before:", before, "-> std:", before.exp())

target_log_std = torch.log(torch.tensor(TARGET_STD))
with torch.no_grad():
    model.policy.log_std.data.fill_(target_log_std.item())

after = model.policy.log_std.data.clone()
print("log_std after: ", after, "-> std:", after.exp())

# Clear stale Adam state (step count, exp_avg, exp_avg_sq) for this specific
# parameter so its first update at the new operating point isn't distorted
# by momentum/variance estimates calibrated for the old (saturated) regime.
opt = model.policy.optimizer
cleared = False
for group in opt.param_groups:
    for p in group["params"]:
        if p is model.policy.log_std:
            if p in opt.state:
                old_step = opt.state[p].get("step")
                del opt.state[p]
                cleared = True
                print(f"Cleared Adam state for log_std (was at step {old_step})")
if not cleared:
    print("No pre-existing Adam state found for log_std (nothing to clear)")

model.save(MODEL_PATH)
print(f"Saved. Reloading fresh from disk to verify...")

reloaded = PPO.load(MODEL_PATH)
verify_log_std = reloaded.policy.log_std.data
print("Verified log_std after save+reload:", verify_log_std, "-> std:", verify_log_std.exp())

opt2 = reloaded.policy.optimizer
for group in opt2.param_groups:
    for p in group["params"]:
        if p is reloaded.policy.log_std:
            state = opt2.state.get(p, {})
            print("Verified optimizer state for log_std after save+reload:", state if state else "(empty, as expected)")

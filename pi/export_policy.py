"""
Export the trained SB3 policy to a plain .npz so the Raspberry Pi needs only
numpy (no torch / stable-baselines3 / mujoco).

Run on the machine that has the project (NOT on the Pi), from the project root:
    python3 pi/export_policy.py [best_model/best_model.zip] [best_model/best_vec_normalize.pkl] [pi/policy.npz]

What gets saved: the 6-64-64-2 tanh MLP weights, plus the VecNormalize
observation mean/var/clip. Deterministic action = clip(MLP(normalized obs), -1, 1),
exactly what PPO.predict(deterministic=True) does in this project's action space.
The script re-checks its own numpy forward pass against SB3 on random
observations and refuses to write the file if they disagree.
"""
import pickle
import sys

import numpy as np
import torch
from stable_baselines3 import PPO

model_zip = sys.argv[1] if len(sys.argv) > 1 else "best_model/best_model.zip"
vecnorm_pkl = sys.argv[2] if len(sys.argv) > 2 else "best_model/best_vec_normalize.pkl"
out = sys.argv[3] if len(sys.argv) > 3 else "pi/policy.npz"

model = PPO.load(model_zip, device="cpu")
with open(vecnorm_pkl, "rb") as f:
    vn = pickle.load(f)

pn = model.policy.mlp_extractor.policy_net
W1, b1 = pn[0].weight.detach().numpy().T, pn[0].bias.detach().numpy()
W2, b2 = pn[2].weight.detach().numpy().T, pn[2].bias.detach().numpy()
W3, b3 = model.policy.action_net.weight.detach().numpy().T, model.policy.action_net.bias.detach().numpy()
mean, var = vn.obs_rms.mean, vn.obs_rms.var
clip, eps = float(vn.clip_obs), float(vn.epsilon)


def forward(obs):
    x = np.clip((obs - mean) / np.sqrt(var + eps), -clip, clip)
    h = np.tanh(x @ W1 + b1)
    h = np.tanh(h @ W2 + b2)
    return np.clip(h @ W3 + b3, -1.0, 1.0)


rng = np.random.default_rng(0)
worst = 0.0
for _ in range(2000):
    o = (rng.normal(size=6) * np.array([0.6, 0.8, 0.5, 2.0, 0.5, 3.0]) + np.array([0, 0, 0.5, 0, 0.5, 0])).astype(np.float32)
    ref, _ = model.predict(((o - mean) / np.sqrt(var + eps)).clip(-clip, clip)[None].astype(np.float32), deterministic=True)
    worst = max(worst, float(np.abs(ref[0] - forward(o)).max()))
print(f"max |numpy - SB3| action difference over 2000 random obs: {worst:.2e}")
assert worst < 1e-4, "export mismatch -- not writing"
np.savez(out, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, mean=mean, var=var, clip=clip, eps=eps)
print("wrote", out)

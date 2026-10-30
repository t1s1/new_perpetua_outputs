"""
Load a trained PPO policy (from train.py) and watch it run in the
interactive viewer, the same real-time-paced viewer setup as
manual_pump.py, but driven by the trained network instead of a hand-coded
rule.

Needs both a model .zip and a matching VecNormalize .pkl (the observation
normalization stats) -- the policy was trained on normalized observations,
so it needs the same normalization applied here or its outputs won't mean
anything.

WHICH CHECKPOINT: defaults to best_model/best_model + best_vec_normalize.pkl
-- the SAME checkpoint evaluate.py and auto_train.py check by default -- not
swing_policy.zip (the raw, unprotected "wherever training last ended"
state). This used to default to swing_policy, which caused real confusion:
after a volatile batch (one round scoring 55.6 deg, the very next scoring
5.0 deg), swing_policy.zip was sitting at a bad moment while best_model.zip
held a good one, so watching swing_policy showed almost no motion even
though evaluate.py was reporting real angles for best_model. Point
MODEL_PATH/VECNORM_PATH at swing_policy.zip/vec_normalize.pkl instead if you
specifically want to watch the raw latest training state rather than the
protected best-so-far checkpoint.

ALSO WORTH KNOWING: this runs in real time (SLOWMO_FACTOR=1.0 below), and
pumping builds up gradually -- even the hand-coded reference controller in
manual_pump.py took until ~120s to reach ~56deg and ~180s to reach ~78deg.
The first minute of watching will look like barely any motion even for a
genuinely good policy, just because amplitude hasn't built up yet -- that's
expected, not a sign anything's wrong.

Run locally:
    mjpython watch_policy.py

(mjpython, not plain python, on macOS -- see manual_pump.py for why.)
"""
import time

import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from swing_env import SwingEnv

MODEL_PATH = "best_model/best_model"
VECNORM_PATH = "best_model/best_vec_normalize.pkl"
# Was TEMPORARILY pointed at checkpoint 13,189,696 (round-2 end /
# swing_policy.zip resume point, README 5.9g) to visually confirm the
# "some deterministic episodes never pump up" finding -- confirmed (and
# explained: README 5.9h), reverted back to the best_model default here.
WATCH_EPISODE_SECONDS = 240.0  # matches swing_env.py's current training
                                 # episode length, so you can watch a full
                                 # pump-up without the view resetting
                                 # mid-cycle
SLOWMO_FACTOR = 1.0

# Rebuild the same VecNormalize wrapper used in training, but load its
# saved running statistics instead of computing fresh ones, then freeze
# it: no further stat updates, no reward normalization needed at
# inference (only observations need to match what the policy trained on).
vec_env = DummyVecEnv([lambda: SwingEnv(episode_seconds=WATCH_EPISODE_SECONDS)])
vec_env = VecNormalize.load(VECNORM_PATH, vec_env)
vec_env.training = False
vec_env.norm_reward = False

model = PPO.load(MODEL_PATH)

# Reach into the underlying SwingEnv for its raw MuJoCo model/data so we
# can hand them directly to MuJoCo's own viewer.
raw_env = vec_env.venv.envs[0]
model_mj = raw_env.model
data_mj = raw_env.data

# One vec_env.step() call already advances frame_skip physics steps
# internally (see swing_env.py) -- that's a coarser, more forgiving unit
# to pace against than the raw 5ms timestep, for the same reason batching
# substeps fixed the fast/slow cycling in manual_pump.py.
sim_seconds_per_step = raw_env.frame_skip * model_mj.opt.timestep

obs = vec_env.reset()

with mujoco.viewer.launch_passive(model_mj, data_mj) as viewer:
    while viewer.is_running():
        step_start = time.time()

        action, _ = model.predict(obs, deterministic=True)
        # VecEnv auto-resets internally when an episode ends and returns
        # the post-reset observation, so there's no manual reset needed
        # here even across episode boundaries.
        obs, reward, done, info = vec_env.step(action)

        viewer.sync()

        target_wall_time = sim_seconds_per_step * SLOWMO_FACTOR
        elapsed = time.time() - step_start
        if elapsed < target_wall_time:
            time.sleep(target_wall_time - elapsed)

import random

import numpy as np
import gymnasium as gym
from thefool.common.env.env_wrapper import EnvWrapper, BatchEnvWrapper
from thefool.common.env.atari_wrapper import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, atari_preprocess
from thefool.DQN.dqn_agent import DQN
def state_preprocess(state):
    state = atari_preprocess(state, [84, 84])
    state = np.array(state, dtype=np.float32)
    return state / 255.0

reward_preprocess = lambda r: np.clip(r, -1.0, 1.0)

env = gym.make("PongNoFrameskip-v4")
env = NoopResetEnv(env, noop_max=30)
env = MaxAndSkipEnv(env)
env = EpisodicLifeEnv(env)
wrapped_env = EnvWrapper(env, [84, 84], 4, r_preprocess=reward_preprocess, s_preprocess=state_preprocess)
batch_train_env = BatchEnvWrapper([wrapped_env])

model = DQN(batch_train_env, batch_train_env)

state = np.array(batch_train_env.reset())
for i in range(100):
    action = model.act(state)
    print(action)
    state, _, _, _ = batch_train_env.step(action)
    state = np.array(state)

env.close()
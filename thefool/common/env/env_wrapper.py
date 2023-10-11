import numpy as np
import copy
from collections import deque
import time


class EnvWrapper:
    def __init__(self, env, client_obs_shape=None, stack_frame=1, r_preprocess=None, s_preprocess=None):
        self.env = env
        self.stack_frame = stack_frame
        if client_obs_shape is not None:
            self.observation_shape = client_obs_shape + [stack_frame]
        else:
            self.observation_shape = env.observation_space.shape
        self.action_space = env.action_space
        self.r_preprocess = r_preprocess
        self.s_preprocess = s_preprocess
        self.index = 0
        if stack_frame > 1:
            self.output_states = deque(np.zeros([stack_frame] + [84, 84], dtype=np.float32).tolist(), maxlen=stack_frame)

    def step(self, action):
        state, reward, done, t, info = self.env.step(action)
        # preprocess reward
        if self.r_preprocess is not None:
            reward = self.r_preprocess(reward)
        # preprocess state
        if self.s_preprocess is not None:
            state = self.s_preprocess(state)
        # if done:
        #     self.reset()
        if self.stack_frame == 1:
            return state, reward, done, info
        self.output_states.append(state)
        states = np.array(list(copy.deepcopy(self.output_states)))
        return np.transpose(states, [1, 2, 0]), reward, done, info

    def reset(self):
        state, _ = self.env.reset(seed=int(time.time()*10))
        # preprocess state
        if self.s_preprocess is not None:
            state = self.s_preprocess(state)
        if self.stack_frame == 1:
            return state
        for _ in range(self.stack_frame):
            self.output_states.append(state)
        states = np.array(list(copy.deepcopy(self.output_states)))
        return np.transpose(states, [1, 2, 0])


class BatchEnvWrapper:
    def __init__(self, envs):
        self.envs = envs
        self.observation_shape = envs[0].observation_shape
        self.action_space = envs[0].action_space
        self.env_nums = len(self.envs)

    def step(self, actions):
        states = []
        rewards = []
        dones = []
        infos = []
        for i, env in enumerate(self.envs):
            state, reward, done, info = env.step(actions[i])
            if done:
                state = env.reset()
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return states, rewards, dones, infos

    def reset_index(self, index):
        return self.envs[index].reset()

    def reset(self):
        return [env.reset() for env in self.envs]

import numpy as np
from thefool.common.dataset.sum_tree import SumTree
import math

class MemoryUnit:
    def __init__(self,
                 obs_shape,
                 size,
                 use_per=False):
        self.obs_shape = obs_shape
        self.obs_t = np.zeros([size] + obs_shape, dtype=np.float32)
        self.actions_t = np.zeros([size], dtype=np.int32)
        self.reward_t = np.zeros([size], dtype=np.float32)
        self.done_t = np.zeros([size], dtype=np.float32)
        self.index = 0
        self.size = size
        self.use_per = use_per
        if use_per:
            self.sum_tree = SumTree(size)

    def add(self, obs, action, reward, done):
        self.obs_t[self.index % self.size] = obs
        self.actions_t[self.index % self.size] = action
        self.reward_t[self.index % self.size] = reward
        self.done_t[self.index % self.size] = done
        if self.use_per:
            self.sum_tree.add(self.index)
        self.index += 1

    def sample(self, batch_size):
        index_max = self.size
        isweight = np.ones(batch_size, dtype=np.float32)
        if self.index < self.size:
            index_max = self.index
        if self.use_per:
            batch_index, isweight = self.sum_tree.sample(batch_size, index_max)
        else:
            batch_index = np.array(np.random.random_integers(0, index_max-1, batch_size))
        next_index = (batch_index + 1) % index_max
        obs_n = self.obs_t[batch_index]
        action_n = self.actions_t[batch_index]
        reward_n = self.reward_t[batch_index]
        done_n = self.done_t[batch_index]
        obs_next_n = self.obs_t[next_index]

        return obs_n, action_n, reward_n, done_n, obs_next_n, isweight

    def n_step_sample(self, batch_size, step=3, gamma=0.99):
        index_max = self.size
        isweight = np.ones(batch_size, dtype=np.float32)
        if self.index < self.size:
            index_max = self.index
        if self.use_per:
            batch_index, isweight = self.sum_tree.sample(batch_size, index_max)
        else:
            batch_index = np.array(np.random.random_integers(0, index_max-1, batch_size))
        obs_n = self.obs_t[batch_index]
        action_n = self.actions_t[batch_index]
        done_n = self.done_t[batch_index]
        reward_n = self.reward_t[batch_index]
        obs_next_n = self.obs_t[(batch_index + step) % index_max]

        # 替换掉最新加入记忆池n-step的index，如果选到的话会导致obs_next_n和reward出错
        newest_index = self.index % self.size
        for i, index in enumerate(batch_index):
            tmp_new_index = newest_index
            if index >= tmp_new_index:
                tmp_new_index += self.size
            if index + step > tmp_new_index:
                batch_index[i] = (batch_index[i] - step) % self.size

        for i in range(step - 1):
            step_index = i + 1
            next_index = (batch_index + step_index) % index_max
            reward_n += self.reward_t[next_index] * math.pow(gamma, step_index) * (1.0 - done_n)
            done_n += self.done_t[next_index]
            done_n = np.minimum(done_n, 1)

        return obs_n, action_n, reward_n, done_n, obs_next_n, isweight



class OffPolicyMemoryPool:
    def __init__(self,
                 nenv,
                 obs_shape,
                 size,
                 n_step=1,
                 use_per=False):
        self.n_step = n_step
        self.nenv = nenv
        self.memory_units = [MemoryUnit(obs_shape, size, use_per) for _ in range(nenv)]

    def add(self, obs_n, action_n, reward_n, done_n):
        for i in range(self.nenv):
            self.memory_units[i].add(
                        obs=obs_n[i],
                        action=action_n[i],
                        reward=reward_n[i],
                        done=1.0 if done_n[i] else 0.0,
                    )

    def sample_batch(self, bathc_size):
        obs_t = []
        actions_t = []
        reward_t = []
        done_t = []
        obs_next_t = []
        isweight_t = []
        for mu in self.memory_units:
            if self.n_step > 1:
                obs_n, action_n, reward_n, done_n, obs_next_n, isweight_n = mu.n_step_sample(bathc_size, self.n_step)
            else:
                obs_n, action_n, reward_n, done_n, obs_next_n, isweight_n = mu.sample(bathc_size)
            obs_t.append(obs_n)
            actions_t.append(action_n)
            reward_t.append(reward_n)
            done_t.append(done_n)
            obs_next_t.append(obs_next_n)
            isweight_t.append(isweight_n)
        return np.concatenate(obs_t, axis=0), np.concatenate(actions_t, axis=0), \
            np.concatenate(reward_t, axis=0), np.concatenate(done_t, axis=0), np.concatenate(obs_next_t, axis=0), \
               np.concatenate(isweight_t, axis=0)

    def sample_unit_batch(self, i, bathc_size):
        if self.n_step > 1:
            return self.memory_units[i].n_step_sample(bathc_size, self.n_step)
        else:
            return self.memory_units[i].sample(bathc_size)

    def sample_update_priority(self, i, priority):
        self.memory_units[i].sum_tree.update(priority)
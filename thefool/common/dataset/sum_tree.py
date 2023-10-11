import copy

import numpy as np
import math
import random

class SumTree:
    def __init__(self, size, beta=0.4):
        self.size = size
        self.tree_depth = int(math.ceil(np.log2(size))) + 1
        self.beta = beta
        self.levels = []
        level_size = 1
        for _ in range(self.tree_depth):
            nodes_at_this_depth = np.zeros(level_size, dtype=np.float32)
            self.levels.append(nodes_at_this_depth)
            level_size *= 2

        self.batch_index = None

    def add(self, upper_index):
        priority = 1.0
        index = upper_index % self.size
        origin_value = self.levels[-1][index]
        value_diff = priority - origin_value
        tmp_index = int(index)
        for i in reversed(range(self.tree_depth)):
            self.levels[i][tmp_index] += value_diff
            tmp_index = int(tmp_index / 2)

    # 原始版本，速度很慢
    def update_origin(self, priority):
        for index in self.batch_index:
            for _ in range(self.tree_depth):
                origin_value = self.levels[-1][index]
                value_diff = priority - origin_value
                tmp_index = int(index)
                for i in reversed(range(self.tree_depth)):
                    self.levels[i][tmp_index] += value_diff
                    tmp_index = int(tmp_index / 2)

    # 原始版本，速度很慢
    def sample_origin(self, batch_size):
        index_values = np.random.random(batch_size) * self.levels[0][0]
        index_out = np.zeros(batch_size, dtype=int)
        for i, v in enumerate(index_values):
            value = v
            left_node = 0
            for t in range(self.tree_depth)[1:]:
                left_node = left_node * 2
                if self.levels[t][left_node] < value:
                    value -= self.levels[t][left_node]
                    left_node += 1
            if left_node > self.size - 1:
                left_node = self.size - 1
            index_out[i] = left_node
        self.batch_index = index_out
        return index_out, index_values

    # 并行更新版本
    def update(self, priority):
        tmp_index = self.batch_index
        for i in reversed(range(self.tree_depth)):
            if i == self.tree_depth - 1:
                self.levels[i][tmp_index] = priority
            else:
                self.levels[i][tmp_index] = self.levels[i+1][tmp_index * 2] + \
                                                            self.levels[i+1][tmp_index * 2 + 1]
            tmp_index = np.int_(tmp_index / 2)

    def sample(self, batch_size, max_index):
        index_values = np.random.random(batch_size) * self.levels[0][0]
        index_out = np.zeros(batch_size, dtype=int)
        for t in range(self.tree_depth)[1:]:
            index_out *= 2
            lsons = self.levels[t][index_out]
            direct = lsons < index_values
            index_values -= lsons * direct
            index_out += direct
        # 以防万一
        index_out = np.minimum(index_out, self.size-1)
        min_prob = np.min(self.levels[-1][:max_index]) + 1e-8
        isweight = np.power(min_prob / (self.levels[-1][index_out] + 1e-8), self.beta)
        self.batch_index = index_out
        return index_out, isweight


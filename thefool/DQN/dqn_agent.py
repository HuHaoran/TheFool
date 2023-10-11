import tensorflow as tf
from thefool.DQN.dqn_network import DQNNetwork
import numpy as np
import random
class DQN:
    def __init__(self,
                 env,
                 test_env,
                 learning_rate=3e-4,
                 memory_size=128000,
                 batch_size=64,
                 gamma=0.99,
                 max_grad_norm=0.5,
                 skip_num=4,
                 replace_step=1000,
                 ):
        self.scope = "ppo"
        self.env = env
        self.test_env = test_env
        self.train_nums = env.env_nums
        self.skip_num = skip_num
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.observation_shape = list(self.env.observation_shape)
        self.num_actions = self.env.action_space.n
        self.max_grad_norm = max_grad_norm
        self.replace_step = replace_step
        self.model = DQNNetwork(self.num_actions)
        self.target_model = DQNNetwork(self.num_actions)
        self.target_model.set_weights(self.model.get_weights())

    def act(self, obs):
        q_values = self.model(obs)
        action_t = tf.argmax(q_values, axis=-1).numpy()
        for i in range(len(action_t)):
            if random.random() < 0.1:
                action_t[i] = random.randint(0, self.num_actions - 1)

        return action_t
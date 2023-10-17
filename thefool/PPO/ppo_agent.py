import tensorflow as tf
from thefool.PPO.ppo_network import PPONetwork
import numpy as np
from thefool.common.dataset.onpolicy_dataset import OnPolicyDataset
from tqdm import tqdm


class PPO:

    def __init__(self,
                 train_env,
                 test_env,
                 learning_rate=3e-4,
                 step_horizon=128,
                 n_epoch=3,
                 batch_size=256,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_range=0.1,
                 value_clip=0.1,
                 normalize_advantage=True,
                 entropy_factor=0.01,
                 value_factor=1,
                 max_grad_norm=0.5,
                 use_sde=False,
                 sde_sample_freq=-1,
                 stats_window_size=100
                 ):
        self.env = train_env
        self.learning_rate = learning_rate
        self.step_horizon = step_horizon
        self.n_epoch = n_epoch
        self.observation_shape = list(self.env.observation_shape)
        self.num_actions = self.env.action_space.n
        self.train_nums = self.env.train_nums
        self.network = PPONetwork(num_actions=self.num_actions)

        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_clip = value_clip
        self.entropy_factor = entropy_factor
        self.value_factor = value_factor
        self.max_grad_norm = max_grad_norm
        self.stats_window_size = stats_window_size

        self.state_tm1 = dict(obs=None, action=None, value=None, log_probs=None, done=None, rnn_state=None)
        self.dataset = OnPolicyDataset(self.train_nums, self.observation_shape, self.step_horizon)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, epsilon=1e-5)
        self.global_step = 0
        self.episode = 0

    def act(self, obs_t, reward_t, done_t):
        action_t, eval_action_t, log_probs_t, value_t = self._act(obs_t)
        value_t = np.reshape(value_t, [-1])

        action_t = action_t[:self.train_nums]
        log_probs_t = log_probs_t[:self.train_nums]
        value_t = value_t[:self.train_nums]
        obs_t = obs_t[:self.train_nums]
        reward_t = reward_t[:self.train_nums]
        done_t = done_t[:self.train_nums]

        eval_action_t = eval_action_t[self.train_nums:]

        if self.state_tm1['obs'] is not None:
            self.dataset.add(states=self.state_tm1['obs'],
                             actions=self.state_tm1['action'],
                             rewards=reward_t,
                             values=self.state_tm1['value'],
                             log_probs=self.state_tm1['log_probs'],
                             dones=done_t)

        if self.global_step > 0 and (self.global_step / self.train_nums) % self.step_horizon == 0:
            bootstrap_values = value_t.copy()
            self.train(bootstrap_values)

        # decay parameters
        self.global_step += self.train_nums
        # self.lr.decay(self.t)
        # self.epsilon.decay(self.t)

        self.state_tm1['obs'] = obs_t
        self.state_tm1['action'] = action_t
        self.state_tm1['value'] = value_t
        self.state_tm1['log_probs'] = log_probs_t
        self.state_tm1['done'] = done_t

        return np.concatenate([action_t, eval_action_t], axis=-1)

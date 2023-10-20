import tensorflow as tf
from thefool.PPO.ppo_network import PPONetwork
import numpy as np
from thefool.common.dataset.onpolicy_dataset import OnPolicyDataset
from thefool.common.process import learn_process


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
        self.test_env = test_env
        self.learning_rate = learning_rate
        self.step_horizon = step_horizon
        self.n_epoch = n_epoch
        self.observation_shape = list(self.env.observation_shape)
        self.num_actions = self.env.action_space.n
        self.train_nums = self.env.env_nums
        self.model = PPONetwork(num_actions=self.num_actions)

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


    @tf.function
    def _tf_act(self, obs_t):
        probs, value = self.model(obs_t)
        log_prob = tf.math.log(probs)
        action = tf.random.categorical(log_prob)
        log_policy = log_prob[:, action]
        return action, log_policy, value

    def act(self, obs_t, reward_t, done_t):
        action_t, log_probs_t, value_t = self._tf_act(obs_t)
        action_t = action_t.numpy()
        log_probs_t = log_probs_t.numpy()
        value_t = value_t.numpy()
        value_t = np.reshape(value_t, [-1])


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

        self.global_step += self.train_nums

        self.state_tm1['obs'] = obs_t
        self.state_tm1['action'] = action_t
        self.state_tm1['value'] = value_t
        self.state_tm1['log_probs'] = log_probs_t
        self.state_tm1['done'] = done_t

        return action_t

    def eval_act(self, obs_t):
        dist, _ = self.model(obs_t)
        probs = dist.probs
        action = tf.argmax(probs, axis=-1)
        return action.numpy()

    def train(self, bootstrap_values):
        obs_t, actions_t, log_probs_t, value_t, tdrets_t, advs_t = \
            self.dataset.rollout_trajectories(bootstrap_values, self.gamma, self.gae_lambda)

        # train network
        local_batch_size = int(self.batch_size / self.train_nums)
        for epoch in range(self.n_epoch):
            # shuffle batch data if without lstm
            indices = np.random.permutation(range(self.step_horizon))
            obs_t = obs_t[:, indices]
            actions_t = actions_t[:, indices]
            log_probs_t = log_probs_t[:, indices]
            value_t = value_t[:, indices]
            tdrets_t = tdrets_t[:, indices]
            advs_t = advs_t[:, indices]
            for i in range(int(self.step_horizon / local_batch_size)):
                batch_actions = self._pick_batch(actions_t, i, local_batch_size)
                batch_log_probs = self._pick_batch(log_probs_t, i, local_batch_size)
                batch_obs = self._pick_batch(obs_t, i, local_batch_size, shape=self.observation_shape)
                batch_values = self._pick_batch(value_t, i, local_batch_size)
                batch_tdrets = self._pick_batch(tdrets_t, i, local_batch_size)
                batch_advs = self._pick_batch(advs_t, i, local_batch_size)
                loss = self._tf_train(batch_obs, batch_actions, batch_values, batch_tdrets, batch_advs, batch_log_probs)

        # clean trajectories
        self.dataset.flush()


    @tf.function
    def _tf_train(self, obs, actions, values, tdrets, advantages, old_log_probs):
        with tf.GradientTape() as tape:
            train_dist, train_value = self.model(obs, training=True)

            value_clipped = values + tf.clip_by_value(train_value - values, -self.value_clip, self.value_clip)
            value_loss1 = tf.reduce_mean(tf.square(train_value - tdrets))
            value_loss2 = tf.reduce_mean(tf.square(value_clipped - tdrets))
            value_loss = self.value_factor * tf.maximum(value_loss1, value_loss2)

            entropy = tf.reduce_mean(train_dist.entropy())
            entropy *= self.entropy_factor

            log_prob = train_dist.log_prob(actions)
            ratio = tf.exp(log_prob - old_log_probs)
            ratio = tf.reshape(ratio, [-1, 1])
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
            surr = tf.minimum(surr1, surr2)
            policy_loss = tf.reduce_mean(surr)

            loss = value_loss - policy_loss - entropy

            gradients = tape.gradient(value_loss, self.model.trainable_variables)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_grad_norm)
            self.optimizer.apply_gradients(zip(clipped_gradients, self.model.trainable_variables))
        return loss

    def _pick_batch(self, data, batch_index, batch_size, flat=True, shape=None):
        start_index = batch_index * batch_size
        batch_data = data[:, start_index:start_index + batch_size]
        if flat:
            if shape is not None:
                return np.reshape(batch_data, [-1] + shape)
            return np.reshape(batch_data, [-1])
        return batch_data


    def learn(self, total_steps: int):
        learn_process(self.env, self.test_env, self.act, self.eval_act, total_steps)
import tensorflow as tf
from thefool.SAC.sac_network import SACNetwork
from thefool.common.dataset.offpolicy_dataset import OffPolicyMemoryPool
from thefool.common.process import learn_process
from thefool.common.uilt.probability import compute_entropy
import numpy as np


class SAC:
    def __init__(self,
                 env,
                 test_env,
                 learning_rate=3e-4,
                 memory_size=64000,
                 batch_size=64,
                 gamma=0.99,
                 max_grad_norm=0.5,
                 skip_num=4,
                 use_dueling=False,
                 use_per=False,
                 n_step=1,
                 soft_replace_rate=0.001,
                 use_average_is=False
                 ):
        self.env = env
        self.test_env = test_env
        self.train_nums = env.env_nums
        self.skip_num = skip_num
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.observation_shape = list(self.env.observation_shape)
        self.num_actions = int(self.env.action_space.n)
        self.max_grad_norm = max_grad_norm
        self.global_step = 0

        self.batch_size = batch_size
        self.gamma = gamma
        self.alpha = 0.05
        self.use_dueling = use_dueling
        self.use_per = use_per
        self.n_step = n_step
        self.soft_replace_rate = np.var(soft_replace_rate, dtype=np.float32)
        self.use_average_is = use_average_is

        self.model = SACNetwork(self.num_actions)
        self.target_model = SACNetwork(self.num_actions)
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, epsilon=1e-5)

        self.state_tm1 = dict(obs=None, action=None, value=None, log_probs=None, done=None)
        self.dataset = OffPolicyMemoryPool(self.train_nums, self.observation_shape,
                                           int(self.memory_size / self.train_nums),
                                           n_step=self.n_step,
                                           use_per=self.use_per)

    @tf.function
    def _tf_act(self, obs_t):
        probs, _, _ = self.model(obs_t, training=False)
        log_probs = tf.math.log(probs)
        action = tf.random.categorical(log_probs, 1)[:, 0]
        return action

    def eval_act(self, obs_t):
        probs, _, _ = self.model(obs_t, training=False)
        action = tf.argmax(probs, axis=-1).numpy()
        return action

    def act(self, obs_t, reward_t, done_t):

        action_t = self._tf_act(obs_t)


        if self.state_tm1['obs'] is not None:
            self.dataset.add(self.state_tm1['obs'],
                                 self.state_tm1['action'],
                                 reward_t,
                                 done_t)

        if self.global_step > 10000:
            if (self.global_step / self.train_nums) % self.skip_num == 0:
                self.train()

        self.global_step += self.train_nums

        self.state_tm1['obs'] = obs_t
        self.state_tm1['action'] = action_t
        self.state_tm1['done'] = done_t

        return action_t

    def train(self):
        for i in range(self.train_nums):
            obs_t, action_t, reward_t, done_t, obs_next_t, isweight_t = self.dataset.sample_unit_batch(i, self.batch_size)
            value_loss = self._tf_train(obs_t, action_t, reward_t, done_t, obs_next_t, isweight_t)

            if self.use_per:
                priority = np.sqrt(np.minimum(value_loss, 1) + 1e-8)
                self.dataset.sample_update_priority(i, priority)

    @tf.function
    def _tf_train(self, obs_t, action_t, reward_t, done_t, obs_next_t, isweight_t):
        with tf.GradientTape() as tape:
            train_probs, train_q1, train_q2 = self.model(obs_t, training=True)
            train_next_probs, _, _ = self.model(obs_next_t, training=True)
            _, target_q1, target_q2 = self.target_model(obs_next_t, training=False)

            min_q_train = tf.stop_gradient(tf.minimum(train_q1, train_q2))
            min_q_target = tf.minimum(target_q1, target_q2)

            q_backup = tf.stop_gradient(reward_t + self.gamma * (1 - done_t) *
                                        (tf.reduce_sum(min_q_target * train_next_probs, axis=-1) +
                                         self.alpha * compute_entropy(train_next_probs)))

            # Soft actor-critic losses
            pi_loss = -tf.reduce_mean(tf.reduce_sum(min_q_train * train_probs, axis=-1)
                                      + self.alpha * compute_entropy(train_probs))
            onehot_action = tf.one_hot(action_t, self.num_actions)
            q1_loss = (q_backup - tf.reduce_sum(train_q1 * onehot_action, axis=-1)) ** 2
            q2_loss = (q_backup - tf.reduce_sum(train_q2 * onehot_action, axis=-1)) ** 2
            value_loss = tf.reduce_mean(q1_loss * isweight_t + q2_loss * isweight_t)

            loss = pi_loss + value_loss

            gradients = tape.gradient(loss, self.model.trainable_variables)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_grad_norm)
            self.optimizer.apply_gradients(zip(clipped_gradients, self.model.trainable_variables))
            if self.use_average_is:
                pri_loss = tf.reduce_mean(value_loss)
            else:
                pri_loss = value_loss
        self.soft_replace_target_weight()
        return pri_loss

    def soft_replace_target_weight(self):
        target_weights = self.target_model.weights
        weights = self.model.weights
        for i in range(len(target_weights)):
            tw = self.soft_replace_rate * weights[i] + (1 - self.soft_replace_rate) * target_weights[i]
            self.target_model.weights[i].assign(tw)

    def learn(self, total_steps: int):
        learn_process(self.env, self.test_env, self.act, self.eval_act, total_steps)

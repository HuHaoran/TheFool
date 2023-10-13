import tensorflow as tf
from thefool.DQN.dqn_network import DQNNetwork
from thefool.common.uilt.scheduler import LinearScheduler
from thefool.common.dataset.offpolicy_dataset import OffPolicyMemoryPool
from thefool.common.process import learn_process
import numpy as np
import random
class DQN:
    def __init__(self,
                 env,
                 test_env,
                 learning_rate=3e-4,
                 memory_size=64000,
                 batch_size=64,
                 gamma=0.99,
                 max_grad_norm=0.5,
                 skip_num=4,
                 replace_step=1000,
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
        self.replace_step = replace_step
        self.linearDecay = LinearScheduler(1.0, 100000, 0.1, "LinearDecay")
        self.global_step = 0

        self.batch_size = batch_size
        self.gamma = gamma
        self.use_per = False

        self.model = DQNNetwork(self.num_actions)
        self.target_model = DQNNetwork(self.num_actions)
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, epsilon=1e-5)

        self.state_tm1 = dict(obs=None, action=None, value=None, log_probs=None, done=None)
        self.dataset = OffPolicyMemoryPool(self.train_nums, self.observation_shape,
                                           int(self.memory_size / self.train_nums))

    def act(self, obs_t, reward_t, done_t):
        q_values = self.model(obs_t, training=False)
        action_t = tf.argmax(q_values, axis=-1).numpy()
        for i in range(len(action_t)):
            if random.random() < self.linearDecay.get_variable():
                action_t[i] = random.randint(0, self.num_actions - 1)

        if self.state_tm1['obs'] is not None:
            self.dataset.add(self.state_tm1['obs'], self.state_tm1['action'], reward_t, done_t)

        if self.global_step > 10000:
            if (self.global_step / self.train_nums) % self.skip_num == 0:
                self.train()
                # print("loss:", loss.numpy())
            if (self.global_step / self.train_nums) % self.replace_step == 0:
                self.replace_target_weight()

        self.global_step += self.train_nums
        self.linearDecay.decay(self.global_step)

        self.state_tm1['obs'] = obs_t
        self.state_tm1['action'] = action_t
        self.state_tm1['done'] = done_t

        return action_t


    def eval_act(self, obs_t):
        q_values = self.model(obs_t, training=False)
        action = tf.argmax(q_values, axis=-1).numpy()
        return action


    def train(self):
        for i in range(self.train_nums):
            obs_t, action_t, reward_t, done_t, obs_next_t, isweight_t = self.dataset.sample_unit_batch(i, self.batch_size)
            self._tf_train(obs_t, action_t, reward_t, done_t, obs_next_t, isweight_t)

            # if self.use_per:
            #     priority = np.sqrt(np.minimum(priority, 1) + 1e-6)
            #     self.dataset.sample_update_priority(i, priority)
            # loss += td_error

    @tf.function
    def _tf_train(self, obs_t, action_t, reward_t, done_t, obs_next_t, isweight_t):
        with tf.GradientTape() as tape:
            train_q_value = self.model(obs_t, training=True)
            target_q_value = self.target_model(obs_next_t)

            y_target = tf.stop_gradient(reward_t + (1 - done_t) * self.gamma * tf.reduce_max(target_q_value, axis=-1))
            num_actions = tf.convert_to_tensor(self.num_actions, dtype=tf.int32)
            a = tf.one_hot(action_t, num_actions)
            b = a * train_q_value
            q_value_action = tf.reduce_sum(b, axis=-1)
            value_loss = tf.reduce_mean(tf.square(y_target - q_value_action))

            gradients = tape.gradient(value_loss, self.model.trainable_variables)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_grad_norm)
            self.optimizer.apply_gradients(zip(clipped_gradients, self.model.trainable_variables))

    def replace_target_weight(self):
        self.target_model.set_weights(self.model.get_weights())

    def learn(self, total_steps: int):
        learn_process(self.env, self.test_env, self.act, self.eval_act, total_steps)

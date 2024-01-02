import tensorflow as tf


class SACNetwork(tf.keras.Model):
    def __init__(self, num_actions, name=None):
        super(SACNetwork, self).__init__(name=name)

        self.num_actions = num_actions

        activation_fn = tf.keras.activations.relu
        self.conv1 = tf.keras.layers.Conv2D(32, [8, 8], strides=4, padding='same',
                                        activation=activation_fn, name='Conv')
        self.conv2 = tf.keras.layers.Conv2D(64, [4, 4], strides=2, padding='same',
                                        activation=activation_fn, name='Conv')
        self.conv3 = tf.keras.layers.Conv2D(64, [3, 3], strides=1, padding='same',
                                        activation=activation_fn, name='Conv')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(512, activation=activation_fn,
                                            name='fully_connected')
        self.policy = tf.keras.layers.Dense(num_actions, activation=tf.keras.activations.softmax, name='policy')

        self.q1 = tf.keras.layers.Dense(num_actions, name='q1')
        self.q2 = tf.keras.layers.Dense(num_actions, name='q2')



    def call(self, state):
        x = tf.cast(state, tf.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        probs = self.policy(x)
        q1 = self.q1(x)
        q2 = self.q2(x)

        return probs, q1, q2


class SACDuelingNetwork(tf.keras.Model):
    def __init__(self, num_actions, name=None):
        super(SACDuelingNetwork, self).__init__(name=name)

        self.num_actions = num_actions

        activation_fn = tf.keras.activations.relu
        self.conv1 = tf.keras.layers.Conv2D(32, [8, 8], strides=4, padding='same',
                                        activation=activation_fn, name='Conv')
        self.conv2 = tf.keras.layers.Conv2D(64, [4, 4], strides=2, padding='same',
                                        activation=activation_fn, name='Conv')
        self.conv3 = tf.keras.layers.Conv2D(64, [3, 3], strides=1, padding='same',
                                        activation=activation_fn, name='Conv')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(512, activation=activation_fn,
                                            name='fully_connected')
        self.policy = tf.keras.layers.Dense(num_actions, activation=tf.keras.activations.softmax, name='policy')

        self.q1_value1 = tf.keras.layers.Dense(512, activation=activation_fn, name='q1_value1')
        self.q1_value2 = tf.keras.layers.Dense(1, name='q1_value2')
        self.q1_adv1 = tf.keras.layers.Dense(512, activation=activation_fn, name='q1_adv1')
        self.q1_adv2 = tf.keras.layers.Dense(num_actions, name='q1_adv2')

        self.q2_value1 = tf.keras.layers.Dense(512, activation=activation_fn, name='q2_value1')
        self.q2_value2 = tf.keras.layers.Dense(1, name='q2_value2')
        self.q2_adv1 = tf.keras.layers.Dense(512, activation=activation_fn, name='q2_adv1')
        self.q2_adv2 = tf.keras.layers.Dense(num_actions, name='q2_adv2')



    def call(self, state):
        x = tf.cast(state, tf.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        p = self.dense(x)
        probs = self.policy(p)

        adv1 = self.q1_adv1(x)
        adv1 = self.q1_adv2(adv1)
        value1 = self.q1_value1(x)
        value1 = self.q1_value2(value1)
        q1 = value1 + (adv1 - tf.reduce_mean(adv1, axis=-1, keepdims=True))

        adv2 = self.q2_adv1(x)
        adv2 = self.q2_adv2(adv2)
        value2 = self.q2_value1(x)
        value2 = self.q2_value2(value2)
        q2 = value2 + (adv2 - tf.reduce_mean(adv2, axis=-1, keepdims=True))


        return probs, q1, q2
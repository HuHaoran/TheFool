import tensorflow as tf


class PPONetwork(tf.keras.Model):
    def __init__(self, num_actions, name=None):
        super(PPONetwork, self).__init__(name=name)

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
        self.value = tf.keras.layers.Dense(1, name='value')


    def call(self, state):
        x = tf.cast(state, tf.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        probs = self.policy(x)
        value = self.value(x)
        return probs, value

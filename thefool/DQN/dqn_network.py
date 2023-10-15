import tensorflow as tf

class DQNNetwork(tf.keras.Model):
    def __init__(self, num_actions, useDueling=False, name=None):
        super(DQNNetwork, self).__init__(name=name)

        self.num_actions = num_actions
        self.useDueling = useDueling

        activation_fn = tf.keras.activations.relu
        self.conv1 = tf.keras.layers.Conv2D(32, [8, 8], strides=4, padding='same',
                                        activation=activation_fn, name='Conv')
        self.conv2 = tf.keras.layers.Conv2D(64, [4, 4], strides=2, padding='same',
                                        activation=activation_fn, name='Conv')
        self.conv3 = tf.keras.layers.Conv2D(64, [3, 3], strides=1, padding='same',
                                        activation=activation_fn, name='Conv')
        self.flatten = tf.keras.layers.Flatten()
        if self.useDueling:
            self.adv1 = tf.keras.layers.Dense(512, activation=activation_fn,
                                        name='adv_fully_connected1')
            self.adv2 = tf.keras.layers.Dense(num_actions, name='adv_fully_connected2')
            self.value1 = tf.keras.layers.Dense(512, activation=activation_fn,
                                        name='value_fully_connected1')
            self.value2 = tf.keras.layers.Dense(1, name='value_fully_connected2')
        else:
            self.dense1 = tf.keras.layers.Dense(512, activation=activation_fn,
                                        name='fully_connected')
            self.dense2 = tf.keras.layers.Dense(num_actions, name='fully_connected')

    def call(self, state):
        x = tf.cast(state, tf.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        if self.useDueling:
            adv = self.adv1(x)
            adv = self.adv2(adv)
            value = self.value1(x)
            value = self.value2(value)
            out = value + (adv - tf.reduce_mean(adv, axis=-1, keepdims=True))
        else:
            x = self.dense1(x)
            out = self.dense2(x)
        return out

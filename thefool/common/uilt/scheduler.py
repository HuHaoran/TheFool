import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class LinearScheduler:
    def __init__(self, initial_value, final_step, final_rate, name):
        self.final_step = final_step
        self.initial_value = initial_value
        self.final_rate = final_rate
        self.variable = tf.Variable(initial_value, name=name)
        self.decayed_ph = tf.placeholder(tf.float32)
        self.decay_op = self.variable.assign(self.decayed_ph)

    def decay(self, step):
        decay = 1.0 - (float(step) / self.final_step)
        if decay < self.final_rate:
            decay = self.final_rate
        feed_dict = {self.decayed_ph: decay * self.initial_value}
        tf.get_default_session().run(self.decay_op, feed_dict=feed_dict)

    def get_variable(self):
        return self.variable

class ConstantScheduler:
    def __init__(self, initial_value, name):
        self.variable = tf.Variable(initial_value, name=name)

    def decay(self, step):
        pass

    def get_variable(self):
        return self.variable
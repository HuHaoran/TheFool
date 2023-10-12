import tensorflow as tf

class LinearScheduler:
    def __init__(self, initial_value, final_step, final_rate, name):
        self.final_step = final_step
        self.initial_value = initial_value
        self.final_rate = final_rate
        self.variable = tf.Variable(initial_value, name=name)

    def decay(self, step):
        decay = 1.0 - (float(step) / self.final_step)
        if decay < self.final_rate:
            decay = self.final_rate
        self.variable.assign(decay)

    def get_variable(self):
        return self.variable.value().numpy()

if __name__ == "__main__":
    ls = LinearScheduler(1.0, 10000, 0.1, "decay")
    for i in range(1000):
        ls.decay(i)
        print(ls.get_variable())


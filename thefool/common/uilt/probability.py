import tensorflow as tf


def compute_entropy(probs):
    log_probs = tf.math.log(probs)
    return tf.reduce_sum(-log_probs*probs, axis=-1)

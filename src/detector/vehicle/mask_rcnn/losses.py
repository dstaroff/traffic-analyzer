import tensorflow as tf


def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), 'float32')
    loss = tf.multiply(0.5, tf.multiply(less_than_one, tf.pow(diff, 2))) + tf.multiply(1 - less_than_one, diff - 0.5)

    return loss

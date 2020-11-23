import tensorflow as tf

def lrelu(x):
    return tf.maximum(x, tf.multiply(x, 0.2))

def relu(x):
    return tf.maximum(0.0, x)

def remove_label(x, img_dim):
    const = 0.7
    x_no_label = tf.Variable(x, validate_shape=False)
    x_no_label = tf.assign(x_no_label[:, img_dim-2, img_dim-1], const*tf.ones_like(x_no_label[:, img_dim-2, img_dim-1]))
    return x_no_label

def label(x, img_dim):
    labels = x[:, img_dim-2, img_dim-1]
    return labels

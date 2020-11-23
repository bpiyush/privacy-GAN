import tensorflow as tf

def binary_cross_entropy(x, z):
    eps = 1e-12
    result =  (-(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps)))
    return result

def information_loss(x, g, delta_mean, delta_sd):
    # to-do : moving avg. update
    orig_mean, orig_var = tf.nn.moments(x, [0])
    gene_mean, gene_var = tf.nn.moments(g, [0])
    
    orig_sd = tf.sqrt(tf.to_float(orig_var))
    gene_sd = tf.sqrt(tf.to_float(gene_var))
    
    mean_diff = tf.subtract(orig_mean, gene_mean)
    sd_diff = tf.subtract(orig_sd, gene_sd)

    mean_loss = tf.norm(mean_diff, ord='euclidean')
    sd_loss = tf.norm(sd_diff, ord='euclidean')
    
    mean_loss = tf.maximum(tf.constant(0.0), mean_loss - delta_mean)
    sd_loss = tf.maximum(tf.constant(0.0), sd_loss - delta_sd)
    
    info_loss = mean_loss + sd_loss
    return info_loss

def classification_loss(lables, predictions, batch_size):
    lables = tf.reshape(lables, [batch_size, 1])
    predictions = tf.reshape(predictions, [batch_size, 1])
    miss_classifications = tf.abs(tf.subtract(lables, predictions))
    class_loss = tf.reduce_mean(miss_classifications)
    return class_loss

def categorical_loss_overlap(x, g, real, zeroes, batch_size):
    _x = tf.reshape(x, [batch_size, -1])
    _g = tf.reshape(g, [batch_size, -1])
    eq = tf.equal(_x[:,real:-zeroes], _g[:,real:-zeroes])
    cat_loss = tf.reduce_sum(tf.cast(eq, tf.float32))
    const = tf.maximum(tf.constant(1.0), tf.cast(tf.size(_x), tf.float32))
    cat_loss = cat_loss / const
    return cat_loss
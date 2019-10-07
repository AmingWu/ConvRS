import tensorflow as tf

def weight_variable_init(shape, name):

    initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable

def bias_variable_init(shape, name):

    initial = tf.constant(0.0, shape = shape)
    return tf.Variable(initial, name=name)
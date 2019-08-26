import tensorflow as tf

def encode_feature(process_feature):
    shape = process_feature.get_shape().as_list()
    two_set = []
    for i in range(0,shape[1]-1,2):
        m1 = process_feature[:,i,:]
        m2 = process_feature[:,i+1,:]
        diff = m2 - m1
        m1 = m1 + tf.nn.relu(diff * -1)
        m2 = m2 + tf.nn.relu(diff)
        two_set.append(tf.expand_dims(m1, 1))
        two_set.append(tf.expand_dims(m2, 1))
    two_set = tf.concat(two_set, 1)

    feature = two_set
    return feature

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

def attention(h, feature3, attention_weights_rnn):

    att_w = []
    sum_w = 0
    shape = feature3.get_shape().as_list()
    for j in range(shape[1]):
        mid1 = tf.matmul(h[:,0,:], attention_weights_rnn['w_a']) + tf.matmul(feature3[:, j, :], attention_weights_rnn['u_a']) + attention_weights_rnn['b_a']
        mid1 = tf.tanh(mid1)
        e_j = tf.matmul(mid1, attention_weights_rnn['w_t'])
        e_j = tf.exp(e_j)
        sum_w += e_j
        att_w.append(e_j)

    att_feature = 0
    for j in range(shape[1]):
        att_feature += (att_w[j] / sum_w) * feature3[:, j, :]

    att_feature = tf.expand_dims(att_feature, 1)
    return att_feature

def encode(process_feature, weights_encode, bias_encode):

    v_shape = process_feature.get_shape().as_list()
    h = tf.zeros([v_shape[0], 512])
    c = tf.zeros([v_shape[0], 512])
    out_h = 0
    out_c = 0
    for i in range(v_shape[1]):

        feature = process_feature[:,i,:]
        f_t = tf.matmul(feature, weights_encode['w_f']) + tf.matmul(h, weights_encode['u_f']) + bias_encode['e_bias']
        f_t = tf.nn.sigmoid(f_t)

        i_t = tf.matmul(feature, weights_encode['w_f']) + tf.matmul(h, weights_encode['u_i']) + bias_encode['e_bias']
        i_t = tf.nn.sigmoid(i_t)

        o_t = tf.matmul(feature, weights_encode['w_f']) + tf.matmul(h, weights_encode['u_o']) + bias_encode['e_bias']
        o_t = tf.nn.sigmoid(o_t)

        new_c = tf.matmul(feature, weights_encode['w_f']) + tf.matmul(h, weights_encode['u_c']) + bias_encode['e_bias']
        new_c = tf.nn.tanh(new_c)

        c = f_t * c + i_t * new_c
        h = o_t * tf.nn.tanh(c)

        out_h = h
        out_c = c

    return out_h, out_c

def decode(feature, attention, h, c, weights_decode, bias_decode):

    f_t = tf.matmul(feature, weights_decode['dw_f']) + tf.matmul(h, weights_decode['du_f']) + bias_decode['d_bias'] + tf.matmul(attention, weights_decode['a_f'])
    f_t = tf.nn.sigmoid(f_t)

    i_t = tf.matmul(feature, weights_decode['dw_i']) + tf.matmul(h, weights_decode['du_i']) + bias_decode['d_bias'] + tf.matmul(attention, weights_decode['a_f'])
    i_t = tf.nn.sigmoid(i_t)

    o_t = tf.matmul(feature, weights_decode['dw_o']) + tf.matmul(h, weights_decode['du_o']) + bias_decode['d_bias'] + tf.matmul(attention, weights_decode['a_f'])
    o_t = tf.nn.sigmoid(o_t)

    new_c = tf.matmul(feature, weights_decode['dw_c']) + tf.matmul(h, weights_decode['du_c']) + bias_decode['d_bias'] + tf.matmul(attention, weights_decode['a_f'])
    new_c = tf.nn.tanh(new_c)

    c = f_t * c + i_t * new_c
    h = o_t * tf.nn.tanh(c)

    return h, c
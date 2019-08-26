import numpy as np
import tensorflow as tf
from utils import *

def weight_variable_init(shape, name):

    initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable

def bias_variable_init(shape, name):

    initial = tf.constant(0.0, shape = shape)
    return tf.Variable(initial, name=name)

d_channel = 512

word_embed = {
    'embed': weight_variable_init([12596, d_channel], 'embed'),
    'd_embed': weight_variable_init([d_channel, 12596], 'd_embed')
}

embed_bias = {
    'bias_embed': bias_variable_init([d_channel], 'bias_embed'),
    'bias_dembed': bias_variable_init([12596], 'bias_dembed')
}

Encode = {
    '1_1_w': weight_variable_init([1,2048,512], '1_1_w'),
    '1_1_b': bias_variable_init([512], '1_1_b'),
    '1_2_w': weight_variable_init([1,512,2048], '1_2_w'),
    '1_2_b': bias_variable_init([2048], '1_2_b')
}

Dilation = {
    '1': weight_variable_init([2,512,256], '1'),
    '1_b': bias_variable_init([256], '1_b'),
    '2': weight_variable_init([2,256,256], '2'),
    '2_b': bias_variable_init([256], '2_b'),
    '3': weight_variable_init([2,256,512], '3'),
    '3_b': bias_variable_init([512], '3_b'),

    '1_n': weight_variable_init([2,512,256], '1_n'),
    '1_b_n': bias_variable_init([256], '1_b_n'),
    '2_n': weight_variable_init([2,256,256], '2_n'),
    '2_b_n': bias_variable_init([256], '2_b_n'),
    '3_n': weight_variable_init([2,256,512], '3_n'),
    '3_b_n': bias_variable_init([512], '3_b_n'),

    '4': weight_variable_init([2,512,512], '4'),
    '4_b': bias_variable_init([512], '4_b'),
    '4_n': weight_variable_init([2,512,512], '4_n'),
    '4_b_n': bias_variable_init([512], '4_b_n'),

    '0': weight_variable_init([2,512,512], '0'),
    '0_b': bias_variable_init([512], '0_b'),
    '0_n': weight_variable_init([2,512,512], '0_n'),
    '0_b_n': bias_variable_init([512], '0_b_n')
}

channel_fusion = {
    'finnal': weight_variable_init([1, 1024, 512], 'finnal')
}


first_attention = {
    'w_a': weight_variable_init([512, 100], 'w_a'),
    'u_a': weight_variable_init([512, 100], 'u_a'),
    'b_a': bias_variable_init([100], 'b_a'),
    'w_t': weight_variable_init([100, 1], 'w_t')
}

fouth_attention = {
    'w_a': weight_variable_init([512, 100], 'w_a'),
    'u_a': weight_variable_init([512, 100], 'u_a'),
    'b_a': bias_variable_init([100], 'b_a'),
    'w_t': weight_variable_init([100, 1], 'w_t')
}

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

Fusion = {
    'w1': weight_variable_init([512,256], 'w1'),
    'w1_b': bias_variable_init([256], 'w1_b'),
    'w2': weight_variable_init([512,256], 'w2'),
    'w2_b': bias_variable_init([256], 'w2_b'),
    'w3': weight_variable_init([256,512], 'w3'),
    'w3_b': bias_variable_init([512], 'w3_b')
    }

def recurent_matrix(text_f, visual_f):

    text_f = tf.matmul(text_f, Fusion['w1']) + Fusion['w1_b']
    visual_f = tf.matmul(visual_f, Fusion['w2']) + Fusion['w2_b']
    shape = text_f.get_shape().as_list()
    text_set = []
    text0 = tf.expand_dims(text_f, 2)
    text_set.append(text0)

    visual_set = []
    visual0 = tf.expand_dims(visual_f, 2)
    visual_set.append(visual0)

    for i in range(1,shape[1]):

        pre=text0[:,(shape[1]-i):,:]
        host=text0[:,0:(shape[1]-i),:]
        text1=tf.concat([pre,host],1)
        text_set.append(text1)

        pre=visual0[:,(shape[1]-i):,:]
        host=visual0[:,0:(shape[1]-i),:]
        visual1=tf.concat([pre,host],1)
        visual_set.append(visual1)

    text_vector=tf.concat(text_set, 2)
    visual_vector=tf.concat(visual_set, 2)

    text_vector=tf.transpose(text_vector, perm=[0,2,1])
    visual_vector=tf.transpose(visual_vector, perm=[0,2,1])

    return visual_vector, text_vector

def MCF(text_f, visual_f):

    visual_vector, text_vector = recurent_matrix(text_f, visual_f)

    text_f = tf.expand_dims(text_f, 2)
    visual_f = tf.expand_dims(visual_f, 2)

    fusion_text = tf.matmul(visual_vector, text_f)
    fusion_visual = tf.matmul(text_vector, visual_f)

    fusion_text = tf.nn.relu(fusion_text)
    fusion_visual = tf.nn.relu(fusion_visual)

    fusion_text = tf.squeeze(fusion_text, 2)
    fusion_visual = tf.squeeze(fusion_visual, 2)

    fusion_v = fusion_text + fusion_visual
    fusion_v = tf.matmul(fusion_v, Fusion['w3']) + Fusion['w3_b']

    return fusion_v

def model(feature, caption, mask, dropout):

    #encoder
    process_feature = feature
    feature_set = encode_feature(process_feature)
    fea4 = tf.nn.conv1d(feature_set, Encode['1_1_w'], 1, 'SAME') + Encode['1_1_b']
    fea4 = tf.nn.relu(fea4)
    recon = tf.nn.conv1d(fea4, Encode['1_2_w'], 1, 'SAME') + Encode['1_2_b']
    recon_loss = tf.sqrt(tf.reduce_mean(tf.square(recon - process_feature)))

    feature3 = fea4
    feature3 = tf.nn.dropout(feature3, dropout)
    #dilation decoder
    loss = 0
    first_layer = []
    second_layer = []
    third_layer = []
    fouth_layer = []
    zero_layer = []
    v_shape = fea4.get_shape().as_list()
    video_hidden = tf.reduce_mean(fea4, 1)
    x = tf.zeros([v_shape[0], v_shape[2]])
    rnn_attention = tf.reduce_mean(feature3, 1)
    TEXT_V = 0
    VISUAL_V = 0
    for i in range(11):
        if i == 0:

            input_f = x + rnn_attention
            input_f = tf.expand_dims(input_f, 1)
            zero_layer.append(input_f)
            shape = zero_layer[i].get_shape().as_list()
            pad = tf.zeros([shape[0], 1, shape[2]])
            new_input = tf.concat([pad, zero_layer[i]], 1)
            f1 = tf.nn.conv1d(new_input, Dilation['0'], 1, 'VALID') + Dilation['0_b']
            f1 = tf.nn.sigmoid(f1)
            f2 = tf.nn.conv1d(new_input, Dilation['0_n'], 1, 'VALID') + Dilation['0_b_n']
            f2 = tf.tanh(f2)
            f = f1 * f2
            hidden = tf.squeeze(f, 1)

            att_feature = attention(tf.expand_dims(hidden, 1), feature3, first_attention)

            first_fusion = tf.concat([tf.expand_dims(hidden, 1), att_feature], 2)
            first_fusion = tf.nn.conv1d(first_fusion, channel_fusion['finnal'], 1, 'SAME')
            first_layer.append(first_fusion)

            input_f = first_layer[i]
            shape = input_f.get_shape().as_list()
            pad = tf.zeros([shape[0], 1, shape[2]])
            new_input = tf.concat([pad, input_f], 1)
            f1 = tf.nn.conv1d(new_input, Dilation['1'], 1, 'VALID') + Dilation['1_b']
            f1 = tf.nn.sigmoid(f1)
            f2 = tf.nn.conv1d(new_input, Dilation['1_n'], 1, 'VALID') + Dilation['1_b_n']
            f2 = tf.tanh(f2)
            f = f1 * f2

            f = tf.nn.dropout(f, dropout)
            second_layer.append(f)

            shape = f.get_shape().as_list()
            m1 = tf.zeros([shape[0], 1, shape[2]])
            m2 = f
            m = tf.concat([m1, m2], 1)
            f1 = tf.nn.conv1d(m, Dilation['2'], 1, 'VALID') + Dilation['2_b']
            f1 = tf.nn.sigmoid(f1)
            f2 = tf.nn.conv1d(m, Dilation['2_n'], 1, 'VALID') + Dilation['2_b_n']
            f2 = tf.tanh(f2)
            f = f1 * f2

            f = tf.nn.dropout(f, dropout)
            third_layer.append(f)

            shape = f.get_shape().as_list()
            m1 = tf.zeros([shape[0], 1, shape[2]])
            m2 = f
            m = tf.concat([m1, m2], 1)
            f1 = tf.nn.conv1d(m, Dilation['3'], 1, 'VALID') + Dilation['3_b']
            f1 = tf.nn.sigmoid(f1)
            f2 = tf.nn.conv1d(m, Dilation['3_n'], 1, 'VALID') + Dilation['3_b_n']
            f2 = tf.tanh(f2)
            f = f1 * f2

            att_feature = attention(f, feature3, fouth_attention)

            fouth_layer.append(f + att_feature)
            shape = f.get_shape().as_list()
            m1 = tf.zeros([shape[0], 1, shape[2]])
            m2 = f
            m = tf.concat([m1, m2], 1)
            f1 = tf.nn.conv1d(m, Dilation['4'], 1, 'VALID') + Dilation['4_b']
            f1 = tf.nn.sigmoid(f1)
            f2 = tf.nn.conv1d(m, Dilation['4_n'], 1, 'VALID') + Dilation['4_b_n']
            f2 = tf.tanh(f2)
            f = f1 * f2

            word_in = f + att_feature
            word_in = tf.nn.dropout(word_in, dropout)
            word = tf.matmul(word_in[:, 0, :], word_embed['d_embed']) + embed_bias['bias_dembed']
            loss1 = tf.nn.softmax_cross_entropy_with_logits(logits=word, labels=caption[:, i+1, :])
            loss1 = loss1 * mask[:, i+1]

            word_hidden = tf.nn.dropout(hidden, dropout)
            word_hidden = tf.matmul(word_hidden, word_embed['d_embed']) + embed_bias['bias_dembed']
            loss2 = tf.nn.softmax_cross_entropy_with_logits(logits=word_hidden, labels=caption[:, i+1, :])
            loss2 = loss2 * mask[:, i+1]

            word_first = tf.nn.dropout(fouth_layer[i], dropout)
            word_first = tf.matmul(word_first[:, 0, :], word_embed['d_embed']) + embed_bias['bias_dembed']
            loss3 = tf.nn.softmax_cross_entropy_with_logits(logits=word_first, labels=caption[:, i+1, :])
            loss3 = loss3 * mask[:, i+1]

            loss += tf.reduce_sum(loss1 * 0.6 + loss2 * 0.2 + loss3 * 0.2)

            x1 = caption[:, i+1, :]
            x = tf.matmul(x1, word_embed['embed']) + embed_bias['bias_embed']
        else:

            input_f=MCF(x, rnn_attention)
            input_f=tf.expand_dims(input_f, 1)

            zero_layer.append(input_f)
            m1 = zero_layer[i-1]
            m2 = zero_layer[i]
            new_input = tf.concat([m1, m2], 1)
            f1 = tf.nn.conv1d(new_input, Dilation['0'], 1, 'VALID') + Dilation['0_b']
            f1 = tf.nn.sigmoid(f1)
            f2 = tf.nn.conv1d(new_input, Dilation['0_n'], 1, 'VALID') + Dilation['0_b_n']
            f2 = tf.tanh(f2)
            f = f1 * f2
            hidden = tf.squeeze(f, 1)

            att_feature = attention(tf.expand_dims(hidden, 1), feature3, first_attention)

            first_fusion = tf.concat([tf.expand_dims(hidden, 1), att_feature], 2)
            first_fusion = tf.nn.conv1d(first_fusion, channel_fusion['finnal'], 1, 'SAME')
            first_layer.append(first_fusion)

            m1 = first_layer[i-1]
            m2 = first_layer[i]
            new_input = tf.concat([m1, m2], 1)
            f1 = tf.nn.conv1d(new_input, Dilation['1'], 1, 'VALID') + Dilation['1_b']
            f1 = tf.nn.sigmoid(f1)
            f2 = tf.nn.conv1d(new_input, Dilation['1_n'], 1, 'VALID') + Dilation['1_b_n']
            f2 = tf.tanh(f2)
            f = f1 * f2
            f = tf.nn.dropout(f, dropout)
            second_layer.append(f)

            if i - 2 < 0:
                shape = f.get_shape().as_list()
                pad = tf.zeros([shape[0], 1, shape[2]])
                new_input = tf.concat([pad, second_layer[i]], 1)
                f1 = tf.nn.conv1d(new_input, Dilation['2'], 1, 'VALID') + Dilation['2_b']
                f1 = tf.nn.sigmoid(f1)
                f2 = tf.nn.conv1d(new_input, Dilation['2_n'], 1, 'VALID') + Dilation['2_b_n']
                f2 = tf.tanh(f2)
                f = f1 * f2
                f = tf.nn.dropout(f, dropout)
                third_layer.append(f)
            else:
                m1 = second_layer[i-2]
                m2 = second_layer[i]
                new_input = tf.concat([m1, m2], 1)
                f1 = tf.nn.conv1d(new_input, Dilation['2'], 1, 'VALID') + Dilation['2_b']
                f1 = tf.nn.sigmoid(f1)
                f2 = tf.nn.conv1d(new_input, Dilation['2_n'], 1, 'VALID') + Dilation['2_b_n']
                f2 = tf.tanh(f2)
                f = f1 * f2
                f = tf.nn.dropout(f, dropout)
                third_layer.append(f)

            att_feature = 0
            if i - 4 < 0:
                shape = f.get_shape().as_list()
                pad = tf.zeros([shape[0], 1, shape[2]])
                new_input = tf.concat([pad, third_layer[i]], 1)
                f1 = tf.nn.conv1d(new_input, Dilation['3'], 1, 'VALID') + Dilation['3_b']
                f1 = tf.nn.sigmoid(f1)
                f2 = tf.nn.conv1d(new_input, Dilation['3_n'], 1, 'VALID') + Dilation['3_b_n']
                f2 = tf.tanh(f2)
                f = f1 * f2
                att_feature = attention(f, feature3, fouth_attention)
                fouth_layer.append(f + att_feature)
            else:
                m1 = third_layer[i-4]
                m2 = third_layer[i]
                new_input = tf.concat([m1, m2], 1)
                f1 = tf.nn.conv1d(new_input, Dilation['3'], 1, 'VALID') + Dilation['3_b']
                f1 = tf.nn.sigmoid(f1)
                f2 = tf.nn.conv1d(new_input, Dilation['3_n'], 1, 'VALID') + Dilation['3_b_n']
                f2 = tf.tanh(f2)
                f = f1 * f2
                att_feature = attention(f, feature3, fouth_attention)
                fouth_layer.append(f + att_feature)

            if i - 2 < 0:
                shape = f.get_shape().as_list()
                pad = tf.zeros([shape[0], 1, shape[2]])
                new_input = tf.concat([pad, fouth_layer[i]], 1)
                f1 = tf.nn.conv1d(new_input, Dilation['4'], 1, 'VALID') + Dilation['4_b']
                f1 = tf.nn.sigmoid(f1)
                f2 = tf.nn.conv1d(new_input, Dilation['4_n'], 1, 'VALID') + Dilation['4_b_n']
                f2 = tf.tanh(f2)
                f = f1 * f2

                word_in = f + att_feature
                word_in = tf.nn.dropout(word_in, dropout)
                word = tf.matmul(word_in[:,0,:], word_embed['d_embed']) + embed_bias['bias_dembed']
                loss1 = tf.nn.softmax_cross_entropy_with_logits(logits=word, labels=caption[:, i+1, :])
                loss1 = loss1 * mask[:, i+1]

                word_hidden = tf.nn.dropout(hidden, dropout)
                word_hidden = tf.matmul(word_hidden, word_embed['d_embed']) + embed_bias['bias_dembed']
                loss2 = tf.nn.softmax_cross_entropy_with_logits(logits=word_hidden, labels=caption[:, i+1, :])
                loss2 = loss2 * mask[:, i+1]

                word_first = tf.nn.dropout(fouth_layer[i], dropout)
                word_first = tf.matmul(word_first[:, 0, :], word_embed['d_embed']) + embed_bias['bias_dembed']
                loss3 = tf.nn.softmax_cross_entropy_with_logits(logits=word_first, labels=caption[:, i+1, :])
                loss3 = loss3 * mask[:, i+1]

                loss += tf.reduce_sum(loss1 * 0.6 + loss2 * 0.2 + loss3 * 0.2)

                x1 = caption[:, i+1, :]
                x = tf.matmul(x1, word_embed['embed']) + embed_bias['bias_embed']
            else:
                m1 = fouth_layer[i-2]
                m2 = fouth_layer[i]
                new_input = tf.concat([m1, m2], 1)
                f1 = tf.nn.conv1d(new_input, Dilation['4'], 1, 'VALID') + Dilation['4_b']
                f1 = tf.nn.sigmoid(f1)
                f2 = tf.nn.conv1d(new_input, Dilation['4_n'], 1, 'VALID') + Dilation['4_b_n']
                f2 = tf.tanh(f2)
                f = f1 * f2

                word_in = f + att_feature
                word_in = tf.nn.dropout(word_in, dropout)
                word = tf.matmul(word_in[:,0,:], word_embed['d_embed']) + embed_bias['bias_dembed']
                loss1 = tf.nn.softmax_cross_entropy_with_logits(logits=word, labels=caption[:, i+1, :])
                loss1 = loss1 * mask[:, i+1]

                word_hidden = tf.nn.dropout(hidden, dropout)
                word_hidden = tf.matmul(word_hidden, word_embed['d_embed']) + embed_bias['bias_dembed']
                loss2 = tf.nn.softmax_cross_entropy_with_logits(logits=word_hidden, labels=caption[:, i+1, :])
                loss2 = loss2 * mask[:, i+1]

                word_first = tf.nn.dropout(fouth_layer[i], dropout)
                word_first = tf.matmul(word_first[:, 0, :], word_embed['d_embed']) + embed_bias['bias_dembed']
                loss3 = tf.nn.softmax_cross_entropy_with_logits(logits=word_first, labels=caption[:, i+1, :])
                loss3 = loss3 * mask[:, i+1]

                loss += tf.reduce_sum(loss1 * 0.6 + loss2 * 0.2 + loss3 * 0.2)

                x1 = caption[:, i+1, :]
                x = tf.matmul(x1, word_embed['embed']) + embed_bias['bias_embed']

    loss = loss / tf.reduce_sum(mask)
    loss = 0.9 * loss + 0.1 * recon_loss

    return loss

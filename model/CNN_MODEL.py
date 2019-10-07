import tensorflow as tf
from model.fusion import recurent_matrix, content_fusion
from model.dataset import next_supervise_batch
from model.preprocess import encode_feature, attention, encode, decode
from model.init import weight_variable_init, bias_variable_init

def modal_fusion(text_f, visual_f, fusion_para, dropout):

    text_f = tf.matmul(text_f, fusion_para['text'])
    visual_f = tf.matmul(visual_f, fusion_para['visual'])

    #First stage
    fusion_visual, fusion_text = content_fusion(text_f, visual_f)
    #fusion
    fusion_v = fusion_visual + fusion_text

    fusion_v = tf.matmul(fusion_v, fusion_para['fusion'])
    fusion_v = tf.nn.dropout(fusion_v, dropout)
    return fusion_v

def forward_feature(feature, caption, mask, dropout, first_attention, fouth_attention, \
    fusion_para, Encode, Dilation, channel_fusion, word_embed, embed_bias):

    #encoder
    process_feature = feature
    feature_set = encode_feature(process_feature)
    #feature_set = tf.nn.dropout(feature_set, 0.7)
    fea4 = tf.nn.conv1d(feature_set, Encode['1_1_w'], 1, 'SAME') + Encode['1_1_b']
    fea4 = tf.nn.relu(fea4)
    recon = tf.nn.conv1d(fea4, Encode['1_2_w'], 1, 'SAME') + Encode['1_2_b']
    recon_loss = tf.sqrt(tf.reduce_mean(tf.square(recon - process_feature)))

    feature3 = fea4
    feature3 = tf.nn.dropout(feature3, dropout)
    #dilation decoder
    loss = 0
    result = []
    h = tf.reduce_mean(fea4, 1)
    h = tf.expand_dims(h, 1)
    first_layer = []
    second_layer = []
    third_layer = []
    fouth_layer = []
    zero_layer = []
    v_shape = fea4.get_shape().as_list()
    true_memory = []
    #hidden, cell = encode(fea4, dropout)
    video_hidden = tf.reduce_mean(fea4, 1)
    x = tf.zeros([v_shape[0], v_shape[2]])
    rnn_attention = tf.reduce_mean(feature3, 1)
    TEXT_V = 0
    VISUAL_V = 0
    for i in range(17):
        if i == 0:

            input_f = x + rnn_attention
            input_f = tf.expand_dims(input_f, 1)
            #input_f = tf.nn.relu(input_f)
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

            result.append(word)
            loss += tf.reduce_sum(loss1 * 0.6 + loss2 * 0.2 + loss3 * 0.2)

            x1 = caption[:, i+1, :]
            x = tf.matmul(x1, word_embed['embed']) + embed_bias['bias_embed']
            true_memory.append(x)
        else:
            input_f=modal_fusion(x, rnn_attention, fusion_para, dropout)
            input_f=tf.expand_dims(input_f, 1)
            #input_f=tf.nn.relu(input_f)

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
            #first_layer.append(tf.expand_dims(hidden, 1) + att_feature)

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

                result.append(word)
                loss += tf.reduce_sum(loss1 * 0.6 + loss2 * 0.2 + loss3 * 0.2)

                x1 = caption[:, i+1, :]
                x = tf.matmul(x1, word_embed['embed']) + embed_bias['bias_embed']
                true_memory.append(x)
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

                result.append(word)
                loss += tf.reduce_sum(loss1 * 0.6 + loss2 * 0.2 + loss3 * 0.2)

                x1 = caption[:, i+1, :]
                x = tf.matmul(x1, word_embed['embed']) + embed_bias['bias_embed']
                true_memory.append(x)

    loss = loss / tf.reduce_sum(mask)

    loss1 = 0
    for i in range(17):
        loss1 += true_memory[i] * tf.expand_dims(mask[:, i+1], 1)
    scalar = tf.reduce_sum(mask, 1)
    scalar = tf.expand_dims(scalar, 1)
    loss1 = tf.div(loss1, scalar)
    loss1 = tf.reduce_mean(tf.abs(video_hidden - tf.matmul(loss1, channel_fusion['correlation'])))
    loss = 0.1 * loss1 + loss + 0.1 * recon_loss
    #loss = 0.3 * recon_loss + loss

    return result, loss
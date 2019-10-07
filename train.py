import tensorflow as tf
import os
import time
import numpy as np
import h5py

from model.dataset import next_supervise_batch
from model.preprocess import encode_feature, attention, encode, decode
from model.init import weight_variable_init, bias_variable_init
from model.fusion import recurent_matrix, content_fusion
from model.CNN_MODEL import forward_feature

train_path = 'msrvtt/data'
train = h5py.File(train_path, 'r')
train_data = train['feature']
train_cap = train['caption']
train_num = train_data.shape[0]
batch_size = 128

d_channel = 512
dic_num = 23308

word_embed = {
    'embed': weight_variable_init([dic_num, d_channel], 'embed'),
    'd_embed': weight_variable_init([d_channel, dic_num], 'd_embed')
}

embed_bias = {
    'bias_embed': bias_variable_init([d_channel], 'bias_embed'),
    'bias_dembed': bias_variable_init([dic_num], 'bias_dembed')
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

    'finnal': weight_variable_init([1, 1024, 512], 'finnal'),
    'correlation': weight_variable_init([512, 512], 'correlation'),
    'w_f': weight_variable_init([1, 1024, 512], 'w_f'),

    'z_f': weight_variable_init([2, 512, 512], 'z_f'),
    'z_f1': weight_variable_init([2, 512, 512], 'z_f1'),

    'n_f': weight_variable_init([512, 1], 'n_f')
}

weights_decode = {
    'dw_f': weight_variable_init([512, 512], 'dw_f'),
    'du_f': weight_variable_init([512, 512], 'du_f'),
    'a_f': weight_variable_init([512, 512], 'a_f'),
    'dw_i': weight_variable_init([512, 512], 'dw_i'),
    'du_i': weight_variable_init([512, 512], 'du_i'),
    'dw_o': weight_variable_init([512, 512], 'dw_o'),
    'du_o': weight_variable_init([512, 512], 'du_o'),
    'dw_c': weight_variable_init([512, 512], 'dw_c'),
    'du_c': weight_variable_init([512, 512], 'du_c')
}

bias_decode = {
    'd_bias': bias_variable_init([512], 'd_bias')
}

weights_encode = {
    'w_f': weight_variable_init([512, 512], 'w_f'),
    'u_f': weight_variable_init([512, 512], 'u_f'),
    'u_i': weight_variable_init([512, 512], 'u_i'),
    'u_o': weight_variable_init([512, 512], 'u_o'),
    'u_c': weight_variable_init([512, 512], 'u_c')
}

bias_encode = {
    'e_bias': bias_variable_init([512], 'e_bias')
}

fusion_para = {
    'visual': weight_variable_init([512, 256], 'visual'),
    'text': weight_variable_init([512, 256], 'text'),
    'fusion': weight_variable_init([256, 512], 'fusion')
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

if __name__ == '__main__':

    l1 = tf.nn.l2_loss(word_embed['embed'])
    l2 = tf.nn.l2_loss(word_embed['d_embed'])
    l3 = tf.nn.l2_loss(Encode['1_1_w'])
    l4 = tf.nn.l2_loss(Dilation['1'])
    l5 = tf.nn.l2_loss(Dilation['2'])
    l6 = tf.nn.l2_loss(Dilation['3'])
    l7 = tf.nn.l2_loss(Dilation['1_n'])
    l8 = tf.nn.l2_loss(Dilation['2_n'])
    l9 = tf.nn.l2_loss(Dilation['3_n'])
    l10 = tf.nn.l2_loss(first_attention['w_a'])
    l11 = tf.nn.l2_loss(first_attention['u_a'])
    l12 = tf.nn.l2_loss(first_attention['w_t'])
    #l13 = tf.nn.l2_loss(channel_fusion['w_f'])
    reg = l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8 + l9 + l10 + l11 + l12
    reg = reg * 1e-3

    feature = tf.placeholder(tf.float32, [batch_size, 20, 2048])
    caption = tf.placeholder(tf.float32, [batch_size, 18, dic_num])
    caption_mask = tf.placeholder(tf.float32, [batch_size, 18])
    dropout = tf.placeholder(tf.float32)

    cap, Loss = forward_feature(feature, caption, caption_mask, dropout, first_attention, fouth_attention, \
        fusion_para, Encode, Dilation, channel_fusion, word_embed, embed_bias)

    lr = 0.0001
    s_sample_num = train_num
    n_step_epoch = s_sample_num / batch_size
    perm_label = np.arange(s_sample_num)
    np.random.shuffle(perm_label)
    index_label = 0

    g_optim = tf.train.AdamOptimizer(lr).minimize(Loss + reg)
    config = tf.ConfigProto()

    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    num_examples = train_num

    saver = tf.train.Saver(max_to_keep=60)
    ckpt = tf.train.get_checkpoint_state('checkpoints/')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        epoch_n = int(ckpt.model_checkpoint_path.split('-')[1])
        print "Restored Epoch ", epoch_n
    else:
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        init = tf.global_variables_initializer()
        sess.run(init)

    n_epoch = 30
    for epoch in range(n_epoch):

        start = time.clock()
        n_batch = 0.0
        train_loss = 0.0
        for i in range(n_step_epoch):

            Data_s, Cap_s, Perm_s, index_s = next_supervise_batch(perm_label=perm_label,index_in_epoch=index_label)
            perm_label = Perm_s
            index_label = index_s

            Cap_t = np.zeros((batch_size, 18, dic_num))
            mask = np.zeros((batch_size, 18))
            for m in range(batch_size):
                for n in range(18):
                    if Cap_s[m][n] > 0:
                        mask[m][n] = 1.0
                    pos = int(Cap_s[m][n])
                    Cap_t[m][n][pos] = 1.0

            Data_s = np.array(Data_s)
            Cap_t = np.array(Cap_t)
            Mask = np.array(mask)
            dropout_level = 0.5

            _, loss1 = sess.run([g_optim, Loss], feed_dict={feature: Data_s, caption: Cap_t, caption_mask: Mask, dropout: dropout_level})

            n_batch += 1.0
            train_loss += loss1

        train_loss = train_loss / n_batch
        end = time.clock()

        print "Small_Time:", end - start, "epoch_num:", epoch, "train_loss:", train_loss
        saver.save(sess, 'checkpoints/model.ckpt', epoch)
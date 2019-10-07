import numpy as np
import h5py
import tensorflow as tf
import os
import time
import json
import cPickle

from model.dataset import next_supervise_batch
from model.preprocess import encode_feature, attention, encode, decode
from model.fusion import recurent_matrix, content_fusion
from model.CNN_MODEL import forward_test
from train import first_attention, fouth_attention, fusion_para, Encode, Dilation, word_embed, embed_bias, channel_fusion

test_path = 'datasets/msrvtt'
test_data = h5py.File(test_path, 'r')
video_test = test_data['data']
vocab_file = 'Dictionary' + '/dic.txt'

voc_f = open(vocab_file, 'r')
data = voc_f.readlines()
dictionary = {}
for i in range(len(data)):
    dictionary[i] = data[i][:-2]

if __name__ == '__main__':

    test_feature = tf.placeholder(tf.float32, [1, 20, 2048])
    cap_test = forward_test(test_feature, first_attention, fouth_attention, fusion_para, Encode, Dilation, channel_fusion, word_embed, embed_bias)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver(max_to_keep=60)
    ckpt = tf.train.get_checkpoint_state('checkpoints/')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        epoch_n = int(ckpt.model_checkpoint_path.split('-')[1])
        #i_iter = (epoch_n+1) * (num_examples/batch_size)
        print "Restored Epoch ", epoch_n

    ## Test and Evaluation
    gen_file = open('gen_caption.txt', 'w')
    dic_set = []
    for i in range(2990):

        Data = video_test[7010 + i]

        Data = np.array(Data)
        Data = np.expand_dims(Data, axis=0)

        caption_result = sess.run(cap_test, feed_dict = {test_feature: Data})

        test_dict = {"image_id": "video" + str(7010 + i)}
        gen_file.write(str(7010 + i) + ' ')
        cap_pos = []
        for j in range(17):
            pos = np.argmax(caption_result[j])
            if pos == 23305:
                continue
            if pos == 0:
                continue
            if pos == 23306:
                continue
            if pos == 23307:
                break

            cap_pos.append(dictionary[pos] + ' ')

            gen_file.write(dictionary[pos] + ' ')

        new_caption = "".join(cap_pos)
        test_dict["caption"] = new_caption
        dic_set.append(test_dict)

        gen_file.write('\n')
        gen_file.write('\n')

        if i % 1000 == 0:
            print i

    gen_file.close()

    test_dict = {"val_predictions": dic_set}
    json_str = json.dumps(test_dict)
    new_dict = json.loads(json_str)

    with open('record.json', 'w') as f:
        json.dump(new_dict, f)

    cammand_path = 'caption_eval/msrvtt_eval.sh' + ' record.json'
    os.system(cammand_path)
import numpy as np
import h5py

train_path = 'msrvtt/data'

train = h5py.File(train_path, 'r')
train_data = train['feature']
train_cap = train['caption']

train_num = train_data.shape[0]

batch_size = 128

perm_label = np.arange(train_num)
def next_supervise_batch(img_data=train_data, caption_data=train_cap, perm_label=perm_label, index_in_epoch=0, num_examples=train_num, batch_size=batch_size):

    start = index_in_epoch
    index_in_epoch += batch_size
    Data=[]; Cap=[]
    if index_in_epoch >= (num_examples - 1):
        np.random.shuffle(perm_label)
        start = 0
        index_in_epoch = batch_size
        for j in range(batch_size):

            Data.append(img_data[perm_label[start + j]])

            Cap.append(caption_data[perm_label[start + j]])
    else:
        for j in range(batch_size):

            Data.append(img_data[perm_label[start + j]])

            Cap.append(caption_data[perm_label[start + j]])

    return Data, Cap, perm_label, index_in_epoch
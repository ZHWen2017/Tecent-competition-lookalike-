"""
@author:hai
@file:Split_Train_Set_To_Ten.PY
@time:2018/05/24
"""

from scipy import sparse
import random,os
import pandas as pd
import numpy as np
# import os
from config import ad_features,user_features,directories

def split_train_sets1(sel):
    if sel in [1,2]:
        if sel == 1:
            split_k = 6
        else:
            split_k = 50

        # Save_dir = "../Nomalizing_dir/train_set_splited/"
        Save_dir = directories["splitted_dir"]

        # source_dir = "../Nomalizing_dir/train_set_for_deepfm/"
        source_dir = directories["train_set_path{}".format(sel)]
    else:
        print("please set right sel!")
        exit()
    # train_labels_file = "../Nomalizing_dir/train_set_for_deepfm/labels.csv"
    if not os.path.exists(Save_dir):
        os.makedirs(Save_dir)
    train_labels_file = source_dir+"labels.csv"
    target_items = ad_features+user_features
    data_0 = sparse.load_npz(source_dir+target_items[0]+"_feature_sparse.npz")

    L,_ = data_0.shape
    print("all length:",L)
    labels = pd.read_csv(train_labels_file,header=None).values

    indexes = list(range(L))
    random.seed(2018)
    random.shuffle(indexes)
    print(indexes[0:10])
    index_list = []
    if L%split_k != 0:
        print('split_k set error!!!')
        exit()
    else:
        sub_size = int(L/split_k)
    for i in range(split_k):
        print("*"*50)

        if sel == 1:
            j =50+i
        else:
            j = i
        tmp_index = [indexes[n] for n in range(i*sub_size,(i+1)*sub_size)]
        sub_labels = labels[tmp_index]
        np.save(Save_dir+"labels_{}.npy".format(j),sub_labels)

        for item in target_items:
            print("The epoch {} changing {}".format(i,item))
            tmp_data = sparse.load_npz(source_dir+item+"_feature_sparse.npz")
            print("Shape of {}".format(item),tmp_data.shape)
            sel_data = tmp_data[tmp_index]
            print("writing ......")
            sparse.save_npz(Save_dir+item+"_sparse_{}.npz".format(j),sel_data)
        print("*" * 50)


if __name__ == "__main__":
    split_train_sets1(sel=1)
    split_train_sets1(sel=2)
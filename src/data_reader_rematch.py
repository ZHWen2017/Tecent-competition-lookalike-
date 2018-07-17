"""
@author:hai
@file:data_reader_rematch.PY
@time:2018/05/24
"""
from scipy import sparse
import numpy as np
import random
import gc
from config import one_hot_features,multi_hot_features

def convert_to_indices_and_value(x):
    coo = x.tocoo()
    indices = np.mat([coo.row,coo.col]).transpose()
    values = (coo.col).astype(float)
    shape = coo.shape
    return [indices,values,shape]


# def get_batch_for_train(batchsize = 724,):
#     # train_sets_indexes = range(10,50)
#     from config import train_sets_indexes_tune
#     for train_sets_index in train_sets_indexes_tune:
#         # print("*"*80)
#         print("reading splitted set {}".format(train_sets_index))
#         train_sub_set = {}
#         for item in titles_onehot:
#             # print(item)
#             train_sub_set[item] = sparse.load_npz(train_data_dir+item+"_sparse_{}.npz".format(train_sets_index))
#         for item in titles_dict:
#             # print(item)
#             train_sub_set[item] = sparse.load_npz(train_data_dir + item + "_sparse_{}.npz".format(train_sets_index))
#         train_labels = np.load(train_labels_file+"labels_{}.npy".format(train_sets_index))
#         sub_L,_ = train_labels.shape
#         sub_indexes = list(range(sub_L))
#         random.seed(2018)
#         random.shuffle(sub_indexes)
#         if sub_L%batchsize != 0:
#             # print("batchsize is not suitable,{} samples will be discarded.".format(sub_L%batchsize))
#             sub_num = int(sub_L/batchsize)
#         else:
#             sub_num = int(sub_L/batchsize)
#         print("total {} iterations".format(sub_num))
#         print('Start yielding {}th sparse set...'.format(train_sets_index))
#         for i in range(sub_num):
#             if i == sub_num - 1:
#                 batch_indexes = sub_indexes[i * batchsize:]
#             else:
#                 batch_indexes = sub_indexes[i*batchsize:(i+1)*batchsize]
#             out_dict = {}
#             for item in titles_onehot:
#                 tmp = train_sub_set[item][batch_indexes]
#
#                 out_dict[item] = convert_to_indices_and_value(tmp)
#
#             for item in titles_dict:
#                 tmp = train_sub_set[item][batch_indexes]
#
#                 out_dict[item] = convert_to_indices_and_value(tmp)
#             out_dict["label"] = train_labels[batch_indexes]
#             # print("label:", out_dict["label"])
#             yield out_dict
#         del train_sub_set
#         gc.collect()
#         # print("*" * 80)
#
#
# def build_valid_set(valid_size=1448):
#     # valid_set_indexes = range(10)
#     from functions.dnn_fm_mvm_lr.config_hai import valid_set_indexes
#     for valid_sets_index in valid_set_indexes:
#         # print("*" * 30+"validation"+"*"*30)
#         print("reading splitted set - validation {}".format(valid_sets_index))
#         valid_sub_set = {}
#         for item in titles_onehot:
#
#             valid_sub_set[item] = sparse.load_npz(train_data_dir + item + "_sparse_{}.npz".format(valid_sets_index))
#         for item in titles_dict:
#
#             valid_sub_set[item] = sparse.load_npz(train_data_dir + item + "_sparse_{}.npz".format(valid_sets_index))
#         valid_labels = np.load(train_labels_file + "labels_{}.npy".format(valid_sets_index))
#
#         sub_L, _ = valid_labels.shape
#         sub_indexes = list(range(sub_L))
#         random.seed(2018)
#         random.shuffle(sub_indexes)
#         if sub_L % valid_size != 0:
#             # print("batchsize is not suitable,{} will be discarded.".format(sub_L % valid_size))
#             sub_num = int(sub_L / valid_size)
#         else:
#             sub_num = int(sub_L / valid_size)
#         print("total {} iterations".format(sub_num))
#         print('Start yielding {}th sparse set...'.format(valid_sets_index))
#         for i in range(sub_num):
#             print("\rvaliding {}/{}".format(i+1,sub_num),end=" ")
#             if sub_num-1 == i:
#                 batch_indexes = sub_indexes[i * valid_size:]
#             else:
#                 batch_indexes = sub_indexes[i * valid_size:(i + 1) * valid_size]
#             out_dict = {}
#             for item in titles_onehot:
#                 tmp = valid_sub_set[item][batch_indexes]
#
#                 out_dict[item] = convert_to_indices_and_value(tmp)
#
#             for item in titles_dict:
#                 tmp = valid_sub_set[item][batch_indexes]
#
#                 out_dict[item] = convert_to_indices_and_value(tmp)
#             out_dict["label"] = valid_labels[batch_indexes]
#             # print("label:", out_dict["label"])
#             yield out_dict
#         print("*" * 30 + "validation" + "*" * 30)

def get_all_batch_for_train(batchsize=724,):
    from config import train_set_index_sub_2,directories
    train_data_dir_2 = directories["splitted_dir"]
    train_labels_file_2 = directories["splitted_dir"]
    import random
    train_set_index_sub = random.sample(train_set_index_sub_2,len(train_set_index_sub_2))
    # random.shuffle(train_set_index_sub)
    print(train_set_index_sub)
    for train_sets_index in train_set_index_sub:
        print("*" * 80)
        print("reading splitted set {}".format(train_sets_index))
        train_sub_set = {}
        for item in one_hot_features:
            # print(item)
            train_sub_set[item] = sparse.load_npz(train_data_dir_2 + item + "_sparse_{}.npz".format(train_sets_index))
        for item in multi_hot_features:
            # print(item)
            train_sub_set[item] = sparse.load_npz(train_data_dir_2 + item + "_sparse_{}.npz".format(train_sets_index))
        train_labels = np.load(train_labels_file_2 + "labels_{}.npy".format(train_sets_index))


        sub_L, _ = train_labels.shape
        sub_indexes = list(range(sub_L))
        random.seed(2018)
        random.shuffle(sub_indexes)
        if sub_L % batchsize != 0:
            sub_num = int(sub_L / batchsize)
        else:
            sub_num = int(sub_L / batchsize)
        print("total {} iterations".format(sub_num))
        print('Start yielding {}th sparse set...'.format(train_sets_index))
        for i in range(sub_num):
            if i == sub_num-1:
                batch_indexes = sub_indexes[i * batchsize:]
            else:
                batch_indexes = sub_indexes[i * batchsize:(i + 1) * batchsize]
            out_dict = {}
            for item in one_hot_features:
                tmp = train_sub_set[item][batch_indexes]

                out_dict[item] = convert_to_indices_and_value(tmp)

            for item in multi_hot_features:
                tmp = train_sub_set[item][batch_indexes]

                out_dict[item] = convert_to_indices_and_value(tmp)
            out_dict["label"] = train_labels[batch_indexes]
            # print("label:", out_dict["label"])
            yield out_dict
        del train_sub_set
        gc.collect()
        print("*" * 80)

def generator_data_for_predict(batchsize = 1448,):
    from config import directories
    test_dir_2 = directories["test_set_path"]
    sparse_test_data = {}
    for item in one_hot_features:
        print("reading:", item + "_feature_sparse.npz")
        file = test_dir_2 + item + "_feature_sparse.npz"
        sparse_test_data[item] = sparse.load_npz(file)
    for item in multi_hot_features:
        print("reading:", item + "_feature_sparse.npz")
        file = test_dir_2 + item + "_feature_sparse.npz"
        sparse_test_data[item] = sparse.load_npz(file)
    test_set_l, _ = sparse_test_data[one_hot_features[1]].shape

    num_subs = int(test_set_l/batchsize)
    remainder = test_set_l%batchsize
    # indexes = list(range(test_set_l))
    for i in range(num_subs+1):
        if i == num_subs:
            tmp_indexes =list(range(num_subs*batchsize,num_subs*batchsize+remainder))
        else:
            tmp_indexes = list(range(i*batchsize,(i+1)*batchsize))
        test_dict = {}
        for item in one_hot_features:
            tmp = sparse_test_data[item][tmp_indexes]

            test_dict[item] = convert_to_indices_and_value(tmp)

        for item in multi_hot_features:
            tmp = sparse_test_data[item][tmp_indexes]
            test_dict[item] = convert_to_indices_and_value(tmp)

        yield test_dict


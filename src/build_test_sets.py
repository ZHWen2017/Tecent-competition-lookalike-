"""
@author:hai
@file:build_test_set.PY
@time:2018/04/22
"""
import pandas as pd
from sklearn.externals import joblib
from scipy import sparse
import json,os,gc
from config import directories,data_directories,ad_features,user_features,multi_hot_user_features,one_hot_features

def build_test_sets(sel):
    if sel in [1,2]:
        # userFeature_file_v1="../Nomalizing_dir/userFeature_V1.csv"
        userFeature_file_v1 = directories["data_to_csv{}".format(sel)]
        # raw_train_file = "../data/train.csv"
        raw_test_file = data_directories["test2_csv{}".format(sel)]

        # adfeature_file = "../data/adFeature.csv"
        adfeature_file = data_directories["ad_feature{}".format(sel)]
        save_path = directories["test_set_path"]
        merge_dicts_path = directories["dicts_merge_dir"]
    else:
        print("please set right sel!")
        exit()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("rerading userfeatures file ......")
    alluser_features_data = pd.read_csv(userFeature_file_v1)

    print("reading raw trian file......")
    test_data = pd.read_csv(raw_test_file)

    print("reading ad feature file......")
    ad_feature_data = pd.read_csv(adfeature_file)

    print(test_data.shape)
    test_data = pd.merge(test_data,ad_feature_data,on=["aid"],how = "left")
    print(test_data.shape)
    test_data = pd.merge(test_data,alluser_features_data,on=["uid"],how = "left")

    print(test_data.shape)



    # *******************************************************************************************
    print("*"*60)
    for feature in one_hot_features:
        print("*" * 60)
        print("feature transform:{}".format(feature))
        # bow_model = joblib.load(models_dir+"{}_bow.pkl".format(bow_feature))
        test_data[feature] = test_data[feature].fillna("NoRecord")
        L = len(test_data[feature])
        print('data length:', len(test_data[feature]))
        dict_path = merge_dicts_path+'{}_dict.json'.format(feature)
        with open(dict_path, 'r') as fr:
            word_to_index = json.load(fr)
        row = []
        col = []
        data = []
        print('build {} sparse mat'.format(feature))
        for i, item in enumerate(test_data[feature].astype(str)):
            print("\r{}/{}".format(i, L), end=" ")
            # items = group.split(' ')
            index = word_to_index[item]
            row.append(i)
            col.append(index)
            data.append(1)
        print('\n')
        print('convert to sparse mat .......')
        sparse_mat = sparse.csr_matrix((data, (row, col)), shape=(len(test_data[feature]), len(word_to_index)))
        print('save sparse mat ......')
        sparse.save_npz(save_path+"{}_feature_sparse.npz".format(feature), sparse_mat)

        del sparse_mat,row,col,data,index,word_to_index
        del test_data[feature]
        gc.collect()
        print("*"*60)

    for bow_feature in multi_hot_user_features:
        print("bow feature vectorizer transform:{}".format(bow_feature))
        # bow_model = joblib.load(models_dir+"{}_bow.pkl".format(bow_feature))
        test_data[bow_feature] = test_data[bow_feature].fillna("NoRecord")
        L = len(test_data[bow_feature])
        print('data length:', len(test_data[bow_feature]))
        dict_path = merge_dicts_path + '{}_dict.json'.format(bow_feature)
        with open(dict_path, 'r') as fr:
            word_to_index = json.load(fr)
        row = []
        col = []
        data = []
        print('build {} sparse mat'.format(bow_feature))
        for i, group in enumerate(test_data[bow_feature]):
            print("\r{}/{}".format(i, L), end=" ")
            items = group.split(' ')
            for item in items:
                index = word_to_index[item]
                row.append(i)
                col.append(index)
                data.append(1)
        print('\n')
        print('convert to sparse mat .......')
        sparse_mat = sparse.csr_matrix((data, (row, col)), shape=(len(test_data[bow_feature]), len(word_to_index)))
        print('save sparse mat ......')
        sparse.save_npz(save_path + "{}_feature_sparse.npz".format(bow_feature), sparse_mat)

print("test features has been built!")

if __name__=="__main__":
    build_test_sets(sel=2)
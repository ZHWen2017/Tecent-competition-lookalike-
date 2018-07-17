"""
@author:Haien Zhang
@file:buildFeature_V1.PY
@time:2018/06/15
"""
import pandas as pd
import json,gc,os
from config import directories,user_features,ad_features,data_directories,one_hot_user_features,multi_hot_user_features


def build_user_dicts(sel):
    if sel in [1,2]:
        source_dir = directories["data_to_csv{}".format(sel)]
        alluser_features_data = pd.read_csv(source_dir)
        save_dir = directories["dicts_dir{}".format(sel)]
    else:
        print("please set right sel!")
        exit()
    for user_feature in one_hot_user_features:
        print("build odict:{}".format(user_feature))
        alluser_features_data[user_feature] = alluser_features_data[user_feature].fillna('NoRecord')

        L = len(alluser_features_data[user_feature])
        words = []
        print("splitting ......")
        for i,item in enumerate(alluser_features_data[user_feature].astype(str)):
            print("\rsplit and construct words:{}/{}".format(i, L), end=' ')
            words.append(item)

        print('\n')
        words = list(set(words))
        words = sorted(words, key=len)
        l_ = len(words)
        word_to_index = {}
        print("construct dictionary ......")
        for i, word in enumerate(words):
            print('\rconstruct dict {}/{}'.format(i, l_), end=" ")
            word_to_index[word] = i
        print('\n')
        print("writting {} dict to dir......".format(user_feature))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        savepath = save_dir+"{}_dict.json".format(user_feature)
        with open(savepath, 'w') as fw:
            json.dump(word_to_index, fw)
        print('\n')
        del alluser_features_data[user_feature]
        gc.collect()

    for bow_feature in multi_hot_user_features:
        print("build bow model:{}".format(bow_feature))
        alluser_features_data[bow_feature] = alluser_features_data[bow_feature].fillna('NoRecord')

        L = len(alluser_features_data[bow_feature])
        words = []
        print("splitting ......")
        for i, group in enumerate(alluser_features_data[bow_feature]):
            print("\rsplit and construct words:{}/{}".format(i, L), end=' ')
            items = group.split(' ')
            for item in items:
                words.append(item)
        print('\n')
        words = list(set(words))
        words = sorted(words, key=len)
        l_ = len(words)
        word_to_index = {}
        print("construct dictionary ......")
        for i, word in enumerate(words):
            print('\rconstruct dict {}/{}'.format(i, l_), end=" ")
            word_to_index[word] = i
        print('\n')
        print("writting {} dict to dir......".format(bow_feature))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        savepath = save_dir+"{}_dict.json".format(bow_feature)
        # if not os.path.exists(savepath):
        #     os.mkdir(savepath)
        with open(savepath, 'w') as fw:
            json.dump(word_to_index, fw)
        del alluser_features_data[bow_feature]
        gc.collect()


    if sel in [1,2]:
        source_dir = data_directories["ad_feature{}".format(sel)]
        allad_features_data = pd.read_csv(source_dir)
        save_dir = directories["dicts_dir{}".format(sel)]
    else:
        print("please set right sel!")
        exit()
    for ad_feature in ad_features:
        print("build dict:{}".format(ad_feature))
        allad_features_data[ad_feature] = allad_features_data[ad_feature].fillna('NoRecord')

        L = len(allad_features_data[ad_feature])
        words = []
        print("splitting ......")
        for i,item in enumerate(allad_features_data[ad_feature].astype(str)):
            print("\rsplit and construct words:{}/{}".format(i, L), end=' ')
            words.append(item)

        print('\n')
        words = list(set(words))
        words = sorted(words, key=len)
        l_ = len(words)
        word_to_index = {}
        print("construct dictionary ......")
        for i, word in enumerate(words):
            print('\rconstruct dict {}/{}'.format(i, l_), end=" ")
            word_to_index[word] = i
        print('\n')
        print("writting {} dict to dir......".format(ad_feature))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        savepath = save_dir+"{}_dict.json".format(ad_feature)
        with open(savepath, 'w') as fw:
            json.dump(word_to_index, fw)
        print('\n')
        del allad_features_data[ad_feature]
        gc.collect()

def merge_dicts():
    titles = ad_features+user_features

    for item in titles:
        file_A = directories["dicts_dir1"] + item + "_dict.json"
        file_B = directories["dicts_dir2"] + item + "_dict.json"

        with open(file_A, 'r') as fr:
            a = json.load(fr)
        with open(file_B, 'r') as fr:
            b = json.load(fr)

        tmp = []
        for key, _ in a.items():
            tmp.append(key)
        for key, _ in b.items():
            tmp.append(key)
        tmp = sorted(list(set(tmp)), key=len)
        l_ = len(tmp)
        print(item, l_)
        word_to_index = {}
        print("construct dictionary ......")
        for i, word in enumerate(tmp):
            print('\rconstruct dict {}/{}'.format(i, l_), end=" ")
            word_to_index[word] = i
        save_dir = directories["dicts_merge_dir"]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        savepath = save_dir+"{}_dict.json".format(item)
        with open(savepath, 'w') as fw:
            json.dump(word_to_index, fw)



if __name__=="__main__":
    build_user_dicts(sel=1)
    build_user_dicts(sel=2)
    merge_dicts()
    print("feature dicts has been built!")
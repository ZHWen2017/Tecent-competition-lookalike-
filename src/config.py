"""
@author:Haien Zhang
@file:buildFeature_V1.PY
@time:2018/06/15
"""
"""dir part """
data_directories = {
    "user_feature1":"../data/data1/userFeature.data",
    "ad_feature1":"../data/data1/adFeature.csv",
    "train1_file":"../data/data1/train.csv",

    "user_feature2":"../data/data2/userFeature.data",
    "ad_feature2":"../data/data2/adFeature.csv",
    "test2_csv2":"../data/data2/test2.csv",
    "train2_file":"../data/data2/train.csv"
}


directories = {
    "data_to_csv1":"../temp_directory/userFeature_csv/userFeature_1.csv",
    "data_to_csv2":"../temp_directory/userFeature_csv/userFeature_2.csv",

    "dicts_dir1":"../temp_directory/word_to_index_dict/1/",
    "dicts_dir2":"../temp_directory/word_to_index_dict/2/",
    "dicts_merge_dir":"../temp_directory/word_to_index_dict/merge/",
    "train_set_path1":"../temp_directory/train_sets1/",
    "train_set_path2":"../temp_directory/train_sets2/",
    "test_set_path":"../temp_directory/test_sets/",

    "splitted_dir":"../temp_directory/splitted_sets/"
}

"""Feature part"""
ad_features = ["advertiserId", "campaignId", "creativeId", "creativeSize", "adCategoryId", "productId","productType"]
one_hot_user_features = ["LBS", 'age', 'gender', 'marriageStatus','education', 'consumptionAbility', 'ct', 'os', 'carrier', 'house']
multi_hot_user_features = ['interest1','interest2','interest3','interest4','interest5',
              'kw1','kw2','kw3','topic1','topic2','topic3','appIdInstall','appIdAction']

user_features = one_hot_user_features+multi_hot_user_features
one_hot_features = ad_features+one_hot_user_features
multi_hot_features = multi_hot_user_features


"""model part"""
titles_0 = ad_features+user_features
activation = "dice"
model_dir = "../models/rpo_model/tfmodel.ckpt"
# train_sets_indexes_tune = [0,1,2,3,4,5,6,7]
# valid_set_indexes = [15,16]

# train for submission
# train_set_index_sub = list(range(50))
train_set_index_sub_2 = list(range(56))


field_config_2={
    "advertiserId":210, "campaignId":543, "creativeId":1005, "creativeSize":16, "adCategoryId":79, "productId":89,"productType":4,
    "LBS":899, 'age':6, 'gender':7, 'marriageStatus':28,'education':8, 'consumptionAbility':3, 'ct':65, 'os':5, 'carrier':4, 'house':2,
    'interest1':123,'interest2':82,'interest3':11,'interest4':11,'interest5':137,
    'kw1':274226,'kw2':51731,'kw3':12615,'topic1':10001,'topic2':9987,'topic3':6036,'appIdInstall':64860,'appIdAction':6218
}
field_config_2['field_length'] = len(one_hot_features)+len(multi_hot_features)

net_config_2 = {}
net_config_2['embedding_size'] = 800
net_config_2['mode'] = "mixture"
net_config_2['decay'] = 0.99


"""Train part"""
Train_for_sub = True

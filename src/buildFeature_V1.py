"""
@author:Haien Zhang
@file:buildFeature_V1.PY
@time:2018/06/15
"""
import pandas as pd
import gc,os
from config import data_directories,user_features,directories

def main(source_dir,save_dir,exit = False):
    fr = open(source_dir, 'r')
    features_values = []
    cnt = 0
    for i,line in enumerate(fr):
        groups = line.strip().split('|')
        userfeature_tmp = {}
        for group in groups:
            items = group.split(' ')
            key = items[0]
            values = ' '.join(items[1:])
            userfeature_tmp[key] = values

        features_values.append(userfeature_tmp)
        if i%100000 == 0:
            print(i)
        # if i>100000:
        #     break
        cnt = i
    print("number of lines=",cnt)
    fr.close()

    features_values = pd.DataFrame(data=features_values)
    print('writing to file.....')
    dir_tmp = save_dir.split("/")
    dir_tmp = '/'.join(dir_tmp[0:-1])
    if not os.path.exists(dir_tmp):
        os.makedirs(dir_tmp)
    features_values.to_csv(save_dir,index=False)
    print("*"*60)
    if exit:
       os._exit(0)



if __name__ == "__main__":
    main(source_dir=data_directories["user_feature1"],save_dir=directories["data_to_csv1"],exit=False)

    main(source_dir=data_directories["user_feature2"], save_dir=directories["data_to_csv2"],exit=True)

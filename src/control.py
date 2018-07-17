"""
@author:hai
@file:feature_set_script.PY
@time:2018/06/15
"""

import os
from os.path import abspath,dirname
os.chdir(abspath(dirname(__file__)))
def main():
    cmd = "python buildFeature_V1.py"
    os.system(cmd)
    print("script_1 over")

    print("script_2 ......")
    cmd = "python build_transform_dict.py"
    os.system(cmd)
    print("script_2 over")

    print("script_3 ......")
    cmd = "python build_train_sets.py"
    os.system(cmd)
    print("script_3 over")

    print("script_4 ......")
    cmd = "python build_test_sets.py"
    os.system(cmd)
    print("script_4 over")

    print("script_5 ......")
    cmd = "python Split_Train_Set_To_K.py"
    os.system(cmd)
    print("script_5 over")

    print("script_6  Training ......")
    cmd = "python train.py"
    os.system(cmd)
    print("script_6 over")

    print("script_7 predict and generate submission......")
    cmd = "python predict_and_generate_submission.py"
    os.system(cmd)
    print("script_7 over")

if __name__=="__main__":
    main()
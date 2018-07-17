"""
@author:Haien Zhang
@file:buildFeature_V1.PY
@time:2018/06/15
"""
from mymodel import deepFM
from config import net_config_2,field_config_2,Train_for_sub

if __name__=="__main__":
    s = 0
    for i,j in enumerate(field_config_2):
        s = s+field_config_2[j]
    print(s)

    m = deepFM(field_config=field_config_2,net_config=net_config_2)
    if Train_for_sub:
        m.train(epoch=1)
    else:
        m.train(epoch=2)


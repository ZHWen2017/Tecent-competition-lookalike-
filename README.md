
#Ready Player One

## 欢迎阅读

### 说明：

### 使用前提：

1： 代码运行环境：

　　* python3.6. 

　　* pandas

　　* tensorflow1.4.0 或者1.7.0(仅在这两种版本上运行过)

　　* scipy

　　* numpy

2: 请在Project目录下新建一个名为data的文件夹，文件夹内包含两个文件夹data1,data2其中data1中存放初赛解  压后的数据,data2中存放复赛解压后的 数据。

### 操作流程：
 mkdir models
 ./run.sh

### 各个脚本及其运行步骤说明：

1：src/control.py  此脚本控制着原始数据转化成用于训练的数据文件,然后训练模型以及预测的整个过程,分为7步(1-5步数据前处理以及构建用于模型输入的训练集和测试集)
    
    1 .data文件数据重排列转化成.csv文件;
    
    2 统计初赛和复赛每一个特征中值的种类构建dict用于one_hot以及multi_hot转换;
    
    3 根据统计出的dict对初赛和复赛的训练数据进行one_hot以及multi_hot转换构建出未分割的训练集(后续需要划分来节省训练时内存的开销);
    
    4 根据统计出的dict将复赛B阶段的test2.csv转换成one_hot以及multi_hot编码后的测试集;
   
    5 将第3步转化好临时训练集进行切分成多个包,以便训练时每次读入单个包,可以节省内存的使用;
    
    6: 训练模型;
    
    7 预测并生成提交文件.
    
2: src/config.py  配置文件

3: src/mymodel.py tensorflow搭建的模型代码

4: src/data_reader_rematch.py 训练模型以及预测结果时用于数据输入

5：src/buildFeature_V1.py　用于生成对原始数据处理过后得到的csv特征文件

6：src/build_transform_dict.py 　control.py中的第2步对应的代码

7：src/build_train_sets.py　用于生成训练集

8：src/build_test_sets.py  用于生成测试集

9：src/Split_Train_Set_To_K.py  训练集切分成多份

10：src/train.py  训练模型     

11：src/predict_and_generate_submission.py  基于train.py所得到的模型，

restore之后，读入测试集数据，进行预测，得到submission.csv





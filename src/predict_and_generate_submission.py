
from mymodel import deepFM
from data_reader_rematch import generator_data_for_predict
from config import net_config_2,field_config_2,data_directories

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model = deepFM(field_config=field_config_2,net_config=net_config_2)
print('preparing test gen')
test_generator = generator_data_for_predict(batchsize=1164)
pre = model.predict(test_data_gen=test_generator)

print(len(pre))
def float_to_6(value):
    return float("%.6f"%value)
import pandas as pd

test_y = pd.read_csv(data_directories["test2_csv2"])
test_y["score"] = list(map(float_to_6,pre))
submission_dir = "../submission.csv"
print("writting submission to csv ......")
test_y.to_csv(submission_dir,index=False)

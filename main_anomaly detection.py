import os
os.chdir('D:\\GitHub\\datascience')
os.getcwd()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from AD.anomal_detection import Local_Outlier_Factor, k_nearest_neighbor, one_class_SVM, Isolation_Forest
from data.read_data import read_data_01, make_random_XY


#############################
####### データの読み込み #######
#############################
x_data, y_data = read_data_01()
x_data, y_data = make_random_XY()

#######################
####### 異常検知 #######
#######################
outlier_index = Local_Outlier_Factor(x_data, y_data, 5, 0.005)
outlier_index = k_nearest_neighbor(x_data, y_data, 2, 0.05)
outlier_index = one_class_SVM(x_data, y_data, 0.1)
outlier_index = Isolation_Forest(x_data, y_data)

###########################
####### 異常値を特定 #######
###########################
df_data = pd.DataFrame({'X': x_data,
                        'Y': y_data})

index_list = list(df_data.index)

outlier_data = list(np.array(index_list)[outlier_index])








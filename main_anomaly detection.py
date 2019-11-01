import os
os.chdir('D:\\GitHub\\datascience')
os.getcwd()
import pandas as pd
import numpy as np

from AD.anomal_detection import Local_Outlier_Factor, k_nearest_neighbor, one_class_SVM, Isolation_Forest, time_series_kNN, time_series_singular_spectrum
from data.read_data import read_data_01, make_random_XY, electro_cardiogram


#############################
####### データの読み込み #######
#############################
# 配列データ
x_data, y_data = read_data_01()
x_data, y_data = make_random_XY()

# 時系列データ
data = electro_cardiogram()


#######################
####### 異常検知 #######
#######################
# 配列データ
outlier_index = Local_Outlier_Factor(x_data, y_data, 5, 0.005)
outlier_index = k_nearest_neighbor(x_data, y_data, 2, 0.05)
outlier_index = one_class_SVM(x_data, y_data, 0.1)
outlier_index = Isolation_Forest(x_data, y_data)

# 時系列データ
time_series_kNN(data)
time_series_singular_spectrum(data)

###########################
####### 異常値を特定 #######
###########################
df_data = pd.DataFrame({'X': x_data,
                        'Y': y_data})

index_list = list(df_data.index)

outlier_data = list(np.array(index_list)[outlier_index])








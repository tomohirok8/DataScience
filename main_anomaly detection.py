import os
os.chdir('D:\\GitHub\\datascience')
os.getcwd()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from AD.anomal_detection import Local_Outlier_Factor, k_nearest_neighbor, one_class_SVM
from data.read_data import read_data_01

#############################
####### データの読み込み #######
#############################
x_data, y_data = read_data_01()


#######################
####### 異常検知 #######
#######################
Local_Outlier_Factor(x_data, y_data, 5, 0.005)
k_nearest_neighbor(x_data, y_data, 2, 0.05)
one_class_SVM(x_data, y_data, 0.5)


###########################
####### 異常値を特定 #######
###########################












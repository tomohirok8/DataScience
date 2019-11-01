import os
os.chdir('D:\\GitHub\\datascience')
os.getcwd()


from ML.basic_ML_clustering import k_means, mean_shift
from data.read_data import read_data_01, read_data_02, make_random_XY


#############################
####### データの読み込み #######
#############################
x_data, y_data = read_data_01()
x_data, y_data = read_data_02()
x_data, y_data = make_random_XY()


#########################
####### クラスタリング #######
#########################
k_means(x_data, y_data)
mean_shift(x_data, y_data)


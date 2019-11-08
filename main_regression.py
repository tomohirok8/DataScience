import os
os.chdir('D:\\GitHub\\datascience')
os.getcwd()

from ML.basic_ML import Random_Forest_Regression
from data.read_data import make_random_sin

#############################
####### データの読み込み #######
#############################
x_train, y_train, x_test, y_test = make_random_sin()


####################
####### 学習 #######
####################
Random_Forest_Regression(x_train, x_test, y_train, y_test)



##########################
####### モデルの評価 #######
##########################



####################
####### 予測 #######
####################




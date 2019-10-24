import os
os.chdir('D:\\GitHub\\datascience')
os.getcwd()
from sklearn.model_selection import StratifiedKFold

from verification.data_split import split_from_combined
from DL.basic_DL import basic_DL_2class
from data.read_data import read_data_MNIST, read_data_Iris


#############################
####### データの読み込み #######
#############################
####### 画像多クラス分類 #######
x_train, y_train, x_test, y_test = read_data_MNIST()
# 2クラス分類とするためラベルデータから1列だけ取り出す
y_train0 = y_train[:,0]
y_test0 = y_test[:,0]

####### 2クラス分類 #######


####### 多クラス分類 #######
df_iris = read_data_Iris()



#######################################
####### 訓練データとテストデータに分割 #######
#######################################
x_train2, x_test2, y_train2, y_test2 = split_from_combined(x_train, y_train)


####################
####### 学習 #######
####################
model = basic_DL_2class(x_train, x_test, y_train0, y_test0)


############################
####### モデルの可視化 #######
############################
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(model).create(prog='dot', format='svg'))
from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)


##########################
####### モデルの評価 #######
##########################
score = model.evaluate(x_test, y_test0, verbose=1)


####################
####### 予測 #######
####################
y_pred = model.predict(x_test)
y_class = model.predict_classes(x_test)
y_proba = model.predict_proba(x_test)


#####################################
####### 層化k分割交差検証で学習 #######
#####################################
# 層化k分割交差検証（訓練データとテストデータが分割されていないものに適用）
kf = StratifiedKFold(n_splits=5, shuffle=True)
kf.split(x_train, y_train0)
score = []
for train_index, test_index in kf.split(x_train, y_train0):
    print('TRAIN:', y_train0[train_index], 'TEST:', y_train0[test_index])
    model = basic_DL_2class(x_train[train_index], x_train[test_index], y_train0[train_index], y_train0[test_index])
    score.append(model.evaluate(x_test, y_test0, verbose=1))













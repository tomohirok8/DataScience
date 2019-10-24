import os
os.chdir('D:\\GitHub\\datascience')
os.getcwd()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from verification.data_split import split_from_combined
from ML.basic_ML import Logistic_Regression, k_NN, SVM_linear, SVM_RBF, SVM_poly, GNB_classify, Decision_Tree, Random_Forest
from ML.ML_grid_search import grid_search_SVM_RBF
from data.read_data import read_data_Iris, make_data_blobs

#############################
####### データの読み込み #######
#############################
####### 2クラス分類 #######


####### 多クラス分類 #######
df_iris = read_data_Iris()

x_train, y_train = make_data_blobs()


#######################################
####### 訓練データとテストデータに分割 #######
#######################################
x_train, x_test, y_train, y_test = split_from_combined(x_train, y_train)


####################
####### 学習 #######
####################
model = Logistic_Regression(x_train, x_test, y_train, y_test)
model = k_NN(x_train, x_test, y_train, y_test, 3)
model = SVM_linear(x_train, x_test, y_train, y_test, 1.0)
model = SVM_RBF(x_train, x_test, y_train, y_test, 0.01, 1.0)
model = SVM_poly(x_train, x_test, y_train, y_test, 0.3, 1.0)
model = GNB_classify(x_train, x_test, y_train, y_test)
model = Decision_Tree(x_train, x_test, y_train, y_test)
model = Random_Forest(x_train, x_test, y_train, y_test)



##########################
####### モデルの評価 #######
##########################
y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred, average='micro'))
print(recall_score(y_test, y_pred, average='micro'))
print(f1_score(y_test, y_pred, average='micro'))


# X軸の最大最小
x_min = x_train[:, 0].min() - 1
x_max = x_train[:, 0].max() + 1
# Y軸の最大最小
y_min = x_train[:, 1].min() - 1
y_max = x_train[:, 1].max() + 1
# meshgrid作成
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# 境界線を描画
plt.figure(figsize=(11,7))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap="Set2", alpha=0.5,linewidths=0)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='Set1')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title('SVC with linear kernel')
plt.show()


####################
####### 予測 #######
####################



##########################
####### グリッドサーチ #######
##########################
# SVMでグリッドサーチ
grid_search_SVM_RBF(x_train, y_train, x_test, y_test)


#####################################
####### 層化k分割交差検証で学習 #######
#####################################
# 層化k分割交差検証（訓練データとテストデータが分割されていないものに適用）
kf = StratifiedKFold(n_splits=5, shuffle=True)
kf.split(x_train, y_train)
score_test = []
score_vali = []
for train_index, test_index in kf.split(x_train, y_train):
    print('TRAIN:', y_train[train_index], 'TEST:', y_train[test_index])
    model = GNB_classify(x_train[train_index], x_train[test_index], y_train[train_index], y_train[test_index])
    y_pred_test = model.predict(x_train[test_index])
    y_pred_vali = model.predict(x_test)
    score_test.append(accuracy_score(y_train[test_index], y_pred_test))
    score_vali.append(accuracy_score(y_test, y_pred_vali))













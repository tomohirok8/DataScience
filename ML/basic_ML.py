import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


# 線形単回帰
def manual_Simple_regression(X, Y):
    X = np.vstack(X)
    # Xを[X 1]の形に
    X = np.array([[value,1] for value in X])
    
    # 最小二乗法の計算を実行
    a, b = np.linalg.lstsq(X, Y)[0]
    
    # 元のデータをプロット
    plt.plot(X, Y,'o')
    # 求めた回帰直線を表示
    plt.plot(X, a*X + b,'r')
    
    # 結果のarrayを取得
    result = np.linalg.lstsq(X,Y)
    # 2つ目の要素が誤差の合計
    error_total = result[1]
    # 誤差の平均値の平方根を計算
    rmse = np.sqrt(error_total/len(X))
    
    print('平均二乗誤差の平方根は、{:0.2f}'.format(rmse[0]))


# 線形重回帰
def Multiple_regression(X, Y):
    lreg = LinearRegression()
    lreg.fit(X, Y)
    
    print('切片の値は{:0.2f}'.format(lreg.intercept_))
    print('係数の数は{}個'.format(len(lreg.coef_)))


# 線形回帰の学習
def Linear_Regression(x_train, x_test, y_train, y_test):
    # インスタンス
    lreg = LinearRegression()
    # 学習
    lreg.fit(x_train,y_train)
    
    pred_train = lreg.predict(x_train)
    pred_test = lreg.predict(x_test)
    
    print('x_trainを使ったモデルの平均二乗誤差＝{:0.2f}'.format(np.mean((y_train - pred_train) ** 2)))
    print('x_testを使ったモデルの平均二乗誤差＝{:0.2f}'.format(np.mean((y_test - pred_test) ** 2)))


# ロジスティック回帰
def Logistic_Regression(x_train, x_test, y_train, y_test):
    # インスタンス
    log_model = LogisticRegression()
    # 学習
    log_model.fit(x_train, y_train)
    # モデルの精度を確認
    log_model.score(x_train, y_train)
    
    log_model.coef_
    
    y_pred = log_model.predict(x_test)
    
    print(metrics.accuracy_score(y_test, y_pred))


# k近傍法
def k_NN(x_train, x_test, y_train, y_test, k):
    # インスタンス
    knn = KNeighborsClassifier(n_neighbors=k)
    # 学習
    knn.fit(x_train,y_train)
    
    y_pred = knn.predict(x_test)
    
    print(metrics.accuracy_score(y_test, y_pred))


# サポートベクターマシン
def SVM_linear(x_train, x_test, y_train, y_test, C):
    # インスタンス作成
    model = SVC(kernel='linear', C=C)
    # 学習
    model.fit(x_train,y_train)
    
    y_pred = model.predict(x_test)
    
    print(metrics.accuracy_score(y_test, y_pred))

def SVM_rbf(x_train, x_test, y_train, y_test, gamma, C):
    # インスタンス作成
    model = SVC(kernel='rbf', gamma=gamma, C=C)
    # 学習
    model.fit(x_train,y_train)
    
    y_pred = model.predict(x_test)
    
    print(metrics.accuracy_score(y_test, y_pred))

def SVM_poly(x_train, x_test, y_train, y_test, degree, C):
    # インスタンス作成
    model = SVC(kernel='poly', degree=degree, C=C)
    # 学習
    model.fit(x_train,y_train)
    
    y_pred = model.predict(x_test)
    
    print(metrics.accuracy_score(y_test, y_pred))


# ナイーブベイズ分類
def NB_classify(x_train, x_test, y_train, y_test):
    # インスタンス作成
    model = GaussianNB()
    # 学習
    model.fit(x_train,y_train)
    
    y_pred = model.predict(x_test)
    
    print(metrics.accuracy_score(y_test, y_pred))





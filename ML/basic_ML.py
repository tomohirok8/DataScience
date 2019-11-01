import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.mixture import GaussianMixture


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
    model = LogisticRegression()
    # 学習
    model.fit(x_train, y_train)
    # モデルの精度を確認
    model.score(x_train, y_train)
    model.coef_
    y_pred = model.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    return model


# k近傍法
def k_NN(x_train, x_test, y_train, y_test, k):
    # インスタンス
    model = KNeighborsClassifier(n_neighbors=k)
    # 学習
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    return model


# サポートベクターマシン
def SVM_linear(x_train, x_test, y_train, y_test, C):
    # インスタンス作成
    model = SVC(kernel='linear', C=C)
    # 学習
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    return model

def SVM_RBF(x_train, x_test, y_train, y_test, gamma, C):
    # インスタンス作成
    model = SVC(kernel='rbf', gamma=gamma, C=C)
    # 学習
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    return model

def SVM_poly(x_train, x_test, y_train, y_test, degree, C):
    # インスタンス作成
    model = SVC(kernel='poly', degree=degree, C=C)
    # 学習
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    return model


# ナイーブベイズ分類
def GNB_classify(x_train, x_test, y_train, y_test):
    # インスタンス作成
    model = GaussianNB()
    # 学習
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    return model


# 決定木
def Decision_Tree(x_train, x_test, y_train, y_test):
    # インスタンス作成
    model = DecisionTreeClassifier(max_depth=4,random_state=0)
    # 学習
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    return model


# ランダムフォレスト
def Random_Forest(x_train, x_test, y_train, y_test):
    # インスタンス作成
    model = RandomForestClassifier(n_estimators=100,random_state=0)
    # 学習
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    return model


# 混合ガウスモデル
def GMM(x_train, x_test, y_train, y_test):
    # クラス数
    num_classes = len(np.unique(y_train))
    # インスタンス作成
    model = GaussianMixture(n_components=num_classes,
                            covariance_type='full',
                            init_params='random',
                            random_state=0,
                            max_iter=20)
    # GMMの平均値を初期化
    model.means_init = np.array([x_train[y_train == i].mean(axis=0) for i in range(num_classes)]) 
    # 学習
    model.fit(x_train)
    y_pred = model.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    return model










# ランダムフォレストによる回帰
def Random_Forest_Regression(x_train, x_test, y_train, y_test):
    # インスタンス作成
    model = RandomForestRegressor(100)
    # 学習
    model.fit(x_train[:, None], y_train)
    # 予測値
    y_pred = model.predict(x_test[:, None])
    # 実際の値
    # Plot
    plt.figure(figsize=(11,7))
    plt.errorbar(x_train, y_train, 0.1, fmt='o')
    plt.plot(x_test, y_pred, '-r')
    plt.plot(x_test, y_test, '-k', alpha=0.5)
    
    

    
    
    
    
    
    
    
    
    
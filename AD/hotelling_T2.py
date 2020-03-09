'''
データが単一の正規分布から発生している
データ中に異常な値を含まないか、含んでいたとしてもごくわずか
1. 自分で設定した誤報率に基づき、閾値を求める
2. 正常と信用できる値の平均値および共分散行列を計算する
3. テストデータに対してマハラノビス距離を計算し、閾値を超えていたら異常と判定する
'''
import numpy as np
from scipy.spatial import distance
from scipy import stats as st
import matplotlib.pyplot as plt



def hotelling(X,Y):
    # X_testの読み込み
    X_test=np.stack([X, Y], axis=1)
    
    # 閾値を設定して下さい
    thr = st.chi2.ppf(0.7, 2)
    
    # X_testの標本値を取得して下さい
    mean = np.mean(X_test, axis=0)
    cov = np.cov(X_test.T)
    
    # X_testの各データの異常度mahを計算して下さい
    mah = [distance.mahalanobis(x, mean, np.linalg.pinv(cov)) for x in X_test]
    
    # X_err, X_normを分類して下さい
    X_err = X_test[mah > thr]
    X_nom = X_test[mah <= thr]
    
    # プロットしています
    plt.plot(X_err[:, 0], X_err[:, 1], "o", color="r")
    plt.plot(X_nom[:, 0], X_nom[:, 1], "o", color="b")
    plt.title("T二乗法によるX_testについての異常値検知")
    plt.show()



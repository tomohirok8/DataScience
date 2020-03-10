'''
標本に異常か正常かのラベルが与えられたもとで、判別したいデータ近傍の局所的なデータについて着目する
判別したいデータとの距離が近いデータk個の中で異常データの割合から異常度を計算し、閾値を超えたら異常と判定する
○ データの分布に関する前提条件がいらないため、実用が簡単
○ 正常なデータが複数箇所のまとまりから構成されていても使用可能
○ 異常度の式が簡単なため、実装が難しくない
× データの分布に関する前提条件がないため、閾値が数式で定まらない
× パラメータ k の厳密なチューニングには複雑な数式が必要
× 怠惰学習(事前にモデルを構築しない学習)のため、新しいデータを分類するための計算量が減らない
'''
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt



def K_neighbor_anomal_detect(X, y):
    # k, 閾値ath
    k = 3
    ath = 0.5
    # データx
    x = np.array([
        [ 1.52,  3.60],
        [-2.50,  0   ],
        [ 5.32, -1.89]
    ])
    
    # x の異常度を計算しています
    # KNeighborsClassifierを用意し、x近傍のラベルを取得しています
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X, y)
    dist, ind = clf.kneighbors(x)
    neighbors = y[ind]
    
    # pi0, pi1, N0, N1 を計算しています
    pi0 = y[y == 0].size / y.size
    pi1 = 1 - pi0
    N1 = neighbors.sum(axis=1) / k
    N0 = 1 - N1
    
    # 異常度の計算
    abnorm = np.log((N1 * pi0) / (N0 * pi1))
    
    # 異常判定し、y_predを作成してください
    y_pred = np.asarray(abnorm > ath, dtype="int")
    
    # 結果の出力
    print("xの異常判定結果:" + str(y_pred))
    x0 = x[y_pred==0]
    x1 = x[y_pred==1]
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "o", color="skyblue")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "o", color="pink")
    plt.plot(x0[:, 0], x0[:, 1], "o", color="b")
    plt.plot(x1[:, 0], x1[:, 1], "o", color="r")
    plt.title("データxの異常判定結果")
    plt.show()


'''
データが単一の正規分布から発生している
データ中に異常な値を含まないか、含んでいたとしてもごくわずか
変数同士の相関が無い
1. 訓練データを用いたベイズ法による重みの計算
2. 検証データを用いた閾値の最適化
3. 異常度の計算、閾値と比較
'''
import numpy as np
import numpy.random as rd


def Naive_Bayes():
    ### 訓練データ
    rd.seed(0)
    # 正規分布に従う正常データの集合X0
    # 100次元×200個
    mean_n = rd.randint(0, 7, 100)
    cov_n = rd.randint(0, 5, 100) * np.identity(100)
    X0 = rd.multivariate_normal(mean_n, cov_n, 200).astype('int64')
    X0[X0 < 0] = 0
    
    # 正規分布に従わない異常データの集合X1
    # 100次元×50個
    X1 = rd.randint(0, 10, (50, 100))
    
    # 重みを0にしないためのゲタ(あとで対数をとるため)
    alpha = 1
    
    
    ### 重みweightの計算
    w0 = (np.sum(X0, axis=0) + alpha) / (np.sum(X0) + X0.shape[1] * alpha)
    w1 = (np.sum(X1, axis=0) + alpha) / (np.sum(X1) + X1.shape[1] * alpha)
    
    weight = np.log(w1 / w0)
    
    
    ### 閾値thresholdの設定
    threshold = 123
    
    
    ### 異常判定したいデータx
    # 100次元×1個
    x = rd.randint(0, 10, 100)
    
    
    ### xの異常度を計算して下さい
    score = np.dot(x, weight)
    
    print("score:" + str(score))
    
    
    ### 閾値を越えているかによって分岐し、結果を出力して下さい
    if score > threshold:
        print("abnormal")
    else:
        print("normal")
    


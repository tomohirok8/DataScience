import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import metrics 


def k_means(x_data, y_data):
    X_con  = np.stack([x_data, y_data], axis=1)
    # シルエットスコアを計算
    scores = [] 
    # 最適なクラスタ数を調べる範囲を定義
    kmin = 2
    kmax = 15
    values = np.arange(kmin, kmax+1) 
    # クラスタ数を変えながらモデル生成、学習を繰り返す
    for num_clusters in values: 
        kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10) 
        kmeans.fit(X_con)
        # クラスタリングモデルについて、ユークリッド距離に基づいたシルエットスコアを推定
        score = metrics.silhouette_score(X_con, kmeans.labels_, metric='euclidean', sample_size=len(X_con)) 
        scores.append(score) 
    max_score_index = np.argmax(scores)
    # クラスタ数ごとのシルエットスコアを可視化
    plt.figure() 
    plt.bar(values, scores, width=0.7, color='black', align='center') 
    plt.title('Silhouette score vs number of clusters') 
    plt.show()

    # クラスタ数の定義
    num_clusters = max_score_index + kmin
    # インスタンス生成 
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10) 
    # 学習
    kmeans.fit(X_con)
    # 予測
    # グリッドサイズ
    step_size = 0.01
    # グリッド定義 
    x_min, x_max = X_con[:, 0].min(), X_con[:, 0].max() 
    y_min, y_max = X_con[:, 1].min(), X_con[:, 1].max() 
    x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size)) 
    # モデルにすべての点を入力して推定
    output = kmeans.predict(np.c_[x_vals.ravel(), y_vals.ravel()]) 
    # 出力を可視化
    output = output.reshape(x_vals.shape)
    plt.figure() 
    plt.clf() 
    plt.imshow(output, interpolation='nearest', 
               extent=(x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()), 
               cmap=plt.cm.Paired, 
               aspect='auto', 
               origin='lower') 
    plt.scatter(X_con[:,0], X_con[:,1], marker='o', facecolors='none', edgecolors='black', s=80) 
    cluster_centers = kmeans.cluster_centers_ 
    plt.scatter(cluster_centers[:,0], cluster_centers[:,1],
                marker='o', s=210, linewidths=4, color='black', 
                zorder=12, facecolors='black') 
    x_min, x_max = X_con[:, 0].min(), X_con[:, 0].max() 
    y_min, y_max = X_con[:, 1].min(), X_con[:, 1].max() 
    plt.title('Boundaries of clusters') 
    plt.xlim(x_min, x_max) 
    plt.ylim(y_min, y_max) 
    plt.xticks(()) 
    plt.yticks(()) 
    plt.show()



def mean_shift(x_data, y_data):
    X_con  = np.stack([x_data, y_data], axis=1)
    # 入力データのバンド幅見積
    bandwidth_X = estimate_bandwidth(X_con, quantile=0.1, n_samples=len(X_con)) 
    # インスタンス生成
    meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True) 
    # 学習
    meanshift_model.fit(X_con)
    # クラスタの中心点を抽出
    cluster_centers = meanshift_model.cluster_centers_ 
    # クラスタの数を抽出
    labels = meanshift_model.labels_ 
    num_clusters = len(np.unique(labels)) 
    # 出力を可視化
    plt.figure() 
    markers = 'o*xvs^' 
    for i, marker in zip(range(num_clusters), markers): 
        plt.scatter(X_con[labels==i, 0], X_con[labels==i, 1], marker=marker, color='black') 
        cluster_center = cluster_centers[i] 
        plt.plot(cluster_center[0], cluster_center[1], marker='o', 
                 markerfacecolor='black', markeredgecolor='black', 
                 markersize=15) 
    plt.title('Clusters') 
    plt.show()






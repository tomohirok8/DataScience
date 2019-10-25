import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest


def Local_Outlier_Factor(X, Y, n_neighbors, contamination):
    X_con  = np.stack([X, Y], axis=1)
    ####### [0,1]に正規化 #######
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(X_con)
     
    # LOFのモデル生成
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    
    # 予測
    pred = model.fit_predict(dataset)
    
    outlier_index = np.where(pred == -1)
    outlier = dataset[outlier_index]
     
    # 出力を可視化
    plt.figure(figsize=(11,7))
    plt.scatter(dataset[:,0], dataset[:,1], c='blue', edgecolor='k', s=20, marker='o', label='normal')
    plt.scatter(outlier[:,0], outlier[:,1], c='red', edgecolor='k', s=100, marker='x', label='anormal')
    plt.axis('tight')
    plt.xlim((-0.1, 1.1))
    plt.ylim((-0.1, 1.1))
    plt.legend(loc="lower right")
    plt.title("Local Outlier Factor")
    plt.show()
    
    return outlier_index[0]



def k_nearest_neighbor(X, Y, k, thresh):
    X_con  = np.stack([X, Y], axis=1)
    ####### [0,1]に正規化 #######
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(X_con)
    
    # 2地点のユークリッド距離の計算
    def distance(a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    
    # 距離のリスト作成
    hankei_list = []
    for data in dataset:
        dist_list = []
        for i in range(len(dataset)):
            dist = distance(dataset[i], data)
            dist_list.append(dist)
        
        # 距離でソートしたインデックスを取得
        sort_index = np.argsort(dist_list)
        
        # k個を含む円の半径を計算
        dist_k = dist_list[sort_index[k-1]]
        
        hankei_list.append(dist_k)
    
    # 距離が閾値を超えていたら外れ値判定
    outlier_list = [0] * len(hankei_list)
    for i in range(len(hankei_list)):
        if hankei_list[i] > thresh:
            outlier_list[i] = 1
    
    outlier_index = np.where(np.array(outlier_list) == 1)
        
    df_dataset_hankei = pd.DataFrame(dataset, columns=['x', 'y'])
    df_dataset_hankei['hankei'] = hankei_list
    df_dataset_hankei['outlier'] = outlier_list
    
    df_abnormal_data = df_dataset_hankei[df_dataset_hankei['outlier'] > thresh]
    
    x = list(df_dataset_hankei.loc[:, 'x'])
    y = list(df_dataset_hankei.loc[:, 'y'])
    
    x_ab = list(df_abnormal_data.loc[:, 'x'])
    y_ab = list(df_abnormal_data.loc[:, 'y'])
    
    # 出力を可視化
    plt.figure(figsize=(11,7)) 
    # 正常データのプロット
    plt.scatter(x, y, c='blue', s=20, marker='o')
    # 異常データのプロット
    plt.scatter(x_ab, y_ab, c='red', s=100, marker='x')
    plt.title('k nearest neighbor')
    plt.show()
    
    return outlier_index[0]
    
    
def one_class_SVM(X, Y, nu):
    X_con  = np.stack([X, Y], axis=1)
    ####### [0,1]に正規化 #######
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(X_con)
    
    model = OneClassSVM(nu=nu, kernel='rbf', gamma='auto')
    model.fit(dataset)
    
    pred = model.predict(dataset)
    
    xx, yy = np.meshgrid(np.linspace(-0.1, 1.1, 100), np.linspace(-0.1, 1.1, 100))
    z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    outlier_index = np.where(pred == -1)
    outlier = dataset[outlier_index]
    
    # 出力を可視化
    plt.figure(figsize=(11,7))
    plt.contourf(xx, yy, z, cmap='Blues_r')
    plt.scatter(dataset[:,0], dataset[:,1], c='blue', edgecolor='k', s=20, marker='o', label='normal')
    plt.scatter(outlier[:,0], outlier[:,1], c='red', edgecolor='k', s=100, marker='x', label='anormal')
    plt.axis('tight')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.legend(loc='lower right')
    plt.title('One class SVM')
    plt.show()
    
    return outlier_index[0]
    


def Isolation_Forest(X, Y):
    X_con  = np.stack([X, Y], axis=1)
    ####### [0,1]に正規化 #######
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(X_con)
    
    model = IsolationForest(n_estimators=100, max_samples=100)
    model.fit(dataset)

    pred = model.predict(dataset)

    outlier_index = np.where(pred == -1)
    outlier = dataset[outlier_index]
    
    xx, yy = np.meshgrid(np.linspace(-0.1, 1.1, 100), np.linspace(-0.1, 1.1, 100))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    # 出力を可視化
    plt.figure(figsize=(11,7))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
    plt.scatter(dataset[:,0], dataset[:,1], c='blue', edgecolor='k', s=20, marker='o', label='normal')
    plt.scatter(outlier[:,0], outlier[:,1], c='red', edgecolor='k', s=100, marker='x', label='anormal')
    plt.axis('tight')
    plt.xlim((-0.1, 1.1))
    plt.ylim((-0.1, 1.1))
    plt.legend(loc="lower right")
    plt.title("Isolation Forest")
    plt.show()
    
    return outlier_index[0]
    
    
    
    
    
    
    
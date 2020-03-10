'''
ラベル無し標本の全体的なデータ
1. SVMの識別器を用意して、各データの異常度を計算
2. 標本データの内何割が異常かをあらかじめ決定して閾値を設定
3. 閾値と異常度によって異常判定
○ 教師なしの異常検知  
○ 機械学習の実用例が多い(= scikit-learn で簡単に実装できる)
○ データが複数箇所のまとまりから構成されていても使用可能  
○ データの次元数が大きくなっても精度を保てる  
× 数式のパラメータの変化によって精度が大きく上下する  
× データが増えると計算量が急上昇する  
'''
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from sklearn.svm import OneClassSVM



def One_Class_SVM2(data):
    # 異常度の計算
    clf = OneClassSVM(kernel="rbf", gamma="auto")
    clf.fit(data)
    y_score = clf.decision_function(data).ravel()
    
    # 閾値は10%分位点
    threshold = st.scoreatpercentile(y_score, 5)
    
    # y_scoreとthresholdによる異常判定
    # bool値のNumPy配列を作ってください(正常ならFalse, 異常ならTrue)
    y_pred = np.array(y_score < threshold)
    
    # 正常データを青色でプロットしてください
    plt.plot(data[y_pred == False][:,0], data[y_pred == False][:,1], "o", color="b")
    # 異常データを赤色でプロットしてください
    plt.plot(data[y_pred][:,0], data[y_pred][:,1], "o", color="r")
    plt.title("データの異常分類")
    plt.show()





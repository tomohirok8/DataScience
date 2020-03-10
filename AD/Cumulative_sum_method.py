'''
1.正常時の目安μ、上振れの目安ν+を与え、変化度を定義する。
2．変化度の上側累積和を求める。
3.与えられた閾値ath+を超えていたら異常と判定する。
・データが正常であるときに安定してμという数を取る
・そこから上にν+振れている際は異常とする
ことが事前知識として予め分かっている必要があります。これらのデータは過去の異常事例の分析から判断します。
'''
import numpy as np
import matplotlib.pyplot as plt



def Cumulative_sum_method(x):
    # 正常時の平均値mu_x、異常時の振れ幅nu_x
    mu_x = 10
    nu_x = 14
    
    # xの標準偏差を計算してください
    std_x = np.std(x)
    
    # 変化度を計算してください(1行)
    change_score = (nu_x / std_x) * ((x - mu_x - nu_x / 2) / std_x)
    
    # 出力
    plt.plot(range(x.size), change_score)
    plt.grid(True)
    plt.title("時系列データxの変化度")
    plt.show()
    
    
    # 上側累積和を格納する配列を用意してください
    score_cumsum = np.array(change_score)
    
    # for文を用いて上側累積和を求めてください
    for i in range(change_score.size - 1):
        score_cumsum[i] = max(0, score_cumsum[i])
        score_cumsum[i + 1] = score_cumsum[i] + change_score[i + 1]
    
    # 出力
    plt.plot(range(change_score.size), score_cumsum)
    plt.grid(True)
    plt.title("上側累積和")
    plt.show()
    
    
    # 閾値
    threshold = 10
    
    # 閾値を超えているデータのラベルを1、それ以外を0としてラベル付けしてください
    pred = np.asarray(score_cumsum > threshold, dtype="int")
    
    # 初めて閾値を超えた点のインデックスを求めてください
    ind_err = np.arange(score_cumsum.size)[pred == 1][0]
    
    # 出力
    print("変化時点:" + str(ind_err))
    # グラフ描画
    plt.plot(range(550), x)
    # 変化点の縦線を重ねる
    plt.axvline(x=ind_err, color='red', linestyle='--')
    plt.grid(True)
    plt.title("時系列データの変化点")
    plt.show()
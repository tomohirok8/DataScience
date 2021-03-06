import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs
import tensorflow as tf
print('tensorflow version', tf.__version__)
if int(tf.__version__.split('.')[0]) >= 2:
    from tensorflow import keras
else:
    import keras
from keras.datasets import mnist
from keras.utils import to_categorical



def read_data_MNIST():
    # MNISTの手書き数字データセット（学習用と評価用に分割済）
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # MNISTの手書き数字のデータを表示
    fig = plt.figure(figsize=(9, 15))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=0.05)
    for i in range(9):
        ax = fig.add_subplot(1, 9, i + 1, xticks=[], yticks=[])
        # 各MNIST画像の上に（タイトルとして）対応するラベルを表示
        ax.set_title(str(y_train[i]))
        ax.imshow(x_train[i], cmap='gray')
    
    # 名義尺度をone-hot表現に変換
    # 入力画像を行列(28x28)からベクトル(長さ784)に変換
    x_train = x_train.reshape(-1, 784) / 255.
    x_test = x_test.reshape(-1, 784) / 255.
    
    # 名義尺度の値をone-hot表現へ変換
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test
 

    
def read_data_Iris():
    iris = load_iris()
    X = iris.data
    Y = iris.target
    
    iris_data = pd.DataFrame(X,columns=['Sepal Length','Sepal Width','Petal Length','Petal Width'])
    iris_target = pd.DataFrame(Y,columns=['Species'])
    
    df_iris = pd.concat([iris_data,iris_target],axis=1)
    
    # 可視化
    sns.pairplot(df_iris,hue='Species',size=2)
    sns.countplot('Petal Length',data=df_iris,hue='Species')
    
    return df_iris



def make_data_blobs():
    X, Y = make_blobs(n_samples=500, centers=4, random_state=8, cluster_std=2.4)
    
    #　表示
    plt.figure(figsize=(11,7))
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='jet')
    plt.grid()
    plt.show()
    
    return X, Y
    


def read_data_01():
    X = np.loadtxt('data/data_quality.txt', delimiter=',')
    
    # データプロット
    plt.figure(figsize=(11,7))
    plt.title('data01')
    plt.scatter(X[:,0], X[:,1], marker='o', facecolors='none', edgecolors='black', s=80) 
    x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1 
    y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
    plt.xlim(x_min, x_max) 
    plt.ylim(y_min, y_max) 
    plt.xticks(()) 
    plt.yticks(()) 
    plt.show()
    
    return X[:,0], X[:,1]



def read_data_02():
    X = np.loadtxt('data/data_clustering.txt', delimiter=',')
    
    # データプロット
    plt.figure(figsize=(11,7))
    plt.title('data02')
    plt.scatter(X[:,0], X[:,1], marker='o', facecolors='none', edgecolors='black', s=80) 
    x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1 
    y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
    plt.xlim(x_min, x_max) 
    plt.ylim(y_min, y_max) 
    plt.xticks(()) 
    plt.yticks(()) 
    plt.show()
    
    return X[:,0], X[:,1]



def read_data_03():
    data = np.loadtxt('data/knnsample.csv', delimiter=',')
    X = data[:, :2]
    Y = data[:, 2:].reshape(data.shape[0])
    
    # データプロット
    plt.figure(figsize=(11,7))
    plt.title('data03')
    plt.scatter(X[:,0], X[:,1], marker='o', facecolors='none', edgecolors='black', s=80) 
    x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1 
    y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
    plt.xlim(x_min, x_max) 
    plt.ylim(y_min, y_max) 
    plt.xticks(()) 
    plt.yticks(()) 
    plt.show()
    
    return X, Y



def read_data_04():
    data = np.loadtxt('data/OCSVM_sample.csv', delimiter=',')
    
    # データプロット
    plt.figure(figsize=(11,7))
    plt.title('data04')
    plt.scatter(data[:,0], data[:,1], marker='o', facecolors='none', edgecolors='black', s=80) 
    x_min, x_max = data[:, 0].min()-1, data[:, 0].max()+1 
    y_min, y_max = data[:, 1].min()-1, data[:, 1].max()+1
    plt.xlim(x_min, x_max) 
    plt.ylim(y_min, y_max) 
    plt.xticks(()) 
    plt.yticks(()) 
    plt.show()
    
    return data



def read_data_05():
    x = np.loadtxt('data/Cumulative_sum.csv', delimiter=',')
    plt.plot(x)
    
    return x


    
def make_random_sin():
    # ダミーデータ生成
    x_train = 10 * np.random.rand(100)
    # サイン波にノイズをのせる関数
    def sin_model(x, sigma=0.2):
        noise = sigma * np.random.randn(len(x))
        return np.sin(5 * x) + np.sin(0.5 * x) + noise
    
    # xからyを計算
    y_train = sin_model(x_train)
    
    # 表示
    plt.figure(figsize=(11,7))
    plt.errorbar(x_train, y_train, 0.1, fmt='o')
    
    # テスト用データ生成
    x_test = np.linspace(0, 10, 1000)
    y_test = sin_model(x_test, 0)
    
    return x_train, y_train, x_test, y_test
    


def make_random_XY():
    # ダミーデータ生成
    X = 0.3 * np.random.randn(100, 2)
    # 外れ値生成
    ANOMALY_DATA_COUNT = 20
    X_outliers = np.random.uniform(low=-4, high=4, size=(ANOMALY_DATA_COUNT, 2))
    X = np.r_[X + 2, X - 2, X_outliers]
    
    return X[:,0], X[:,1]



def electro_cardiogram():
    data = np.loadtxt("data/qtdbsel102.txt",delimiter="\t")

    return data



def read_imdb():
    imdb = keras.datasets.imdb
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    
    print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
    
    # A dictionary mapping words to an integer index
    word_index = imdb.get_word_index()
    # The first indices are reserved
    word_index = {k:(v+3) for k,v in word_index.items()}
    word_index["<pad>"] = 0
    word_index["<start>"] = 1
    word_index["<unk>"] = 2  # unknown
    word_index["<unused>"] = 3
     
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
     
    def decode_review(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])
    	
    decode_review(train_data[0])
    
    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=word_index["<pad>"],
                                                            padding='post',
                                                            maxlen=256)
     
    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                           value=word_index["<pad>"],
                                                           padding='post',
                                                           maxlen=256)
    
    return train_data, test_data, train_labels, test_labels, reverse_word_index












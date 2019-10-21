import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris
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
    sns.pairplot(iris,hue='Species',size=2)
    sns.countplot('Petal Length',data=iris,hue='Species')
    
    return df_iris
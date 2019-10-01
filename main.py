import os
os.chdir('D:\\GitHub\\datascience')
os.getcwd()
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from DL.basic_DL import basic_DL_2class


############################
####### データの読み込み #######
############################
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

# 2クラス分類とするためラベルデータから1列だけ取り出す
y_train0 = y_train[:,0]
y_test0 = y_test[:,0]


####################
####### 学習 #######
####################
model = basic_DL_2class(x_train, x_test, y_train0, y_test0)


############################
####### モデルの可視化 #######
############################
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(model).create(prog='dot', format='svg'))
from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)


##########################
####### モデルの評価 #######
##########################
score = model.evaluate(x_test, y_test0, verbose=1)


####################
####### 予測 #######
####################
y_pred = model.predict(x_test)















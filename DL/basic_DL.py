import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

from keras.utils import plot_model

############################
####### 2クラス分類 #######
############################
def basic_DL_2class(x_train, x_test, y_train, y_test):
    epochs = 10
    model = Sequential()
    
    ####### モデル構築 #######
    # 多層パーセプトロンを構成する
    model.add(Dense(units=256, input_shape=(x_train.shape[1],),activation='relu'))
    model.add(Dense(units=100,activation='relu'))
    model.add(Dense(units=1,activation='sigmoid'))
    
    # モデルの学習方法
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    # モデルの概要
    model.summary()
    
    ####### 学習 #######
    history = model.fit(x_train, y_train,
                        batch_size=1000,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test)
                        )
    
    # 学習の経過を表示
    loss = history.history
    plt.figure(figsize=(14,8))
    plt.subplot(211)
    plt.plot(loss['val_loss'], label='val loss')
    plt.plot(loss['loss'], label='loss')
    plt.legend()
    plt.xlim(0, epochs)
    plt.ylim(0, 2)
    plt.ylabel('loss')
    plt.grid()
    plt.subplot(212)
    plt.plot(loss['val_acc'], label='val acc')
    plt.plot(loss['acc'], label='acc')
    plt.legend()
    plt.xlim(0, epochs)
    plt.ylim(0, 1)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()
    plt.show()
    
    # モデルの可視化
    plot_model(model, to_file='model.png', show_shapes=True)
    
    return model

    
    



















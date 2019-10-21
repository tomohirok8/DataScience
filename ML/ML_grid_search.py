from sklearn.svm import SVC
import numpy as np


def grid_search_SVM_RBF(x_train, y_train, x_test, y_test):
    param_list_gamma = list(np.arange(0,10,0.001))
    param_list_C = list(np.arange(0,100,0.1))
    
    best_score = 0
    best_parameters = {}
    for gamma in param_list_gamma:
        for C in param_list_C:
            model = SVC(kernel='rbf', gamma=gamma, C=C)
            model.fit(x_train, y_train)
            score = model.score(x_test, y_test)
            print('gamma = ', gamma, 'C = ', C)
            print('score = ', score)
            # 最も良いスコアのパラメータとスコアを更新
            if score > best_score:
                best_score = score
                best_parameters = {'gamma' : gamma, 'C' : C}
    
    print('Best score: {}'.format(best_score))
    print('Best parameters: {}'.format(best_parameters))






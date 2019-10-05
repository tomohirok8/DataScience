from sklearn.svm import SVC


def grid_search_SVM(x_train, y_train, x_test, y_test):
    param_list = [0.001, 0.01, 0.1, 1, 10, 100]
    
    best_score = 0
    best_parameters = {}
    for gamma in param_list:
        for C in param_list:
            svm = SVC(gamma=gamma, C=C)
            svm.fit(x_train, y_train)
            score = svm.score(x_test, y_test)
            print('gamma = ', gamma, 'C = ', C)
            print('score = ', score)
            # 最も良いスコアのパラメータとスコアを更新
            if score > best_score:
                best_score = score
                best_parameters = {'gamma' : gamma, 'C' : C}
    
    print('Best score: {}'.format(best_score))
    print('Best parameters: {}'.format(best_parameters))






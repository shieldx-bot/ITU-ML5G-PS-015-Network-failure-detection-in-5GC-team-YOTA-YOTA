from sklearn import svm
import os
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report



def main():
#     ####データの合成なし
#     X_train = np.loadtxt('/ap_data/01_c_train_data.txt')
#     y_train = np.loadtxt('/ap_data/01_c_train_label.txt', dtype='int64')

    X_test = np.loadtxt('../ap_data/01_a_test_data.txt')
    y_test = np.loadtxt('../ap_data/01_a_test_label.txt', dtype='int64')

    #データの合成あり
    X_train_a = np.loadtxt('../ap_data/01_a_train_data.txt')
    y_train_a = np.loadtxt('../ap_data/01_a_train_label.txt', dtype='int64')

    X_test_a = np.loadtxt('../ap_data/01_a_test_data.txt')
    y_test_a = np.loadtxt('../ap_data/01_a_test_label.txt', dtype='int64')

    X_train_c = np.loadtxt('../ap_data/01_c_train_data.txt')
    y_train_c = np.loadtxt('../ap_data/01_c_train_label.txt', dtype='int64')

    X_test_c = np.loadtxt('../ap_data/01_c_test_data.txt')
    y_test_c = np.loadtxt('../ap_data/01_c_test_label.txt', dtype='int64')

    X_train = np.concatenate([X_train_a, X_train_c])
    y_train = np.concatenate([y_train_a, y_train_c])

#     X_test = np.concatenate([X_test_a, X_test_c])
#     y_test = np.concatenate([y_test_a, y_test_c])


    fin_xgboost = svm.SVC()


    # モデル訓練
    fin_xgboost.fit(X_train, y_train,verbose=True)

    # テストデータで推測値を算出
    fin_test_pred = fin_xgboost.predict(X_test)

    # 混同行列で確認
    confusion_matrix(y_test, fin_test_pred, labels=[1, 0])

    print(accuracy_score(y_test, fin_test_pred))
    print(classification_report(y_test, fin_test_pred))

    print(confusion_matrix(y_test,  fin_test_pred))


if __name__ == '__main__':
    main()

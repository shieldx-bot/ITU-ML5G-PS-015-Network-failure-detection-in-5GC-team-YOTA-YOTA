from sklearn import svm
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import time




def main():
 # ===== LOAD TOÀN BỘ DỮ LIỆU =====
    print("Loading data...")
    # =========================
    # LOAD TRAIN DATA
    # =========================

    # Dataset A
    X_train_a = np.loadtxt('/kaggle/input/datasets/nguyenvananhxa/data-faults/01_a_train_data.txt')
    y_train_a = np.loadtxt(
        '/kaggle/input/datasets/nguyenvananhxa/data-faults/01_a_train_label.txt',
        dtype='int64'
    )

    # Dataset B
    X_train_b = np.loadtxt('/kaggle/input/datasets/nguyenvananhxa/data-faults/02_b_train_data.txt')
    y_train_b = np.loadtxt(
        '/kaggle/input/datasets/nguyenvananhxa/data-faults/02_b_train_label.txt',
        dtype='int64'
    )

    # Dataset C
    X_train_c = np.loadtxt('/kaggle/input/datasets/nguyenvananhxa/data-faults/01_c_train_data.txt')
    y_train_c = np.loadtxt(
        '/kaggle/input/datasets/nguyenvananhxa/data-faults/01_c_train_label.txt',
        dtype='int64'
    )

    # Merge train datasets
    X_train = np.concatenate([
        X_train_a,
        X_train_b,
        X_train_c
    ])

    y_train = np.concatenate([
        y_train_a,
        y_train_b,
        y_train_c
    ])

    # =========================
    # LOAD TEST DATA
    # =========================

    # Dataset A
    X_test_a = np.loadtxt('/kaggle/input/datasets/nguyenvananhxa/data-faults/01_a_test_data.txt')
    y_test_a = np.loadtxt(
        '/kaggle/input/datasets/nguyenvananhxa/data-faults/01_a_test_label.txt',
        dtype='int64'
    )

    # Dataset B
    X_test_b = np.loadtxt('/kaggle/input/datasets/nguyenvananhxa/data-faults/02_b_test_data.txt')
    y_test_b = np.loadtxt(
        '/kaggle/input/datasets/nguyenvananhxa/data-faults/02_b_test_label.txt',
        dtype='int64'
    )

    # Dataset C
    X_test_c = np.loadtxt('/kaggle/input/datasets/nguyenvananhxa/data-faults/01_c_test_data.txt')
    y_test_c = np.loadtxt(
        '/kaggle/input/datasets/nguyenvananhxa/data-faults/01_c_test_label.txt',
        dtype='int64'
    )

    # Merge test datasets
    X_test = np.concatenate([
        X_test_a,
        X_test_b,
        X_test_c
    ])

    y_test = np.concatenate([
        y_test_a,
        y_test_b,
        y_test_c
    ])


    fin_xgboost = KNeighborsClassifier(n_neighbors=3)

    print(f"Training samples: {len(y_train)}") 
    print(f"Test samples: {len(y_test)}")

    # モデル訓練
    start_train = time.time()
    fin_xgboost.fit(X_train, y_train)
    train_time = time.time() - start_train
    # テストデータで推測値を算出

    start_pred = time.time()
    fin_test_pred = fin_xgboost.predict(X_test)
    predict_time = time.time() - start_pred

    # 混同行列で確認
    confusion_matrix(y_test, fin_test_pred, labels=[1, 0])

    print(accuracy_score(y_test, fin_test_pred))
    print(classification_report(y_test, fin_test_pred))

    print(confusion_matrix(y_test,  fin_test_pred))
    acc = accuracy_score(y_test, fin_test_pred)

    report = classification_report(y_test, fin_test_pred)

    cm = confusion_matrix(y_test, fin_test_pred)

    with open("Logistic Regression.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")

        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n")

        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\n")

        f.write(f"Training time: {train_time:.2f} seconds\n")
        f.write(f"Prediction time: {predict_time:.2f} seconds\n")


if __name__ == '__main__':
    main()

from sklearn.ensemble import RandomForestClassifier
import os
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import time
 


def main():
    # =========================
    # LOAD TRAIN DATA
    # =========================

    # Dataset A
    X_train_a = np.loadtxt('../ap_data/01_a_train_data.txt')
    y_train_a = np.loadtxt(
        '../ap_data/01_a_train_label.txt',
        dtype='int64'
    )

    # Dataset B
    X_train_b = np.loadtxt('../ap_data/02_b_train_data.txt')
    y_train_b = np.loadtxt(
        '../ap_data/02_b_train_label.txt',
        dtype='int64'
    )

    # Dataset C
    X_train_c = np.loadtxt('../ap_data/01_c_train_data.txt')
    y_train_c = np.loadtxt(
        '../ap_data/01_c_train_label.txt',
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
    X_test_a = np.loadtxt('../ap_data/01_a_test_data.txt')
    y_test_a = np.loadtxt(
        '../ap_data/01_a_test_label.txt',
        dtype='int64'
    )

    # Dataset B
    X_test_b = np.loadtxt('../ap_data/02_b_test_data.txt')
    y_test_b = np.loadtxt(
        '../ap_data/02_b_test_label.txt',
        dtype='int64'
    )

    # Dataset C
    X_test_c = np.loadtxt('../ap_data/01_c_test_data.txt')
    y_test_c = np.loadtxt(
        '../ap_data/01_c_test_label.txt',
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

    print("Length X_train:", len(X_train));
    print("Length X_test:", len(X_test));


    start = time.time()

    fin_xgboost = RandomForestClassifier(    
    n_estimators=500,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
        )


    # モデル訓練
    fin_xgboost.fit(X_train, y_train)
    end = time.time()
    train_time = end - start
    print("train_time:", train_time)
 
    # テストデータで推測値を算出

    start_pred = time.time()
    fin_test_pred = fin_xgboost.predict(X_test)
    end_pred = time.time(); 
    predict_time = end_pred - start_pred

    print("predict_time:", predict_time)

    # 混同行列で確認
    confusion_matrix(y_test, fin_test_pred, labels=[1, 0])

    print(accuracy_score(y_test, fin_test_pred))
    print(classification_report(y_test, fin_test_pred))

    print(confusion_matrix(y_test,  fin_test_pred))
    acc = accuracy_score(y_test, fin_test_pred)

    report = classification_report(y_test, fin_test_pred)

    cm = confusion_matrix(y_test, fin_test_pred)

    with open("RandomForest.txt", "w") as f:
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

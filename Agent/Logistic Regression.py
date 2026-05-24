import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report





def main():
    # ===== LOAD TOÀN BỘ DỮ LIỆU =====
    print("Loading data...")

    # helper to try several candidate data directories (relative to this script)
    base = os.path.dirname(os.path.abspath(__file__))
    candidate_dirs = [
        os.path.join(base, '..', 'ap_data'),
        os.path.join(base, 'ap_data'),
        os.path.join(base, '..', '..', 'ap_data'),
        os.path.join(os.getcwd(), 'ap_data'),
    ]

    def find_file(name):
        for d in candidate_dirs:
            p = os.path.abspath(os.path.join(d, name))
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Could not find {name} in candidate dirs: {candidate_dirs}")

    X_train_a = np.loadtxt(find_file('01_a_train_data.txt'))
    y_train_a = np.loadtxt(find_file('01_a_train_label.txt'), dtype='int64')

    X_train_c = np.loadtxt(find_file('01_c_train_data.txt'))
    y_train_c = np.loadtxt(find_file('01_c_train_label.txt'), dtype='int64')

    X_test = np.loadtxt(find_file('01_a_test_data.txt'))
    y_test = np.loadtxt(find_file('01_a_test_label.txt'), dtype='int64')

    X_train = np.concatenate([X_train_a, X_train_c])
    y_train = np.concatenate([y_train_a, y_train_c])
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")

    # ===== TẠO DATALOADER (KHÔNG GIỚI HẠN) =====
    BATCH_SIZE = 512  # Điều chỉnh: 128 nếu vẫn OOM, 512 nếu muốn nhanh

    # Use numpy arrays with scikit-learn (no torch required)
    X_train_used = X_train.astype(np.float32)
    y_train_used = y_train.ravel()  # ensure 1-D labels

    clf = LogisticRegression(random_state=0, max_iter=1000)
    clf.fit(X_train_used[0:5000], y_train_used[0:5000])
    # prepare test data and evaluate
    X_test_used = X_test.astype(np.float32)
    y_test_used = y_test.ravel()
    preds = clf.predict(X_test_used[:10])
    mae = np.mean(np.abs(preds - y_test_used[:10]))

    print("Predictions (first 10):", mae)
    print(f"Test accuracy: {clf.score(X_test_used, y_test_used):.4f}")



if __name__ == '__main__':
    main()

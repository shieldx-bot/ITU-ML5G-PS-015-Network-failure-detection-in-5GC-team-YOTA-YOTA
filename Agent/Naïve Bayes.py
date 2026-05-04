import os
import numpy as np
from sklearn.naive_bayes import GaussianNB




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

  
    # Use numpy arrays with scikit-learn (no torch required)
    X_train_used = X_train.astype(np.float32)
    y_train_used = y_train.ravel()  # ensure 1-D labels
    X_test_used = X_test.astype(np.float32)
    y_test_used = y_test.ravel()
    gnb = GaussianNB()
    gnb.fit(X_train[0:10000], y_train[0:10000])
    y_pred = gnb.predict(X_test_used[:100])
    mae = np.mean(np.abs(y_pred - y_test_used[:100]))
    print("mae = ", mae)

if __name__ == "__main__":
    main()

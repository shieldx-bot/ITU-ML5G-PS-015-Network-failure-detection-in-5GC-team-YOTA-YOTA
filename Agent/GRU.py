import json
import pandas as pd
from pathlib import Path
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report




print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
input_size=28
sequence_length =28
num_layers=2
hidden_size=256

learning_rate = 0.001
num_epochs = 5

num_classes =10
batch_size = 64
class SimpleGRU(nn.Module):
    def __init__(self, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes, sequence_length=sequence_length):
        super(SimpleGRU, self).__init__()
        self.hidden_size  = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out,_ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.fc1(out)
        return out

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

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor[:5000], y_train_tensor[:5000])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Number of batches per epoch: {len(train_loader)}")
    # ====================================

    model = SimpleGRU(input_size=1, hidden_size=100, num_layers=1, num_classes=num_classes, sequence_length=sequence_length)
    model = model.to(device)

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 100

    # ===== HUẤN LUYỆN VỚI TOÀN BỘ DỮ LIỆU =====
    current_loss = 0
    for epoch in range(num_epochs):
        for data, target in train_loader:
            data = data.to(device=device)
            target = target.to(device=device).long().squeeze(-1)

            score = model(data)
            loss = loss_criterion(score, target)
            current_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
        print(f"At epoch: {epoch}, loss: {current_loss}")

    # ===== DỰ ĐOÁN TRÊN TEST =====
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Tạo DataLoader cho test để tránh OOM
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            pred = model(batch_X)
            all_predictions.append(pred.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    predicted_logits = np.concatenate(all_predictions)
    y_test_np = np.concatenate(all_targets).flatten().astype(np.int64)

    predicted_classes = np.argmax(predicted_logits, axis=1)

      print(accuracy_score(y_test_np, predicted_classes))
      print(classification_report(y_test_np, predicted_classes))

      print(confusion_matrix(y_test_np,  predicted_classes))
    # =============================

if __name__ == '__main__':
    main()

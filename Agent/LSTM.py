import json
import pandas as pd
from pathlib import Path
import os
import torch
import torch.nn as nn 
from torch.utils.data import TensorDataset, DataLoader
import numpy as np 
import matplotlib.pyplot as plt

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class LSTMModel(nn.Module): 
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        
    def forward(self, x, h0=None, c0=None): 
        batch_size = x.size(0)
        if h0 is None or c0 is None or h0.size(1) != batch_size or c0.size(1) != batch_size:
            h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim, device=x.device)
            c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim, device=x.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out, hn, cn 

def main(): 
    # ===== LOAD TOÀN BỘ DỮ LIỆU =====
    print("Loading data...")
    X_train_a = np.loadtxt('01_a_train_data.txt')
    y_train_a = np.loadtxt('01_a_train_label.txt', dtype='int64')
    
    X_train_c = np.loadtxt('01_c_train_data.txt')
    y_train_c = np.loadtxt('01_c_train_label.txt', dtype='int64')
    
    X_test = np.loadtxt('01_a_test_data.txt')
    y_test = np.loadtxt('01_a_test_label.txt', dtype='int64')
    
    X_train = np.concatenate([X_train_a, X_train_c])
    y_train = np.concatenate([y_train_a, y_train_c])
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")

    # ===== TẠO DATALOADER (KHÔNG GIỚI HẠN) =====
    BATCH_SIZE = 512  # Điều chỉnh: 128 nếu vẫn OOM, 512 nếu muốn nhanh
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Number of batches per epoch: {len(train_loader)}")
    # ====================================

    model = LSTMModel(input_dim=1, hidden_dim=100, layer_dim=1, output_dim=1)
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 1000
    
    # ===== HUẤN LUYỆN VỚI TOÀN BỘ DỮ LIỆU =====
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs, _, _ = model(batch_X)  # Mỗi batch reset hidden state
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}')
    # ===========================================

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
            pred, _, _ = model(batch_X)
            all_predictions.append(pred.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    predicted_np = np.concatenate(all_predictions).flatten()
    y_test_np = np.concatenate(all_targets).flatten()
    
    mae = np.mean(np.abs(predicted_np - y_test_np))
    print(f"\nMAE: {mae:.4f}")
    print(f"Predictions range: [{predicted_np.min():.4f}, {predicted_np.max():.4f}]")
    print(f"First 10 predictions: {predicted_np[:10]}")
    # =============================

if __name__ == '__main__':
    main()
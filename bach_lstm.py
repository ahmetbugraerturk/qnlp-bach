import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm # Progress bar for cluster logs

# 1. Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Classical model training on: {device}")

# 2. Dataset Preparation (Assuming X_padded and y_tensor are ready from previous step)
dataset = TensorDataset(X_padded, y_tensor)
# Increase num_workers and pin_memory for faster cluster data loading
train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

class BachLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(BachLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        out = self.fc(hidden[-1])
        return out

# 3. Model Initialization on CUDA
model = BachLSTM(vocab_size, embed_dim=16, hidden_dim=64).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. Training Loop
def train_classical(epochs=50):
    model.train()
    for epoch in range(epochs):
        loop = tqdm(train_loader, leave=True)
        for batch_idx, (data, target) in enumerate(loop):
            # Move tensors to GPU
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

train_classical()
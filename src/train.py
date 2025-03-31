import torch
import torch.nn as nn
import torch.optim as optim
from model import BeeClassifierCNN
from dataset import get_data_loaders

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--data_dir', type=str, default='data')
args = parser.parse_args()

train_loader, val_loader = get_data_loaders(args.data_dir)

model = BeeClassifierCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {total_loss:.4f}")


torch.save(model.state_dict(), 'model.pth')
print("✅ Model saved to model.pth")

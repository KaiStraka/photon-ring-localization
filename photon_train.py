# KAI STRAKA 2026

from photonModel import photonUNET
from photonModel import photonDataset

import glob
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime

train_data = r"C:\Users\Kai\Desktop\astroAI\blackholeML\model\train"
image_paths = glob.glob(os.path.join(train_data, "*.npz"))

train_dataset = photonDataset(image_paths)
train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)

model = photonUNET()

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

start_time = datetime.now()

epochs = 5     # number of cycles

for epoch in range(epochs):

    model.train()
    training_loss = 0.0

    for degraded, mask in train_loader:

        optimizer.zero_grad()
        prediction = model(degraded)            # predict photon ring mask
        loss = criterion(prediction, mask)

        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        print(".")

    training_loss /= len(train_loader)
    print(f"Epoch {epoch + 1}: " f"Loss = {training_loss:.4}")

torch.save(model.state_dict(), "photon_model.pth")
print("training complete.")
print("start time: ", start_time)
print("finish time: ", datetime.now())
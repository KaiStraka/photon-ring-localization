# KAI STRAKA 2026

from photonModel import photonUNET
from photonModel import photonDataset

import glob
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime

val_data = r"C:\Users\Kai\Desktop\astroAI\blackholeML\model\val"
image_paths = glob.glob(os.path.join(val_data, "*.npz"))

val_dataset = photonDataset(image_paths)
val_loader = DataLoader(val_dataset, batch_size = 16, shuffle = True)

model = photonUNET()

model.load_state_dict(torch.load("photon_model.pth"))
criterion = nn.BCEWithLogitsLoss()

start_time = datetime.now()

model.eval()

val_loss = 0.0
val_correct = 0
val_total = 0

with torch.no_grad():

    for degraded, mask in val_loader:

        prediction = model(degraded)        # predict photon ring mask
        loss = criterion(prediction, mask)
        val_loss += loss.item()

        probability = torch.sigmoid(prediction)     # convert logit to percentages
        predicted_mask = (probability >= 0.5).float()       # 0.5 since mask pixels are binary
        val_correct += (predicted_mask == mask).sum().item()
        val_total += mask.numel()
        print(".")

val_loss /= len(val_loader)
val_accuracy = val_correct / val_total

print("validation complete.")
print("start time: ", start_time)
print("finish time: ", datetime.now())
print(f"average loss: {val_loss:.4f}")
print(f"accuracy: {val_accuracy:.4f}")
print(f"correct: {val_correct} / {val_total}")
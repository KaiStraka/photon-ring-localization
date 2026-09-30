# KAI STRAKA 2026

from photonModel import photonUNET
from photonModel import photonDataset

import glob
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime


val_data = r"C:\Users\Kai\Desktop\astroAI\blackholeML\model\val"            # load validation data
val_paths = glob.glob(os.path.join(val_data, "*.npz"))
val_dataset = photonDataset(val_paths)
val_loader = DataLoader(val_dataset, batch_size = 16, shuffle = False)

test_data = r"C:\Users\Kai\Desktop\astroAI\blackholeML\model\test"          # load test data
test_paths = glob.glob(os.path.join(test_data, "*.npz"))
test_dataset = photonDataset(test_paths)
test_loader = DataLoader(test_dataset, batch_size = 16, shuffle = False)

model = photonUNET()        # load model
model.load_state_dict(torch.load("photon_model.pth"))       # load trained model

criterion = nn.BCEWithLogitsLoss()      # loss function

model.eval()        # set model to evaluation mode


def val_loop():     # validation function

    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_start_time = datetime.now()

    with torch.no_grad():       # validation

        for degraded, mask in val_loader:       # validation loop

            prediction = model(degraded)        # predict photon ring mask
            loss = criterion(prediction, mask)
            val_loss += loss.item()

            probability = torch.sigmoid(prediction)             # convert logit to percentages
            predicted_mask = (probability >= 0.5).float()       # 0.5 since mask pixels are binary
            val_correct += (predicted_mask == mask).sum().item()
            val_total += mask.numel()           # num of elements
            print(".")

    val_loss /= len(val_loader)
    val_accuracy = val_correct / val_total
    val_end_time = datetime.now()

    return val_start_time, val_end_time, val_loss, val_accuracy, val_correct, val_total


def test_loop():    # testing function

    test_loss = 0.0
    test_correct = 0
    test_total = 0
    test_start_time = datetime.now()

    with torch.no_grad():

        for degraded, mask in test_loader:      # testing loop

            prediction = model(degraded)        # predict photon ring mask
            loss = criterion(prediction, mask)
            test_loss += loss.item()

            probability = torch.sigmoid(prediction)             # convert logit to percentages
            predicted_mask = (probability >= 0.5).float()       # 0.5 since mask pixels are binary
            test_correct += (predicted_mask == mask).sum().item()
            test_total += mask.numel()          # num of elements
            print(".")

    test_loss /= len(test_loader)
    test_accuracy = test_correct / test_total
    test_end_time = datetime.now()

    return test_start_time, test_end_time, test_loss, test_accuracy, test_correct, test_total


def finished_info(name, x_start_time, x_end_time, x_loss, x_accuracy, x_correct, x_total):      # print complete loop info

    print(f"{name} complete.")
    print("start time: ", x_start_time)
    print("finish time: ", x_end_time)
    print(f"loss: {x_loss:.4f}")
    print(f"accuracy: {x_accuracy:.4f}")
    print(f"correct: {x_correct} / {x_total}")


val_start_time, val_end_time, val_loss, val_accuracy, val_correct, val_total = val_loop()               # run validation loop
test_start_time, test_end_time, test_loss, test_accuracy, test_correct, test_total = test_loop()        # run testing loop

finished_info("validation", val_start_time, val_end_time, val_loss, val_accuracy, val_correct, val_total)       # print validation info
finished_info("test", test_start_time, test_end_time, test_loss, test_accuracy, test_correct, test_total)       # print testing info
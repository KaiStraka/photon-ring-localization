# KAI STRAKA 2026

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

from photonModel import photonUNET
import photonDegrade
import photonImage


image = r"C:\Users\Kai\Desktop\astroAI\blackholeML\model\localize\zj74lfymz0.npz"       # image path

model = photonUNET()        # load model
model.load_state_dict(torch.load("photon_model.pth"))       # load trained model
model.eval()                # set model to evaluation mode

img_clean = photonImage.pixel_list(image)               # load clean simulated image
img_degraded = photonDegrade.degraded_image(image)      # load degraded image

mask_clean = photonImage.ring_mask(img_clean)           # load clean mask
degraded_tensor = photonImage.pixel_tensor(img_degraded)

with torch.no_grad():       # predict degraded mask

    prediction = model(degraded_tensor.unsqueeze(0))
    probability = torch.sigmoid(prediction)

probabilities_np = probability.squeeze().cpu().numpy()

predicted_mask = photonImage.ring_mask(probabilities_np)


def display_img(clean, clean_mask, degraded, degraded_mask, cmap):

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 4, 1)        # clean image
    plt.imshow(clean, origin = "lower", cmap = cmap)
    plt.colorbar(label = "Intensity")
    plt.title("Clean")

    plt.subplot(1, 4, 2)        # clean mask
    plt.imshow(clean_mask, origin = "lower", cmap = cmap)
    plt.colorbar(label = "Intensity")
    plt.title("Clean Mask")

    plt.subplot(1, 4, 3)        # degraded image
    plt.imshow(degraded, origin = "lower", cmap = cmap)
    plt.colorbar(label = "Intensity")
    plt.title("Gauss + Scatter")

    plt.subplot(1, 4, 4)        # degraded mask
    plt.imshow(degraded_mask, origin = "lower", cmap = cmap)
    plt.colorbar(label = "Intensity")
    plt.title("Predicted Mask")

    plt.tight_layout()
    plt.show()


display_img(img_clean, mask_clean, img_degraded, predicted_mask, "afmhot")
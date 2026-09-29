# KAI STRAKA 2026

import numpy as np
import torch
import matplotlib.pyplot as plt


def pixel_list(image):          # generate pixel list | use to load image

    data = np.load(image)
    pixel_values = data["I"].astype(np.float64)         # extract pixel values
    pixel_values /= pixel_values.max()      # normalize list

    return pixel_values


def pixel_tensor(image):        # generate pixel tensor

    pixels = torch.tensor(image, dtype = torch.float32)      # convert to tensor
    pixels = pixels.unsqueeze(0)        # match torch dimension

    return pixels


def ring_mask(pixels):       # generate ring mask

    threshold = np.percentile(pixels, 99.4)      # find nth percentile brightest pixels
    mask = np.zeros_like(pixels, dtype = np.float32)
    mask[pixels >= threshold] = 1.0

    return mask


def display_img():      # plot images | internal testing only

    image_path = r"C:\Users\Kai\Desktop\astroAI\blackholeML\model\train\0b4wnw3wlz.npz"

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(pixel_list(image_path), origin = "lower", cmap = "gray")
    plt.colorbar(label = "Intensity")
    plt.title("Image")

    plt.subplot(1, 2, 2)
    plt.imshow(ring_mask(pixel_list(image_path)), origin = "lower", cmap = "gray")
    plt.colorbar(label = "")
    plt.title("Ring Mask")

    plt.tight_layout()
    plt.show()


# display_img()